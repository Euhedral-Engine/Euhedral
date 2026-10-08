package io.euhedral_execution.inference.core.model.qwen38.prefix;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model.qwen38.AttentionStates;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionFixtures;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.GdnStates;
import io.euhedral_execution.inference.core.model.qwen38.HeldWork;
import io.euhedral_execution.inference.core.model.qwen38.MemoryGpu;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.model.qwen38.speculative.MtpCheckpoint;
import io.euhedral_execution.inference.core.model.qwen38.speculative.SpeculativeCheckpoint;
import io.euhedral_execution.inference.core.prefix.PrefixNode;
import io.euhedral_execution.inference.core.runtime.graph.FrameLake;
import io.euhedral_execution.inference.core.runtime.graph.FrameSeeds;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

@Timeout(10)
class PrefixCacheTest {
    private static final Qwen38Config CONFIG =
            ExecutionFixtures.statefulCompactWeights(8).config();

    /// A lake driven by hand: published frames wait until [#drive] runs them, in arrival order, and every frame
    /// ever published is kept for the test to inspect.
    static final class Frames implements FrameLake {
        final ConcurrentLinkedQueue<AbstractFrame> ready = new ConcurrentLinkedQueue<>();
        final List<AbstractFrame> published = new ArrayList<>();
        /// Runs before each frame (a test's interference, such as a cancellation).
        Runnable beforeEach = () -> {};
        /// Frames the lake refuses to take (a closing lake, a full partition).
        java.util.function.Predicate<AbstractFrame> refuse = frame -> false;
        /// Frames the lattice rejects without running them (their worker retired).
        java.util.function.Predicate<AbstractFrame> reject = frame -> false;

        @Override
        public synchronized void publish(AbstractFrame frame) {
            if (this.refuse.test(frame)) throw new IllegalStateException("the inference lake is closed");
            this.published.add(frame);
            this.ready.add(frame);
        }

        @Override
        public void publishFromCallback(AbstractFrame frame) {
            publish(frame);
        }

        @Override
        public void admit() {}

        @Override
        public void admitDuringDrain() {}

        @Override
        public void terminated() {}

        void drive() {
            for (AbstractFrame frame; (frame = this.ready.poll()) != null; ) {
                this.beforeEach.run();
                if (this.reject.test(frame)) {
                    frame.doFinallyWithError(new java.util.concurrent.RejectedExecutionException("no worker"));
                    continue;
                }
                try {
                    frame.execute();
                    frame.doFinally();
                } catch (Throwable failure) {
                    frame.doFinallyWithError(failure);
                }
            }
        }
    }

    /// What the cache told one call, and whether the frame it continued with ran.
    static final class Outcome extends AbstractFrame implements PrefixCache.Steps {
        PrefixCache.Hit hit;
        Boolean restored;
        PrefixNode captured;
        Throwable failure;
        boolean continued;

        Outcome() {
            super(FrameSeeds.ID_HASH);
        }

        @Override
        public void found(PrefixCache.Hit hit) {
            this.hit = hit;
        }

        @Override
        public void restored(boolean restored, Throwable failure) {
            this.restored = restored;
            this.failure = failure;
        }

        @Override
        public void captured(PrefixNode node, Throwable failure) {
            this.captured = node;
            this.failure = failure;
        }

        @Override
        public void execute() {
            this.continued = true;
        }
    }

    /// The stream the next cache opens: the GPU's own (inline) unless a test holds its boundaries.
    private GpuStream cacheStream;
    /// Device-to-host copies left before one fails to queue; negative: none fails.
    private int copiesBeforeFailure = -1;

    private final MemoryGpu gpu = new MemoryGpu() {
        @Override
        public GpuStream openStream() {
            return cacheStream != null ? cacheStream : super.openStream();
        }

        @Override
        public void copyDeviceToHostAsync(long hostAddress, long deviceAddress, long bytes) {
            if (copiesBeforeFailure == 0) throw new IllegalStateException("the copy could not be queued");
            if (copiesBeforeFailure > 0) copiesBeforeFailure--;
            super.copyDeviceToHostAsync(hostAddress, deviceAddress, bytes);
        }
    };
    private final Frames frames = new Frames();

    private PrefixCache cache(long bytes, int interval) {
        Arena arena = Arena.ofShared();
        return new PrefixCache(this.gpu, this.frames, CONFIG, arena.allocate(bytes), arena::close, interval);
    }

    private PrefixNode capture(PrefixCache cache, Sequence sequence, PrefixNode parent, int[] tokens, int position) {
        return capture(cache, sequence, parent, tokens, position, null);
    }

    private PrefixNode capture(
            PrefixCache cache,
            Sequence sequence,
            PrefixNode parent,
            int[] tokens,
            int position,
            SpeculativeCheckpoint speculative) {
        var outcome = new Outcome();
        cache.capture(sequence, parent, tokens, position, speculative, outcome, outcome);
        this.frames.drive();
        assertTrue(outcome.continued, "the capture threw its next frame");
        assertNull(outcome.failure);
        return outcome.captured;
    }

    private PrefixCache.Hit lookup(PrefixCache cache, int[] prompt) {
        return lookup(cache, prompt, null);
    }

    private PrefixCache.Hit lookup(PrefixCache cache, int[] prompt, String speculation) {
        var outcome = new Outcome();
        cache.lookup(prompt, speculation, outcome, outcome);
        this.frames.drive();
        assertTrue(outcome.continued, "the lookup threw its next frame");
        return outcome.hit;
    }

    private Outcome restore(PrefixCache cache, Sequence target, PrefixCache.Hit hit) {
        return restore(cache, target, hit, null);
    }

    private Outcome restore(PrefixCache cache, Sequence target, PrefixCache.Hit hit, SpeculativeCheckpoint spec) {
        var outcome = new Outcome();
        cache.restore(plan(), target, hit, spec, outcome, outcome);
        this.frames.drive();
        assertTrue(outcome.continued, "the restore threw its next frame");
        return outcome;
    }

    private static boolean restored(Outcome outcome) {
        assertNull(outcome.failure);
        return outcome.restored;
    }

    private ExecutionPlan plan() {
        return new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
    }

    /// A sequence with `rows` committed rows and recognizable state, as a prefill leaves it.
    private Sequence sequence(int rows, int seed) {
        var sequence = new Sequence(1);
        var held = HeldWork.admit(sequence);
        Quantum.attachSequenceState(plan(), sequence, this.gpu);
        var attention = (AttentionStates) sequence.kvCacheState();
        var gdn = (GdnStates) sequence.recurrentState();
        AttentionKvState kv = attention.forLayer(1);
        kv.prepareAppend(0, rows);
        kv.appendSubmitted(rows);
        kv.commitSubmitted();
        this.gpu.fill(
                gdn.forLayer(0).recurrentStateAddress(), (int) gdn.forLayer(0).recurrentBytes(), seed);
        this.gpu.fill(
                gdn.forLayer(0).convolutionStateAddress(), (int) gdn.forLayer(0).convolutionBytes(), seed + 1);
        for (int i = 0; i < kv.pageAddresses().size(); i++)
            this.gpu.fill(kv.pageAddresses().get(i), (int) (2 * kv.planePageBytes()), seed + 2 + i);
        held.commit(rows);
        return sequence;
    }

    @Test
    void capturesACheckpointAndRestoresItIntoAFreshSequence() {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence source = sequence(1024, 4);
        PrefixNode node = capture(cache, source, cache.root(), tokens, 1024);
        assertEquals(1024, node.position());
        assertEquals(1, cache.stats().captured());

        var hit = lookup(cache, tokens);
        assertNotNull(hit);
        assertEquals(1024, hit.position());
        assertSame(node, hit.cursor());

        var target = new Sequence(2);
        assertTrue(restored(restore(cache, target, hit)));
        assertEquals(1024, target.currentTokenPosition());
        assertFalse(target.inFlight());
        var restoredKv = ((AttentionStates) target.kvCacheState()).forLayer(1);
        var sourceKv = ((AttentionStates) source.kvCacheState()).forLayer(1);
        assertEquals(1024, restoredKv.length());
        int pageBytes = (int) (2 * sourceKv.planePageBytes());
        for (int page = 0; page < 4; page++)
            assertArrayEquals(
                    this.gpu.bytes(sourceKv.pageAddresses().get(page), pageBytes),
                    this.gpu.bytes(restoredKv.pageAddresses().get(page), pageBytes),
                    "page " + page);
        var sourceGdn = ((GdnStates) source.recurrentState()).forLayer(0);
        var restoredGdn = ((GdnStates) target.recurrentState()).forLayer(0);
        assertArrayEquals(
                this.gpu.bytes(sourceGdn.recurrentStateAddress(), (int) sourceGdn.recurrentBytes()),
                this.gpu.bytes(restoredGdn.recurrentStateAddress(), (int) restoredGdn.recurrentBytes()));
        assertEquals(1, cache.stats().hits());
        assertEquals(1024, cache.stats().reusedTokens());
    }

    @Test
    void aSecondCheckpointExtendsTheFirstAndRestoresBothChainsPages() {
        PrefixCache cache = cache(16L << 20, 512);
        int[] tokens = IntStream.range(0, 1400).toArray();
        Sequence source = sequence(1280, 4);
        PrefixNode first = capture(cache, source, cache.root(), tokens, 512);
        PrefixNode second = capture(cache, source, first, tokens, 1280);
        assertSame(first, second.parent());
        var hit = lookup(cache, tokens);
        assertEquals(1280, hit.position());
        var target = new Sequence(2);
        assertTrue(restored(restore(cache, target, hit)));
        assertEquals(1280, target.currentTokenPosition());
    }

    @Test
    void anExistingNodeIsNotCapturedTwice() {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence source = sequence(1024, 4);
        PrefixNode first = capture(cache, source, cache.root(), tokens, 1024);
        PrefixNode second = capture(cache, source, cache.root(), tokens, 1024);
        assertSame(first, second);
        assertEquals(1, cache.stats().captured());
    }

    @Test
    void aBudgetSmallerThanACheckpointSkipsTheCaptureAndCountsIt() {
        PrefixCache cache = cache(8192, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        PrefixNode parent = cache.root();
        PrefixNode result = capture(cache, sequence(1024, 4), parent, tokens, 1024);
        assertSame(parent, result, "the cursor stays where it was");
        assertEquals(1, cache.stats().skipped());
        assertEquals(0, cache.stats().captured());
        assertNull(lookup(cache, tokens));
    }

    @Test
    void aSequenceShorterThanTheCheckpointIsNotCaptured() {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        PrefixNode result = capture(cache, sequence(512, 4), cache.root(), tokens, 1024);
        assertSame(cache.root(), result);
        assertEquals(0, cache.stats().captured());
    }

    @Test
    void aFailedCopyAbortsTheNodeAndFreesItsBytes() {
        var stream = new ExecutionFixtures.HoldingStream();
        this.cacheStream = stream;
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence source = sequence(1024, 4);
        var outcome = new Outcome();
        cache.capture(source, cache.root(), tokens, 1024, null, outcome, outcome);
        this.frames.drive();
        assertEquals(1, stream.held(), "the copies wait for their device completion");
        stream.release(new IllegalStateException("the device copy failed"));
        this.frames.drive();
        assertTrue(outcome.continued);
        assertSame(cache.root(), outcome.captured);
        assertEquals(1, cache.stats().failed());
        assertEquals(0, cache.stats().usedBytes());
        assertNull(lookup(cache, tokens));
        assertFalse(source.inFlight());
    }

    @Test
    void aSequenceCancelledBeforeTheRestoreRestoresNothingAndIsNotAnError() {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        capture(cache, sequence(1024, 4), cache.root(), tokens, 1024);
        var hit = lookup(cache, tokens);
        var target = new Sequence(3);
        target.cancel();
        assertFalse(restored(restore(cache, target, hit)), "cancelled: no restore, and no failure");
        assertEquals(Sequence.TerminalState.CANCELLED, target.terminalState());
        assertFalse(target.inFlight());
    }

    @Test
    void aCaptureHoldsTheSequenceSoACancellationCannotReleaseItsStateUnderTheCopies() {
        var stream = new ExecutionFixtures.HoldingStream();
        this.cacheStream = stream;
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence source = sequence(1024, 4);
        var outcome = new Outcome();
        cache.capture(source, cache.root(), tokens, 1024, null, outcome, outcome);
        this.frames.drive();
        // Cancelled while the capture's copies run, as a client that leaves mid-prefill does.
        source.cancel();
        assertEquals(Sequence.TerminalState.ACTIVE, source.terminalState(), "released under the copies");
        stream.release(null);
        this.frames.drive();
        assertEquals(1024, outcome.captured.position(), "the copies finished before the release: the node is whole");
        assertFalse(source.inFlight());
        assertEquals(Sequence.TerminalState.CANCELLED, source.terminalState(), "released after the copies");
    }

    @Test
    void aSequenceCancelledBeforeTheCaptureIsNotCaptured() {
        PrefixCache cache = cache(8L << 20, 1024);
        Sequence source = sequence(1024, 4);
        source.cancel();
        assertSame(
                cache.root(),
                capture(cache, source, cache.root(), IntStream.range(0, 1100).toArray(), 1024));
        assertEquals(0, cache.stats().captured());
    }

    @Test
    void aCancellationDuringTheRestoreStopsItWithoutAnError() {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        capture(cache, sequence(1024, 4), cache.root(), tokens, 1024);
        var hit = lookup(cache, tokens);
        var target = new Sequence(3);
        var outcome = new Outcome();
        cache.restore(plan(), target, hit, null, outcome, outcome);
        // Cancelled once the restore's copies began: the restore is in flight, so cancel() only flags the sequence.
        target.cancel();
        this.frames.drive();
        assertTrue(outcome.continued);
        assertFalse(restored(outcome));
        assertFalse(target.inFlight(), "the restore retired");
        assertEquals(Sequence.TerminalState.CANCELLED, target.terminalState());
        assertFalse(hit.cursor().pinned(), "the hit was released");
    }

    @Test
    void aFailedPieceReleasesTheHitAndFailsTheSequence() {
        var stream = new ExecutionFixtures.HoldingStream();
        this.cacheStream = stream;
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        var capture = new Outcome();
        cache.capture(sequence(1024, 4), cache.root(), tokens, 1024, null, capture, capture);
        this.frames.drive();
        stream.release(null);
        this.frames.drive();
        var hit = lookup(cache, tokens);
        assertTrue(hit.cursor().pinned());
        var target = new Sequence(3);
        var outcome = new Outcome();
        cache.restore(plan(), target, hit, null, outcome, outcome);
        this.frames.drive();
        assertNull(outcome.restored, "nothing is restored before the copies' device completion");
        assertTrue(target.inFlight());
        var failure = new IllegalStateException("the device copy failed");
        stream.release(failure);
        this.frames.drive();
        assertTrue(outcome.continued);
        assertFalse(outcome.restored);
        assertSame(failure, outcome.failure);
        assertEquals(Sequence.TerminalState.FAILED, target.terminalState());
        assertFalse(target.inFlight());
        assertFalse(hit.cursor().pinned(), "the hit was released");
    }

    @Test
    void copiesCompleteThroughDeviceCompletionFrames() {
        var stream = new ExecutionFixtures.HoldingStream();
        this.cacheStream = stream;
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        var capture = new Outcome();
        cache.capture(sequence(1024, 4), cache.root(), tokens, 1024, null, capture, capture);
        this.frames.drive();
        assertFalse(capture.continued, "nothing is published before the copies' device completion");
        assertNull(lookup(cache, tokens), "the node is not visible yet");
        stream.release(null);
        this.frames.drive();
        assertTrue(capture.continued);
        assertEquals(1024, capture.captured.position());

        var hit = lookup(cache, tokens);
        var target = new Sequence(3);
        var restore = new Outcome();
        cache.restore(plan(), target, hit, null, restore, restore);
        this.frames.drive();
        assertFalse(restore.continued, "nothing is restored before the copies' device completion");
        assertEquals(0, target.currentTokenPosition());
        stream.release(null);
        this.frames.drive();
        assertTrue(restored(restore));
        assertEquals(1024, target.currentTokenPosition());
    }

    @Test
    void treeOperationsRunOnTheOwnersHash() {
        var stream = new ExecutionFixtures.HoldingStream();
        this.cacheStream = stream;
        PrefixCache cache = cache(8192, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        // Aborted: the budget holds no checkpoint, then a failed copy; published; looked up; restored, released.
        capture(cache, sequence(1024, 4), cache.root(), tokens, 1024);
        PrefixCache roomy = cache(8L << 20, 1024);
        var failed = new Outcome();
        roomy.capture(sequence(1024, 4), roomy.root(), tokens, 1024, null, failed, failed);
        this.frames.drive();
        stream.release(new IllegalStateException("the device copy failed"));
        this.frames.drive();
        var published = new Outcome();
        roomy.capture(sequence(1024, 4), roomy.root(), tokens, 1024, null, published, published);
        this.frames.drive();
        stream.release(null);
        this.frames.drive();
        var hit = lookup(roomy, tokens);
        var restore = new Outcome();
        roomy.restore(plan(), new Sequence(3), hit, null, restore, restore);
        this.frames.drive();
        stream.release(null);
        this.frames.drive();
        assertTrue(restored(restore));
        int owner = 0;
        for (AbstractFrame frame : this.frames.published) {
            if (frame instanceof Outcome) continue;
            if (frame.getIdHash() != PrefixCache.HASH) continue;
            if (frame instanceof PrefixCopies || frame instanceof PrefixCopies.Retired) continue;
            owner++;
            assertEquals(
                    PrefixCache.HASH, frame.getRoutingHash(), frame.getClass().getSimpleName());
        }
        // Reserve x3 (one per capture), abort, publish, lookup, release.
        assertTrue(owner >= 7, "owner frames: " + owner);
        for (AbstractFrame frame : this.frames.published)
            if (frame instanceof PrefixCopies)
                assertEquals(PrefixCache.HASH, frame.getIdHash(), "the cache owns its copies, which run anywhere");
    }

    @Test
    void checkpointsFallOnTheIntervalAndAtTheLastChunkBeforeThePromptEnds() {
        PrefixCache cache = cache(1 << 20, 2048);
        assertTrue(cache.wantsCheckpoint(2048, 10_000), "on the interval");
        assertFalse(cache.wantsCheckpoint(512, 10_000));
        assertFalse(cache.wantsCheckpoint(9216, 10_000), "two chunks remain");
        assertTrue(cache.wantsCheckpoint(9728, 10_000), "the next chunk is the last: 272 tokens");
        assertFalse(cache.wantsCheckpoint(10_000, 10_000), "the end of a prompt off the interval");
        assertTrue(cache.wantsCheckpoint(4096, 4096), "the end of a prompt on the interval");
        assertTrue(cache.wantsCheckpoint(1024, 1500), "a short prompt's last full chunk boundary");
    }

    @Test
    void closeFreesTheArena() {
        var released = new AtomicBoolean();
        Arena arena = Arena.ofShared();
        var cache =
                new PrefixCache(this.gpu, this.frames, CONFIG, arena.allocate(1 << 20), () -> released.set(true), 512);
        cache.close();
        cache.close();
        assertTrue(released.get());
    }

    @Test
    void rejectsAnIntervalOffTheChunkGrid() {
        Arena arena = Arena.ofShared();
        assertThrows(
                IllegalArgumentException.class,
                () -> new PrefixCache(this.gpu, this.frames, CONFIG, arena.allocate(1 << 20), arena::close, 1000));
    }

    // --- MTP state

    private PrefixCache mtpCache(long bytes) {
        Arena arena = Arena.ofShared();
        return new PrefixCache(this.gpu, this.frames, CONFIG, arena.allocate(bytes), arena::close, 512);
    }

    /// A sequence as a speculative prefill leaves it: `rows` base rows, `mtpRows` MTP rows, recognizable bytes.
    private Sequence sequenceWithMtp(int rows, int mtpRows, int seed) {
        var sequence = new Sequence(1);
        var held = HeldWork.admit(sequence);
        attachMtpStates(sequence);
        var attention = (AttentionStates) sequence.kvCacheState();
        var gdn = (GdnStates) sequence.recurrentState();
        AttentionKvState kv = attention.forLayer(1);
        kv.prepareAppend(0, rows);
        kv.appendSubmitted(rows);
        kv.commitSubmitted();
        AttentionKvState mtp = attention.forLayer(CONFIG.numHiddenLayers());
        mtp.prepareAppend(0, mtpRows);
        mtp.appendSubmitted(mtpRows);
        mtp.commitSubmitted();
        this.gpu.fill(
                gdn.forLayer(0).recurrentStateAddress(), (int) gdn.forLayer(0).recurrentBytes(), seed);
        this.gpu.fill(
                gdn.forLayer(0).convolutionStateAddress(), (int) gdn.forLayer(0).convolutionBytes(), seed + 1);
        for (int i = 0; i < kv.pageAddresses().size(); i++)
            this.gpu.fill(kv.pageAddresses().get(i), (int) (2 * kv.planePageBytes()), seed + 2 + i);
        for (int i = 0; i < mtp.pageAddresses().size(); i++)
            this.gpu.fill(mtp.pageAddresses().get(i), (int) (2 * mtp.planePageBytes()), seed + 50 + i);
        held.commit(rows);
        return sequence;
    }

    private void attachMtpStates(Sequence sequence) {
        sequence.setRecurrentState(GdnStates.allocate(
                this.gpu,
                CONFIG.layerTypes(),
                CONFIG.linearNumKeyHeads(),
                CONFIG.linearNumValueHeads(),
                CONFIG.linearKeyHeadDim(),
                CONFIG.linearValueHeadDim(),
                CONFIG.linearConvKernelDim()));
        sequence.setKvCacheState(AttentionStates.allocate(
                this.gpu, CONFIG.layerTypes(), CONFIG.numKeyValueHeads() * CONFIG.attentionHeadDim(), true));
    }

    private static MtpCheckpoint mtp(long seedRow) {
        var checkpoint = new MtpCheckpoint(CONFIG);
        checkpoint.seedRow(seedRow);
        return checkpoint;
    }

    private long seedRow(int seed) {
        long address = this.gpu.allocate(CONFIG.hiddenSize() * 2L);
        this.gpu.fill(address, CONFIG.hiddenSize() * 2, seed);
        return address;
    }

    @Test
    void anMtpCheckpointRestoresTheMtpCacheAndTheSeedRowOfItsLastPosition() {
        PrefixCache cache = mtpCache(8L << 20);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence source = sequenceWithMtp(1024, 1023, 4);
        long seed = seedRow(9);
        PrefixNode node = capture(cache, source, cache.root(), tokens, 1024, mtp(seed));
        assertEquals(MtpCheckpoint.KIND, node.speculation());

        var hit = lookup(cache, tokens, MtpCheckpoint.KIND);
        assertNotNull(hit);
        var target = new Sequence(2);
        var held = HeldWork.admit(target);
        attachMtpStates(target);
        held.commit(0);
        assertTrue(restored(restore(cache, target, hit, new MtpCheckpoint(CONFIG))));
        var sourceMtp = ((AttentionStates) source.kvCacheState()).forLayer(CONFIG.numHiddenLayers());
        var restoredStates = (AttentionStates) target.kvCacheState();
        var restoredMtp = restoredStates.forLayer(CONFIG.numHiddenLayers());
        assertEquals(1023, restoredMtp.length(), "the MTP cache stops one row short of the position");
        assertEquals(1024, restoredStates.forLayer(1).length());
        int pageBytes = (int) (2 * sourceMtp.planePageBytes());
        for (int page = 0; page < 4; page++)
            assertArrayEquals(
                    this.gpu.bytes(sourceMtp.pageAddresses().get(page), pageBytes),
                    this.gpu.bytes(restoredMtp.pageAddresses().get(page), pageBytes),
                    "MTP page " + page);
        long restoredSeed = restoredStates.draftSeedRows(1, CONFIG.hiddenSize());
        assertArrayEquals(this.gpu.bytes(seed, 256), this.gpu.bytes(restoredSeed, 256));
    }

    @Test
    void aSequenceWithoutTheMtpCacheIsStoredWithoutMtpState() {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        PrefixNode node = capture(cache, sequence(1024, 4), cache.root(), tokens, 1024, mtp(seedRow(9)));
        assertNull(node.speculation());
    }

    @Test
    void aSequenceWhoseMtpCacheLagsIsStoredWithoutMtpState() {
        PrefixCache cache = mtpCache(8L << 20);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence lagging = sequenceWithMtp(1024, 700, 4);
        PrefixNode node = capture(cache, lagging, cache.root(), tokens, 1024, mtp(seedRow(9)));
        assertNull(node.speculation(), "rows [0, 1023) of the MTP cache are not all there");
        assertNull(lookup(cache, tokens, MtpCheckpoint.KIND));
        assertNotNull(lookup(cache, tokens));
    }

    @Test
    void aSpanStoredWithAndWithoutMtpStateKeepsBothVariants() {
        PrefixCache cache = mtpCache(16L << 20);
        int[] tokens = IntStream.range(0, 1100).toArray();
        PrefixNode plain = capture(cache, sequence(1024, 4), cache.root(), tokens, 1024);
        PrefixNode withMtp =
                capture(cache, sequenceWithMtp(1024, 1023, 4), cache.root(), tokens, 1024, mtp(seedRow(9)));
        assertNull(plain.speculation());
        assertEquals(MtpCheckpoint.KIND, withMtp.speculation());
        assertEquals(2, cache.stats().captured());
        assertEquals(
                withMtp.position(), lookup(cache, tokens, MtpCheckpoint.KIND).position());
    }

    @Test
    void aClosedCacheStoresAndRestoresNothing() {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        capture(cache, sequence(1024, 4), cache.root(), tokens, 1024);
        var hit = lookup(cache, tokens);
        cache.close();
        assertSame(cache.root(), capture(cache, sequence(1024, 5), cache.root(), tokens, 1024));
        var target = new Sequence(3);
        var outcome = restore(cache, target, hit);
        assertInstanceOf(IllegalStateException.class, outcome.failure);
        assertEquals(Sequence.TerminalState.ACTIVE, target.terminalState(), "nothing was claimed");
        assertFalse(target.inFlight());
    }

    @Test
    void aFrameThatRunsAfterCloseCopiesNothing() {
        // Freed native memory stays addressable from Java, so only the cache's own check can stop the copy.
        Arena arena = Arena.ofShared();
        PrefixCache cache = new PrefixCache(this.gpu, this.frames, CONFIG, arena.allocate(8L << 20), () -> {}, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence source = sequence(1024, 4);
        var outcome = new Outcome();
        cache.capture(source, cache.root(), tokens, 1024, null, outcome, outcome);
        // Closed after the reservation was published, before its copies ran.
        this.frames.beforeEach = () -> {
            if (this.frames.published.stream().anyMatch(PrefixCopies.class::isInstance)) cache.close();
        };
        this.frames.drive();
        assertSame(cache.root(), outcome.captured, "the copy failed, so the node was not published");
        assertEquals(1, cache.stats().failed());
    }

    @Test
    void aSamplingPromptReusesAnMtpNodeInsteadOfStoringASecondOne() {
        PrefixCache cache = mtpCache(16L << 20);
        int[] tokens = IntStream.range(0, 1100).toArray();
        PrefixNode withMtp =
                capture(cache, sequenceWithMtp(1024, 1023, 4), cache.root(), tokens, 1024, mtp(seedRow(9)));
        PrefixNode reused = capture(cache, sequence(1024, 4), cache.root(), tokens, 1024);
        assertSame(withMtp, reused, "an MTP node is a superset of a plain one");
        assertEquals(1, cache.stats().captured());
    }

    @Test
    void aPieceTheLakeRefusesStillRunsAndTheCaptureCompletes() {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        this.frames.refuse = PrefixCopies.class::isInstance;
        var outcome = new Outcome();
        cache.capture(sequence(1024, 4), cache.root(), tokens, 1024, null, outcome, outcome);
        this.frames.drive();
        assertTrue(outcome.continued, "the capture threw its next frame");
        assertEquals(1024, outcome.captured.position());
    }

    @Test
    void anOwnerFrameTheLatticeRejectsStillTellsItsCallAndThrowsItsNextFrame() {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        capture(cache, sequence(1024, 4), cache.root(), tokens, 1024);
        this.frames.reject = PrefixCache.Owned.class::isInstance;
        var outcome = new Outcome();
        cache.lookup(tokens, null, outcome, outcome);
        this.frames.drive();
        assertTrue(outcome.continued, "the lookup threw its next frame");
        assertNotNull(outcome.hit);
    }

    @Test
    void aPieceThatFailsPartwayWaitsForWhatItQueued() {
        var stream = new ExecutionFixtures.HoldingStream();
        this.cacheStream = stream;
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence source = sequence(1024, 4);
        this.copiesBeforeFailure = 1;
        var outcome = new Outcome();
        cache.capture(source, cache.root(), tokens, 1024, null, outcome, outcome);
        this.frames.drive();
        assertFalse(outcome.continued, "the node's bytes are not given back under a queued copy");
        assertEquals(1, stream.held());
        stream.release(null);
        this.frames.drive();
        assertTrue(outcome.continued);
        assertSame(cache.root(), outcome.captured);
        assertEquals(1, cache.stats().failed());
        assertEquals(0, cache.stats().usedBytes());
    }
}
