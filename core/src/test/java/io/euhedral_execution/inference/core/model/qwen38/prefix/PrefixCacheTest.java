package io.euhedral_execution.inference.core.model.qwen38.prefix;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
import io.euhedral_execution.inference.core.prefix.PrefixFrames;
import io.euhedral_execution.inference.core.prefix.PrefixNode;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import java.lang.foreign.Arena;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

class PrefixCacheTest {
    private static final Qwen38Config CONFIG =
            ExecutionFixtures.statefulCompactWeights(8).config();
    private static final PrefixFrames INLINE = new PrefixFrames() {
        @Override
        public <T> CompletableFuture<T> run(Supplier<T> work) {
            try {
                return CompletableFuture.completedFuture(work.get());
            } catch (RuntimeException | Error failure) {
                return CompletableFuture.failedFuture(failure);
            }
        }
    };

    private final MemoryGpu gpu = new MemoryGpu();

    private PrefixCache cache(long bytes, int interval) {
        Arena arena = Arena.ofShared();
        return new PrefixCache(this.gpu, CONFIG, arena.allocate(bytes), arena::close, interval);
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
    void capturesACheckpointAndRestoresItIntoAFreshSequence() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence source = sequence(1024, 4);
        PrefixNode node =
                cache.capture(INLINE, source, cache.root(), tokens, 1024).get();
        assertEquals(1024, node.position());
        assertEquals(1, cache.stats().captured());

        var hit = cache.lookup(tokens);
        assertNotNull(hit);
        assertEquals(1024, hit.position());
        assertSame(node, hit.cursor());

        var target = new Sequence(2);
        assertTrue(cache.restore(INLINE, plan(), target, hit).get());
        cache.release(hit);
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
    void aSecondCheckpointExtendsTheFirstAndRestoresBothChainsPages() throws Exception {
        PrefixCache cache = cache(16L << 20, 512);
        int[] tokens = IntStream.range(0, 1400).toArray();
        Sequence source = sequence(1280, 4);
        PrefixNode first =
                cache.capture(INLINE, source, cache.root(), tokens, 512).get();
        PrefixNode second = cache.capture(INLINE, source, first, tokens, 1280).get();
        assertSame(first, second.parent());
        var hit = cache.lookup(tokens);
        assertEquals(1280, hit.position());
        var target = new Sequence(2);
        assertTrue(cache.restore(INLINE, plan(), target, hit).get());
        cache.release(hit);
        assertEquals(1280, target.currentTokenPosition());
    }

    @Test
    void anExistingNodeIsNotCapturedTwice() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence source = sequence(1024, 4);
        PrefixNode first =
                cache.capture(INLINE, source, cache.root(), tokens, 1024).get();
        PrefixNode second =
                cache.capture(INLINE, source, cache.root(), tokens, 1024).get();
        assertSame(first, second);
        assertEquals(1, cache.stats().captured());
    }

    @Test
    void aBudgetSmallerThanACheckpointSkipsTheCaptureAndCountsIt() throws Exception {
        PrefixCache cache = cache(8192, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        PrefixNode parent = cache.root();
        PrefixNode result =
                cache.capture(INLINE, sequence(1024, 4), parent, tokens, 1024).get();
        assertSame(parent, result, "the cursor stays where it was");
        assertEquals(1, cache.stats().skipped());
        assertEquals(0, cache.stats().captured());
        assertNull(cache.lookup(tokens));
    }

    @Test
    void aSequenceShorterThanTheCheckpointIsNotCaptured() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        PrefixNode result = cache.capture(INLINE, sequence(512, 4), cache.root(), tokens, 1024)
                .get();
        assertSame(cache.root(), result);
        assertEquals(0, cache.stats().captured());
    }

    @Test
    void aFailedCopyAbortsTheNodeAndFreesItsBytes() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence source = sequence(1024, 4);
        PrefixFrames failing = new PrefixFrames() {
            @Override
            public <T> CompletableFuture<T> run(Supplier<T> work) {
                return CompletableFuture.failedFuture(new IllegalStateException("lattice rejected the frame"));
            }
        };
        PrefixNode result =
                cache.capture(failing, source, cache.root(), tokens, 1024).get();
        assertSame(cache.root(), result);
        assertEquals(1, cache.stats().failed());
        assertEquals(0, cache.stats().usedBytes());
        assertNull(cache.lookup(tokens));
    }

    @Test
    void aSequenceCancelledBeforeTheRestoreRestoresNothingAndIsNotAnError() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        cache.capture(INLINE, sequence(1024, 4), cache.root(), tokens, 1024).get();
        var hit = cache.lookup(tokens);
        var target = new Sequence(3);
        target.cancel();
        assertFalse(cache.restore(INLINE, plan(), target, hit).get(), "cancelled: no restore, and no failure");
        cache.release(hit);
        assertEquals(Sequence.TerminalState.CANCELLED, target.terminalState());
        assertFalse(target.inFlight());
    }

    @Test
    void aCaptureHoldsTheSequenceSoACancellationCannotReleaseItsStateUnderTheCopies() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence source = sequence(1024, 4);
        var released = new java.util.concurrent.atomic.AtomicBoolean();
        // Cancelled while the capture's copies run, as a client that leaves mid-prefill does.
        PrefixFrames cancelling = new PrefixFrames() {
            @Override
            public <T> CompletableFuture<T> run(Supplier<T> work) {
                source.cancel();
                if (source.terminalState() != Sequence.TerminalState.ACTIVE) released.set(true);
                return INLINE.run(work);
            }
        };
        PrefixNode node =
                cache.capture(cancelling, source, cache.root(), tokens, 1024).get();
        assertFalse(released.get(), "the sequence's state was released while the capture copied it");
        assertEquals(1024, node.position(), "the copies finished before the release, so the node is whole");
        assertFalse(source.inFlight());
        assertEquals(Sequence.TerminalState.CANCELLED, source.terminalState(), "released after the copies");
    }

    @Test
    void aSequenceCancelledBeforeTheCaptureIsNotCaptured() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        Sequence source = sequence(1024, 4);
        source.cancel();
        assertSame(
                cache.root(),
                cache.capture(
                                INLINE,
                                source,
                                cache.root(),
                                IntStream.range(0, 1100).toArray(),
                                1024)
                        .get());
        assertEquals(0, cache.stats().captured());
    }

    @Test
    void aCancellationDuringTheRestoreStopsItWithoutAnError() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        cache.capture(INLINE, sequence(1024, 4), cache.root(), tokens, 1024).get();
        var hit = cache.lookup(tokens);
        var target = new Sequence(3);
        // Cancelled once the restore's copies began: the restore is in flight, so cancel() only flags the sequence.
        PrefixFrames cancelling = new PrefixFrames() {
            @Override
            public <T> CompletableFuture<T> run(Supplier<T> work) {
                target.cancel();
                return INLINE.run(work);
            }
        };
        assertFalse(cache.restore(cancelling, plan(), target, hit).get());
        cache.release(hit);
        assertFalse(target.inFlight(), "the restore retired");
        assertEquals(Sequence.TerminalState.CANCELLED, target.terminalState());
    }

    @Test
    void aFailedRestoreCopyFailsTheSequenceAndRetiresTheRestore() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        cache.capture(INLINE, sequence(1024, 4), cache.root(), tokens, 1024).get();
        var hit = cache.lookup(tokens);
        var target = new Sequence(3);
        PrefixFrames failing = new PrefixFrames() {
            @Override
            public <T> CompletableFuture<T> run(Supplier<T> work) {
                return CompletableFuture.failedFuture(new IllegalStateException("lattice rejected the frame"));
            }
        };
        assertThrows(
                ExecutionException.class,
                () -> cache.restore(failing, plan(), target, hit).get());
        cache.release(hit);
        assertEquals(Sequence.TerminalState.FAILED, target.terminalState());
        assertFalse(target.inFlight());
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
        var cache = new PrefixCache(this.gpu, CONFIG, arena.allocate(1 << 20), () -> released.set(true), 512);
        cache.close();
        cache.close();
        assertTrue(released.get());
    }

    @Test
    void rejectsAnIntervalOffTheChunkGrid() {
        Arena arena = Arena.ofShared();
        assertThrows(
                IllegalArgumentException.class,
                () -> new PrefixCache(this.gpu, CONFIG, arena.allocate(1 << 20), arena::close, 1000));
    }

    // --- MTP state

    private PrefixCache mtpCache(long bytes) {
        Arena arena = Arena.ofShared();
        return new PrefixCache(this.gpu, CONFIG, arena.allocate(bytes), arena::close, 512);
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
    void anMtpCheckpointRestoresTheMtpCacheAndTheSeedRowOfItsLastPosition() throws Exception {
        PrefixCache cache = mtpCache(8L << 20);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence source = sequenceWithMtp(1024, 1023, 4);
        long seed = seedRow(9);
        PrefixNode node = cache.capture(INLINE, source, cache.root(), tokens, 1024, mtp(seed))
                .get();
        assertEquals(MtpCheckpoint.KIND, node.speculation());

        var hit = cache.lookup(tokens, new MtpCheckpoint(CONFIG));
        assertNotNull(hit);
        var target = new Sequence(2);
        var held = HeldWork.admit(target);
        attachMtpStates(target);
        held.commit(0);
        assertTrue(cache.restore(INLINE, plan(), target, hit, new MtpCheckpoint(CONFIG))
                .get());
        cache.release(hit);
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
    void aSequenceWithoutTheMtpCacheIsStoredWithoutMtpState() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        PrefixNode node = cache.capture(INLINE, sequence(1024, 4), cache.root(), tokens, 1024, mtp(seedRow(9)))
                .get();
        assertNull(node.speculation());
    }

    @Test
    void aSequenceWhoseMtpCacheLagsIsStoredWithoutMtpState() throws Exception {
        PrefixCache cache = mtpCache(8L << 20);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence lagging = sequenceWithMtp(1024, 700, 4);
        PrefixNode node = cache.capture(INLINE, lagging, cache.root(), tokens, 1024, mtp(seedRow(9)))
                .get();
        assertNull(node.speculation(), "rows [0, 1023) of the MTP cache are not all there");
        assertNull(cache.lookup(tokens, new MtpCheckpoint(CONFIG)));
        assertNotNull(cache.lookup(tokens));
    }

    @Test
    void aSpanStoredWithAndWithoutMtpStateKeepsBothVariants() throws Exception {
        PrefixCache cache = mtpCache(16L << 20);
        int[] tokens = IntStream.range(0, 1100).toArray();
        PrefixNode plain = cache.capture(INLINE, sequence(1024, 4), cache.root(), tokens, 1024)
                .get();
        PrefixNode withMtp = cache.capture(
                        INLINE, sequenceWithMtp(1024, 1023, 4), cache.root(), tokens, 1024, mtp(seedRow(9)))
                .get();
        assertNull(plain.speculation());
        assertEquals(MtpCheckpoint.KIND, withMtp.speculation());
        assertEquals(2, cache.stats().captured());
        assertEquals(
                withMtp.position(),
                cache.lookup(tokens, new MtpCheckpoint(CONFIG)).position());
    }

    @Test
    void aClosedCacheStoresAndRestoresNothing() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        cache.capture(INLINE, sequence(1024, 4), cache.root(), tokens, 1024).get();
        var hit = cache.lookup(tokens);
        cache.close();
        assertSame(
                cache.root(),
                cache.capture(INLINE, sequence(1024, 5), cache.root(), tokens, 1024)
                        .get());
        var target = new Sequence(3);
        assertThrows(
                ExecutionException.class,
                () -> cache.restore(INLINE, plan(), target, hit).get());
        assertEquals(Sequence.TerminalState.ACTIVE, target.terminalState(), "nothing was claimed");
        assertFalse(target.inFlight());
    }

    @Test
    void aFrameThatRunsAfterCloseCopiesNothing() throws Exception {
        // Freed native memory stays addressable from Java, so only the cache's own check can stop the copy.
        Arena arena = Arena.ofShared();
        PrefixCache cache = new PrefixCache(this.gpu, CONFIG, arena.allocate(8L << 20), () -> {}, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        Sequence source = sequence(1024, 4);
        PrefixFrames closing = new PrefixFrames() {
            @Override
            public <T> CompletableFuture<T> run(Supplier<T> work) {
                cache.close();
                return INLINE.run(work);
            }
        };
        PrefixNode result =
                cache.capture(closing, source, cache.root(), tokens, 1024).get();
        assertSame(cache.root(), result, "the copy failed, so the node was not published");
        assertEquals(1, cache.stats().failed());
    }

    @Test
    void aSamplingPromptReusesAnMtpNodeInsteadOfStoringASecondOne() throws Exception {
        PrefixCache cache = mtpCache(16L << 20);
        int[] tokens = IntStream.range(0, 1100).toArray();
        PrefixNode withMtp = cache.capture(
                        INLINE, sequenceWithMtp(1024, 1023, 4), cache.root(), tokens, 1024, mtp(seedRow(9)))
                .get();
        PrefixNode reused = cache.capture(INLINE, sequence(1024, 4), cache.root(), tokens, 1024)
                .get();
        assertSame(withMtp, reused, "an MTP node is a superset of a plain one");
        assertEquals(1, cache.stats().captured());
    }
}
