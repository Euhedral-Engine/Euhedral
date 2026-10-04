package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.prefix.PrefixNode;
import java.lang.foreign.Arena;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

class PrefixCacheTest {
    private static final QwenConfig CONFIG =
            QwenExecutionFixtures.statefulCompactWeights(8).config();
    private static final PrefixCache.Frames INLINE = new PrefixCache.Frames() {
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

    private QwenExecutionPlan plan() {
        return new QwenExecutionPlan(QwenExecutionFixtures.statefulCompactWeights(8));
    }

    /// A sequence with `rows` committed rows and recognizable state, as a prefill leaves it.
    private QwenSequenceState sequence(int rows, int seed) {
        var sequence = new QwenSequenceState(1);
        var lease = sequence.claimExecution(0);
        QwenExecutionContext.attachSequenceState(plan(), sequence, lease, this.gpu);
        var attention = (AttentionSequenceStates) sequence.kvCacheState();
        var gdn = (GdnSequenceStates) sequence.recurrentState();
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
        sequence.releaseExecution(lease, rows);
        return sequence;
    }

    @Test
    void capturesACheckpointAndRestoresItIntoAFreshSequence() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        QwenSequenceState source = sequence(1024, 4);
        PrefixNode node =
                cache.capture(INLINE, source, cache.root(), tokens, 1024).get();
        assertEquals(1024, node.position());
        assertEquals(1, cache.stats().captured());

        var hit = cache.lookup(tokens, false);
        assertNotNull(hit);
        assertEquals(1024, hit.position());
        assertSame(node, hit.cursor());

        var target = new QwenSequenceState(2);
        assertTrue(cache.restore(INLINE, plan(), target, hit).get());
        cache.release(hit);
        assertEquals(1024, target.currentTokenPosition());
        assertFalse(target.isExecutionClaimed());
        var restoredKv = ((AttentionSequenceStates) target.kvCacheState()).forLayer(1);
        var sourceKv = ((AttentionSequenceStates) source.kvCacheState()).forLayer(1);
        assertEquals(1024, restoredKv.length());
        int pageBytes = (int) (2 * sourceKv.planePageBytes());
        for (int page = 0; page < 4; page++)
            assertArrayEquals(
                    this.gpu.bytes(sourceKv.pageAddresses().get(page), pageBytes),
                    this.gpu.bytes(restoredKv.pageAddresses().get(page), pageBytes),
                    "page " + page);
        var sourceGdn = ((GdnSequenceStates) source.recurrentState()).forLayer(0);
        var restoredGdn = ((GdnSequenceStates) target.recurrentState()).forLayer(0);
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
        QwenSequenceState source = sequence(1280, 4);
        PrefixNode first =
                cache.capture(INLINE, source, cache.root(), tokens, 512).get();
        PrefixNode second = cache.capture(INLINE, source, first, tokens, 1280).get();
        assertSame(first, second.parent());
        var hit = cache.lookup(tokens, false);
        assertEquals(1280, hit.position());
        var target = new QwenSequenceState(2);
        assertTrue(cache.restore(INLINE, plan(), target, hit).get());
        cache.release(hit);
        assertEquals(1280, target.currentTokenPosition());
    }

    @Test
    void anExistingNodeIsNotCapturedTwice() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        QwenSequenceState source = sequence(1024, 4);
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
        assertNull(cache.lookup(tokens, false));
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
        QwenSequenceState source = sequence(1024, 4);
        PrefixCache.Frames failing = new PrefixCache.Frames() {
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
        assertNull(cache.lookup(tokens, false));
    }

    @Test
    void aCancelledSequenceRestoresNothing() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        cache.capture(INLINE, sequence(1024, 4), cache.root(), tokens, 1024).get();
        var hit = cache.lookup(tokens, false);
        var target = new QwenSequenceState(3);
        target.cancel();
        assertThrows(
                ExecutionException.class,
                () -> cache.restore(INLINE, plan(), target, hit).get());
        cache.release(hit);
        assertEquals(QwenSequenceState.TerminalState.CANCELLED, target.terminalState());
        assertFalse(target.isExecutionClaimed());
    }

    @Test
    void aFailedRestoreCopyFailsTheSequenceAndReleasesItsLease() throws Exception {
        PrefixCache cache = cache(8L << 20, 1024);
        int[] tokens = IntStream.range(0, 1100).toArray();
        cache.capture(INLINE, sequence(1024, 4), cache.root(), tokens, 1024).get();
        var hit = cache.lookup(tokens, false);
        var target = new QwenSequenceState(3);
        PrefixCache.Frames failing = new PrefixCache.Frames() {
            @Override
            public <T> CompletableFuture<T> run(Supplier<T> work) {
                return CompletableFuture.failedFuture(new IllegalStateException("lattice rejected the frame"));
            }
        };
        assertThrows(
                ExecutionException.class,
                () -> cache.restore(failing, plan(), target, hit).get());
        cache.release(hit);
        assertEquals(QwenSequenceState.TerminalState.FAILED, target.terminalState());
        assertFalse(target.isExecutionClaimed());
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
}
