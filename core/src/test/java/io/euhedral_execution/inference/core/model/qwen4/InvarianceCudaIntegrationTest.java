package io.euhedral_execution.inference.core.model.qwen4;

import static io.euhedral_execution.inference.core.model.qwen4.Reference.bf;
import static io.euhedral_execution.inference.core.model.qwen4.TestSupport.error;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen4.loader.HostBudget;
import io.euhedral_execution.inference.core.model.qwen4.loader.Mode;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/// What must not change the output: how a prompt is cut into prefill chunks (within BF16 noise: the
/// engine's linears switch kernels at nine rows), and what the expert cache holds (not at all: bit
/// for bit).
class InvarianceCudaIntegrationTest {

    private static final int CONTEXT = 4096;

    private static int[] prompt() throws Exception {
        assumeTrue(TestSupport.hasArtifact(), "no artifact");
        Path directory = TestSupport.modelFixtureRoot().resolve("chunks_a");
        assumeTrue(ReferenceFixtures.exists(directory), "no fixtures " + directory);
        return Arrays.stream(new ReferenceFixtures(directory).i64("c0/tokens"))
                .mapToInt(t -> (int) t)
                .toArray();
    }

    /// The last row's logits after prefilling `tokens` in chunks of the given sizes.
    private static short[] prefill(
            CudaGpuMemory gpu, ExecutionPlan executor, int[] tokens, int[] chunks, int decodeSteps) throws Exception {
        int vocabulary = executor.vocabularySize();
        var readback = gpu.allocateReadbackBuffer((long) vocabulary * 2);
        try (Sequence sequence = executor.newSequence()) {
            int at = 0;
            for (int chunk : chunks) {
                Blocking.step(
                        executor,
                        sequence,
                        tokens,
                        at,
                        chunk,
                        address -> gpu.copyDeviceToReadback(readback, address, vocabulary * 2L));
                at += chunk;
            }
            assertEquals(tokens.length, at);
            short[] logits = new short[vocabulary];
            MemorySegment.copy(readback.segment(), ValueLayout.JAVA_SHORT, 0, logits, 0, vocabulary);
            return logits;
        } finally {
            readback.close();
        }
    }

    private static int argmax(short[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) if (bf(logits[i]) > bf(logits[best])) best = i;
        return best;
    }

    private static int[] chunks(int total, int first, int then) {
        List<Integer> sizes = new ArrayList<>();
        int left = total;
        sizes.add(Math.min(first, left));
        left -= sizes.get(0);
        while (left > 0) {
            int size = Math.min(then, left);
            sizes.add(size);
            left -= size;
        }
        return sizes.stream().mapToInt(Integer::intValue).toArray();
    }

    @Test
    void chunkBoundariesChangeTheResultOnlyByNoise() throws Exception {
        int[] tokens = prompt();
        try (CudaGpuMemory gpu = TestSupport.openGpu();
                Qwen4Model model = Qwen4Model.open(
                        TestSupport.artifactPath(),
                        gpu,
                        gpu.deviceMemoryInfo().freeBytes(),
                        HostBudget.system(),
                        Mode.TEXT,
                        CONTEXT);
                TestLattice.Run run = TestLattice.shared().run(gpu, model, CONTEXT);
                ExecutionPlan executor = run.plan()) {
            short[] whole = prefill(gpu, executor, tokens, new int[] {tokens.length}, 0);
            int expected = argmax(whole);
            for (int[] cut : new int[][] {
                {32, 32, 6},
                {1, 1, 1, 1, 1, 1, 1, 1, 62},
                chunks(tokens.length, 16, 16),
                chunks(tokens.length, 7, 9),
                chunks(tokens.length, 64, 1)
            }) {
                short[] logits = prefill(gpu, executor, tokens, cut, 0);
                var difference = error(whole, logits);
                System.out.println("chunks " + Arrays.toString(cut) + ": " + difference + ", greedy " + argmax(logits)
                        + " vs " + expected);
                assertEquals(expected, argmax(logits), "greedy token for chunks " + Arrays.toString(cut));
                assertTrue(difference.relativeRms() < 0.2, difference.toString());
            }
            // The reference's own fixture for the same prompt.
            ReferenceFixtures fixture =
                    new ReferenceFixtures(TestSupport.modelFixtureRoot().resolve("chunks_a"));
            assertEquals(argmax(fixture.bf16("c0/logits")), expected, "greedy token against the reference");
        }
    }

    @Test
    void theExpertCacheDoesNotChangeTheOutput() throws Exception {
        int[] tokens = prompt();
        int[] cut = {32, 32, 6};
        short[] roomy;
        try (CudaGpuMemory gpu = TestSupport.openGpu();
                Qwen4Model model = Qwen4Model.open(
                        TestSupport.artifactPath(),
                        gpu,
                        gpu.deviceMemoryInfo().freeBytes(),
                        HostBudget.system(),
                        Mode.TEXT,
                        CONTEXT);
                TestLattice.Run run = TestLattice.shared().run(gpu, model, CONTEXT);
                ExecutionPlan executor = run.plan()) {
            roomy = prefill(gpu, executor, tokens, cut, 0);
            // Cold, then warm: the second run finds most experts resident.
            long hits = model.expertCache().stats().snapshot().hits();
            assertArrayEquals(roomy, prefill(gpu, executor, tokens, cut, 0), "a warm cache changed the result");
            assertTrue(model.expertCache().stats().snapshot().hits() > hits, "the second run should hit the cache");
        }
        // The smallest cache the planner allows: 5 GiB of device and a 262,144-token context leave 20 slots.
        try (CudaGpuMemory gpu = TestSupport.openGpu();
                Qwen4Model model = Qwen4Model.open(
                        TestSupport.artifactPath(),
                        gpu,
                        Math.min(5L << 30, gpu.deviceMemoryInfo().freeBytes()),
                        HostBudget.system(),
                        Mode.TEXT,
                        262144);
                TestLattice.Run run = TestLattice.shared().run(gpu, model, CONTEXT);
                ExecutionPlan executor = run.plan()) {
            System.out.println("small cache: " + model.expertCache().slotCount() + " slots");
            assertTrue(model.expertCache().slotCount() <= 64, "expected a small cache");
            short[] small = prefill(gpu, executor, tokens, cut, 0);
            assertArrayEquals(roomy, small, "a small cache changed the result");
            System.out.println(
                    "evictions: " + model.expertCache().stats().snapshot().evictions());
            assertTrue(model.expertCache().stats().snapshot().evictions() > 0);
        }
    }

    /// What a sequence holds on the device is what the residency plan reserved for it: the
    /// recurrent state and the indexer keys when it opens, the KV pages as it grows.
    @Test
    void aSequenceHoldsWhatThePlanReserved() throws Exception {
        int[] tokens = prompt();
        try (CudaGpuMemory gpu = TestSupport.openGpu();
                Qwen4Model model = Qwen4Model.open(
                        TestSupport.artifactPath(),
                        gpu,
                        gpu.deviceMemoryInfo().freeBytes(),
                        HostBudget.system(),
                        Mode.TEXT,
                        CONTEXT);
                TestLattice.Run run = TestLattice.shared().run(gpu, model, CONTEXT);
                ExecutionPlan executor = run.plan()) {
            var config = model.artifact().config();
            long before = gpu.allocatedBytes();
            long scratchBefore = gpu.retainedScratchBytes();
            try (Sequence sequence = executor.newSequence()) {
                long opened = gpu.allocatedBytes() - before;
                long accounted =
                        io.euhedral_execution.inference.core.model.qwen4.loader.SequenceState.gdnStateBytes(config)
                                + io.euhedral_execution.inference.core.model.qwen4.loader.SequenceState.indexerBytes(
                                        config, CONTEXT);
                System.out.println("sequence at open: " + opened + " bytes, plan " + accounted);
                // The tails of incomplete indexer blocks and the PLE history are a few hundred KiB the plan's workspace
                // covers.
                assertTrue(opened >= accounted && opened <= accounted + (2L << 20), opened + " vs " + accounted);
                Blocking.step(executor, sequence, tokens, 0, tokens.length, null);
                // The shared native-kernel scratch grows on first use: the workspace, not the sequence.
                long grown = gpu.allocatedBytes() - before - (gpu.retainedScratchBytes() - scratchBefore);
                long kv = io.euhedral_execution.inference.core.model.qwen4.loader.SequenceState.kvBytes(
                        config, tokens.length);
                System.out.println(
                        "after " + tokens.length + " tokens: +" + (grown - opened) + " bytes, plan KV " + kv);
                assertTrue(grown - opened <= kv + 65536, "KV pages beyond the plan: " + (grown - opened) + " vs " + kv);
            }
            assertEquals(
                    before + (gpu.retainedScratchBytes() - scratchBefore),
                    gpu.allocatedBytes(),
                    "closing a sequence must return its storage");
        }
    }
}
