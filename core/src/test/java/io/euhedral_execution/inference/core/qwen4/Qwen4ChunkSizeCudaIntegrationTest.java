package io.euhedral_execution.inference.core.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.qwen4.HostBudget;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Mode;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Model;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

/// The prompt's chunking is not part of the answer: the real model reads the same prompt in chunks of 512
/// tokens and in the plan's own (longer) chunk, and the greedy continuation is the same. A prompt of 1500
/// tokens is a few chunks of 512 and one chunk in the plan's workspace, which is where a longer chunk's
/// larger waves and scratch differ from the short ones'. Opt-in with the artifact present.
class Qwen4ChunkSizeCudaIntegrationTest {

    private static final int PROMPT = 1500;
    private static final int STEPS = 8;
    private static final int CONTEXT = PROMPT + STEPS + 16;

    private static float bf(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }

    private static int argmax(short[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) if (bf(logits[i]) > bf(logits[best])) best = i;
        return best;
    }

    /// The greedy tokens after prefilling `prompt` in chunks of `chunk`, and the first step's logits.
    private static int[] greedy(
            CudaGpuMemory gpu, Qwen4ExecutionPlan executor, int[] prompt, int chunk, short[] firstLogits)
            throws Exception {
        int vocabulary = executor.vocabularySize();
        var readback = gpu.allocateReadbackBuffer((long) vocabulary * 2);
        Qwen4ExecutionPlan.LogitsSink sink = address -> gpu.copyDeviceToReadback(readback, address, vocabulary * 2L);
        int[] out = new int[STEPS];
        try (Qwen4Sequence sequence = executor.newSequence()) {
            int at = 0;
            while (at < prompt.length) {
                int rows = Math.min(chunk, prompt.length - at);
                Qwen4Blocking.step(executor, sequence, prompt, at, rows, at + rows == prompt.length ? sink : null);
                at += rows;
            }
            short[] logits = new short[vocabulary];
            MemorySegment.copy(readback.segment(), ValueLayout.JAVA_SHORT, 0, logits, 0, vocabulary);
            System.arraycopy(logits, 0, firstLogits, 0, vocabulary);
            int[] token = new int[1];
            for (int step = 0; step < STEPS; step++) {
                out[step] = argmax(logits);
                token[0] = out[step];
                Qwen4Blocking.step(executor, sequence, token, 0, 1, sink);
                MemorySegment.copy(readback.segment(), ValueLayout.JAVA_SHORT, 0, logits, 0, vocabulary);
            }
            return out;
        } finally {
            readback.close();
        }
    }

    @Test
    void theGreedyContinuationDoesNotDependOnTheChunkSize() throws Exception {
        assumeTrue(Qwen4TestSupport.hasArtifact(), "no artifact");
        int[] prompt = Qwen4PerformanceCudaIntegrationTest.corpus(PROMPT);
        try (Qwen4TestLattice lattice = Qwen4TestLattice.start(8);
                CudaGpuMemory gpu = Qwen4TestSupport.openGpu();
                Qwen4Model model = Qwen4Model.open(
                        Qwen4TestSupport.artifactPath(),
                        gpu,
                        gpu.deviceMemoryInfo().freeBytes(),
                        HostBudget.system(),
                        Qwen4Mode.TEXT,
                        CONTEXT);
                Qwen4TestLattice.Run run = lattice.run(gpu, model, CONTEXT);
                Qwen4ExecutionPlan executor = run.plan()) {
            System.out.println("plan chunk: " + executor.maxRows() + " tokens");
            assertTrue(executor.maxRows() >= PROMPT, "the plan's chunk holds the prompt: " + executor.maxRows());
            short[] small = new short[executor.vocabularySize()];
            short[] whole = new short[executor.vocabularySize()];
            int[] shortChunks = greedy(gpu, executor, prompt, 512, small);
            int[] oneChunk = greedy(gpu, executor, prompt, PROMPT, whole);
            int[] odd = greedy(gpu, executor, prompt, 333, new short[executor.vocabularySize()]);
            System.out.println("512-token chunks: " + java.util.Arrays.toString(shortChunks));
            System.out.println("one chunk:        " + java.util.Arrays.toString(oneChunk));
            System.out.println("333-token chunks: " + java.util.Arrays.toString(odd));
            double worst = 0;
            for (int i = 0; i < small.length; i++) worst = Math.max(worst, Math.abs(bf(small[i]) - bf(whole[i])));
            System.out.println("largest difference between the two prompts' final logits: " + worst);
            assertEquals(java.util.Arrays.toString(shortChunks), java.util.Arrays.toString(oneChunk));
            assertEquals(java.util.Arrays.toString(shortChunks), java.util.Arrays.toString(odd));
        }
    }
}
