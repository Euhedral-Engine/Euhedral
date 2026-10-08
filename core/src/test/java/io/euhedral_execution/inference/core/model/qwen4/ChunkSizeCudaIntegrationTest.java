package io.euhedral_execution.inference.core.model.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The prompt's chunking is not part of the answer: the real model reads the same prompt in chunks of 512
/// tokens and in the plan's own (longer) chunk, and the greedy continuation is the same. A prompt of 700
/// tokens is a full chunk of 512 and a partial one, and one chunk (past 512 rows) in the plan's workspace, which is
/// where a longer chunk's larger waves and scratch differ from the short ones'. (700 rather than 1,500: every
/// prefill chunk streams every expert of the model through the cache, so the chunk count is the cost, and past 512 rows
/// a chunk already routes to nearly every expert of a layer.) Opt-in with the artifact present.
@ModelGroup.FlashNext
@Timeout(value = 5, unit = TimeUnit.MINUTES)
class ChunkSizeCudaIntegrationTest {

    private static final int PROMPT = 700;
    private static final int STEPS = 8;

    private static float bf(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }

    private static int argmax(short[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) if (bf(logits[i]) > bf(logits[best])) best = i;
        return best;
    }

    /// The greedy tokens after prefilling `prompt` in chunks of `chunk`, and the first step's logits.
    private static int[] greedy(CudaGpuMemory gpu, ExecutionPlan executor, int[] prompt, int chunk, short[] firstLogits)
            throws Exception {
        int vocabulary = executor.vocabularySize();
        var readback = gpu.allocateReadbackBuffer((long) vocabulary * 2);
        ExecutionPlan.LogitsSink sink = address -> gpu.copyDeviceToReadback(readback, address, vocabulary * 2L);
        int[] out = new int[STEPS];
        try (Sequence sequence = executor.newSequence()) {
            int at = 0;
            while (at < prompt.length) {
                int rows = Math.min(chunk, prompt.length - at);
                Blocking.step(executor, sequence, prompt, at, rows, at + rows == prompt.length ? sink : null);
                at += rows;
            }
            short[] logits = new short[vocabulary];
            MemorySegment.copy(readback.segment(), ValueLayout.JAVA_SHORT, 0, logits, 0, vocabulary);
            System.arraycopy(logits, 0, firstLogits, 0, vocabulary);
            int[] token = new int[1];
            for (int step = 0; step < STEPS; step++) {
                out[step] = argmax(logits);
                token[0] = out[step];
                Blocking.step(executor, sequence, token, 0, 1, sink);
                MemorySegment.copy(readback.segment(), ValueLayout.JAVA_SHORT, 0, logits, 0, vocabulary);
            }
            return out;
        } finally {
            readback.close();
        }
    }

    @Test
    void theGreedyContinuationDoesNotDependOnTheChunkSize() throws Exception {
        assumeTrue(TestSupport.hasArtifact(), "no artifact");
        long begun = System.nanoTime();
        int[] prompt = PerformanceCudaIntegrationTest.corpus(PROMPT);
        System.out.println("TIMING corpus " + (System.nanoTime() - begun) / 1_000_000 + " ms");
        var loaded = SharedFlashNext.model(SharedFlashNext.ROOMY);
        var gpu = loaded.gpu();
        System.out.println("TIMING corpus+model " + (System.nanoTime() - begun) / 1_000_000 + " ms");
        ExecutionPlan executor = loaded.plan();
        System.out.println("plan chunk: " + executor.maxRows() + " tokens");
        assertTrue(executor.maxRows() >= PROMPT, "the plan's chunk holds the prompt: " + executor.maxRows());
        short[] small = new short[executor.vocabularySize()];
        short[] whole = new short[executor.vocabularySize()];
        long t0 = System.nanoTime();
        int[] shortChunks = greedy(gpu, executor, prompt, 512, small);
        int[] oneChunk = greedy(gpu, executor, prompt, PROMPT, whole);
        System.out.println("TIMING two runs " + (System.nanoTime() - t0) / 1_000_000 + " ms");
        int[] odd = greedy(gpu, executor, prompt, 411, new short[executor.vocabularySize()]);
        System.out.println("512-token chunks: " + java.util.Arrays.toString(shortChunks));
        System.out.println("one chunk:        " + java.util.Arrays.toString(oneChunk));
        System.out.println("411-token chunks: " + java.util.Arrays.toString(odd));
        double worst = 0;
        for (int i = 0; i < small.length; i++) worst = Math.max(worst, Math.abs(bf(small[i]) - bf(whole[i])));
        System.out.println("largest difference between the two prompts' final logits: " + worst);
        assertEquals(java.util.Arrays.toString(shortChunks), java.util.Arrays.toString(oneChunk));
        assertEquals(java.util.Arrays.toString(shortChunks), java.util.Arrays.toString(odd));
    }
}
