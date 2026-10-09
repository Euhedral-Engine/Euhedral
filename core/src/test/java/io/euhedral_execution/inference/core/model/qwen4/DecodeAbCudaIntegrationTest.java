package io.euhedral_execution.inference.core.model.qwen4;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.model.qwen4.loader.Mode;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/// Decode at 4096 context with two arms of the plan taken in turn, a block of tokens each, on one sequence in one
/// process: the difference of neighbouring blocks cancels what moves on the machine. Opt-in:
/// `EUHEDRAL_QWEN4_AB=A,B` (the plan's variants, ints), `EUHEDRAL_QWEN4_AB_BLOCKS` (pairs, 24), `_BLOCK` (tokens, 16).
@ModelGroup.OwnJvm
class DecodeAbCudaIntegrationTest {

    @Test
    void arms() throws Exception {
        String arms = System.getenv("EUHEDRAL_QWEN4_AB");
        assumeTrue(arms != null);
        int a = Integer.parseInt(arms.split(",")[0].trim());
        int b = Integer.parseInt(arms.split(",")[1].trim());
        int pairs = Integer.parseInt(System.getenv().getOrDefault("EUHEDRAL_QWEN4_AB_BLOCKS", "24"));
        int block = Integer.parseInt(System.getenv().getOrDefault("EUHEDRAL_QWEN4_AB_BLOCK", "16"));
        int context = 4096;
        int[] prompt = PerformanceCudaIntegrationTest.corpus(context + 64);
        int total = context + 2 * pairs * block + 64;
        var spec = new SharedFlashNext.Spec("ab", 0, Mode.TEXT, total, total);
        var loaded = SharedFlashNext.model(spec);
        ExecutionPlan plan = loaded.plan();
        var gpu = loaded.gpu();
        int vocabulary = plan.vocabularySize();
        var readback = gpu.allocateReadbackBuffer((long) vocabulary * 2);
        ExecutionPlan.LogitsSink sink = address -> gpu.copyDeviceToReadback(readback, address, vocabulary * 2L);
        int chunk = plan.maxRows();
        try (Sequence sequence = plan.newSequence()) {
            int at = 0;
            while (at < context) {
                int rows = Math.min(chunk, context - at);
                Blocking.step(plan, sequence, prompt, at, rows, null);
                at += rows;
            }
            int[] token = new int[1];
            int step = 0;
            // warm-up: both arms, so the shapes exist and the cache has settled
            for (int warm = 0; warm < 2 * block; warm++) {
                plan.variant(warm < block ? a : b);
                token[0] = prompt[(context + (step++) * 97) % prompt.length];
                Blocking.step(plan, sequence, token, 0, 1, sink);
            }
            List<double[]> results = new ArrayList<>();
            for (int pair = 0; pair < pairs; pair++) {
                double[] ms = new double[2];
                boolean aFirst = pair % 2 == 0;
                for (int turn = 0; turn < 2; turn++) {
                    boolean armA = (turn == 0) == aFirst;
                    plan.variant(armA ? a : b);
                    long begin = System.nanoTime();
                    for (int i = 0; i < block; i++) {
                        token[0] = prompt[(context + (step++) * 97) % prompt.length];
                        Blocking.step(plan, sequence, token, 0, 1, sink);
                        short[] logits = new short[16];
                        MemorySegment.copy(readback.segment(), ValueLayout.JAVA_SHORT, 0, logits, 0, 16);
                    }
                    ms[armA ? 0 : 1] = (System.nanoTime() - begin) / 1e6 / block;
                }
                results.add(ms);
            }
            double meanA = results.stream().mapToDouble(r -> r[0]).average().orElse(0);
            double meanB = results.stream().mapToDouble(r -> r[1]).average().orElse(0);
            int bAhead = (int) results.stream().filter(r -> r[1] < r[0]).count();
            double[] diff = results.stream().mapToDouble(r -> r[0] - r[1]).toArray();
            double mean = java.util.Arrays.stream(diff).average().orElse(0);
            double sd = Math.sqrt(java.util.Arrays.stream(diff)
                            .map(d -> (d - mean) * (d - mean))
                            .sum()
                    / (diff.length - 1));
            System.err.printf(
                    "AB variant %d: %.2f ms/token (%.1f tok/s) | variant %d: %.2f ms/token (%.1f tok/s) | change %+.1f%% (+-%.1f%%), B ahead in %d of %d pairs%n",
                    a,
                    meanA,
                    1000 / meanA,
                    b,
                    meanB,
                    1000 / meanB,
                    100 * (meanA / meanB - 1),
                    100 * sd / Math.sqrt(diff.length) / meanB,
                    bAhead,
                    results.size());
        } finally {
            readback.close();
        }
    }
}
