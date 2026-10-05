package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4ExpertReference.HIDDEN;
import static io.euhedral_execution.inference.core.qwen4.Qwen4ExpertReference.INTER;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bf;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bits;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The routed-expert kernels on synthetic NVFP4 experts: every stage against a CPU reference written from upstream, and
/// the bit-exactness the design promises (a pair's result does not depend on the wave split, the work items or the
/// chunking of rows). Real weights against upstream fixtures are in Qwen4ExpertFixtureCudaIntegrationTest.
class Qwen4ExpertCudaIntegrationTest {

    private static CudaGpuMemory open() {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null, "euhedral.cuda.library is not set");
        return new CudaGpuMemory(Path.of(library));
    }

    private static Qwen4ExpertHarness.Records records(byte[][] records) {
        return (expert, destination) ->
                MemorySegment.copy(records[expert], 0, destination, ValueLayout.JAVA_BYTE, 0, records[expert].length);
    }

    private static byte[][] randomRecords(int experts, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        byte[][] records = new byte[experts][];
        for (int e = 0; e < experts; e++) records[e] = Qwen4ExpertReference.randomRecord(random);
        return records;
    }

    /// Router output: `topK` distinct experts per row, drawn with probability falling with the id when `skewed`; one
    /// entry in `paddingEvery` rows is the padding id.
    static void route(
            SplittableRandom random, int rows, int topK, int experts, boolean skewed, int[] ids, short[] weights) {
        for (int r = 0; r < rows; r++) {
            for (int k = 0; k < topK; k++) {
                int id;
                boolean fresh;
                do {
                    id = skewed
                            ? (int) Math.min(experts - 1, Math.floor(Math.pow(random.nextDouble(), 2.5) * experts))
                            : random.nextInt(experts);
                    fresh = true;
                    for (int j = 0; j < k; j++) if (ids[r * topK + j] == id) fresh = false;
                } while (!fresh);
                ids[r * topK + k] = id;
                weights[r * topK + k] = bits(0.02f + 0.3f * random.nextFloat());
            }
        }
    }

    /// Within two BF16 steps: a rounding that flips in gate or up moves the activation by one step and the product by
    /// another.
    private static void assertNear(short[] expected, short[] actual, String what) {
        assertEquals(expected.length, actual.length, what);
        int different = 0;
        for (int i = 0; i < expected.length; i++) {
            float e = bf(expected[i]), a = bf(actual[i]);
            if (Math.abs(e - a) > Math.abs(e) * 0.0157 + 1e-20)
                throw new AssertionError(what + ": element " + i + " expected " + e + " but was " + a);
            if (expected[i] != actual[i]) different++;
        }
        assertTrue(
                different <= expected.length * 0.03 + 1,
                what + ": " + different + " of " + expected.length + " differ");
    }

    @Test
    void everyStageMatchesTheReference() throws IOException {
        int experts = 7, topK = 3, rows = 12;
        SplittableRandom random = new SplittableRandom(17);
        byte[][] records = randomRecords(experts, 41);
        Qwen4ExpertReference.Expert[] expanded = new Qwen4ExpertReference.Expert[experts];
        for (int e = 0; e < experts; e++) expanded[e] = Qwen4ExpertReference.expand(records[e]);
        short[] x = Qwen4ExpertReference.randomBf16(random, rows * HIDDEN, 1.0);
        int[] ids = new int[rows * topK];
        short[] weights = new short[rows * topK];
        route(random, rows, topK, experts, false, ids, weights);
        ids[4] = experts; // a padding entry is skipped
        ids[17] = experts;
        try (CudaGpuMemory gpu = open();
                Qwen4ExpertHarness harness = new Qwen4ExpertHarness(gpu, experts, topK, rows, 3, 24)) {
            // the weighted rows of every pair as the kernels produced them, to replay the routed sum
            short[][] weightedByPair = new short[rows * topK][];
            int[] pairOfEntry = new int[rows * topK];
            short[] sum = harness.run(rows, ids, weights, x, records(records), (wave, plan, act, weighted) -> {
                int pairs = plan.wavePairCount(wave);
                short[] actBits = harness.download(act, pairs * INTER);
                short[] weightedBits = harness.download(weighted, pairs * HIDDEN);
                int pair = 0;
                for (int i = 0; i < plan.waveExpertCount(wave); i++) {
                    int expert = plan.waveExpert(wave, i);
                    for (int entry = 0; entry < rows * topK; entry++) {
                        if (ids[entry] != expert) continue;
                        int row = entry / topK;
                        short[] expectedAct = Qwen4ExpertReference.act(expanded[expert], x, row * HIDDEN);
                        short[] gotAct = java.util.Arrays.copyOfRange(actBits, pair * INTER, (pair + 1) * INTER);
                        assertNear(expectedAct, gotAct, "act of row " + row + " expert " + expert);
                        // the down projection and weighting on the kernel's own act
                        short[] expectedWeighted =
                                Qwen4ExpertReference.weighted(expanded[expert], gotAct, weights[entry]);
                        short[] gotWeighted =
                                java.util.Arrays.copyOfRange(weightedBits, pair * HIDDEN, (pair + 1) * HIDDEN);
                        assertNear(expectedWeighted, gotWeighted, "weighted of row " + row + " expert " + expert);
                        weightedByPair[entry] = gotWeighted;
                        pair++;
                    }
                }
                assertEquals(pairs, pair);
            });
            for (int row = 0; row < rows; row++) {
                short[] expected = new short[HIDDEN];
                for (int expert = 0; expert < experts; expert++)
                    for (int k = 0; k < topK; k++)
                        if (ids[row * topK + k] == expert)
                            Qwen4ExpertReference.accumulate(expected, weightedByPair[row * topK + k]);
                assertArrayEquals(
                        expected, java.util.Arrays.copyOfRange(sum, row * HIDDEN, (row + 1) * HIDDEN), "row " + row);
            }
        }
    }

    /// The same chunk played with different wave limits, in one piece, in chunks and row by row gives the same bits.
    @Test
    void resultsAreBitwiseIndependentOfWavesChunksAndWorkItems() throws IOException {
        int experts = 12, topK = 4, rows = 70;
        SplittableRandom random = new SplittableRandom(23);
        byte[][] records = randomRecords(experts, 59);
        short[] x = Qwen4ExpertReference.randomBf16(random, rows * HIDDEN, 1.0);
        int[] ids = new int[rows * topK];
        short[] weights = new short[rows * topK];
        route(random, rows, topK, experts, true, ids, weights);
        short[] baseline;
        try (CudaGpuMemory gpu = open();
                Qwen4ExpertHarness all = new Qwen4ExpertHarness(gpu, experts, topK, rows, experts, rows * topK)) {
            baseline = all.run(rows, ids, weights, x, records(records), null);
            assertTrue(all.plan.expertPairCount(0) > 8 * all.plan.expertPairCount(experts - 1) / 2, "skewed routing");
            int[][] limits = {{5, 300}, {1, rows}, {3, 100}, {2, 70}};
            for (int[] limit : limits) {
                try (Qwen4ExpertHarness split = new Qwen4ExpertHarness(gpu, experts, topK, rows, limit[0], limit[1])) {
                    short[] result = split.run(rows, ids, weights, x, records(records), null);
                    assertTrue(split.plan.waveCount() > 1, "the limits split the chunk");
                    assertArrayEquals(baseline, result, "waves of " + limit[0] + " experts, " + limit[1] + " pairs");
                }
            }
            for (int chunk : new int[] {7, 1}) {
                try (Qwen4ExpertHarness part = new Qwen4ExpertHarness(gpu, experts, topK, chunk, 4, chunk * topK)) {
                    short[] stitched = new short[rows * HIDDEN];
                    for (int first = 0; first < rows; first += chunk) {
                        int n = Math.min(chunk, rows - first);
                        short[] result = part.run(
                                n,
                                java.util.Arrays.copyOfRange(ids, first * topK, (first + n) * topK),
                                java.util.Arrays.copyOfRange(weights, first * topK, (first + n) * topK),
                                java.util.Arrays.copyOfRange(x, first * HIDDEN, (first + n) * HIDDEN),
                                records(records),
                                null);
                        System.arraycopy(result, 0, stitched, first * HIDDEN, n * HIDDEN);
                    }
                    assertArrayEquals(baseline, stitched, "chunks of " + chunk + " rows");
                }
            }
        }
    }

    /// A full 512-row chunk with skewed routing: bit-identical across wave splits, and three rows against the CPU.
    @Test
    void aFullChunkMatchesTheReferenceAndEveryWaveSplit() throws IOException {
        int experts = 16, topK = 10, rows = 512;
        SplittableRandom random = new SplittableRandom(29);
        byte[][] records = randomRecords(experts, 73);
        short[] x = Qwen4ExpertReference.randomBf16(random, rows * HIDDEN, 1.0);
        int[] ids = new int[rows * topK];
        short[] weights = new short[rows * topK];
        route(random, rows, topK, experts, true, ids, weights);
        try (CudaGpuMemory gpu = open();
                Qwen4ExpertHarness big = new Qwen4ExpertHarness(gpu, experts, topK, rows, 16, 5120);
                Qwen4ExpertHarness small = new Qwen4ExpertHarness(gpu, experts, topK, rows, 3, 700)) {
            short[] result = big.run(rows, ids, weights, x, records(records), null);
            short[] split = small.run(rows, ids, weights, x, records(records), null);
            assertTrue(small.plan.waveCount() >= 6);
            assertArrayEquals(result, split);
            Qwen4ExpertReference.Expert[] expanded = new Qwen4ExpertReference.Expert[experts];
            for (int e = 0; e < experts; e++) expanded[e] = Qwen4ExpertReference.expand(records[e]);
            for (int row : new int[] {0, 311, 511}) {
                short[] expected = new short[HIDDEN];
                List<Integer> order = new ArrayList<>();
                for (int k = 0; k < topK; k++) order.add(k);
                order.sort((a, b) -> Integer.compare(ids[row * topK + a], ids[row * topK + b]));
                for (int k : order) {
                    int expert = ids[row * topK + k];
                    short[] act = Qwen4ExpertReference.act(expanded[expert], x, row * HIDDEN);
                    Qwen4ExpertReference.accumulate(
                            expected, Qwen4ExpertReference.weighted(expanded[expert], act, weights[row * topK + k]));
                }
                short[] actual = java.util.Arrays.copyOfRange(result, row * HIDDEN, (row + 1) * HIDDEN);
                // ten BF16 additions of values near 1 accumulate a few ulps of difference from flipped roundings
                int different = 0;
                for (int j = 0; j < HIDDEN; j++) {
                    float e = bf(expected[j]), a = bf(actual[j]);
                    assertTrue(
                            Math.abs(e - a) <= 0.02f + Math.abs(e) * 0.03f,
                            "row " + row + " column " + j + ": " + e + " vs " + a);
                    if (expected[j] != actual[j]) different++;
                }
                assertTrue(different < HIDDEN / 4, "row " + row + ": " + different + " columns differ");
            }
        }
    }
}
