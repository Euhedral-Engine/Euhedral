package io.euhedral_execution.inference.core.model.qwen4;

import static io.euhedral_execution.inference.core.model.qwen4.QsaTestSupport.downloadFloats;
import static io.euhedral_execution.inference.core.model.qwen4.QsaTestSupport.downloadInts;
import static io.euhedral_execution.inference.core.model.qwen4.QsaTestSupport.downloadShorts;
import static io.euhedral_execution.inference.core.model.qwen4.QsaTestSupport.open;
import static io.euhedral_execution.inference.core.model.qwen4.QsaTestSupport.random;
import static io.euhedral_execution.inference.core.model.qwen4.QsaTestSupport.upload;
import static io.euhedral_execution.inference.core.model.qwen4.Reference.bf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The QSA operators on the GPU against CPU references written from the upstream source
/// (QsaReference), on synthetic data: norm and RoPE, block-key pooling across chunk
/// boundaries, block scores, top-k selection (score aware), the sparse attention and its split
/// merge. The whole layer is in QsaLayerCudaIntegrationTest.
class QsaOperatorCudaIntegrationTest {

    /// Elements agree within one BF16 step of the reference (a value on a rounding boundary may
    /// land on the other side).
    private static void assertBf16Close(short[] expected, short[] actual, String what) {
        assertEquals(expected.length, actual.length, what);
        int off = 0;
        for (int i = 0; i < expected.length; i++) {
            float e = bf(expected[i]), a = bf(actual[i]);
            if (Math.abs(e - a) > Math.abs(e) * 0.0079 + 1e-30)
                throw new AssertionError(what + ": element " + i + " expected " + e + " but was " + a);
            if (expected[i] != actual[i]) off++;
        }
        assertTrue(
                off <= expected.length * 0.01 + 2, what + ": " + off + " of " + expected.length + " elements differ");
    }

    @Test
    void headNormRopeMatchesTheReference() {
        SplittableRandom rng = new SplittableRandom(5);
        // rows, heads, width, row stride, head stride, rotary, position, step, in place
        int[][] cases = {
            {7, 24, 256, 12288, 512, 64, 0, 1, 1},
            {5, 2, 256, 512, 256, 64, 262000, 1, 0},
            {9, 4, 128, 640, 128, 64, 3000, 1, 1},
            {6, 1, 128, 128, 128, 64, 4000, 4, 1},
            {3, 4, 128, 512, 128, 0, 0, 1, 0},
            {4, 24, 256, 12288, 512, 64, 16_000_000, 1, 0},
        };
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            for (int[] c : cases) {
                int rows = c[0], heads = c[1], width = c[2], rowStride = c[3], headStride = c[4];
                int rotary = c[5], position = c[6], step = c[7];
                boolean inPlace = c[8] == 1;
                short[] input = random(rng, rows * rowStride, 2.0);
                short[] weight = random(rng, width, 0.3);
                long in = upload(gpu, arena, input), w = upload(gpu, arena, weight);
                long out = inPlace ? in : upload(gpu, arena, new short[rows * rowStride]);
                try {
                    QsaOps.headNormRope(
                            gpu,
                            in,
                            w,
                            out,
                            rows,
                            heads,
                            width,
                            rowStride,
                            headStride,
                            rowStride,
                            headStride,
                            rotary,
                            position,
                            step,
                            QsaReference.EPSILON,
                            QsaReference.THETA);
                    short[] actual = downloadShorts(gpu, arena, out, (long) rows * rowStride);
                    for (int row = 0; row < rows; row++)
                        for (int head = 0; head < heads; head++) {
                            int offset = row * rowStride + head * headStride;
                            short[] expected =
                                    QsaReference.normRope(input, offset, width, weight, position + row * step, rotary);
                            assertBf16Close(
                                    expected,
                                    Arrays.copyOfRange(actual, offset, offset + width),
                                    Arrays.toString(c) + " row " + row + " head " + head);
                        }
                    if (inPlace && headStride > width) {
                        // the gate halves between the heads are untouched
                        for (int row = 0; row < rows; row++)
                            for (int head = 0; head < heads; head++)
                                for (int d = width; d < headStride; d++)
                                    assertEquals(
                                            input[row * rowStride + head * headStride + d],
                                            actual[row * rowStride + head * headStride + d]);
                    }
                } finally {
                    if (!inPlace) gpu.free(out);
                    gpu.free(w);
                    gpu.free(in);
                }
            }
        }
    }

    /// The pattern repeated until `total` tokens are covered, the last chunk cut short.
    private static int[] chunking(int total, int... pattern) {
        java.util.List<Integer> chunks = new java.util.ArrayList<>();
        for (int sum = 0, i = 0; sum < total; i++) {
            int rows = Math.min(pattern[i % pattern.length], total - sum);
            chunks.add(rows);
            sum += rows;
        }
        return chunks.stream().mapToInt(Integer::intValue).toArray();
    }

    /// Chunks of every small size, blocks completing across their boundaries: the pooled block keys
    /// and the tail equal the keys computed from the whole raw sequence.
    @Test
    void pooledBlockKeysAndTailAreChunkIndependent() {
        SplittableRandom rng = new SplittableRandom(6);
        int tokens = 97, rawStride = 640;
        int[][] chunkings = {
            chunking(tokens, 1, 2, 1, 3, 4, 5, 1, 1, 1, 1, 7, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12),
            chunking(tokens, 97),
            chunking(tokens, 3),
            chunking(tokens, 4),
            chunking(tokens, 5),
            chunking(tokens, 1),
            chunking(tokens, 1, 4),
            chunking(tokens, 2, 3),
        };
        short[][] raw = new short[tokens][];
        for (int t = 0; t < tokens; t++) raw[t] = random(rng, QsaReference.INDEX_DIM, 1.5);
        short[] kNorm = random(rng, QsaReference.INDEX_DIM, 0.3);
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            long weight = upload(gpu, arena, kNorm);
            for (int[] chunks : chunkings) {
                int sum = Arrays.stream(chunks).sum();
                assertEquals(tokens, sum, "chunking sums to the sequence");
                int blocksTotal = tokens / 4;
                long blocks = gpu.allocate((long) blocksTotal * 128 * 2);
                long tail0 = gpu.allocate(3L * 128 * 2), tail1 = gpu.allocate(3L * 128 * 2);
                long tailIn = tail0, tailOut = tail1;
                try {
                    int start = 0;
                    for (int rows : chunks) {
                        short[] chunk = new short[rows * rawStride];
                        for (int r = 0; r < rows; r++)
                            System.arraycopy(raw[start + r], 0, chunk, r * rawStride + 512, 128);
                        long raw0 = upload(gpu, arena, chunk);
                        try {
                            long rawKeys = raw0 + 512 * 2;
                            QsaOps.poolKeys(gpu, rawKeys, tailIn, blocks, rows, start, rawStride);
                            int completed = QsaOps.completedBlocks(start, rows);
                            if (completed > 0) {
                                long fresh = blocks + (long) (start / 4) * 128 * 2;
                                QsaOps.headNormRope(
                                        gpu,
                                        fresh,
                                        weight,
                                        fresh,
                                        completed,
                                        1,
                                        128,
                                        128,
                                        128,
                                        128,
                                        128,
                                        64,
                                        (start / 4) * 4,
                                        4,
                                        QsaReference.EPSILON,
                                        QsaReference.THETA);
                            }
                            QsaOps.tail(gpu, rawKeys, tailIn, tailOut, rows, start, rawStride);
                        } finally {
                            gpu.free(raw0);
                        }
                        long swap = tailIn;
                        tailIn = tailOut;
                        tailOut = swap;
                        start += rows;
                    }
                    short[] actual = downloadShorts(gpu, arena, blocks, (long) blocksTotal * 128);
                    for (int j = 0; j < blocksTotal; j++)
                        assertBf16Close(
                                QsaReference.blockKey(raw, j, kNorm),
                                Arrays.copyOfRange(actual, j * 128, (j + 1) * 128),
                                Arrays.toString(chunks) + " block " + j);
                    // 97 tokens: one raw token is left of the incomplete block (token 96)
                    short[] tail = downloadShorts(gpu, arena, tailIn, 128);
                    assertEquals(0, Arrays.compare(raw[96], tail), "tail holds the raw key of token 96");
                } finally {
                    gpu.free(tail1);
                    gpu.free(tail0);
                    gpu.free(blocks);
                }
            }
            gpu.free(weight);
        }
    }

    private record ScoreCase(String name, int start, int rows, int blocksTotal, int tileSplit, boolean ties) {}

    @Test
    void blockScoresAndSelectionMatchTheReference() {
        SplittableRandom rng = new SplittableRandom(7);
        ScoreCase[] cases = {
            new ScoreCase("first block", 0, 5, 1, 5, false),
            new ScoreCase("under budget", 100, 30, 32, 30, false),
            new ScoreCase("crossing the budget", 2040, 24, 516, 10, false),
            new ScoreCase("over budget", 2600, 40, 660, 40, false),
            new ScoreCase("budget exactly", 2047, 6, 513, 6, false),
            new ScoreCase("ties", 2600, 8, 652, 8, true),
            new ScoreCase("long history", 262_000, 3, 65_500, 3, false),
        };
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            for (ScoreCase c : cases) {
                int blocks = c.blocksTotal();
                short[][] keys = new short[blocks][];
                short[] flatKeys = new short[blocks * 128];
                short[] sharedKey = random(rng, 128, 1.0);
                for (int j = 0; j < blocks; j++) {
                    keys[j] = c.ties() ? (j % 3 == 0 ? sharedKey : sharedKey.clone()) : random(rng, 128, 1.0);
                    if (c.ties() && j % 5 == 0) keys[j] = random(rng, 128, 1.0);
                    System.arraycopy(keys[j], 0, flatKeys, j * 128, 128);
                }
                int rows = c.rows(), qStride = 640;
                short[] queries = random(rng, rows * qStride, 1.0);
                long keyAddress = upload(gpu, arena, flatKeys), queryAddress = upload(gpu, arena, queries);
                int stride = (blocks + 31) & ~31;
                long scoreAddress = gpu.allocate((long) rows * stride * 4);
                long ids = gpu.allocate((long) rows * 512 * 4), counts = gpu.allocate((long) rows * 4);
                try {
                    for (int begin = 0; begin < rows; begin += c.tileSplit()) {
                        int count = Math.min(c.tileSplit(), rows - begin);
                        int tileBlocks = Math.max(1, (c.start() + begin + count) / 4);
                        QsaOps.scores(
                                gpu,
                                queryAddress,
                                keyAddress,
                                scoreAddress,
                                begin,
                                count,
                                c.start(),
                                qStride,
                                stride,
                                tileBlocks);
                        QsaOps.select(gpu, scoreAddress, ids, counts, begin, count, c.start(), stride, 512);
                        float[] gpuScores = downloadFloats(gpu, arena, scoreAddress, (long) count * stride);
                        for (int r = begin; r < begin + count; r++) {
                            int p = c.start() + r, nb = (p + 1) / 4;
                            if (nb == 0) continue;
                            short[][] q = new short[4][];
                            for (int h = 0; h < 4; h++)
                                q[h] = Arrays.copyOfRange(queries, r * qStride + h * 128, r * qStride + h * 128 + 128);
                            double[] expected = QsaReference.scores(q, keys, nb);
                            double scale = Arrays.stream(expected).max().orElse(0);
                            double worst = 0;
                            for (int j = 0; j < nb; j++) {
                                double got = gpuScores[(r - begin) * stride + j];
                                worst = Math.max(worst, Math.abs(got - expected[j]));
                            }
                            assertTrue(
                                    worst <= 2e-5 * scale + 1e-9,
                                    c.name() + " row " + r + ": score error " + worst + " of " + scale);
                            int count0 = downloadInts(gpu, arena, counts + 4L * r, 1)[0];
                            int[] selected = downloadInts(gpu, arena, ids + 2048L * r, count0);
                            QsaReference.checkSelection(
                                    expected, selected, count0, 512, 4e-5 * scale + 1e-9, c.name() + " row " + r);
                            if (c.ties() || nb <= 512) {
                                // an exactly tied or fully selected row has a unique answer under the lower-id rule
                                int[] unique = QsaReference.select(
                                        c.ties() ? gpuTieScores(gpuScores, (r - begin) * stride, nb) : expected, 512);
                                assertEquals(
                                        Arrays.toString(unique),
                                        Arrays.toString(selected),
                                        c.name() + " row " + r + " ids");
                            }
                        }
                    }
                } finally {
                    gpu.free(counts);
                    gpu.free(ids);
                    gpu.free(scoreAddress);
                    gpu.free(queryAddress);
                    gpu.free(keyAddress);
                }
            }
        }
    }

    /// The GPU's own scores as doubles: selection is a function of them, so the lower-id rule is
    /// exact on them.
    private static double[] gpuTieScores(float[] scores, int offset, int n) {
        double[] values = new double[n];
        for (int i = 0; i < n; i++) values[i] = scores[offset + i];
        return values;
    }

    @Test
    void selectionBreaksTiesToTheLowerBlockId() {
        // Every block scores exactly 0 (zero keys): the selection is blocks 0 ..< 512 of every row.
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            int blocks = 3000, rows = 4, start = 11_000, stride = (blocks + 31) & ~31;
            long keys = upload(gpu, arena, new short[blocks * 128]);
            long queries = upload(gpu, arena, random(new SplittableRandom(3), rows * 640, 1.0));
            long scores = gpu.allocate((long) rows * stride * 4);
            long ids = gpu.allocate((long) rows * 512 * 4), counts = gpu.allocate(rows * 4L);
            try {
                QsaOps.scores(gpu, queries, keys, scores, 0, rows, start, 640, stride, (start + rows) / 4);
                QsaOps.select(gpu, scores, ids, counts, 0, rows, start, stride, 512);
                int[] selected = downloadInts(gpu, arena, ids, rows * 512L);
                for (int r = 0; r < rows; r++) {
                    assertEquals(512, downloadInts(gpu, arena, counts + 4L * r, 1)[0]);
                    for (int i = 0; i < 512; i++) assertEquals(i, selected[r * 512 + i], "row " + r + " slot " + i);
                }
            } finally {
                gpu.free(counts);
                gpu.free(ids);
                gpu.free(scores);
                gpu.free(queries);
                gpu.free(keys);
            }
        }
    }
}
