package io.euhedral_execution.inference.core.model.qwen4;

import static io.euhedral_execution.inference.core.model.qwen4.QsaTestSupport.downloadInts;
import static io.euhedral_execution.inference.core.model.qwen4.QsaTestSupport.downloadShorts;
import static io.euhedral_execution.inference.core.model.qwen4.QsaTestSupport.open;
import static io.euhedral_execution.inference.core.model.qwen4.QsaTestSupport.random;
import static io.euhedral_execution.inference.core.model.qwen4.QsaTestSupport.upload;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/// The whole QSA attention block of layer 3 of the real artifact (NVFP4 projections, BF16 indexer):
/// against the fixtures of tools/flash_next_reference.py (upstream's unmodified eager
/// implementation, `layer_qsa_short` and `layer_qsa_long`) and for independence from the chunking
/// of the sequence. The linears run row-exact, so a row's projections do not depend on the chunk it
/// is in. Skipped without the artifact (or the fixtures).
class QsaLayerCudaIntegrationTest {

    private static final int LAYER = 3;
    /// A block may be selected or omitted against the reference only if its score is within this
    /// fraction of the row's highest score of the k-th score. Scores come from BF16 queries and
    /// keys that agree with the reference's to a few BF16 steps, so near-tied blocks at the
    /// boundary may swap.
    private static final double SELECTION_EPSILON = 2e-3;

    /// Fixture roots in order of preference; the first fixture of a case recorded with the NVFP4
    /// cache wins.
    private static List<Path> fixtureRoots() {
        List<Path> roots = new ArrayList<>();
        String configured = System.getProperty("euhedral.qwen4.fixtures");
        if (configured != null) roots.add(Path.of(configured));
        String home = System.getProperty("user.home");
        roots.add(Path.of(home, "fixtures", "flash-next-qsa-nvfp4"));
        roots.add(Path.of(home, "fixtures", "flash-next-nvfp4kv"));
        roots.add(Path.of(home, "fixtures", "flash-next"));
        return roots;
    }

    private static ReferenceFixtures fixture(String name) throws IOException {
        ReferenceFixtures fallback = null;
        for (Path root : fixtureRoots()) {
            Path directory = root.resolve(name);
            if (!ReferenceFixtures.exists(directory)) continue;
            ReferenceFixtures fixtures = new ReferenceFixtures(directory);
            if (fixtures.metadata().path("kv_format").asText("bf16").equals("nvfp4")) return fixtures;
            if (fallback == null) fallback = fixtures;
        }
        assumeTrue(fallback != null, "no fixtures for " + name);
        return fallback;
    }

    /// The layer's weights on the device, loaded by the first test and kept for the others (immutable: a test
    /// that selects row-exact numerics restores them).
    private static QsaTestSupport.Loaded shared;

    private static synchronized QsaTestSupport.Loaded shared() throws IOException {
        if (shared == null) shared = QsaTestSupport.Loaded.load(open(), LAYER);
        return shared;
    }

    @AfterAll
    static synchronized void freeWeights() {
        if (shared == null) return;
        CudaGpuMemory gpu = shared.gpu();
        shared.close();
        gpu.close();
        shared = null;
    }

    private record Stat(String what, double relativeRms, double worstInRms) {}

    private static final List<Stat> STATS = new ArrayList<>();

    private static Stat compare(String what, short[] expected, short[] actual) {
        assertEquals(expected.length, actual.length, what);
        Stat stat = new Stat(
                what, QsaReference.relativeRms(expected, actual), QsaReference.maxDifferenceInRms(expected, actual));
        STATS.add(stat);
        return stat;
    }

    /// Per tensor name, the largest relative RMS error over the chunks and the largest single
    /// difference in RMS units.
    private static void report(String title) {
        java.util.Map<String, double[]> worst = new java.util.TreeMap<>();
        for (Stat stat : STATS) {
            double[] w = worst.computeIfAbsent(stat.what(), k -> new double[2]);
            w[0] = Math.max(w[0], stat.relativeRms());
            w[1] = Math.max(w[1], stat.worstInRms());
        }
        for (var entry : worst.entrySet())
            System.out.printf(
                    "%s %-14s relative RMS %.3e  worst %.3f RMS%n",
                    title, entry.getKey(), entry.getValue()[0], entry.getValue()[1]);
        STATS.clear();
    }

    private static short[] columns(short[] matrix, int rows, int stride, int offset, int width) {
        short[] out = new short[rows * width];
        for (int r = 0; r < rows; r++) System.arraycopy(matrix, r * stride + offset, out, r * width, width);
        return out;
    }

    private static short[] qHeadParts(short[] qProj, int rows, int part) {
        // [row][24 heads][q 256 | gate 256]: part 0 = q, 1 = gate
        short[] out = new short[rows * 24 * 256];
        for (int r = 0; r < rows; r++)
            for (int h = 0; h < 24; h++)
                System.arraycopy(qProj, r * 12288 + h * 512 + part * 256, out, (r * 24 + h) * 256, 256);
        return out;
    }

    @Test
    void shortFixtureMatchesUpstream() throws IOException {
        runFixture("layer_qsa_short");
    }

    /// Layer 3 of the model case `short` (real activations of a real prompt, not synthetic rows).
    @Test
    void modelCaseMatchesUpstream() throws IOException {
        runFixture("short");
    }

    @Test
    void longFixtureMatchesUpstream() throws IOException {
        runFixture("layer_qsa_long");
    }

    private void runFixture(String name) throws IOException {
        ReferenceFixtures fixtures = fixture(name);
        boolean nvfp4 = fixtures.metadata().path("kv_format").asText("bf16").equals("nvfp4");
        int[] chunks = new int[fixtures.metadata().get("chunks").size()];
        for (int i = 0; i < chunks.length; i++)
            chunks[i] = fixtures.metadata().get("chunks").get(i).asInt();
        int total = Arrays.stream(chunks).sum(),
                maxRows = Arrays.stream(chunks).max().orElseThrow();
        QsaTestSupport.Loaded loaded = shared();
        CudaGpuMemory gpu = loaded.gpu();
        try (Arena arena = Arena.ofConfined()) {
            gpu.selectRowExact(true);
            QsaLayer layer = new QsaLayer(QsaLayer.Config.of(loaded.artifact().config(), total));
            long scratchBytes = layer.scratchBytes(maxRows);
            long base = gpu.allocate(scratchBytes);
            QsaLayer.Scratch scratch = layer.scratch(base, maxRows);
            long output = gpu.allocate((long) maxRows * 2560 * 2), core = gpu.allocate((long) maxRows * 6144 * 2);
            try (QsaState state = new QsaState(gpu, 512, total)) {
                int start = 0;
                for (int k = 0; k < chunks.length; k++) {
                    int rows = chunks[k];
                    String p = "c" + k + "/L" + LAYER + "/attn/";
                    long input = upload(gpu, arena, fixtures.bf16(p + "in"));
                    try {
                        layer.run(gpu, loaded.weights(), state, input, rows, output, scratch, core);
                        compare(
                                "out",
                                fixtures.bf16(p + "out"),
                                downloadShorts(gpu, arena, output, (long) rows * 2560));
                        compare(
                                "core",
                                fixtures.bf16(p + "core"),
                                downloadShorts(gpu, arena, core, (long) rows * 6144));
                        if (fixtures.has(p + "q")) {
                            short[] qProj = downloadShorts(gpu, arena, scratch.qProj(), (long) rows * 12288);
                            compare("q", fixtures.bf16(p + "q"), qHeadParts(qProj, rows, 0));
                            compare("gate", fixtures.bf16(p + "gate"), qHeadParts(qProj, rows, 1));
                            compare(
                                    "k",
                                    fixtures.bf16(p + "k"),
                                    downloadShorts(gpu, arena, scratch.kNormed(), rows * 512L));
                            compare(
                                    "v",
                                    fixtures.bf16(p + "v"),
                                    downloadShorts(gpu, arena, scratch.vProj(), rows * 512L));
                            short[] index = downloadShorts(gpu, arena, scratch.indexProj(), rows * 640L);
                            compare(
                                    "index_q_rope",
                                    fixtures.bf16(p + "index_q_rope"),
                                    columns(index, rows, 640, 0, 512));
                            compare(
                                    "index_raw_k",
                                    fixtures.bf16(p + "index_raw_k"),
                                    columns(index, rows, 640, 512, 128));
                            compare(
                                    "gated",
                                    fixtures.bf16(p + "gated"),
                                    downloadShorts(gpu, arena, scratch.gated(), (long) rows * 6144));
                            if (fixtures.has(p + "index_block_keys")) {
                                short[] expected = fixtures.bf16(p + "index_block_keys");
                                int blocks = expected.length / 128;
                                compare(
                                        "block_keys",
                                        expected,
                                        downloadShorts(gpu, arena, state.blockKeys(), blocks * 128L));
                            }
                            if (fixtures.has(p + "kv_k")) compareCache(gpu, arena, fixtures, p, state, start + rows);
                            if (start + rows <= 2048)
                                compareExact(
                                        gpu,
                                        arena,
                                        p,
                                        state,
                                        start,
                                        rows,
                                        qProj,
                                        downloadShorts(gpu, arena, core, (long) rows * 6144));
                        }
                        if (start + rows - 1 >= 2051) checkSelection(gpu, arena, fixtures, p, scratch, start, rows);
                    } finally {
                        gpu.free(input);
                    }
                    state.commit();
                    start += rows;
                }
            } finally {
                gpu.free(core);
                gpu.free(output);
                gpu.free(base);
                gpu.selectRowExact(false);
            }
        }
        report(name + (nvfp4 ? " (nvfp4 cache)" : " (bf16 cache)"));
    }

    /// The device's cache pages, decoded, against the fixture's cache: the fixture holds the
    /// codec's result rounded to BF16, the pages the exact represented values.
    private static void compareCache(
            CudaGpuMemory gpu, Arena arena, ReferenceFixtures fixtures, String p, QsaState state, int length)
            throws IOException {
        for (String plane : new String[] {"kv_k", "kv_v"}) {
            short[] expected = fixtures.bf16(p + plane);
            long table = plane.equals("kv_k") ? state.keyPages() : state.valuePages();
            double error = 0, norm = 0;
            for (int head = 0; head < 2; head++) {
                double[][] decoded = QsaTestSupport.decodeCache(gpu, arena, table, length, 2, head);
                for (int t = 0; t < length; t++)
                    for (int d = 0; d < 256; d++) {
                        double e = Reference.bf(expected[(head * length + t) * 256 + d]);
                        error += (e - decoded[t][d]) * (e - decoded[t][d]);
                        norm += e * e;
                    }
            }
            STATS.add(new Stat("cache_" + plane.substring(3), Math.sqrt(error / norm), 0));
        }
    }

    /// The core of a chunk against double-precision attention over the device's own decoded cache,
    /// for the dense rows (no selection) and the device's own queries.
    private static void compareExact(
            CudaGpuMemory gpu,
            Arena arena,
            String p,
            QsaState state,
            int start,
            int rows,
            short[] qProj,
            short[] core) {
        int length = start + rows;
        double[][][] k = new double[2][][], v = new double[2][][];
        for (int h = 0; h < 2; h++) {
            k[h] = QsaTestSupport.decodeCache(gpu, arena, state.keyPages(), length, 2, h);
            v[h] = QsaTestSupport.decodeCache(gpu, arena, state.valuePages(), length, 2, h);
        }
        short[] q = qHeadParts(qProj, rows, 0);
        double[] expected = new double[rows * 6144];
        for (int r = 0; r < rows; r++)
            for (int head = 0; head < 24; head++)
                System.arraycopy(
                        QsaReference.attend(q, (r * 24 + head) * 256, k[head / 12], v[head / 12], start + r, null, 0),
                        0,
                        expected,
                        r * 6144 + head * 256,
                        256);
        STATS.add(new Stat("core_vs_exact", QsaReference.relativeRms(expected, core), 0));
    }

    /// The GPU's block ids of a selecting chunk against the fixture's: score aware where the
    /// fixture has the scores.
    private static void checkSelection(
            CudaGpuMemory gpu,
            Arena arena,
            ReferenceFixtures fixtures,
            String p,
            QsaLayer.Scratch scratch,
            int start,
            int rows)
            throws IOException {
        int[] counts = downloadInts(gpu, arena, scratch.counts(), rows);
        int[] ids = downloadInts(gpu, arena, scratch.ids(), (long) rows * 512);
        int[] reference = fixtures.i32(p + (fixtures.has(p + "block_ids_sorted") ? "block_ids_sorted" : "block_ids"));
        boolean scored = fixtures.has(p + "index_scores");
        float[] scores = scored ? fixtures.f32(p + "index_scores") : null;
        int width = scored ? (int) fixtures.shape(p + "index_scores")[1] : 0;
        int differing = 0, rowsWithDifferences = 0;
        double worstGap = 0;
        for (int r = 0; r < rows; r++) {
            int nb = (start + r + 1) / 4, k = Math.min(512, nb);
            assertEquals(k, counts[r], p + " row " + r + " selected blocks");
            int[] mine = Arrays.copyOfRange(ids, r * 512, r * 512 + k);
            if (scored) {
                double[] rowScores = new double[nb];
                for (int j = 0; j < nb; j++) rowScores[j] = scores[r * width + j];
                double max = Arrays.stream(rowScores).max().orElse(0);
                QsaReference.Check check =
                        QsaReference.checkSelection(rowScores, mine, k, 512, SELECTION_EPSILON * max, p + " row " + r);
                if (check.differing() > 0) rowsWithDifferences++;
                differing += check.differing();
                worstGap = Math.max(worstGap, check.gap() / max);
            } else {
                boolean[] chosen = new boolean[nb];
                for (int id : mine) chosen[id] = true;
                int differ = 0;
                for (int i = 0; i < k; i++) if (!chosen[reference[r * 512 + i]]) differ++;
                differing += differ;
                if (differ > 0) rowsWithDifferences++;
            }
        }
        System.out.printf(
                "%s selection: %d rows, %d blocks differ from the reference's top-k in %d rows (%s, worst score gap %.2e"
                        + " of the row's maximum)%n",
                p, rows, differing, rowsWithDifferences, scored ? "score aware" : "set difference", worstGap);
    }

    /// Sequence of `tokens` random rows run in the given chunking; returns the outputs.
    private static short[] runSequence(
            CudaGpuMemory gpu, Arena arena, QsaTestSupport.Loaded loaded, short[] x, int tokens, int[] chunking) {
        QsaLayer layer = new QsaLayer(QsaLayer.Config.of(loaded.artifact().config(), tokens));
        int maxRows = Arrays.stream(chunking).max().orElseThrow();
        long base = gpu.allocate(layer.scratchBytes(maxRows));
        QsaLayer.Scratch scratch = layer.scratch(base, maxRows);
        long output = gpu.allocate((long) maxRows * 2560 * 2);
        short[] result = new short[tokens * 2560];
        try (QsaState state = new QsaState(gpu, 512, tokens)) {
            int start = 0;
            for (int rows : chunking) {
                long input = upload(gpu, arena, Arrays.copyOfRange(x, start * 2560, (start + rows) * 2560));
                try {
                    layer.run(gpu, loaded.weights(), state, input, rows, output, scratch, 0);
                    short[] out = downloadShorts(gpu, arena, output, (long) rows * 2560);
                    System.arraycopy(out, 0, result, start * 2560, out.length);
                } finally {
                    gpu.free(input);
                }
                state.commit();
                start += rows;
            }
        } finally {
            gpu.free(output);
            gpu.free(base);
        }
        return result;
    }

    private static int[] chunking(int total, int... pattern) {
        List<Integer> chunks = new ArrayList<>();
        for (int sum = 0, i = 0; sum < total; i++) {
            int rows = Math.min(pattern[i % pattern.length], total - sum);
            chunks.add(rows);
            sum += rows;
        }
        return chunks.stream().mapToInt(Integer::intValue).toArray();
    }

    /// Any chunking of 2,300 tokens (past the budget of 2,048) gives the one-shot result: chunks of
    /// 1 to 5 rows, blocks completing across chunk boundaries, prefill chunks followed by decode
    /// rows.
    @Test
    void chunkingDoesNotChangeTheResult() throws IOException {
        int tokens = 2300;
        SplittableRandom rng = new SplittableRandom(99);
        short[] x = random(rng, tokens * 2560, 0.6);
        QsaTestSupport.Loaded loaded = shared();
        CudaGpuMemory gpu = loaded.gpu();
        try (Arena arena = Arena.ofConfined()) {
            gpu.selectRowExact(true);
            try {
                short[] oneShot = runSequence(gpu, arena, loaded, x, tokens, new int[] {tokens});
                int[][] chunkings = {
                    chunking(tokens, 512),
                    chunking(tokens, 1),
                    chunking(tokens, 2),
                    chunking(tokens, 3),
                    chunking(tokens, 4),
                    chunking(tokens, 5),
                    chunking(tokens, 1, 2, 3, 4, 5, 7, 16),
                    chunking(tokens, 31, 1, 1, 1, 64, 4, 3, 130),
                    concat(chunking(2200, 512), chunking(100, 1)),
                };
                for (int[] chunks : chunkings) {
                    short[] result = runSequence(gpu, arena, loaded, x, tokens, chunks);
                    double rms = QsaReference.relativeRms(oneShot, result);
                    double worst = QsaReference.maxDifferenceInRms(oneShot, result);
                    System.out.printf(
                            "chunking %s...: relative RMS %.3e, worst %.3f RMS%n",
                            Arrays.toString(Arrays.copyOf(chunks, Math.min(8, chunks.length))), rms, worst);
                    assertTrue(
                            rms < 5e-3,
                            "chunking " + Arrays.toString(Arrays.copyOf(chunks, 8)) + " relative RMS " + rms);
                }
            } finally {
                gpu.selectRowExact(false);
            }
        }
    }

    /// The layer at the end of the maximum context (262,144 tokens, 65,500 blocks), on random rows:
    /// the selection of every row of a chunk against the reference scores of the device's own block
    /// keys, and the attention against the decoded cache of the selected tokens.
    @Test
    void layerAtFullContextMatchesTheReference() throws IOException {
        int maxTokens = 262144, history = 261632, rows = 8;
        SplittableRandom rng = new SplittableRandom(17);
        QsaTestSupport.Loaded loaded = shared();
        CudaGpuMemory gpu = loaded.gpu();
        try (Arena arena = Arena.ofConfined()) {
            QsaLayer layer = new QsaLayer(QsaLayer.Config.of(loaded.artifact().config(), maxTokens));
            int maxRows = 512;
            long base = gpu.allocate(layer.scratchBytes(maxRows));
            QsaLayer.Scratch s = layer.scratch(base, maxRows);
            long input = gpu.allocate((long) maxRows * 2560 * 2), output = gpu.allocate((long) maxRows * 2560 * 2);
            long core = gpu.allocate((long) maxRows * 6144 * 2);
            java.lang.foreign.MemorySegment chunk = arena.allocate((long) maxRows * 2560 * 2, 16);
            try (QsaState state = new QsaState(gpu, 512, maxTokens)) {
                while (state.length() < history) {
                    int step = Math.min(maxRows, history - state.length());
                    QsaTestSupport.fillRandom(chunk, rng, step, 2560);
                    gpu.copyHostToDevice(input, chunk, (long) step * 2560 * 2);
                    layer.run(gpu, loaded.weights(), state, input, step, output, s, 0);
                    gpu.synchronize();
                    state.commit();
                }
                QsaTestSupport.fillRandom(chunk, rng, rows, 2560);
                gpu.copyHostToDevice(input, chunk, (long) rows * 2560 * 2);
                layer.run(gpu, loaded.weights(), state, input, rows, output, s, core);
                gpu.synchronize();
                int start = history, end = start + rows, blocks = end / 4;
                short[] blockKeys = downloadShorts(gpu, arena, state.blockKeys(), (long) blocks * 128);
                short[][] keys = new short[blocks][];
                for (int j = 0; j < blocks; j++) keys[j] = Arrays.copyOfRange(blockKeys, j * 128, (j + 1) * 128);
                short[] index = downloadShorts(gpu, arena, s.indexProj(), rows * 640L);
                short[] qProj = downloadShorts(gpu, arena, s.qProj(), (long) rows * 12288);
                short[] actualCore = downloadShorts(gpu, arena, core, (long) rows * 6144);
                int[] counts = downloadInts(gpu, arena, s.counts(), rows);
                int[] ids = downloadInts(gpu, arena, s.ids(), rows * 512L);
                boolean[] wanted = new boolean[end];
                int[][] selected = new int[rows][];
                double worstGap = 0;
                for (int r = 0; r < rows; r++) {
                    int p = start + r, nb = (p + 1) / 4;
                    short[][] q = new short[4][];
                    for (int h = 0; h < 4; h++)
                        q[h] = Arrays.copyOfRange(index, r * 640 + h * 128, r * 640 + (h + 1) * 128);
                    double[] scores = QsaReference.scores(q, keys, nb);
                    double max = Arrays.stream(scores).max().orElse(0);
                    assertEquals(512, counts[r]);
                    selected[r] = Arrays.copyOfRange(ids, r * 512, (r + 1) * 512);
                    QsaReference.Check check = QsaReference.checkSelection(
                            scores, selected[r], 512, 512, 1e-5 * max, "full context row " + r);
                    worstGap = Math.max(worstGap, check.gap() / max);
                    for (int id : selected[r]) for (int u = 0; u < 4; u++) wanted[4 * id + u] = true;
                    for (int t = 4 * nb; t <= p; t++) wanted[t] = true;
                }
                double[][][] k = new double[2][][], v = new double[2][][];
                for (int h = 0; h < 2; h++) {
                    k[h] = QsaTestSupport.decodeTokens(gpu, arena, state.keyPages(), end, 2, h, wanted);
                    v[h] = QsaTestSupport.decodeTokens(gpu, arena, state.valuePages(), end, 2, h, wanted);
                }
                short[] q = qHeadParts(qProj, rows, 0);
                double[] expected = new double[rows * 6144];
                for (int r = 0; r < rows; r++)
                    for (int head = 0; head < 24; head++)
                        System.arraycopy(
                                QsaReference.attend(
                                        q,
                                        (r * 24 + head) * 256,
                                        k[head / 12],
                                        v[head / 12],
                                        start + r,
                                        selected[r],
                                        512),
                                0,
                                expected,
                                r * 6144 + head * 256,
                                256);
                double rms = QsaReference.relativeRms(expected, actualCore);
                System.out.printf(
                        "full context: selection score gap %.2e of the maximum, core relative RMS %.3e%n",
                        worstGap, rms);
                assertTrue(rms < 3e-3, "core relative RMS " + rms);
            } finally {
                gpu.free(core);
                gpu.free(output);
                gpu.free(input);
                gpu.free(base);
            }
        }
    }

    private static int[] concat(int[] a, int[] b) {
        int[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}
