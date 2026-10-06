package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4ExpertReference.HIDDEN;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Artifact;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4ArtifactReader;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.junit.jupiter.api.Test;

/// The routed experts of layer 0 of the real artifact (records read straight from the file into device slots)
/// against the upstream fixtures of `layer_moe` (`~/fixtures/flash-next/layer_moe`, or `EUHEDRAL_QWEN4_FIXTURES`):
/// per expert the weighted outputs, per row the routed sum. The upstream product is an FP32 matmul over exactly
/// expanded
/// weights, the kernels accumulate in FP32 in another order, so values agree to BF16 rounding.
class Qwen4ExpertFixtureCudaIntegrationTest {

    private static Path fixtures() {
        String override = System.getenv("EUHEDRAL_QWEN4_FIXTURES");
        return override != null
                ? Path.of(override)
                : Path.of(System.getProperty("user.home"), "fixtures", "flash-next", "layer_moe");
    }

    /// Steps between two BF16 values of the same sign: 0 when equal.
    private static int ulps(short a, short b) {
        int x = a & 0xffff, y = b & 0xffff;
        int ax = (x & 0x8000) != 0 ? 0x8000 - (x & 0x7fff) : 0x8000 + x;
        int ay = (y & 0x8000) != 0 ? 0x8000 - (y & 0x7fff) : 0x8000 + y;
        return Math.abs(ax - ay);
    }

    /// Agreement of two BF16 tensors given per row. A rounding of a gate or up sum that flips (the sums differ in
    /// the last FP32 bits with the accumulation order) moves one activation by a step, which moves every output
    /// of the down projection by a fraction of a step times the weight. An output is held to two steps plus 2%
    /// of its row's largest value; values above 1/64 of that maximum are also counted in steps.
    private static final class Stats {
        long count, exact, large, largeWithinOne;
        int worstLargeSteps;
        double worstAbsolute;
        boolean bounded = true;

        void addRow(short[] expected, int expectedAt, short[] actual, int actualAt, int width) {
            float rowMax = 0;
            for (int j = 0; j < width; j++) rowMax = Math.max(rowMax, Math.abs(bf(expected[expectedAt + j])));
            for (int j = 0; j < width; j++) {
                short e = expected[expectedAt + j], a = actual[actualAt + j];
                int u = ulps(e, a);
                count++;
                if (u == 0) exact++;
                double error = Math.abs(bf(e) - bf(a));
                worstAbsolute = Math.max(worstAbsolute, error / rowMax);
                if (Math.abs(bf(e)) >= rowMax / 64) {
                    large++;
                    if (u <= 1) largeWithinOne++;
                    worstLargeSteps = Math.max(worstLargeSteps, u);
                }
                if (error > Math.abs(bf(e)) * 0.0157 + rowMax * 0.02) bounded = false;
            }
        }

        void add(Stats other) {
            count += other.count;
            exact += other.exact;
            large += other.large;
            largeWithinOne += other.largeWithinOne;
            worstLargeSteps = Math.max(worstLargeSteps, other.worstLargeSteps);
            worstAbsolute = Math.max(worstAbsolute, other.worstAbsolute);
            bounded &= other.bounded;
        }

        @Override
        public String toString() {
            return String.format(
                    "%d values, %.4f%% bit-equal; values above 1/64 of their row's largest: %.4f%% within one step, "
                            + "worst %d steps; worst absolute error %.2e of the row's largest",
                    count,
                    100.0 * exact / count,
                    100.0 * largeWithinOne / Math.max(1, large),
                    worstLargeSteps,
                    worstAbsolute);
        }
    }

    @Test
    void layerZeroExpertsMatchTheUpstreamFixtures() throws IOException {
        Path directory = fixtures();
        assumeTrue(ReferenceFixtures.exists(directory), "no fixtures at " + directory);
        Path artifactPath = Path.of(System.getProperty(
                "euhedral.qwen4.artifact", "/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl"));
        assumeTrue(Files.isRegularFile(artifactPath), "no Flash-Next artifact at " + artifactPath);
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null, "euhedral.cuda.library is not set");
        ReferenceFixtures fixtures = new ReferenceFixtures(directory);
        Qwen4Artifact artifact = Qwen4ArtifactReader.read(artifactPath);
        ExpertBank bank = artifact.bank("text/layers/0/moe/experts").orElseThrow();
        assertEquals(
                Qwen4ExpertOps.Geometry.flashNext(),
                Qwen4ExpertOps.Geometry.of(bank),
                "the kernels' record offsets are the bank's");
        int topK = 10;
        Stats allWeighted = new Stats(), allSums = new Stats();
        try (FileChannel file = FileChannel.open(artifactPath, StandardOpenOption.READ);
                CudaGpuMemory gpu = new CudaGpuMemory(Path.of(library))) {
            for (int chunk = 0; chunk < 3; chunk++) {
                String prefix = "c" + chunk + "/L0/moe/";
                if (!fixtures.has(prefix + "in")) continue;
                short[] x = fixtures.bf16(prefix + "in");
                int rows = x.length / HIDDEN;
                int[] ids = fixtures.i32(prefix + "topk_ids");
                short[] weights = fixtures.bf16(prefix + "topk_weights");
                short[] routedSum = fixtures.bf16(prefix + "routed_sum");
                assertEquals(rows * topK, ids.length);
                Stats weightedStats = new Stats();
                // 16 expert slots: a chunk of 64 rows names more experts, so slots are reused between experts
                try (Qwen4ExpertHarness harness = new Qwen4ExpertHarness(gpu, bank.expertCount(), topK, 64, 16)) {
                    short[] out = harness.run(
                            rows,
                            ids,
                            weights,
                            x,
                            (expert, destination) -> {
                                ByteBuffer buffer = destination.asByteBuffer().limit((int) bank.recordBytes(expert));
                                long position = bank.fileOffset(expert);
                                while (buffer.hasRemaining()) {
                                    int n = file.read(buffer, position + buffer.position());
                                    if (n < 0) throw new IOException("short read of expert " + expert);
                                }
                            },
                            (plan, act, weighted) -> {
                                short[] weightedBits = harness.download(weighted, plan.pairCount() * HIDDEN);
                                int pair = 0;
                                for (int i = 0; i < plan.activeExperts(); i++) {
                                    int expert = plan.activeExpert(i);
                                    int[] tokens = fixtures.i32(prefix + "expert/" + expert + "/tokens");
                                    short[] expected = fixtures.bf16(prefix + "expert/" + expert + "/weighted");
                                    for (int row = 0; row < rows; row++) {
                                        boolean routed = false;
                                        for (int k = 0; k < topK; k++) routed |= ids[row * topK + k] == expert;
                                        if (!routed) continue;
                                        int slot = -1;
                                        for (int t = 0; t < tokens.length; t++) if (tokens[t] == row) slot = t;
                                        assertTrue(slot >= 0, "fixture lists row " + row + " for expert " + expert);
                                        weightedStats.addRow(
                                                expected, slot * HIDDEN, weightedBits, pair * HIDDEN, HIDDEN);
                                        pair++;
                                    }
                                }
                                assertEquals(plan.pairCount(), pair);
                            });
                    System.out.println("chunk " + chunk + " (" + rows + " rows, " + harness.plan.activeExperts()
                            + " experts) weighted: " + weightedStats);
                    Stats sums = new Stats();
                    for (int row = 0; row < rows; row++)
                        sums.addRow(routedSum, row * HIDDEN, out, row * HIDDEN, HIDDEN);
                    System.out.println("chunk " + chunk + " routed sum: " + sums);
                    allWeighted.add(weightedStats);
                    allSums.add(sums);
                    // BF16 results of FP32 sums: nearly every value is bit-equal and none is beyond rounding noise
                    assertTrue(weightedStats.bounded, "weighted: " + weightedStats);
                    assertTrue(weightedStats.exact >= weightedStats.count * 0.98, "weighted: " + weightedStats);
                    assertTrue(sums.bounded, "routed sum: " + sums);
                    assertTrue(sums.exact >= sums.count * 0.97, "routed sum: " + sums);
                }
            }
        }
        System.out.println("all chunks weighted: " + allWeighted);
        System.out.println("all chunks routed sum: " + allSums);
        assertTrue(allWeighted.count > 0);
    }
}
