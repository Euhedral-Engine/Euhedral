package io.euhedral_execution.inference.core.model.qwen4;

import static io.euhedral_execution.inference.core.model.qwen4.Reference.bf;
import static io.euhedral_execution.inference.core.model.qwen4.TestSupport.downloadBf16;
import static io.euhedral_execution.inference.core.model.qwen4.TestSupport.downloadFloat;
import static io.euhedral_execution.inference.core.model.qwen4.TestSupport.error;
import static io.euhedral_execution.inference.core.model.qwen4.TestSupport.upload;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/// The Gated DeltaNet block of layer 0 and its gated residuals on the real weights against the upstream fixtures
/// (tools/flash_next_reference.py, case layer_gdn). Every component is fed the reference's own input.
class GdnFixtureCudaIntegrationTest {

    @Test
    void theGatedResidualAndTheGdnBlockFollowTheReference() throws Exception {
        Path directory = TestSupport.fixtureRoot().resolve("layer_gdn");
        assumeTrue(TestSupport.hasArtifact() && ReferenceFixtures.exists(directory), "no artifact or fixtures");
        ReferenceFixtures fixture = new ReferenceFixtures(directory);
        int chunks = fixture.metadata().get("chunks").size();
        for (boolean exact : new boolean[] {true, false}) {
            TestSupport.Report report =
                    new TestSupport.Report("layer 0 GDN block, " + (exact ? "exact" : "production") + " numerics");
            try (CudaGpuMemory gpu = TestSupport.openGpu();
                    Arena arena = Arena.ofConfined();
                    TestSupport.Weights weights = new TestSupport.Weights(gpu)) {
                gpu.selectExactNumerics(exact);
                Qwen4Config config = weights.artifact().config();
                GdnLayer gdn = new GdnLayer(
                        config.text().hiddenSize(),
                        config.gdn().numKeyHeads(),
                        config.gdn().numValueHeads(),
                        config.gdn().keyHeadDim(),
                        config.gdn().valueHeadDim(),
                        config.gdn().convKernelDim(),
                        TestSupport.epsilon(weights.artifact()),
                        config.attention().outputGate());
                String p = "text/layers/0/gdn/";
                var gdnWeights = new GdnLayer.Weights(
                        weights.weight(p + "in_proj_qkv"),
                        weights.weight(p + "in_proj_z"),
                        weights.weight(p + "out_proj"),
                        weights.weight(p + "in_proj_a"),
                        weights.weight(p + "in_proj_b"),
                        weights.weight(p + "conv1d"),
                        weights.weight(p + "a_log"),
                        weights.weight(p + "dt_bias"),
                        weights.weight(p + "norm"));
                var state = gdn.allocateState(gpu);
                try {
                    for (int chunk = 0; chunk < chunks; chunk++) {
                        double bound = exact ? 3e-3 : 2e-2;
                        String at = "c" + chunk + "/L0/";
                        TestSupport.checkHyperConnection(
                                gpu, arena, weights, fixture, report, chunk, 0, "attn_hc", at + "after_ple", bound);
                        short[] mixed = fixture.bf16(at + "gdn/in");
                        int rows = mixed.length / config.text().hiddenSize();
                        long input = upload(gpu, arena, mixed);
                        long output = gpu.allocate((long) rows * config.text().hiddenSize() * 2);
                        long scratchAddress = gpu.allocate(gdn.scratchBytes(rows));
                        try {
                            var scratch = gdn.scratch(scratchAddress, rows);
                            gdn.run(gpu, gdnWeights, state, input, rows, scratch, output);
                            int channels = gdn.convolutionChannels(), valueWidth = gdn.valueWidth();
                            report.check(
                                    at + "gdn/qkv_proj",
                                    error(
                                            fixture.bf16(at + "gdn/qkv_proj"),
                                            downloadBf16(gpu, arena, scratch.qkv(), rows * channels)),
                                    bound);
                            report.check(
                                    at + "gdn/z_proj",
                                    error(
                                            fixture.bf16(at + "gdn/z_proj"),
                                            downloadBf16(gpu, arena, scratch.z(), rows * valueWidth)),
                                    bound);
                            report.check(
                                    at + "gdn/a_proj",
                                    error(
                                            fixture.bf16(at + "gdn/a_proj"),
                                            downloadBf16(gpu, arena, scratch.a(), rows * 48)),
                                    bound);
                            report.check(
                                    at + "gdn/b_proj",
                                    error(
                                            fixture.bf16(at + "gdn/b_proj"),
                                            downloadBf16(gpu, arena, scratch.b(), rows * 48)),
                                    bound);
                            report.check(
                                    at + "gdn/conv_out",
                                    error(
                                            fixture.bf16(at + "gdn/conv_out"),
                                            downloadBf16(gpu, arena, scratch.convolved(), rows * channels)),
                                    bound);
                            report.check(
                                    at + "gdn/beta",
                                    error(
                                            Arrays.copyOf(bf16ToFloat(fixture.bf16(at + "gdn/beta")), rows * 48),
                                            downloadFloat(gpu, arena, scratch.beta(), rows * 48)),
                                    bound);
                            float[] g = fixture.f32(at + "gdn/g");
                            float[] alpha = new float[g.length];
                            for (int i = 0; i < g.length; i++) alpha[i] = (float) Math.exp(g[i]);
                            report.check(
                                    at + "gdn/alpha",
                                    error(alpha, downloadFloat(gpu, arena, scratch.alpha(), rows * 48)),
                                    1e-6);
                            report.check(
                                    at + "gdn/core_out",
                                    error(
                                            fixture.bf16(at + "gdn/core_out"),
                                            downloadBf16(gpu, arena, scratch.core(), rows * valueWidth)),
                                    bound);
                            report.check(
                                    at + "gdn/norm_out",
                                    error(
                                            fixture.bf16(at + "gdn/norm_out"),
                                            downloadBf16(gpu, arena, scratch.normed(), rows * valueWidth)),
                                    bound);
                            report.check(
                                    at + "gdn/out",
                                    error(
                                            fixture.bf16(at + "gdn/out"),
                                            downloadBf16(
                                                    gpu,
                                                    arena,
                                                    output,
                                                    rows * config.text().hiddenSize())),
                                    bound);
                            // States after the chunk. The reference keeps four convolution columns per channel, ours
                            // the last three rows.
                            short[] referenceConv = fixture.bf16(at + "gdn/conv_state");
                            short[] ours = downloadBf16(gpu, arena, state.convolutionHistory(), 3 * channels);
                            short[] expectedConv = new short[3 * channels];
                            for (int c = 0; c < channels; c++)
                                for (int i = 0; i < 3; i++)
                                    expectedConv[i * channels + c] = referenceConv[c * 4 + 1 + i];
                            report.check(at + "gdn/conv_state", error(expectedConv, ours), bound);
                            float[] recurrent = fixture.f32(at + "gdn/recurrent_state"); // [value head][key][value]
                            float[] actual = downloadFloat(
                                    gpu, arena, state.recurrent(), recurrent.length); // [head][value][key]
                            float[] transposed = new float[recurrent.length];
                            for (int h = 0; h < 48; h++)
                                for (int k = 0; k < 128; k++)
                                    for (int v = 0; v < 128; v++)
                                        transposed[(h * 128 + v) * 128 + k] = recurrent[(h * 128 + k) * 128 + v];
                            report.check(at + "gdn/recurrent_state", error(transposed, actual), bound);
                        } finally {
                            gpu.free(scratchAddress);
                            gpu.free(output);
                            gpu.free(input);
                        }
                        TestSupport.checkHyperConnection(
                                gpu, arena, weights, fixture, report, chunk, 0, "mlp_hc", at + "mid", bound);
                    }
                } finally {
                    gdn.freeState(gpu, state);
                    gpu.selectExactNumerics(false);
                }
            }
            report.finish();
        }
    }

    private static float[] bf16ToFloat(short[] values) {
        float[] floats = new float[values.length];
        for (int i = 0; i < values.length; i++) floats[i] = bf(values[i]);
        return floats;
    }
}
