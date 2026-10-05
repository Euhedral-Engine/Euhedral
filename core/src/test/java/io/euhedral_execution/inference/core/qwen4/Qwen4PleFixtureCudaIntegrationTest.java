package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bf;
import static io.euhedral_execution.inference.core.qwen4.Qwen4TestSupport.downloadBf16;
import static io.euhedral_execution.inference.core.qwen4.Qwen4TestSupport.error;
import static io.euhedral_execution.inference.core.qwen4.Qwen4TestSupport.upload;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Config;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/// The per-layer embedding of layer 1 and its gated residuals on the real weights against the upstream fixtures
/// (tools/flash_next_reference.py, case layer_ple): n-gram rows across end-of-sequence tokens and chunk boundaries,
/// the projections, the gate and the dilated convolution with its history.
class Qwen4PleFixtureCudaIntegrationTest {

    @Test
    void thePerLayerEmbeddingFollowsTheReferenceAcrossChunksAndEndOfSequenceTokens() throws Exception {
        Path directory = Qwen4TestSupport.fixtureRoot().resolve("layer_ple");
        assumeTrue(Qwen4TestSupport.hasArtifact() && ReferenceFixtures.exists(directory), "no artifact or fixtures");
        ReferenceFixtures fixture = new ReferenceFixtures(directory);
        int chunks = fixture.metadata().get("chunks").size();
        for (boolean exact : new boolean[] {true, false}) {
            Qwen4TestSupport.Report report =
                    new Qwen4TestSupport.Report("layer 1 PLE, " + (exact ? "exact" : "production") + " numerics");
            try (CudaGpuMemory gpu = Qwen4TestSupport.openGpu();
                    Arena arena = Arena.ofConfined();
                    Qwen4TestSupport.Weights weights = new Qwen4TestSupport.Weights(gpu);
                    var store = io.euhedral_execution.inference.core.model_loader.qwen4.NgramStore.open(
                            weights.path(),
                            weights.artifact(),
                            io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4ResidencyPlan.NgramMode
                                    .MAPPED_FILE,
                            null)) {
                gpu.selectExactNumerics(exact);
                Qwen4Config config = weights.artifact().config();
                Qwen4Config.Ngram ngram = config.ngram();
                var ids = new Qwen4NgramIds(
                        ngram.size(),
                        ngram.headsPerNgram(),
                        ngram.layerMultipliers(),
                        ngram.headsVocabSizes(),
                        ngram.headsOffsets(),
                        config.text().eosTokenId());
                int hidden = config.text().hiddenSize(),
                        streams = config.hyperConnection().count();
                var ple = new Qwen4Ple(
                        store,
                        ids,
                        hidden,
                        streams,
                        config.ple().embedDim(),
                        config.ple().convKernelSize(),
                        ngram.size(),
                        Qwen4TestSupport.epsilon(weights.artifact()));
                String p = "text/layers/" + config.ple().layers()[0] + "/ple/";
                var pleWeights = new Qwen4Ple.Weights(
                        weights.address(p + "key_proj"),
                        weights.bytes(p + "key_proj"),
                        weights.address(p + "value_proj"),
                        weights.bytes(p + "value_proj"),
                        weights.address(p + "norm_key"),
                        weights.address(p + "norm_query"),
                        weights.address(p + "norm_conv"),
                        weights.address(p + "conv1d"));
                long historyAddress = gpu.allocate(ple.historyBytes());
                gpu.zeroDeviceMemory(historyAddress, ple.historyBytes());
                var state = ple.newState(historyAddress);
                try {
                    double bound = exact ? 3e-3 : 2e-2;
                    int[] allTokens = new int[0];
                    for (int chunk = 0; chunk < chunks; chunk++) {
                        String at = "c" + chunk + "/L1/";
                        Qwen4TestSupport.checkHyperConnection(
                                gpu, arena, weights, fixture, report, chunk, 1, "attn_hc", at + "after_ple", bound);
                        Qwen4TestSupport.checkHyperConnection(
                                gpu, arena, weights, fixture, report, chunk, 1, "mlp_hc", at + "mid", bound);
                        long[] tokens64 = fixture.i64("c" + chunk + "/tokens");
                        int[] tokens =
                                Arrays.stream(tokens64).mapToInt(t -> (int) t).toArray();
                        int rows = tokens.length;
                        short[] streamState = fixture.bf16(at + "in");
                        long streamAddress = upload(gpu, arena, streamState);
                        long output = gpu.allocate((long) rows * ple.stateWidth() * 2);
                        long scratchAddress = gpu.allocate(ple.scratchBytes(rows));
                        try {
                            var scratch = ple.scratch(scratchAddress, rows);
                            var staged =
                                    ple.apply(gpu, pleWeights, state, tokens, 0, rows, streamAddress, scratch, output);
                            try {
                                gpu.synchronize();
                                int width = ple.stateWidth();
                                report.check(
                                        at + "ple/rows",
                                        error(
                                                fixture.bf16(at + "ple/rows"),
                                                downloadBf16(gpu, arena, scratch.embedding(), rows * 2560)),
                                        1e-9);
                                report.check(
                                        at + "ple/emb",
                                        error(
                                                fixture.bf16(at + "ple/emb"),
                                                downloadBf16(gpu, arena, scratch.embedding(), rows * 2560)),
                                        1e-9);
                                report.check(
                                        at + "ple/key_normed",
                                        error(
                                                fixture.bf16(at + "ple/key_normed"),
                                                downloadBf16(gpu, arena, scratch.keyNormed(), rows * width)),
                                        bound);
                                report.check(
                                        at + "ple/value",
                                        error(
                                                fixture.bf16(at + "ple/value"),
                                                downloadBf16(gpu, arena, scratch.value(), rows * hidden)),
                                        bound);
                                report.check(
                                        at + "ple/query_normed",
                                        error(
                                                fixture.bf16(at + "ple/query_normed"),
                                                downloadBf16(gpu, arena, scratch.query(), rows * width)),
                                        1e-6);
                                report.check(
                                        at + "ple/gated_value",
                                        error(
                                                fixture.bf16(at + "ple/gated_value"),
                                                downloadBf16(gpu, arena, scratch.gated(), rows * width)),
                                        bound);
                                report.check(
                                        at + "ple/gated_normed",
                                        error(
                                                fixture.bf16(at + "ple/gated_normed"),
                                                downloadBf16(gpu, arena, scratch.gatedNormed(), rows * width)),
                                        bound);
                                report.check(
                                        at + "ple/out",
                                        error(
                                                fixture.bf16(at + "ple/out"),
                                                downloadBf16(gpu, arena, output, rows * width)),
                                        bound);
                                // State: the reference keeps the history channel-major, ours is time-major.
                                short[] referenceHistory = fixture.bf16(at + "ple/conv_state");
                                int historyRows = ple.historyRows();
                                short[] expectedHistory = new short[historyRows * width];
                                for (int c = 0; c < width; c++)
                                    for (int i = 0; i < historyRows; i++)
                                        expectedHistory[i * width + c] = referenceHistory[c * historyRows + i];
                                report.check(
                                        at + "ple/conv_state",
                                        error(
                                                expectedHistory,
                                                downloadBf16(gpu, arena, state.history(), historyRows * width)),
                                        bound);
                                long[] referenceContext = fixture.i64(at + "ple/ngram_context");
                                for (int i = 0; i < referenceContext.length; i++)
                                    if (referenceContext[i] != state.context().token(i))
                                        throw new AssertionError(at + "ngram context " + i + ": "
                                                + state.context().token(i) + " != " + referenceContext[i]);
                            } finally {
                                staged.close();
                            }
                        } finally {
                            gpu.free(scratchAddress);
                            gpu.free(output);
                            gpu.free(streamAddress);
                        }
                    }
                } finally {
                    gpu.free(historyAddress);
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
