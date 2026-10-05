package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bf;
import static io.euhedral_execution.inference.core.qwen4.Qwen4TestSupport.error;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.qwen4.HostBudget;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Mode;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Model;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/// The whole model on the real artifact against the upstream fixtures of the model cases (`short`, `eos`), recorded
/// through the NVFP4 KV codec the engine's cache applies: every layer fed the reference's own input (so a layer's error
/// is its own), and the full forward from tokens (prefill chunks and decode steps) compared at every layer boundary and
/// at the logits.
///
/// Bounds: a layer fed the reference's own input agrees to about 1% (GDN layers) to a few percent. A sparse-attention
/// layer's attention block differs from the reference by 0.5% (the BF16 rounding of upstream's own score tensors and
/// the
/// KV codec's), and the MoE block that follows is chaotic in that difference: its router sees an input one rounding
/// step
/// away, which can choose another expert for a near-tied token. The MoE block alone on the reference's input agrees to
/// 3e-4 (`Qwen4MoeFixtureCudaIntegrationTest`), so the layer bound is 10%, not a kernel tolerance.
class Qwen4ModelFixtureCudaIntegrationTest {

    private static final int CONTEXT = 8192;

    private static int chunkCount(ReferenceFixtures fixture) {
        int count = 0;
        while (fixture.has("c" + count + "/tokens")) count++;
        return count;
    }

    private static int[] tokens(ReferenceFixtures fixture, int chunk) throws Exception {
        return Arrays.stream(fixture.i64("c" + chunk + "/tokens"))
                .mapToInt(t -> (int) t)
                .toArray();
    }

    private static Qwen4Model open(CudaGpuMemory gpu) throws Exception {
        return Qwen4Model.open(
                Qwen4TestSupport.artifactPath(),
                gpu,
                gpu.deviceMemoryInfo().freeBytes(),
                HostBudget.system(),
                Qwen4Mode.TEXT,
                CONTEXT);
    }

    private static ReferenceFixtures fixtures(String name) throws Exception {
        assumeTrue(Qwen4TestSupport.hasArtifact(), "no artifact");
        Path directory = Qwen4TestSupport.modelFixtureRoot().resolve(name);
        assumeTrue(ReferenceFixtures.exists(directory), "no fixtures " + directory);
        return new ReferenceFixtures(directory);
    }

    @Test
    void everyLayerFollowsTheReferenceWhenFedItsInput() throws Exception {
        for (String name : new String[] {"short", "eos"}) {
            ReferenceFixtures fixture = fixtures(name);
            Qwen4TestSupport.Report report = new Qwen4TestSupport.Report("layers fed their input, case " + name);
            try (CudaGpuMemory gpu = Qwen4TestSupport.openGpu();
                    Qwen4Model model = open(gpu);
                    Qwen4Executor executor = new Qwen4Executor(gpu, model, CONTEXT);
                    Qwen4Sequence sequence = executor.newSequence()) {
                int layers = model.artifact().config().text().numLayers();
                for (int chunk = 0; chunk < chunkCount(fixture); chunk++) {
                    int[] tokens = tokens(fixture, chunk);
                    for (int layer = 0; layer < layers; layer++) {
                        short[] in = fixture.bf16("c" + chunk + "/L" + layer + "/in");
                        short[] out = executor.runSingleLayer(sequence, layer, tokens, 0, tokens.length, in);
                        report.check(
                                "c" + chunk + "/L" + String.format("%02d", layer) + "/out",
                                error(fixture.bf16("c" + chunk + "/L" + layer + "/out"), out),
                                0.1);
                    }
                    executor.finishChunk(sequence, tokens.length);
                }
            }
            report.finish();
        }
    }

    /// Where a layer's error enters: the state after the attention block (before the MoE block) against the
    /// reference's, for the layers the case captures in detail.
    @Test
    void attentionBlocksFollowTheReferenceBeforeTheMoeBlock() throws Exception {
        for (String name : new String[] {"short", "eos"}) {
            ReferenceFixtures fixture = fixtures(name);
            Qwen4TestSupport.Report report =
                    new Qwen4TestSupport.Report("state after the attention block, case " + name);
            try (CudaGpuMemory gpu = Qwen4TestSupport.openGpu();
                    Arena arena = Arena.ofConfined();
                    Qwen4Model model = open(gpu);
                    Qwen4Executor executor = new Qwen4Executor(gpu, model, CONTEXT);
                    Qwen4Sequence sequence = executor.newSequence()) {
                int[] chunkHolder = new int[1];
                executor.observeMid((layer, state, rows) -> {
                    String at = "c" + chunkHolder[0] + "/L" + layer + "/mid";
                    if (!fixture.has(at)) return;
                    try {
                        short[] actual = Qwen4TestSupport.downloadBf16(gpu, arena, state, rows * 10240);
                        report.check(at, error(fixture.bf16(at), actual), 0.03);
                    } catch (Exception failure) {
                        throw new IllegalStateException(failure);
                    }
                });
                for (int chunk = 0; chunk < chunkCount(fixture); chunk++) {
                    chunkHolder[0] = chunk;
                    int[] tokens = tokens(fixture, chunk);
                    for (int layer = 0; layer < 4; layer++) {
                        short[] in = fixture.bf16("c" + chunk + "/L" + layer + "/in");
                        executor.runSingleLayer(sequence, layer, tokens, 0, tokens.length, in);
                    }
                    executor.finishChunk(sequence, tokens.length);
                }
            }
            report.finish();
        }
    }

    /// The forward from tokens: prefill and decode steps with the sequence state carried. The error accumulates
    /// with depth the way the reference's own does between chunkings (BF16 noise); the greedy token must agree at
    /// every step.
    @Test
    void theFullForwardFollowsTheReference() throws Exception {
        ReferenceFixtures fixture = fixtures("short");
        Qwen4TestSupport.Report report = new Qwen4TestSupport.Report("full forward, case short");
        try (CudaGpuMemory gpu = Qwen4TestSupport.openGpu();
                Arena arena = Arena.ofConfined();
                Qwen4Model model = open(gpu);
                Qwen4Executor executor = new Qwen4Executor(gpu, model, CONTEXT);
                Qwen4Sequence sequence = executor.newSequence()) {
            int vocabulary = executor.vocabularySize();
            var readback = gpu.allocateReadbackBuffer((long) vocabulary * 2);
            int[] chunkHolder = new int[1];
            executor.observe((layer, state, rows) -> {
                try {
                    short[] actual = Qwen4TestSupport.downloadBf16(gpu, arena, state, rows * 10240);
                    short[] expected = fixture.bf16("c" + chunkHolder[0] + "/L" + layer + "/out");
                    report.check(
                            "c" + chunkHolder[0] + "/L" + String.format("%02d", layer) + "/out",
                            error(expected, actual),
                            0.3);
                } catch (Exception failure) {
                    throw new IllegalStateException(failure);
                }
            });
            for (int chunk = 0; chunk < chunkCount(fixture); chunk++) {
                chunkHolder[0] = chunk;
                int[] tokens = tokens(fixture, chunk);
                executor.step(
                        sequence,
                        tokens,
                        0,
                        tokens.length,
                        address -> gpu.copyDeviceToReadback(readback, address, vocabulary * 2L));
                short[] actual = new short[vocabulary];
                MemorySegment.copy(readback.segment(), ValueLayout.JAVA_SHORT, 0, actual, 0, vocabulary);
                short[] expected = fixture.bf16("c" + chunk + "/logits");
                report.check("c" + chunk + "/logits", error(expected, actual), 0.2);
                int[] top = topK(expected, 5);
                int[] mine = topK(actual, 5);
                System.out.println("chunk " + chunk + " reference top " + Arrays.toString(top) + " engine top "
                        + Arrays.toString(mine));
                assertEquals(top[0], mine[0], "greedy token of chunk " + chunk);
            }
            readback.close();
        }
        report.finish();
    }

    private static int[] topK(short[] logits, int k) {
        int[] best = new int[k];
        Arrays.fill(best, -1);
        for (int i = 0; i < logits.length; i++) {
            int at = k;
            while (at > 0 && (best[at - 1] < 0 || bf(logits[i]) > bf(logits[best[at - 1]]))) at--;
            if (at < k) {
                System.arraycopy(best, at, best, at + 1, k - at - 1);
                best[at] = i;
            }
        }
        return best;
    }
}
