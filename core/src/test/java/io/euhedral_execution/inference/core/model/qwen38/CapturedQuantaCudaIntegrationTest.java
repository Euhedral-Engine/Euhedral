package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.speculative.MtpDecoder;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.testing.SharedQwen38Mtp;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Captured quanta (docs/CUDA_GRAPHS.md) compute what stage-by-stage submission computes. Speculative and
/// ordinary greedy generation on a runtime that replays captured graphs must produce the same tokens as on
/// one that never captures, and the same drafts: every verification accepts as many of them, which the
/// output tokens alone would not show (the verifier keeps the output exact whatever the drafts are).
@ModelGroup.CompactQ3
class CapturedQuantaCudaIntegrationTest {

    private record Run(List<List<Integer>> tokens, List<long[]> accepted, long replayed) {}

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void replayedQuantaGenerateTheSameTokensAndDraftsAsStageByStageSubmission() throws Throwable {
        Path tokenizerDirectory =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        QwenTokenizer tokenizer = QwenTokenizer.load(tokenizerDirectory);
        String frameModel = Files.readString(
                SpeculativeVerifyCudaIntegrationTest.repositoryRoot().resolve("docs/FRAME_MODEL.md"));
        List<int[]> prompts = List.of(
                tokenizer.encodeWithModelSpecialTokens("<|im_start|>user\nWrite a Java method that reverses a"
                        + " singly linked list, with a short explanation.<|im_end|>\n<|im_start|>assistant\n"
                        + "<think>\n\n</think>\n\n"),
                // Crosses a KV page boundary and the 2048-key attention switch while generating.
                Arrays.copyOf(tokenizer.encodeText(frameModel), 1990));
        long hostMiB = Long.getLong("euhedral.speculative.host-mib", 1024L);
        var loaded = SharedQwen38Mtp.q3(hostMiB);
        CudaGpuMemory gpu = loaded.gpu();
        Qwen38Model model = loaded.model();
        var artifactData = loaded.artifact();
        try (var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            int depth = ArtifactProfile.of(artifactData).speculativeDepth();
            Run captured = generate(gpu, lattice, plan, tokenizer, prompts, depth, true);
            Run submitted = generate(gpu, lattice, plan, tokenizer, prompts, depth, false);
            assertTrue(captured.replayed() > 100, "replayed quanta: " + captured.replayed());
            assertEquals(0, submitted.replayed());
            assertEquals(submitted.tokens(), captured.tokens(), "generated tokens");
            for (int prompt = 0; prompt < prompts.size(); prompt++)
                assertArrayEquals(
                        submitted.accepted().get(prompt),
                        captured.accepted().get(prompt),
                        "accepted drafts per verification, prompt " + prompt);
        }
    }

    private static Run generate(
            CudaGpuMemory gpu,
            PullingLattice lattice,
            ExecutionPlan plan,
            QwenTokenizer tokenizer,
            List<int[]> prompts,
            int depth,
            boolean capture)
            throws Exception {
        var runtime = new Execution(lattice, plan, gpu, EuhedralInferenceRuntime.laneCount(), capture);
        List<List<Integer>> tokens = new ArrayList<>();
        List<long[]> accepted = new ArrayList<>();
        try {
            long id = capture ? 300 : 400;
            for (int[] prompt : prompts) {
                var speculative = new Sequence(++id);
                try (var decoder =
                        new MtpDecoder(runtime, plan, gpu, speculative, tokenizer::isGenerationEosToken, depth, 512)) {
                    tokens.add(decoder.generate(prompt, 160, token -> {}));
                    accepted.add(decoder.statistics().acceptedDrafts.clone());
                } finally {
                    speculative.complete();
                }
                // Ordinary one-row decode quanta as well: 48 tokens of the short prompt, and for the long one 72, which
                // still crosses the 2048-key attention switch (58 tokens in) as the speculative run does.
                var ordinary = new Sequence(++id);
                try (var logits = new HostLogits(gpu, plan.weights().config().vocabSize())) {
                    logits.selectOnDevice(true);
                    tokens.add(SpeculativeDecodeCudaIntegrationTest.greedy(
                            runtime, plan, ordinary, prompt, prompt.length > 1000 ? 72 : 48, logits, tokenizer));
                } finally {
                    ordinary.complete();
                }
            }
            return new Run(tokens, accepted, runtime.replayedQuanta());
        } finally {
            runtime.close();
        }
    }
}
