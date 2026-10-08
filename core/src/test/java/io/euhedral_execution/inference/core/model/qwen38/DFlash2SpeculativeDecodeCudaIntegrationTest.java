package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.ResidencyPlanner;
import io.euhedral_execution.inference.core.model.qwen38.speculative.DFlash2Decoder;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// DFlash2 speculative decoding is an optimization of greedy decode: for each prompt, ordinary one-row greedy
/// decode and DFlash2 speculative decode on fresh sequences produce exactly the same token IDs and leave exactly
/// the same GDN and attention state. Skipped unless -Peuhedral.qwen.dflash2-artifact names a DFlash2 artifact.
@ModelGroup.OwnJvm // loads the DFlash2 artifact, which no shared group holds
class DFlash2SpeculativeDecodeCudaIntegrationTest {

    static Qwen38Model load(Path artifact, CudaGpuMemory gpu, int contextTokens) throws Exception {
        var data = ArtifactReader.read(artifact);
        var profile = ArtifactProfile.of(data);
        assumeTrue(profile.speculation() == ArtifactProfile.Speculation.DFLASH2, "not a DFlash2 artifact");
        var residency =
                ResidencyPlanner.plan(data, profile, gpu.deviceMemoryInfo().freeBytes(), contextTokens);
        return Qwen38Model.load(artifact, data, gpu, ArtifactProfile.Speculation.DFLASH2, residency.hostBacked());
    }

    @Test
    @Timeout(value = 3600, unit = TimeUnit.SECONDS)
    void speculativeDecodeEqualsGreedyDecode() throws Throwable {
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        Path artifact = Path.of(System.getProperty("euhedral.qwen.dflash2-artifact", ""));
        assumeTrue(Files.isRegularFile(artifact), "no DFlash2 artifact: " + artifact);
        Path tokenizerDirectory =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        int budget = Integer.getInteger("euhedral.speculative.tokens", 160);
        QwenTokenizer tokenizer = QwenTokenizer.load(tokenizerDirectory);
        Path root = SpeculativeVerifyCudaIntegrationTest.repositoryRoot();
        int[] text = tokenizer.encodeText(Files.readString(root.resolve("docs/FRAME_MODEL.md")));
        List<int[]> prompts = List.of(
                tokenizer.encodeWithModelSpecialTokens("<|im_start|>user\nWhat is the capital of France? Answer with"
                        + " one word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
                Arrays.copyOf(text, 600),
                tokenizer.encodeWithModelSpecialTokens("<|im_start|>user\nWrite a Java method that reverses a"
                        + " singly linked list, with a short explanation.<|im_end|>\n<|im_start|>assistant\n"
                        + "<think>\n\n</think>\n\n"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Qwen38Model model = load(artifact, gpu, 8192);
                var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new Execution(lattice, plan, gpu);
            try {
                long id = 300;
                for (int[] prompt : prompts) {
                    var ordinary = new Sequence(++id);
                    List<Integer> expected;
                    List<byte[]> expectedState;
                    int expectedNext;
                    long ordinaryPosition;
                    try (var logits =
                            new HostLogits(gpu, model.weights().config().vocabSize())) {
                        logits.selectOnDevice(true);
                        expected = SpeculativeDecodeCudaIntegrationTest.greedy(
                                runtime, plan, ordinary, prompt, budget, logits, tokenizer);
                        expectedNext = SpeculativeDecodeCudaIntegrationTest.probe(runtime, plan, ordinary, logits);
                        expectedState = SpeculativeDecodeCudaIntegrationTest.snapshot(
                                gpu, model.weights().config().layerTypes(), ordinary);
                    } finally {
                        ordinaryPosition = ordinary.currentTokenPosition() - 1;
                        ordinary.complete();
                    }
                    var speculative = new Sequence(++id);
                    try (var decoder =
                            new DFlash2Decoder(runtime, plan, gpu, speculative, tokenizer::isGenerationEosToken, 512)) {
                        List<Integer> actual = decoder.generate(prompt, budget, token -> {}, null);
                        System.out.println("DFLASH2 prompt of " + prompt.length + " tokens, " + expected.size()
                                + " generated: " + decoder.statistics());
                        assertEquals(expected, actual, "speculative tokens");
                        assertEquals(ordinaryPosition, speculative.currentTokenPosition());
                        try (var logits =
                                new HostLogits(gpu, model.weights().config().vocabSize())) {
                            logits.selectOnDevice(true);
                            assertEquals(
                                    expectedNext,
                                    SpeculativeDecodeCudaIntegrationTest.probe(runtime, plan, speculative, logits),
                                    "decode after generation");
                        }
                        List<byte[]> actualState = SpeculativeDecodeCudaIntegrationTest.snapshot(
                                gpu, model.weights().config().layerTypes(), speculative);
                        assertEquals(expectedState.size(), actualState.size());
                        for (int i = 0; i < expectedState.size(); i++)
                            assertArrayEquals(expectedState.get(i), actualState.get(i), "state block " + i);
                    } finally {
                        speculative.complete();
                    }
                }
            } finally {
                runtime.close();
            }
        }
    }
}
