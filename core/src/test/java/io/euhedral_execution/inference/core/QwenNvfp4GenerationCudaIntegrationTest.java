package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.SystemInfo;
import io.euhedral_execution.inference.core.model_loader.WeightResidency;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.QwenGenerationSession;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The NVFP4 artifact generates through the engine with only its executed objects resident: prefill
/// (tile kernels) and decode (GEMV) both run NVFP4 projections and the NVFP4 LM head. Skipped unless
/// -Peuhedral.qwen.nvfp4-artifact names an artifact.
class QwenNvfp4GenerationCudaIntegrationTest {
    @Test
    @Timeout(900)
    void nvfp4ArtifactAnswersAFactualPromptGreedily() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)));
        Path artifact = Path.of(System.getProperty("euhedral.qwen.nvfp4-artifact", ""));
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(artifact) && Files.isRegularFile(tokenizer.resolve("tokenizer.json")));
        BitSet cpus = SystemInfo.getPCpuSet();
        var tuning = InferenceTuning.defaults(cpus).withWeightResidency(WeightResidency.EXECUTED);
        var config = new InferenceConfig(artifact, tokenizer, Path.of(library), tuning, Duration.ofSeconds(10));
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            StringBuilder output = new StringBuilder();
            try (QwenGenerationSession session = engine.createSession(GenerationConfig.greedy(91L))) {
                var tokens = session.generate("The capital of France is", 8, output::append);
                assertEquals(8, tokens.size());
            }
            System.out.println("Generated text: " + output);
            assertTrue(output.toString().contains("Paris"), output.toString());
        }
    }
}
