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
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The NVFP4 artifact generates through the engine with only its executed objects resident: prefill
/// (tile kernels) and decode (GEMV) both run NVFP4 projections and the NVFP4 LM head. Skipped unless
/// -Peuhedral.qwen.nvfp4-artifact names an artifact.
class QwenNvfp4GenerationCudaIntegrationTest {
    @Test
    @Timeout(900)
    void nvfp4ArtifactAnswersAFactualPromptGreedilyResidentOrHostBacked() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)));
        Path artifact = Path.of(System.getProperty("euhedral.qwen.nvfp4-artifact", ""));
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(artifact) && Files.isRegularFile(tokenizer.resolve("tokenizer.json")));
        BitSet cpus = SystemInfo.getPCpuSet();
        var tuning = InferenceTuning.defaults(cpus).withWeightResidency(WeightResidency.EXECUTED);
        List<Integer> resident = generate(artifact, tokenizer, library, tuning, "resident");
        // About 1 GiB of GDN projections stay in host memory and are staged per use: same tokens.
        List<Integer> staged =
                generate(artifact, tokenizer, library, tuning.withHostWeights(1L << 30, 4), "host-backed");
        assertEquals(resident, staged);
    }

    private static List<Integer> generate(
            Path artifact, Path tokenizer, String library, InferenceTuning tuning, String label) throws Exception {
        var config = new InferenceConfig(artifact, tokenizer, Path.of(library), tuning, Duration.ofSeconds(10));
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            StringBuilder output = new StringBuilder();
            List<Integer> tokens;
            try (QwenGenerationSession session = engine.createSession(GenerationConfig.greedy(91L))) {
                tokens = session.generate("The capital of France is", 16, output::append);
                assertEquals(16, tokens.size());
            }
            System.out.println(
                    label + " generated text: " + output + " (device " + engine.allocatedDeviceBytes() + " bytes)");
            assertTrue(output.toString().contains("Paris"), output.toString());
            return tokens;
        }
    }
}
