package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model.qwen38.Session;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The NVFP4 artifact generates through the engine with only its executed objects resident: prefill
/// (tile kernels) and decode (GEMV) both run NVFP4 projections and the NVFP4 LM head. Skipped unless
/// -Peuhedral.qwen.nvfp4-artifact names an artifact.
@ModelGroup.Nvfp4Engine
class Nvfp4GenerationCudaIntegrationTest {
    @Test
    @Timeout(240)
    void nvfp4ArtifactAnswersAFactualPromptGreedilyResidentOrHostBacked() throws Exception {
        List<Integer> resident = generate(4096, "resident");
        // A 128K context leaves no device room for all weights: part of the projections stay in host
        // memory and are staged per use. Same tokens.
        List<Integer> staged = generate(131072, "host-backed");
        assertEquals(resident, staged);
    }

    private static List<Integer> generate(int maxContextTokens, String label) throws Exception {
        InferenceEngine engine = CoreEngines.nvfp4(maxContextTokens).engine();
        StringBuilder output = new StringBuilder();
        List<Integer> tokens;
        try (Session session = engine.createSession(GenerationConfig.greedy(91L))) {
            tokens = session.generate("The capital of France is", 16, output::append);
            assertEquals(16, tokens.size());
        }
        System.out.println(
                label + " generated text: " + output + " (device " + engine.allocatedDeviceBytes() + " bytes)");
        assertTrue(output.toString().contains("Paris"), output.toString());
        return tokens;
    }
}
