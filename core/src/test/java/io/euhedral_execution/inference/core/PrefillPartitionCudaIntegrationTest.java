package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.EngineExecutionFixture;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.model.qwen38.SequenceStateProbe;
import io.euhedral_execution.inference.core.model.qwen38.Session;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

@ModelGroup.CompactQ3Engine
class PrefillPartitionCudaIntegrationTest {
    private static final int PROMPT_TOKENS = 1536;

    /// Prefills the same 1536 tokens split into chunks of several sizes and reports how the resulting
    /// sequence state differs from the 512-token partition a cold run uses. 1536 is a multiple of the
    /// 256-token page, so every KV row is in a full page and is compared.
    @Test
    @Timeout(120)
    void reportsWhetherPrefillStateDependsOnTheChunkPartition() throws Exception {
        // Chunks up to 1536 tokens: the workspace is sized at load for the widest.
        var held = CoreEngines.q3Wide(PROMPT_TOKENS);
        InferenceEngine engine = held.engine();
        String text = "The quick brown fox jumps over the lazy dog while the engine counts tokens. ".repeat(400);
        int[] encoded = engine.tokenizer().encodeWithModelSpecialTokens(text);
        assertTrue(encoded.length >= PROMPT_TOKENS, "the prompt text is too short");
        int[] prompt = Arrays.copyOf(encoded, PROMPT_TOKENS);
        List<String> baseline = stateAfterPrefill(engine, held.gpu(), prompt, 512);
        System.out.println("prefill partition 512 (baseline): " + baseline.size() + " buffers compared");
        for (int chunk : new int[] {512, 1024, 256, 700, 1000, 1536}) {
            List<String> other = stateAfterPrefill(engine, held.gpu(), prompt, chunk);
            System.out.println("prefill partition " + chunk + ": " + describe(baseline, other));
        }
    }

    private static List<String> stateAfterPrefill(InferenceEngine engine, ExecutionGpu gpu, int[] prompt, int chunk)
            throws Exception {
        try (Session session = engine.createSession(GenerationConfig.greedy(1L), chunk)) {
            session.generate(prompt, 0, text -> {}, null);
            return SequenceStateProbe.committedDigests(
                    gpu,
                    EngineExecutionFixture.sequence(session),
                    ((Qwen38Runtime) engine.modelRuntime()).config().layerTypes());
        }
    }

    private static String describe(List<String> expected, List<String> actual) {
        if (expected.size() != actual.size())
            return "different buffer counts " + expected.size() + " vs " + actual.size();
        List<String> differing = new java.util.ArrayList<>();
        for (int i = 0; i < expected.size(); i++)
            if (!expected.get(i).equals(actual.get(i)))
                differing.add(expected.get(i).replaceAll(" [0-9a-f]{16}$", ""));
        if (differing.isEmpty()) return "identical (" + expected.size() + " buffers)";
        return differing.size() + " of " + expected.size() + " buffers differ, first: "
                + differing.subList(0, Math.min(3, differing.size()));
    }
}
