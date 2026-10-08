package io.euhedral_execution.inference.core.model.qwen4;

import static io.euhedral_execution.inference.core.model.qwen4.GreedyAgreementCudaIntegrationTest.NOISE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.InferenceEngine;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.model.qwen4.GreedyAgreementCudaIntegrationTest.Case;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.testing.SharedFlashNextEngine;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The free-running half of the end-to-end gate (see [GreedyAgreementCudaIntegrationTest]): the engine's own
/// tokenizer, sampler and sessions generate from the fixed prompts, and every token equals the upstream model's up to
/// the first flip, which must be an interchangeable one (a margin within BF16 noise).
@ModelGroup.FlashNextEngine
@Timeout(value = 5, unit = TimeUnit.MINUTES)
class GreedyAgreementEngineCudaIntegrationTest {

    @Test
    void freeRunningGenerationThroughTheEngineAgreesUntilAnInterchangeableFlip() throws Exception {
        List<Case> cases = GreedyAgreementCudaIntegrationTest.cases();
        InferenceEngine engine = SharedFlashNextEngine.get(4096);
        int matched = 0, total = 0;
        for (Case reference : cases) {
            int[] prompt = engine.tokenizePromptAsync(reference.prompt()).get(30, TimeUnit.SECONDS);
            assertArrayEquals(reference.promptIds(), prompt, "tokenization of " + reference.prompt());
            GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(1));
            List<Integer> tokens = session.generateAsync(
                            prompt, reference.steps().size(), text -> {}, null, null)
                    .get(10, TimeUnit.MINUTES);
            session.close();
            int step = 0;
            while (step < tokens.size()
                    && tokens.get(step) == reference.steps().get(step).token()) step++;
            matched += step;
            total += reference.steps().size();
            if (step < tokens.size())
                assertTrue(
                        reference.steps().get(step).margin(tokens.get(step)) < NOISE,
                        "free-running step " + step + " of '"
                                + reference
                                        .prompt()
                                        .substring(
                                                0,
                                                Math.min(30, reference.prompt().length()))
                                + "': engine " + tokens.get(step) + " vs reference "
                                + reference.steps().get(step).token());
        }
        System.out.println("free-running greedy tokens equal before the first flip: " + matched + " of " + total);
    }
}
