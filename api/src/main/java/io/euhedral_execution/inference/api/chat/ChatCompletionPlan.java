package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.List;

/// A validated request, fully resolved before any session is opened.
/// `promptTokens + maxTokens` never exceeds the model context. With `reasoning` the prompt ends inside the
/// model's think block, so the output begins as reasoning.
public record ChatCompletionPlan(
        String id,
        long created,
        String model,
        InferenceBackend.EncodedPrompt prompt,
        int maxTokens,
        GenerationConfig sampling,
        List<String> stops,
        boolean stream,
        boolean includeUsage,
        ToolCalling tools,
        boolean reasoning) {

    public ChatCompletionPlan {
        stops = List.copyOf(stops);
    }

    public int promptTokens() {
        return this.prompt.tokenCount();
    }
}
