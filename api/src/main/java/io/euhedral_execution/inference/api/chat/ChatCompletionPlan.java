package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.List;

/// A validated request, fully resolved before any session is opened.
/// `promptTokens + maxTokens` never exceeds the model context. With `reasoning` the prompt ends inside the
/// model's think block, so the output begins as reasoning. `grammar` (llguidance Lark, null for free text)
/// constrains the answer: the tool-call envelope, or the JSON document of `format`.
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
        boolean reasoning,
        ResponseFormat format,
        String grammar) {

    public ChatCompletionPlan {
        stops = List.copyOf(stops);
    }

    public int promptTokens() {
        return this.prompt.tokenCount();
    }
}
