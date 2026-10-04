package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.List;

/// A validated request, fully resolved before any session is opened: its encoded prompt and how the output is
/// generated and read back. `promptTokens + maxTokens` never exceeds the model context. With `reasoning` the
/// prompt ends inside the model's think block, so the output begins as reasoning, capped at `reasoningBudget`
/// tokens. `grammar` (llguidance Lark, null for free text) constrains the answer: the tool-call envelope, or the
/// JSON document of `format`. `id` names the request in logs; each API derives its public ID from it.
public record GenerationPlan(
        String id,
        InferenceBackend.EncodedPrompt prompt,
        int maxTokens,
        GenerationConfig sampling,
        List<String> stops,
        boolean stream,
        ToolCalling tools,
        boolean reasoning,
        int reasoningBudget,
        ResponseFormat format,
        String grammar) {

    public GenerationPlan {
        stops = List.copyOf(stops);
    }

    public int promptTokens() {
        return this.prompt.tokenCount();
    }
}
