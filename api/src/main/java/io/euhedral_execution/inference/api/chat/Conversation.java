package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.List;
import java.util.Objects;

/// A request as every API surface states it once validated: the turns to render and how to answer them.
///
/// Each surface maps its own request format into this; everything after it (rendering, grammar, encoding, the
/// generation, prefix reuse) is shared. `reasoningBudget` caps reasoning tokens (`Integer.MAX_VALUE`: no cap); a
/// null `maxTokens` takes the server's default, and `maxTokensParam` names the request field in errors.
public record Conversation(
        List<QwenChatTemplate.Turn> turns,
        ToolCalling tools,
        QwenChatTemplate.Thinking thinking,
        int reasoningBudget,
        ResponseFormat format,
        List<String> stops,
        GenerationConfig sampling,
        Integer maxTokens,
        String maxTokensParam,
        boolean stream) {

    public Conversation {
        turns = List.copyOf(turns);
        stops = List.copyOf(stops);
        Objects.requireNonNull(tools, "tools");
        Objects.requireNonNull(thinking, "thinking");
        Objects.requireNonNull(format, "format");
        Objects.requireNonNull(sampling, "sampling");
        if (reasoningBudget < 0) throw new IllegalArgumentException("reasoningBudget must not be negative");
    }
}
