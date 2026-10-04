package io.euhedral_execution.inference.api.openai;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Map;

/// One request message. `content` is a string, a list of content parts, or null.
/// `name` is accepted and ignored because the checkpoint template does not render participant names.
/// `tool_calls` (assistant), `tool_call_id` (tool) and `reasoning_content` (assistant: the reasoning of an
/// earlier response, replayed) stay untyped JSON for precise validation.
public record ChatMessage(
        String role,
        Object content,
        String name,
        @JsonProperty("tool_calls") Object toolCalls,
        @JsonProperty("tool_call_id") Object toolCallId,
        @JsonProperty("reasoning_content") Object reasoningContent,
        @JsonAnySetter Map<String, Object> otherFields) {

    public ChatMessage {
        otherFields = otherFields == null ? Map.of() : otherFields;
    }
}
