package io.euhedral_execution.inference.api.anthropic;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Map;

/// An Anthropic Messages request (`POST /v1/messages`, and `/v1/messages/count_tokens` without `max_tokens`).
///
/// Content stays untyped JSON so [MessagesMapper] can report precise errors; every other top-level field lands in
/// `otherFields`, where the mapper accepts the ones that cannot change the output and refuses the rest.
public record MessagesRequest(
        String model,
        List<Object> messages,
        Object system,
        @JsonProperty("max_tokens") Integer maxTokens,
        @JsonProperty("stop_sequences") Object stopSequences,
        Boolean stream,
        Double temperature,
        @JsonProperty("top_p") Double topP,
        @JsonProperty("top_k") Integer topK,
        Object tools,
        @JsonProperty("tool_choice") Object toolChoice,
        Object thinking,
        @JsonProperty("output_config") Object outputConfig,
        @JsonProperty("output_format") Object outputFormat,
        @JsonAnySetter Map<String, Object> otherFields) {

    public MessagesRequest {
        otherFields = otherFields == null ? Map.of() : otherFields;
    }
}
