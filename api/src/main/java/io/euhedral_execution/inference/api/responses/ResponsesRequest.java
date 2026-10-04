package io.euhedral_execution.inference.api.responses;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Map;

/// An OpenAI Responses request (`POST /v1/responses`). Input items stay untyped JSON so [ResponsesMapper] can report
/// precise errors; every other top-level field lands in `otherFields`, where the mapper accepts metadata and
/// refuses the rest.
public record ResponsesRequest(
        String model,
        Object input,
        String instructions,
        Object tools,
        @JsonProperty("tool_choice") Object toolChoice,
        @JsonProperty("parallel_tool_calls") Boolean parallelToolCalls,
        @JsonProperty("max_output_tokens") Integer maxOutputTokens,
        Double temperature,
        @JsonProperty("top_p") Double topP,
        Boolean stream,
        Object text,
        Object reasoning,
        Object include,
        @JsonAnySetter Map<String, Object> otherFields) {

    public ResponsesRequest {
        otherFields = otherFields == null ? Map.of() : otherFields;
    }
}
