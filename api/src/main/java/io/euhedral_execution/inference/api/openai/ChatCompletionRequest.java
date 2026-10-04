package io.euhedral_execution.inference.api.openai;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Map;

/// Supported subset of an OpenAI Chat Completions request.
///
/// `stop` is a string or a list of strings. `tools` and `tool_choice` stay untyped JSON (maps, lists, and
/// scalars) so `ChatRequestMapper` can report precise errors and render tool schemas in their original
/// key order. Every other top-level field is captured in `otherFields` and classified by
/// `ChatRequestMapper` as ignorable metadata, a neutral default, or a rejected feature.
public record ChatCompletionRequest(
        String model,
        List<ChatMessage> messages,
        Boolean stream,
        @JsonProperty("stream_options") StreamOptions streamOptions,
        @JsonProperty("max_tokens") Integer maxTokens,
        @JsonProperty("max_completion_tokens") Integer maxCompletionTokens,
        Double temperature,
        @JsonProperty("top_p") Double topP,
        Long seed,
        Object stop,
        Object tools,
        @JsonProperty("tool_choice") Object toolChoice,
        @JsonProperty("parallel_tool_calls") Boolean parallelToolCalls,
        @JsonProperty("reasoning_effort") Object reasoningEffort,
        @JsonProperty("chat_template_kwargs") Object chatTemplateKwargs,
        @JsonProperty("response_format") Object responseFormat,
        @JsonAnySetter Map<String, Object> otherFields) {

    public ChatCompletionRequest {
        otherFields = otherFields == null ? Map.of() : otherFields;
    }

    public record StreamOptions(
            @JsonProperty("include_usage") Boolean includeUsage,
            @JsonAnySetter Map<String, Object> otherFields) {

        public StreamOptions {
            otherFields = otherFields == null ? Map.of() : otherFields;
        }
    }
}
