package io.euhedral_execution.inference.api.openai;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

/// Tokenizer-derived token accounting for one completion. `prompt_tokens_details.cached_tokens` counts the prompt
/// tokens restored from the prefix cache instead of prefilled (part of `prompt_tokens`). `completion_tokens_details`
/// appears when the output opened in the model's think block; its reasoning tokens are part of `completion_tokens`.
@JsonInclude(JsonInclude.Include.NON_NULL)
public record Usage(
        @JsonProperty("prompt_tokens") int promptTokens,
        @JsonProperty("completion_tokens") int completionTokens,
        @JsonProperty("total_tokens") int totalTokens,
        @JsonProperty("prompt_tokens_details") PromptDetails promptDetails,
        @JsonProperty("completion_tokens_details") CompletionDetails completionDetails) {

    /// `reasoningTokens` null leaves out the completion details.
    public static Usage of(int promptTokens, int cachedTokens, int completionTokens, Integer reasoningTokens) {
        return new Usage(
                promptTokens,
                completionTokens,
                promptTokens + completionTokens,
                new PromptDetails(cachedTokens),
                reasoningTokens == null ? null : new CompletionDetails(reasoningTokens));
    }

    public record PromptDetails(
            @JsonProperty("cached_tokens") int cachedTokens) {}

    public record CompletionDetails(
            @JsonProperty("reasoning_tokens") int reasoningTokens) {}
}
