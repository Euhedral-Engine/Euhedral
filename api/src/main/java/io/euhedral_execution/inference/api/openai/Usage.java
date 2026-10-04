package io.euhedral_execution.inference.api.openai;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

/// Tokenizer-derived token accounting for one completion. `completion_tokens_details` appears when the output
/// opened in the model's think block; its reasoning tokens are part of `completion_tokens`.
@JsonInclude(JsonInclude.Include.NON_NULL)
public record Usage(
        @JsonProperty("prompt_tokens") int promptTokens,
        @JsonProperty("completion_tokens") int completionTokens,
        @JsonProperty("total_tokens") int totalTokens,
        @JsonProperty("completion_tokens_details") CompletionDetails completionDetails) {

    public static Usage of(int promptTokens, int completionTokens) {
        return of(promptTokens, completionTokens, null);
    }

    /// `reasoningTokens` null leaves out the completion details.
    public static Usage of(int promptTokens, int completionTokens, Integer reasoningTokens) {
        return new Usage(
                promptTokens,
                completionTokens,
                promptTokens + completionTokens,
                reasoningTokens == null ? null : new CompletionDetails(reasoningTokens));
    }

    public record CompletionDetails(
            @JsonProperty("reasoning_tokens") int reasoningTokens) {}
}
