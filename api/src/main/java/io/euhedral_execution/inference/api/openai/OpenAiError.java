package io.euhedral_execution.inference.api.openai;

import com.fasterxml.jackson.annotation.JsonInclude;

/// OpenAI error envelope: `{"error": {"message", "type", "param", "code"}}`.
public record OpenAiError(Body error) {

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record Body(String message, String type, String param, String code) {}

    /// The OpenAI error object of an API failure.
    public static OpenAiError of(io.euhedral_execution.inference.api.chat.ApiException failure) {
        return new OpenAiError(new Body(failure.getMessage(), failure.type(), failure.param(), failure.code()));
    }
}
