package io.euhedral_execution.inference.api.anthropic;

import io.euhedral_execution.inference.api.chat.ApiException;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.http.HttpStatusCode;

/// Anthropic's error object: `{"type": "error", "error": {"type", "message"}}`, the type named after the status.
public final class AnthropicErrors {
    private AnthropicErrors() {}

    public static Map<String, Object> body(ApiException failure) {
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("type", type(failure.status().value()));
        error.put("message", failure.getMessage());
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("type", "error");
        body.put("error", error);
        return body;
    }

    /// Anthropic answers an overloaded API with 529, and its clients treat that status as overloaded and retry; the
    /// server's 503 (at capacity, shutting down) is that condition.
    public static HttpStatusCode status(ApiException failure) {
        return failure.status().value() == 503 ? HttpStatusCode.valueOf(529) : failure.status();
    }

    static String type(int status) {
        return switch (status) {
            case 400, 405, 406, 415 -> "invalid_request_error";
            case 401 -> "authentication_error";
            case 403 -> "permission_error";
            case 404 -> "not_found_error";
            case 413 -> "request_too_large";
            case 429 -> "rate_limit_error";
            case 503 -> "overloaded_error";
            default -> "api_error";
        };
    }
}
