package io.euhedral_execution.inference.api.openai;

import org.springframework.http.HttpStatus;

/// An API failure with its HTTP status and OpenAI error fields. Messages are client-safe by construction.
public final class OpenAiException extends RuntimeException {
    private final HttpStatus status;
    private final String type;
    private final String param;
    private final String code;

    private OpenAiException(HttpStatus status, String type, String message, String param, String code) {
        super(message);
        this.status = status;
        this.type = type;
        this.param = param;
        this.code = code;
    }

    public static OpenAiException invalidRequest(String message, String param) {
        return new OpenAiException(HttpStatus.BAD_REQUEST, "invalid_request_error", message, param, null);
    }

    public static OpenAiException requestTooLarge() {
        return new OpenAiException(
                HttpStatus.CONTENT_TOO_LARGE,
                "invalid_request_error",
                "The chat completion request body exceeds the configured size limit.",
                null,
                "request_too_large");
    }

    public static OpenAiException unsupportedParameter(String param) {
        return new OpenAiException(
                HttpStatus.BAD_REQUEST,
                "invalid_request_error",
                "Unsupported parameter: '" + param + "' is not supported by this server.",
                param,
                "unsupported_parameter");
    }

    public static OpenAiException unrecognizedArgument(String param) {
        return new OpenAiException(
                HttpStatus.BAD_REQUEST,
                "invalid_request_error",
                "Unrecognized request argument supplied: " + param,
                param,
                "unrecognized_argument");
    }

    public static OpenAiException contextLengthExceeded(String message, String param) {
        return new OpenAiException(
                HttpStatus.BAD_REQUEST, "invalid_request_error", message, param, "context_length_exceeded");
    }

    public static OpenAiException modelNotFound(String model) {
        return new OpenAiException(
                HttpStatus.NOT_FOUND,
                "invalid_request_error",
                "The model '" + model + "' does not exist or you do not have access to it.",
                "model",
                "model_not_found");
    }

    public static OpenAiException notFound(String message) {
        return new OpenAiException(HttpStatus.NOT_FOUND, "invalid_request_error", message, null, null);
    }

    public static OpenAiException unavailable(String message) {
        return new OpenAiException(
                HttpStatus.SERVICE_UNAVAILABLE, "service_unavailable_error", message, null, "engine_unavailable");
    }

    public static OpenAiException serverError() {
        return new OpenAiException(
                HttpStatus.INTERNAL_SERVER_ERROR,
                "server_error",
                "The server had an error while processing your request.",
                null,
                null);
    }

    /// The model's output began a tool call that cannot be returned for the offered tools. The reason
    /// describes only the generated text and the request's own definitions.
    public static OpenAiException invalidToolCall(String reason) {
        return new OpenAiException(
                HttpStatus.INTERNAL_SERVER_ERROR,
                "server_error",
                "The model produced an invalid tool call: " + reason + ".",
                null,
                "invalid_tool_call");
    }

    /// The model's output under a JSON response format is not a JSON document; never expected, since the
    /// grammar admits only documents, but never returned as a success either.
    public static OpenAiException invalidStructuredOutput() {
        return new OpenAiException(
                HttpStatus.INTERNAL_SERVER_ERROR,
                "server_error",
                "The model's output is not valid JSON for the requested response format.",
                null,
                "invalid_structured_output");
    }

    public static OpenAiException timeout() {
        return new OpenAiException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "service_unavailable_error",
                "The request timed out before generation completed.",
                null,
                "timeout");
    }

    public HttpStatus status() {
        return this.status;
    }

    public OpenAiError toError() {
        return new OpenAiError(new OpenAiError.Body(getMessage(), this.type, this.param, this.code));
    }
}
