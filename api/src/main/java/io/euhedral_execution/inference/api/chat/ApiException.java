package io.euhedral_execution.inference.api.chat;

import org.springframework.http.HttpStatus;

/// An API failure with its HTTP status and error fields, in OpenAI's vocabulary (`type`, `param`, `code`); each API
/// surface serializes it in its own error format. Messages are client-safe by construction.
public final class ApiException extends RuntimeException {
    private final HttpStatus status;
    private final String type;
    private final String param;
    private final String code;

    private ApiException(HttpStatus status, String type, String message, String param, String code) {
        super(message);
        this.status = status;
        this.type = type;
        this.param = param;
        this.code = code;
    }

    /// A protocol error the web framework raised (405, 415, ...), with its status.
    public static ApiException protocol(org.springframework.http.HttpStatusCode status, String message) {
        return new ApiException(HttpStatus.valueOf(status.value()), "invalid_request_error", message, null, null);
    }

    public static ApiException invalidRequest(String message, String param) {
        return new ApiException(HttpStatus.BAD_REQUEST, "invalid_request_error", message, param, null);
    }

    public static ApiException requestTooLarge() {
        return new ApiException(
                HttpStatus.CONTENT_TOO_LARGE,
                "invalid_request_error",
                "The chat completion request body exceeds the configured size limit.",
                null,
                "request_too_large");
    }

    public static ApiException unsupportedParameter(String param) {
        return new ApiException(
                HttpStatus.BAD_REQUEST,
                "invalid_request_error",
                "Unsupported parameter: '" + param + "' is not supported by this server.",
                param,
                "unsupported_parameter");
    }

    public static ApiException unrecognizedArgument(String param) {
        return new ApiException(
                HttpStatus.BAD_REQUEST,
                "invalid_request_error",
                "Unrecognized request argument supplied: " + param,
                param,
                "unrecognized_argument");
    }

    public static ApiException contextLengthExceeded(String message, String param) {
        return new ApiException(
                HttpStatus.BAD_REQUEST, "invalid_request_error", message, param, "context_length_exceeded");
    }

    public static ApiException modelNotFound(String model) {
        return new ApiException(
                HttpStatus.NOT_FOUND,
                "invalid_request_error",
                "The model '" + model + "' does not exist or you do not have access to it.",
                "model",
                "model_not_found");
    }

    public static ApiException notFound(String message) {
        return new ApiException(HttpStatus.NOT_FOUND, "invalid_request_error", message, null, null);
    }

    public static ApiException unavailable(String message) {
        return new ApiException(
                HttpStatus.SERVICE_UNAVAILABLE, "service_unavailable_error", message, null, "engine_unavailable");
    }

    public static ApiException serverError() {
        return new ApiException(
                HttpStatus.INTERNAL_SERVER_ERROR,
                "server_error",
                "The server had an error while processing your request.",
                null,
                null);
    }

    /// The model's output began a tool call that cannot be returned for the offered tools. The reason
    /// describes only the generated text and the request's own definitions.
    public static ApiException invalidToolCall(String reason) {
        return new ApiException(
                HttpStatus.INTERNAL_SERVER_ERROR,
                "server_error",
                "The model produced an invalid tool call: " + reason + ".",
                null,
                "invalid_tool_call");
    }

    /// The model's output under a JSON response format is not a JSON document; never expected, since the
    /// grammar admits only documents, but never returned as a success either.
    public static ApiException invalidStructuredOutput() {
        return new ApiException(
                HttpStatus.INTERNAL_SERVER_ERROR,
                "server_error",
                "The model's output is not valid JSON for the requested response format.",
                null,
                "invalid_structured_output");
    }

    public static ApiException timeout() {
        return new ApiException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "service_unavailable_error",
                "The request timed out before generation completed.",
                null,
                "timeout");
    }

    /// The API failure a failed request future carries: itself, the engine's unavailability, or a server error.
    public static ApiException from(Throwable failure) {
        Throwable cause = failure instanceof java.util.concurrent.CompletionException && failure.getCause() != null
                ? failure.getCause()
                : failure;
        if (cause instanceof ApiException api) return api;
        if (cause instanceof io.euhedral_execution.inference.api.engine.InferenceUnavailableException unavailable)
            return unavailable("The inference engine is unavailable: " + unavailable.getMessage());
        org.slf4j.LoggerFactory.getLogger(ApiException.class).error("Request failed", cause);
        return serverError();
    }

    public HttpStatus status() {
        return this.status;
    }

    public String type() {
        return this.type;
    }

    public String param() {
        return this.param;
    }

    public String code() {
        return this.code;
    }
}
