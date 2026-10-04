package io.euhedral_execution.inference.api.web;

import io.euhedral_execution.inference.api.anthropic.AnthropicErrors;
import io.euhedral_execution.inference.api.chat.ApiException;
import io.euhedral_execution.inference.api.engine.InferenceUnavailableException;
import io.euhedral_execution.inference.api.openai.OpenAiError;
import jakarta.servlet.http.HttpServletRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.ErrorResponse;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.context.request.async.AsyncRequestNotUsableException;
import org.springframework.web.servlet.resource.NoResourceFoundException;

/// Maps every failure to the error object of the surface the request addressed: Anthropic's for `/v1/messages`,
/// OpenAI's otherwise. Internal exception text is logged, never returned.
@RestControllerAdvice
public class ApiErrorHandler {
    private static final Logger LOG = LoggerFactory.getLogger(ApiErrorHandler.class);

    @ExceptionHandler(ApiException.class)
    ResponseEntity<Object> api(ApiException exception, HttpServletRequest request) {
        if (request.getRequestURI().startsWith("/v1/messages"))
            return body(AnthropicErrors.status(exception), AnthropicErrors.body(exception));
        return body(exception.status(), OpenAiError.of(exception));
    }

    @ExceptionHandler(InferenceUnavailableException.class)
    ResponseEntity<Object> unavailable(InferenceUnavailableException exception, HttpServletRequest request) {
        return api(ApiException.unavailable("The inference engine is unavailable: " + exception.getMessage()), request);
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    ResponseEntity<Object> unreadable(HttpMessageNotReadableException exception, HttpServletRequest request) {
        for (Throwable cause = exception; cause != null; cause = cause.getCause()) {
            if (cause instanceof ChatRequestBodyLimit.TooLarge) return api(ApiException.requestTooLarge(), request);
        }
        LOG.debug("Unreadable request body", exception);
        return api(
                ApiException.invalidRequest(
                        "We could not parse the JSON body of your request. Check that it is valid JSON and that each"
                                + " field has the documented type.",
                        null),
                request);
    }

    @ExceptionHandler(NoResourceFoundException.class)
    ResponseEntity<Object> notFound(HttpServletRequest request) {
        return api(
                ApiException.notFound(
                        "Unknown request URL: " + request.getMethod() + " " + request.getRequestURI() + "."),
                request);
    }

    /// The client disconnected; there is nobody to answer.
    @ExceptionHandler(AsyncRequestNotUsableException.class)
    void clientGone() {}

    /// Spring MVC protocol errors (405, 415, 406, ...) implement `ErrorResponse` and keep their status.
    @ExceptionHandler(Exception.class)
    ResponseEntity<Object> unexpected(Exception exception, HttpServletRequest request) {
        if (exception instanceof ErrorResponse framework
                && framework.getStatusCode().is4xxClientError()) {
            String detail = framework.getBody().getDetail();
            return api(
                    ApiException.protocol(framework.getStatusCode(), detail == null ? "Invalid request." : detail),
                    request);
        }
        LOG.error("Unhandled API failure", exception);
        return api(ApiException.serverError(), request);
    }

    private static ResponseEntity<Object> body(HttpStatusCode status, Object error) {
        // Explicit type: an SSE Accept header must not turn an error into a 406.
        return ResponseEntity.status(status)
                .contentType(MediaType.APPLICATION_JSON)
                .body(error);
    }
}
