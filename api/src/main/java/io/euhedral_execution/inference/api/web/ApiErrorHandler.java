package io.euhedral_execution.inference.api.web;

import io.euhedral_execution.inference.api.engine.InferenceUnavailableException;
import io.euhedral_execution.inference.api.openai.OpenAiError;
import io.euhedral_execution.inference.api.openai.OpenAiException;
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

/// Maps every failure to an OpenAI error object. Internal exception text is logged, never returned.
@RestControllerAdvice
public class OpenAiErrorHandler {
    private static final Logger LOG = LoggerFactory.getLogger(OpenAiErrorHandler.class);

    @ExceptionHandler(OpenAiException.class)
    ResponseEntity<OpenAiError> openAi(OpenAiException exception) {
        return body(exception.status(), exception.toError());
    }

    @ExceptionHandler(InferenceUnavailableException.class)
    ResponseEntity<OpenAiError> unavailable(InferenceUnavailableException exception) {
        return openAi(OpenAiException.unavailable("The inference engine is unavailable: " + exception.getMessage()));
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    ResponseEntity<OpenAiError> unreadable(HttpMessageNotReadableException exception) {
        for (Throwable cause = exception; cause != null; cause = cause.getCause()) {
            if (cause instanceof ChatRequestBodyLimit.TooLarge) return openAi(OpenAiException.requestTooLarge());
        }
        LOG.debug("Unreadable request body", exception);
        return openAi(OpenAiException.invalidRequest(
                "We could not parse the JSON body of your request. Check that it is valid JSON and that each"
                        + " field has the documented type.",
                null));
    }

    @ExceptionHandler(NoResourceFoundException.class)
    ResponseEntity<OpenAiError> notFound(HttpServletRequest request) {
        return openAi(OpenAiException.notFound(
                "Unknown request URL: " + request.getMethod() + " " + request.getRequestURI() + "."));
    }

    /// The client disconnected; there is nobody to answer.
    @ExceptionHandler(AsyncRequestNotUsableException.class)
    void clientGone() {}

    /// Spring MVC protocol errors (405, 415, 406, ...) implement `ErrorResponse` and keep their status.
    @ExceptionHandler(Exception.class)
    ResponseEntity<OpenAiError> unexpected(Exception exception) {
        if (exception instanceof ErrorResponse framework
                && framework.getStatusCode().is4xxClientError()) {
            String detail = framework.getBody().getDetail();
            return body(
                    framework.getStatusCode(),
                    new OpenAiError(new OpenAiError.Body(
                            detail == null ? "Invalid request." : detail, "invalid_request_error", null, null)));
        }
        LOG.error("Unhandled API failure", exception);
        return openAi(OpenAiException.serverError());
    }

    private static ResponseEntity<OpenAiError> body(HttpStatusCode status, OpenAiError error) {
        // Explicit type: an SSE Accept header must not turn an error into a 406.
        return ResponseEntity.status(status)
                .contentType(MediaType.APPLICATION_JSON)
                .body(error);
    }
}
