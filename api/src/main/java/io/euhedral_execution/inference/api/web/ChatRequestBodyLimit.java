package io.euhedral_execution.inference.api.web;

import io.euhedral_execution.inference.api.anthropic.MessagesRequest;
import io.euhedral_execution.inference.api.chat.ApiException;
import io.euhedral_execution.inference.api.engine.ApiProperties;
import io.euhedral_execution.inference.api.openai.ChatCompletionRequest;
import io.euhedral_execution.inference.api.responses.ResponsesRequest;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import org.springframework.core.MethodParameter;
import org.springframework.http.HttpInputMessage;
import org.springframework.http.converter.HttpMessageConverter;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.servlet.mvc.method.annotation.RequestBodyAdviceAdapter;

/// Limits generation request bytes (Chat Completions and Messages) while Jackson reads the body, before it allocates
/// tool schemas or renders
/// the prompt. A declared length is rejected immediately; chunked bodies are counted as they arrive.
@ControllerAdvice
final class ChatRequestBodyLimit extends RequestBodyAdviceAdapter {
    private final int maxBytes;

    ChatRequestBodyLimit(ApiProperties properties) {
        this.maxBytes = properties.maxRequestBytes();
    }

    @Override
    public boolean supports(
            MethodParameter method, Type targetType, Class<? extends HttpMessageConverter<?>> converter) {
        return targetType == ChatCompletionRequest.class
                || targetType == MessagesRequest.class
                || targetType == ResponsesRequest.class;
    }

    @Override
    public HttpInputMessage beforeBodyRead(
            HttpInputMessage message,
            MethodParameter method,
            Type targetType,
            Class<? extends HttpMessageConverter<?>> converter)
            throws IOException {
        if (message.getHeaders().getContentLength() > this.maxBytes) throw ApiException.requestTooLarge();
        return new HttpInputMessage() {
            @Override
            public InputStream getBody() throws IOException {
                return new FilterInputStream(message.getBody()) {
                    private int remaining = maxBytes;

                    @Override
                    public int read() throws IOException {
                        int next = super.read();
                        if (next < 0) return next;
                        if (this.remaining-- == 0) throw new TooLarge();
                        return next;
                    }

                    @Override
                    public int read(byte[] bytes, int offset, int length) throws IOException {
                        if (length == 0) return 0;
                        int count = this.in.read(bytes, offset, Math.min(length, this.remaining + 1));
                        if (count < 0) return count;
                        if (count > this.remaining) throw new TooLarge();
                        this.remaining -= count;
                        return count;
                    }
                };
            }

            @Override
            public org.springframework.http.HttpHeaders getHeaders() {
                return message.getHeaders();
            }
        };
    }

    static final class TooLarge extends IOException {
        TooLarge() {
            super("Chat completion request body exceeds the configured size limit.");
        }
    }
}
