package io.euhedral_execution.inference.api.web;

import io.euhedral_execution.inference.api.anthropic.MessagesMapper;
import io.euhedral_execution.inference.api.anthropic.MessagesRequest;
import io.euhedral_execution.inference.api.chat.ApiException;
import io.euhedral_execution.inference.api.chat.GenerationService;
import io.euhedral_execution.inference.api.engine.ApiProperties;
import jakarta.servlet.http.HttpServletRequest;
import java.util.Map;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.context.request.async.DeferredResult;

/// Anthropic-compatible Messages surface under `/v1`, so Anthropic clients use `http://host:port` as their base
/// URL. Requests become the same conversations and generations as Chat Completions; only the formats differ.
/// Authentication headers (`x-api-key`) and `anthropic-version`/`anthropic-beta` are accepted and not checked.
@RestController
@RequestMapping("/v1")
public class AnthropicController {
    private final GenerationService generations;
    private final MessagesMapper messages;
    private final long timeoutMillis;

    public AnthropicController(GenerationService generations, MessagesMapper messages, ApiProperties properties) {
        this.generations = generations;
        this.messages = messages;
        this.timeoutMillis = properties.requestTimeout().toMillis();
    }

    @PostMapping(path = "/messages", consumes = MediaType.APPLICATION_JSON_VALUE)
    public DeferredResult<Object> messages(@RequestBody MessagesRequest request, HttpServletRequest servlet) {
        return this.generations.submit("messages", this.messages.planAsync(request), ServletClientLink.of(servlet));
    }

    /// `{"input_tokens": N}`: the rendered prompt's tokens, encoded on the workers without generating.
    @PostMapping(path = "/messages/count_tokens", consumes = MediaType.APPLICATION_JSON_VALUE)
    public DeferredResult<Object> countTokens(@RequestBody MessagesRequest request) {
        var result = new DeferredResult<Object>(this.timeoutMillis);
        result.onTimeout(() -> result.setErrorResult(ApiException.timeout()));
        this.messages.countTokensAsync(request).whenComplete((tokens, failure) -> {
            if (failure != null) result.setErrorResult(ApiException.from(failure));
            else
                result.setResult(ResponseEntity.ok()
                        .contentType(MediaType.APPLICATION_JSON)
                        .body(Map.of("input_tokens", tokens)));
        });
        return result;
    }
}
