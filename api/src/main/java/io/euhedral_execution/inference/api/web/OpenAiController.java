package io.euhedral_execution.inference.api.web;

import io.euhedral_execution.inference.api.chat.ChatCompletionService;
import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.api.openai.ChatCompletionRequest;
import io.euhedral_execution.inference.api.openai.ModelList;
import io.euhedral_execution.inference.api.openai.OpenAiException;
import java.time.Instant;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.context.request.async.DeferredResult;

/// OpenAI-compatible surface rooted at `/v1`, so clients use `http://host:port/v1` as their base URL.
///
/// All `/v1` routes live here; authentication can later be added as a filter on `/v1/**` without
/// touching generation code.
@RestController
@RequestMapping("/v1")
public class OpenAiController {
    private final InferenceBackend backend;
    private final ChatCompletionService completions;
    private final long createdAt = Instant.now().getEpochSecond();

    public OpenAiController(InferenceBackend backend, ChatCompletionService completions) {
        this.backend = backend;
        this.completions = completions;
    }

    @GetMapping("/models")
    public ModelList models() {
        return ModelList.of(ModelList.Model.of(this.backend.modelId(), this.createdAt));
    }

    @GetMapping("/models/{model}")
    public ModelList.Model model(@PathVariable String model) {
        if (!model.equals(this.backend.modelId())) throw OpenAiException.modelNotFound(model);
        return ModelList.Model.of(model, this.createdAt);
    }

    /// Hands the request to the workers and returns its deferred response, releasing this thread: the workers
    /// plan the request and resolve it to a JSON body, an `SseEmitter` for `stream=true`, or an OpenAI error.
    @PostMapping(path = "/chat/completions", consumes = MediaType.APPLICATION_JSON_VALUE)
    public DeferredResult<Object> chatCompletions(@RequestBody ChatCompletionRequest request) {
        return this.completions.complete(request);
    }
}
