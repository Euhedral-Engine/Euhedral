package io.euhedral_execution.inference.api.web;

import io.euhedral_execution.inference.api.chat.ApiException;
import io.euhedral_execution.inference.api.chat.GenerationService;
import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.api.openai.ChatCompletionRequest;
import io.euhedral_execution.inference.api.openai.ChatRequestMapper;
import io.euhedral_execution.inference.api.openai.ModelList;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
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
    private final GenerationService generations;
    private final ChatRequestMapper chatRequests;
    private final long createdAt = Instant.now().getEpochSecond();

    public OpenAiController(InferenceBackend backend, GenerationService generations, ChatRequestMapper chatRequests) {
        this.backend = backend;
        this.generations = generations;
        this.chatRequests = chatRequests;
    }

    /// The served model, in OpenAI's list format, or Anthropic's for a client that sends `anthropic-version`.
    @GetMapping("/models")
    public Object models(@RequestHeader(name = "anthropic-version", required = false) String anthropicVersion) {
        if (anthropicVersion != null) {
            Map<String, Object> list = new LinkedHashMap<>();
            list.put("data", List.of(anthropicModel(this.backend.modelId())));
            list.put("has_more", false);
            list.put("first_id", this.backend.modelId());
            list.put("last_id", this.backend.modelId());
            return list;
        }
        return ModelList.of(ModelList.Model.of(this.backend.modelId(), this.createdAt));
    }

    @GetMapping("/models/{model}")
    public Object model(
            @PathVariable String model,
            @RequestHeader(name = "anthropic-version", required = false) String anthropicVersion) {
        if (!model.equals(this.backend.modelId())) throw ApiException.modelNotFound(model);
        return anthropicVersion != null ? anthropicModel(model) : ModelList.Model.of(model, this.createdAt);
    }

    private Map<String, Object> anthropicModel(String id) {
        Map<String, Object> model = new LinkedHashMap<>();
        model.put("type", "model");
        model.put("id", id);
        model.put("display_name", id);
        model.put("created_at", Instant.ofEpochSecond(this.createdAt).toString());
        return model;
    }

    /// Hands the request to the workers and returns its deferred response, releasing this thread: the workers
    /// plan the request and resolve it to a JSON body, an `SseEmitter` for `stream=true`, or an OpenAI error.
    @PostMapping(path = "/chat/completions", consumes = MediaType.APPLICATION_JSON_VALUE)
    public DeferredResult<Object> chatCompletions(@RequestBody ChatCompletionRequest request) {
        return this.generations.submit(this.chatRequests.planAsync(request));
    }
}
