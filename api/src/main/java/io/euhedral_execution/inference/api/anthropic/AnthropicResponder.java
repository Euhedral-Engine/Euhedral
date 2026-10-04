package io.euhedral_execution.inference.api.anthropic;

import io.euhedral_execution.inference.api.chat.ApiException;
import io.euhedral_execution.inference.api.chat.CompletionSink;
import io.euhedral_execution.inference.api.chat.Finish;
import io.euhedral_execution.inference.api.chat.GeneratedCall;
import io.euhedral_execution.inference.api.chat.GenerationPlan;
import io.euhedral_execution.inference.api.chat.GenerationService;
import io.euhedral_execution.inference.api.chat.TokenUsage;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.context.request.async.DeferredResult;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import tools.jackson.databind.json.JsonMapper;

/// Writes a generation as an Anthropic `message`, or as its stream of events.
///
/// Content blocks come in output order: a `thinking` block (the reasoning, with an opaque `signature`), then a
/// `text` block or `tool_use` blocks. `usage.input_tokens` counts the prompt tokens that were prefilled and
/// `cache_read_input_tokens` those restored from the prefix cache; nothing is ever written to a cache on request, so
/// `cache_creation_input_tokens` is 0. A stream's `message_start` is written when generation produced its first
/// token, by when the prefix-cache restore is known.
record AnthropicResponder(String model, boolean stream) implements GenerationService.Responder {
    private static final MediaType JSON = MediaType.APPLICATION_JSON;
    private static final JsonMapper MAPPER = JsonMapper.shared();

    @Override
    public CompletionSink json(GenerationPlan plan, DeferredResult<Object> result) {
        return new JsonSink(plan, result);
    }

    @Override
    public CompletionSink stream(GenerationPlan plan, SseEmitter emitter) {
        return new StreamSink(plan, emitter);
    }

    static String stopReason(Finish.Reason reason) {
        return switch (reason) {
            case END -> "end_turn";
            case STOP_SEQUENCE -> "stop_sequence";
            case LENGTH -> "max_tokens";
            case TOOL_CALLS -> "tool_use";
        };
    }

    static Map<String, Object> usage(int promptTokens, int cachedTokens, int outputTokens) {
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("input_tokens", promptTokens - cachedTokens);
        usage.put("cache_creation_input_tokens", 0);
        usage.put("cache_read_input_tokens", cachedTokens);
        usage.put("output_tokens", outputTokens);
        return usage;
    }

    /// An opaque token naming the thinking it accompanies; clients send it back with the block, unchecked.
    static String signature(String thinking) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(thinking.getBytes(StandardCharsets.UTF_8));
            return "euhedral-" + Base64.getUrlEncoder().withoutPadding().encodeToString(digest);
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException(impossible);
        }
    }

    /// A JSON object with its keys in the given order (`type` first, as Anthropic writes them).
    private static Map<String, Object> ordered(Object... keysAndValues) {
        Map<String, Object> object = new LinkedHashMap<>();
        for (int index = 0; index < keysAndValues.length; index += 2)
            object.put((String) keysAndValues[index], keysAndValues[index + 1]);
        return object;
    }

    static String toolUseId() {
        return "toolu_" + UUID.randomUUID().toString().replace("-", "");
    }

    private static Map<String, Object> toolUse(String id, GeneratedCall call) {
        Map<String, Object> block = new LinkedHashMap<>();
        block.put("type", "tool_use");
        block.put("id", id);
        block.put("name", call.name());
        block.put("input", MAPPER.readValue(call.arguments(), Map.class));
        return block;
    }

    private static Map<String, Object> message(String id, String model, List<Object> content) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("id", id);
        message.put("type", "message");
        message.put("role", "assistant");
        message.put("model", model);
        message.put("content", content);
        return message;
    }

    private final class JsonSink implements CompletionSink {
        private final GenerationPlan plan;
        private final DeferredResult<Object> result;
        private final StringBuilder thinking = new StringBuilder();
        private final StringBuilder text = new StringBuilder();
        private final List<Object> calls = new ArrayList<>();

        private JsonSink(GenerationPlan plan, DeferredResult<Object> result) {
            this.plan = plan;
            this.result = result;
        }

        @Override
        public void start(int cachedPromptTokens) {}

        @Override
        public void reasoning(String delta) {
            this.thinking.append(delta);
        }

        @Override
        public void text(String delta) {
            this.text.append(delta);
        }

        @Override
        public void toolCall(int index, GeneratedCall call) {
            this.calls.add(toolUse(toolUseId(), call));
        }

        @Override
        public void finish(Finish finish) {
            List<Object> content = new ArrayList<>();
            if (!this.thinking.isEmpty())
                content.add(ordered(
                        "type", "thinking",
                        "thinking", this.thinking.toString(),
                        "signature", signature(this.thinking.toString())));
            if (!this.text.isEmpty()) content.add(ordered("type", "text", "text", this.text.toString()));
            content.addAll(this.calls);
            Map<String, Object> message = message("msg_" + this.plan.id(), model, content);
            message.put("stop_reason", stopReason(finish.reason()));
            message.put("stop_sequence", finish.stopSequence());
            TokenUsage usage = finish.usage();
            message.put("usage", usage(usage.promptTokens(), usage.cachedPromptTokens(), usage.completionTokens()));
            this.result.setResult(ResponseEntity.ok().contentType(JSON).body(message));
        }

        @Override
        public void fail(ApiException error) {
            this.result.setErrorResult(error);
        }
    }

    /// Events are written as output is decoded; one content block is open at a time. A failed write means the
    /// client is gone.
    private final class StreamSink implements CompletionSink {
        private final GenerationPlan plan;
        private final SseEmitter emitter;
        private final StringBuilder thinking = new StringBuilder();
        private int index = -1;
        private String open;

        private StreamSink(GenerationPlan plan, SseEmitter emitter) {
            this.plan = plan;
            this.emitter = emitter;
        }

        /// Begins the message once generation reported how much of the prompt the prefix cache restored.
        @Override
        public void start(int cachedTokens) throws IOException {
            Map<String, Object> message = message("msg_" + this.plan.id(), model, List.of());
            message.put("stop_reason", null);
            message.put("stop_sequence", null);
            message.put("usage", usage(this.plan.promptTokens(), cachedTokens, 0));
            send("message_start", ordered("type", "message_start", "message", message));
        }

        @Override
        public void reasoning(String delta) throws IOException {
            if (!"thinking".equals(this.open))
                openBlock("thinking", ordered("type", "thinking", "thinking", "", "signature", ""));
            this.thinking.append(delta);
            delta(ordered("type", "thinking_delta", "thinking", delta));
        }

        @Override
        public void text(String delta) throws IOException {
            if (!"text".equals(this.open)) openBlock("text", ordered("type", "text", "text", ""));
            delta(ordered("type", "text_delta", "text", delta));
        }

        @Override
        public void toolCall(int index, GeneratedCall call) throws IOException {
            Map<String, Object> block = new LinkedHashMap<>();
            block.put("type", "tool_use");
            block.put("id", toolUseId());
            block.put("name", call.name());
            block.put("input", Map.of());
            openBlock("tool_use", block);
            delta(ordered("type", "input_json_delta", "partial_json", call.arguments()));
            closeBlock();
        }

        @Override
        public void finish(Finish finish) throws IOException {
            closeBlock();
            Map<String, Object> delta = new LinkedHashMap<>();
            delta.put("stop_reason", stopReason(finish.reason()));
            delta.put("stop_sequence", finish.stopSequence());
            TokenUsage usage = finish.usage();
            send(
                    "message_delta",
                    ordered(
                            "type",
                            "message_delta",
                            "delta",
                            delta,
                            "usage",
                            usage(usage.promptTokens(), usage.cachedPromptTokens(), usage.completionTokens())));
            send("message_stop", ordered("type", "message_stop"));
            try {
                this.emitter.complete();
            } catch (IllegalStateException alreadyCompleted) {
                throw new IOException("stream already completed", alreadyCompleted);
            }
        }

        /// Anthropic's own keep-alive event.
        @Override
        public void keepAlive() throws IOException {
            send("ping", ordered("type", "ping"));
        }

        /// Headers are already committed, so errors are reported in-band as an `error` event.
        @Override
        public void fail(ApiException error) {
            try {
                send("error", AnthropicErrors.body(error));
                this.emitter.complete();
            } catch (IOException clientGone) {
                // Nobody is listening; the container completes the request.
            }
        }

        private void openBlock(String type, Map<String, Object> block) throws IOException {
            closeBlock();
            this.index++;
            this.open = type;
            send(
                    "content_block_start",
                    ordered("type", "content_block_start", "index", this.index, "content_block", block));
        }

        private void closeBlock() throws IOException {
            if (this.open == null) return;
            if (this.open.equals("thinking"))
                delta(ordered("type", "signature_delta", "signature", signature(this.thinking.toString())));
            send("content_block_stop", ordered("type", "content_block_stop", "index", this.index));
            this.open = null;
        }

        private void delta(Map<String, Object> delta) throws IOException {
            send("content_block_delta", ordered("type", "content_block_delta", "index", this.index, "delta", delta));
        }

        private void send(String event, Object payload) throws IOException {
            try {
                this.emitter.send(SseEmitter.event().name(event).data(payload, JSON));
            } catch (IllegalStateException alreadyCompleted) {
                throw new IOException("stream already completed", alreadyCompleted);
            }
        }
    }
}
