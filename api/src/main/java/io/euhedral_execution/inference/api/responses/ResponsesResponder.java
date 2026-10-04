package io.euhedral_execution.inference.api.responses;

import io.euhedral_execution.inference.api.chat.ApiException;
import io.euhedral_execution.inference.api.chat.CompletionSink;
import io.euhedral_execution.inference.api.chat.Finish;
import io.euhedral_execution.inference.api.chat.GeneratedCall;
import io.euhedral_execution.inference.api.chat.GenerationPlan;
import io.euhedral_execution.inference.api.chat.GenerationService;
import io.euhedral_execution.inference.api.chat.TokenUsage;
import io.euhedral_execution.inference.api.openai.OpenAiError;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.context.request.async.DeferredResult;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/// Writes a generation as an OpenAI `response` object, or as its stream of `response.*` events.
///
/// Output items come in output order: a `reasoning` item (its text as one `reasoning_text` part, an empty `summary`,
/// and with `include: ["reasoning.encrypted_content"]` an opaque token of it), then a `message` with one
/// `output_text` part or `function_call` items. A generation that ran out of `max_output_tokens` is `incomplete`.
record ResponsesResponder(long created, String model, boolean includeEncrypted, Map<String, Object> echo)
        implements GenerationService.Responder {
    private static final MediaType JSON = MediaType.APPLICATION_JSON;

    @Override
    public CompletionSink json(GenerationPlan plan, DeferredResult<Object> result) {
        return new JsonSink(new Output(plan), result);
    }

    @Override
    public CompletionSink stream(GenerationPlan plan, SseEmitter emitter) {
        return new StreamSink(new Output(plan), emitter);
    }

    private static String id(String prefix) {
        return prefix + UUID.randomUUID().toString().replace("-", "");
    }

    private static Map<String, Object> ordered(Object... keysAndValues) {
        Map<String, Object> object = new LinkedHashMap<>();
        for (int index = 0; index < keysAndValues.length; index += 2)
            object.put((String) keysAndValues[index], keysAndValues[index + 1]);
        return object;
    }

    /// The response's items as they are produced; one item is open at a time.
    private final class Output {
        final GenerationPlan plan;
        final String id;
        final List<Map<String, Object>> items = new ArrayList<>();
        final StringBuilder openText = new StringBuilder();
        Map<String, Object> open;

        Output(GenerationPlan plan) {
            this.plan = plan;
            this.id = "resp_" + plan.id();
        }

        Map<String, Object> reasoningItem() {
            return ordered("id", id("rs_"), "type", "reasoning", "summary", List.of(), "content", List.of());
        }

        Map<String, Object> messageItem() {
            return ordered(
                    "id",
                    id("msg_"),
                    "type",
                    "message",
                    "status",
                    "in_progress",
                    "role",
                    "assistant",
                    "content",
                    List.of());
        }

        static Map<String, Object> outputText(String text) {
            return ordered("type", "output_text", "text", text, "annotations", List.of());
        }

        Map<String, Object> callItem(GeneratedCall call, String status) {
            return ordered(
                    "id",
                    id("fc_"),
                    "type",
                    "function_call",
                    "status",
                    status,
                    "call_id",
                    id("call_"),
                    "name",
                    call.name(),
                    "arguments",
                    call.arguments());
        }

        /// Fills the open item's final content and adds it to the response.
        Map<String, Object> close() {
            Map<String, Object> item = this.open;
            this.open = null;
            if (item == null) return null;
            String text = this.openText.toString();
            this.openText.setLength(0);
            if (item.get("type").equals("reasoning")) {
                item.put("content", List.of(ordered("type", "reasoning_text", "text", text)));
                if (includeEncrypted) item.put("encrypted_content", ResponsesMapper.encrypt(text));
            } else {
                item.put("status", "completed");
                item.put("content", List.of(outputText(text)));
            }
            this.items.add(item);
            return item;
        }

        Map<String, Object> response(String status, Finish finish) {
            Map<String, Object> response = new LinkedHashMap<>();
            response.put("id", this.id);
            response.put("object", "response");
            response.put("created_at", created);
            response.put("status", status);
            response.put("error", null);
            response.put(
                    "incomplete_details",
                    finish != null && finish.reason() == Finish.Reason.LENGTH
                            ? Map.of("reason", "max_output_tokens")
                            : null);
            response.put("model", model);
            response.put("output", finish == null ? List.of() : List.copyOf(this.items));
            response.putAll(echo);
            response.put("usage", finish == null ? null : usage(finish.usage()));
            return response;
        }

        static String status(Finish finish) {
            return finish.reason() == Finish.Reason.LENGTH ? "incomplete" : "completed";
        }

        static Map<String, Object> usage(TokenUsage usage) {
            return ordered(
                    "input_tokens", usage.promptTokens(),
                    "input_tokens_details", Map.of("cached_tokens", usage.cachedPromptTokens()),
                    "output_tokens", usage.completionTokens(),
                    "output_tokens_details",
                            Map.of("reasoning_tokens", usage.reasoningTokens() == null ? 0 : usage.reasoningTokens()),
                    "total_tokens", usage.promptTokens() + usage.completionTokens());
        }
    }

    private final class JsonSink implements CompletionSink {
        private final Output output;
        private final DeferredResult<Object> result;

        private JsonSink(Output output, DeferredResult<Object> result) {
            this.output = output;
            this.result = result;
        }

        @Override
        public void start(int cachedPromptTokens) {}

        @Override
        public void reasoning(String delta) {
            if (this.output.open == null) this.output.open = this.output.reasoningItem();
            this.output.openText.append(delta);
        }

        @Override
        public void text(String delta) {
            if (this.output.open != null && !this.output.open.get("type").equals("message")) this.output.close();
            if (this.output.open == null) this.output.open = this.output.messageItem();
            this.output.openText.append(delta);
        }

        @Override
        public void toolCall(int index, GeneratedCall call) {
            this.output.close();
            this.output.items.add(this.output.callItem(call, "completed"));
        }

        @Override
        public void finish(Finish finish) {
            this.output.close();
            this.result.setResult(
                    ResponseEntity.ok().contentType(JSON).body(this.output.response(Output.status(finish), finish)));
        }

        @Override
        public void fail(ApiException error) {
            this.result.setErrorResult(error);
        }
    }

    /// Events are written as output is decoded, each with its `sequence_number`. A failed write means the client is
    /// gone.
    private final class StreamSink implements CompletionSink {
        private final Output output;
        private final SseEmitter emitter;
        private int sequence;

        private StreamSink(Output output, SseEmitter emitter) {
            this.output = output;
            this.emitter = emitter;
        }

        @Override
        public void start(int cachedPromptTokens) throws IOException {
            Map<String, Object> response = this.output.response("in_progress", null);
            send("response.created", ordered("response", response));
            send("response.in_progress", ordered("response", response));
        }

        @Override
        public void reasoning(String delta) throws IOException {
            if (this.output.open == null) open(this.output.reasoningItem(), "reasoning_text");
            this.output.openText.append(delta);
            send("response.reasoning_text.delta", partEvent(delta));
        }

        @Override
        public void text(String delta) throws IOException {
            if (this.output.open != null && !this.output.open.get("type").equals("message")) closeOpen();
            if (this.output.open == null) open(this.output.messageItem(), "output_text");
            this.output.openText.append(delta);
            send("response.output_text.delta", partEvent(delta));
        }

        @Override
        public void toolCall(int index, GeneratedCall call) throws IOException {
            closeOpen();
            Map<String, Object> item = this.output.callItem(call, "in_progress");
            int outputIndex = this.output.items.size();
            Map<String, Object> added = new LinkedHashMap<>(item);
            added.put("arguments", "");
            send("response.output_item.added", ordered("output_index", outputIndex, "item", added));
            send(
                    "response.function_call_arguments.delta",
                    ordered("item_id", item.get("id"), "output_index", outputIndex, "delta", call.arguments()));
            send(
                    "response.function_call_arguments.done",
                    ordered("item_id", item.get("id"), "output_index", outputIndex, "arguments", call.arguments()));
            item.put("status", "completed");
            this.output.items.add(item);
            send("response.output_item.done", ordered("output_index", outputIndex, "item", item));
        }

        @Override
        public void finish(Finish finish) throws IOException {
            closeOpen();
            String status = Output.status(finish);
            send("response." + status, ordered("response", this.output.response(status, finish)));
            try {
                this.emitter.complete();
            } catch (IllegalStateException alreadyCompleted) {
                throw new IOException("stream already completed", alreadyCompleted);
            }
        }

        @Override
        public void keepAlive() throws IOException {
            try {
                this.emitter.send(SseEmitter.event().comment("keep-alive"));
            } catch (IllegalStateException alreadyCompleted) {
                throw new IOException("stream already completed", alreadyCompleted);
            }
        }

        /// Headers are already committed, so a failure is an in-band `error` event.
        @Override
        public void fail(ApiException error) {
            try {
                OpenAiError.Body body = OpenAiError.of(error).error();
                send("error", ordered("code", body.code(), "message", body.message(), "param", body.param()));
                this.emitter.complete();
            } catch (IOException clientGone) {
                // Nobody is listening; the container completes the request.
            }
        }

        private void open(Map<String, Object> item, String partType) throws IOException {
            this.output.open = item;
            int outputIndex = this.output.items.size();
            send("response.output_item.added", ordered("output_index", outputIndex, "item", item));
            send(
                    "response.content_part.added",
                    ordered(
                            "item_id",
                            item.get("id"),
                            "output_index",
                            outputIndex,
                            "content_index",
                            0,
                            "part",
                            partType.equals("output_text")
                                    ? Output.outputText("")
                                    : ordered("type", partType, "text", "")));
        }

        private void closeOpen() throws IOException {
            Map<String, Object> item = this.output.open;
            if (item == null) return;
            int outputIndex = this.output.items.size();
            String text = this.output.openText.toString();
            boolean reasoning = item.get("type").equals("reasoning");
            Map<String, Object> done =
                    ordered("item_id", item.get("id"), "output_index", outputIndex, "content_index", 0, "text", text);
            send(reasoning ? "response.reasoning_text.done" : "response.output_text.done", done);
            send(
                    "response.content_part.done",
                    ordered(
                            "item_id",
                            item.get("id"),
                            "output_index",
                            outputIndex,
                            "content_index",
                            0,
                            "part",
                            reasoning ? ordered("type", "reasoning_text", "text", text) : Output.outputText(text)));
            Map<String, Object> closed = this.output.close();
            send("response.output_item.done", ordered("output_index", outputIndex, "item", closed));
        }

        private Map<String, Object> partEvent(String delta) {
            Map<String, Object> item = this.output.open;
            return ordered(
                    "item_id",
                    item.get("id"),
                    "output_index",
                    this.output.items.size(),
                    "content_index",
                    0,
                    "delta",
                    delta);
        }

        private void send(String type, Map<String, Object> fields) throws IOException {
            Map<String, Object> event = new LinkedHashMap<>();
            event.put("type", type);
            event.put("sequence_number", this.sequence++);
            event.putAll(fields);
            try {
                this.emitter.send(SseEmitter.event().name(type).data(event, JSON));
            } catch (IllegalStateException alreadyCompleted) {
                throw new IOException("stream already completed", alreadyCompleted);
            }
        }
    }
}
