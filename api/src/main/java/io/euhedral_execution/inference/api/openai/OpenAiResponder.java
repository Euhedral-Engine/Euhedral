package io.euhedral_execution.inference.api.openai;

import io.euhedral_execution.inference.api.chat.ApiException;
import io.euhedral_execution.inference.api.chat.CompletionSink;
import io.euhedral_execution.inference.api.chat.Finish;
import io.euhedral_execution.inference.api.chat.GeneratedCall;
import io.euhedral_execution.inference.api.chat.GenerationPlan;
import io.euhedral_execution.inference.api.chat.GenerationService;
import io.euhedral_execution.inference.api.chat.TokenUsage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.context.request.async.DeferredResult;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/// Writes a generation as an OpenAI `chat.completion` body or `chat.completion.chunk` stream.
record OpenAiResponder(long created, String model, boolean includeUsage) implements GenerationService.Responder {
    private static final MediaType JSON = MediaType.APPLICATION_JSON;

    @Override
    public CompletionSink json(GenerationPlan plan, DeferredResult<Object> result) {
        return new JsonSink("chatcmpl-" + plan.id(), result);
    }

    @Override
    public CompletionSink stream(GenerationPlan plan, SseEmitter emitter) {
        return new StreamSink("chatcmpl-" + plan.id(), emitter);
    }

    static String finishReason(Finish.Reason reason) {
        return switch (reason) {
            case END, STOP_SEQUENCE -> "stop";
            case LENGTH -> "length";
            case TOOL_CALLS -> "tool_calls";
        };
    }

    static Usage usage(TokenUsage usage) {
        return Usage.of(
                usage.promptTokens(), usage.cachedPromptTokens(), usage.completionTokens(), usage.reasoningTokens());
    }

    private static ToolCall toolCall(GeneratedCall call) {
        return ToolCall.function(
                "call_" + UUID.randomUUID().toString().replace("-", ""), call.name(), call.arguments());
    }

    private final class JsonSink implements CompletionSink {
        private final String id;
        private final DeferredResult<Object> result;
        private final StringBuilder reasoning = new StringBuilder();
        private final StringBuilder content = new StringBuilder();
        private final List<ToolCall> toolCalls = new ArrayList<>();

        private JsonSink(String id, DeferredResult<Object> result) {
            this.id = id;
            this.result = result;
        }

        @Override
        public void start(int cachedPromptTokens) {}

        @Override
        public void reasoning(String delta) {
            this.reasoning.append(delta);
        }

        @Override
        public void text(String delta) {
            this.content.append(delta);
        }

        @Override
        public void toolCall(int index, GeneratedCall call) {
            this.toolCalls.add(OpenAiResponder.toolCall(call));
        }

        @Override
        public void finish(Finish finish) {
            var response = ChatCompletionResponse.of(
                    this.id,
                    created,
                    model,
                    this.reasoning.toString(),
                    this.content.toString(),
                    this.toolCalls,
                    finishReason(finish.reason()),
                    usage(finish.usage()));
            this.result.setResult(ResponseEntity.ok().contentType(JSON).body(response));
        }

        @Override
        public void fail(ApiException error) {
            this.result.setErrorResult(error);
        }
    }

    /// Chunks are written as they are decoded. A failed write means the client is gone.
    private final class StreamSink implements CompletionSink {
        private final String id;
        private final SseEmitter emitter;

        private StreamSink(String id, SseEmitter emitter) {
            this.id = id;
            this.emitter = emitter;
        }

        @Override
        public void start(int cachedPromptTokens) throws IOException {
            send(ChatCompletionChunk.role(this.id, created, model));
        }

        @Override
        public void reasoning(String delta) throws IOException {
            send(ChatCompletionChunk.reasoning(this.id, created, model, delta));
        }

        @Override
        public void text(String delta) throws IOException {
            send(ChatCompletionChunk.content(this.id, created, model, delta));
        }

        @Override
        public void toolCall(int index, GeneratedCall call) throws IOException {
            send(ChatCompletionChunk.toolCall(this.id, created, model, index, OpenAiResponder.toolCall(call)));
        }

        @Override
        public void finish(Finish finish) throws IOException {
            send(ChatCompletionChunk.finish(this.id, created, model, finishReason(finish.reason())));
            if (includeUsage) send(ChatCompletionChunk.usage(this.id, created, model, usage(finish.usage())));
            try {
                this.emitter.send(SseEmitter.event().data("[DONE]", MediaType.TEXT_PLAIN));
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

        /// Headers are already committed, so errors are reported in-band as an OpenAI error event.
        @Override
        public void fail(ApiException error) {
            try {
                send(OpenAiError.of(error));
                this.emitter.complete();
            } catch (IOException clientGone) {
                // Nobody is listening; the container completes the request.
            }
        }

        private void send(Object payload) throws IOException {
            try {
                this.emitter.send(SseEmitter.event().data(payload, JSON));
            } catch (IllegalStateException alreadyCompleted) {
                // The request timed out or failed on a container thread.
                throw new IOException("stream already completed", alreadyCompleted);
            }
        }
    }
}
