package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.engine.ApiProperties;
import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.api.openai.ChatCompletionChunk;
import io.euhedral_execution.inference.api.openai.ChatCompletionResponse;
import io.euhedral_execution.inference.api.openai.OpenAiException;
import io.euhedral_execution.inference.api.openai.ToolCall;
import io.euhedral_execution.inference.api.openai.Usage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.context.request.async.DeferredResult;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/// Runs planned completions on a bounded generation executor so servlet threads return immediately.
///
/// Both response modes use Spring MVC async handling: JSON through `DeferredResult`, streaming through
/// `SseEmitter`. The blocking core API runs on this service's own threads. Because this service depends on
/// the backend, Spring destroys it before the engine: queued work is discarded and running work interrupted
/// before the engine closes.
@Service
public class ChatCompletionService implements DisposableBean {
    private static final Logger LOG = LoggerFactory.getLogger(ChatCompletionService.class);
    private static final MediaType JSON = MediaType.APPLICATION_JSON;

    private final InferenceBackend backend;
    private final long requestTimeoutMillis;
    private final ThreadPoolExecutor executor;

    public ChatCompletionService(InferenceBackend backend, ApiProperties properties) {
        this.backend = backend;
        this.requestTimeoutMillis = properties.requestTimeout().toMillis();
        BlockingQueue<Runnable> queue = properties.maxQueuedGenerations() == 0
                ? new SynchronousQueue<>()
                : new ArrayBlockingQueue<>(properties.maxQueuedGenerations());
        var threadIds = new AtomicInteger();
        this.executor = new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, queue, task -> {
            Thread thread = new Thread(task, "euhedral-generation-" + threadIds.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        });
    }

    /// Non-streaming completion. The servlet thread is released while generation runs.
    public DeferredResult<ResponseEntity<ChatCompletionResponse>> complete(ChatCompletionPlan plan) {
        var result = new DeferredResult<ResponseEntity<ChatCompletionResponse>>(this.requestTimeoutMillis);
        var job = new ChatGeneration(plan, open(plan), new JsonSink(plan, result));
        result.onTimeout(() -> {
            job.abandon();
            result.setErrorResult(OpenAiException.timeout());
        });
        result.onError(failure -> job.abandon());
        submit(job);
        return result;
    }

    /// Streaming completion as OpenAI `chat.completion.chunk` server-sent events.
    public SseEmitter stream(ChatCompletionPlan plan) {
        var emitter = new SseEmitter(this.requestTimeoutMillis);
        var sink = new StreamSink(plan, emitter);
        var job = new ChatGeneration(plan, open(plan), sink);
        emitter.onTimeout(() -> {
            job.abandon();
            sink.fail(OpenAiException.timeout());
        });
        emitter.onError(failure -> job.abandon());
        submit(job);
        return emitter;
    }

    private InferenceBackend.Generation open(ChatCompletionPlan plan) {
        if (!plan.tools().parsesOutput()) return this.backend.openGeneration(plan.sampling());
        var constraint = new InferenceBackend.ToolConstraint(
                plan.tools().callable().stream().map(FunctionTool::name).toList(),
                plan.tools().choice() != ToolCalling.Choice.AUTO,
                plan.tools().parallel());
        return this.backend.openGeneration(plan.sampling(), constraint);
    }

    private void submit(ChatGeneration job) {
        try {
            this.executor.execute(job);
        } catch (RejectedExecutionException saturated) {
            OpenAiException busy = OpenAiException.unavailable("The server is at generation capacity; retry later.");
            job.discard(busy);
            throw busy;
        }
    }

    @Override
    public void destroy() throws InterruptedException {
        // Graceful web shutdown has already drained what it could; stop the rest before the engine closes.
        List<Runnable> queued = this.executor.shutdownNow();
        for (Runnable task : queued)
            ((ChatGeneration) task).discard(OpenAiException.unavailable("The server is shutting down."));
        if (!this.executor.awaitTermination(30, TimeUnit.SECONDS))
            LOG.warn("Generation threads did not stop within 30s; engine shutdown will cancel their sessions");
    }

    private static final class JsonSink implements CompletionSink {
        private final ChatCompletionPlan plan;
        private final DeferredResult<ResponseEntity<ChatCompletionResponse>> result;
        private final StringBuilder content = new StringBuilder();
        private final List<ToolCall> toolCalls = new ArrayList<>();

        private JsonSink(ChatCompletionPlan plan, DeferredResult<ResponseEntity<ChatCompletionResponse>> result) {
            this.plan = plan;
            this.result = result;
        }

        @Override
        public void start() {}

        @Override
        public void text(String delta) {
            this.content.append(delta);
        }

        @Override
        public void toolCall(int index, ToolCall call) {
            this.toolCalls.add(call);
        }

        @Override
        public void finish(String finishReason, Usage usage) {
            var response = ChatCompletionResponse.of(
                    this.plan.id(),
                    this.plan.created(),
                    this.plan.model(),
                    this.content.toString(),
                    this.toolCalls,
                    finishReason,
                    usage);
            this.result.setResult(ResponseEntity.ok().contentType(JSON).body(response));
        }

        @Override
        public void fail(OpenAiException error) {
            this.result.setErrorResult(error);
        }
    }

    /// Chunks are written as they are decoded. A failed write means the client is gone.
    private static final class StreamSink implements CompletionSink {
        private final ChatCompletionPlan plan;
        private final SseEmitter emitter;

        private StreamSink(ChatCompletionPlan plan, SseEmitter emitter) {
            this.plan = plan;
            this.emitter = emitter;
        }

        @Override
        public void start() throws IOException {
            send(ChatCompletionChunk.role(this.plan.id(), this.plan.created(), this.plan.model()));
        }

        @Override
        public void text(String delta) throws IOException {
            send(ChatCompletionChunk.content(this.plan.id(), this.plan.created(), this.plan.model(), delta));
        }

        @Override
        public void toolCall(int index, ToolCall call) throws IOException {
            send(ChatCompletionChunk.toolCall(this.plan.id(), this.plan.created(), this.plan.model(), index, call));
        }

        @Override
        public void finish(String finishReason, Usage usage) throws IOException {
            send(ChatCompletionChunk.finish(this.plan.id(), this.plan.created(), this.plan.model(), finishReason));
            if (this.plan.includeUsage())
                send(ChatCompletionChunk.usage(this.plan.id(), this.plan.created(), this.plan.model(), usage));
            sendDone();
        }

        /// Headers are already committed, so errors are reported in-band as an OpenAI error event.
        @Override
        public void fail(OpenAiException error) {
            try {
                send(error.toError());
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

        private void sendDone() throws IOException {
            try {
                this.emitter.send(SseEmitter.event().data("[DONE]", MediaType.TEXT_PLAIN));
                this.emitter.complete();
            } catch (IllegalStateException alreadyCompleted) {
                throw new IOException("stream already completed", alreadyCompleted);
            }
        }
    }
}
