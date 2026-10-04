package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.engine.ApiProperties;
import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.api.engine.InferenceUnavailableException;
import io.euhedral_execution.inference.api.openai.ChatCompletionChunk;
import io.euhedral_execution.inference.api.openai.ChatCompletionRequest;
import io.euhedral_execution.inference.api.openai.ChatCompletionResponse;
import io.euhedral_execution.inference.api.openai.OpenAiException;
import io.euhedral_execution.inference.api.openai.ToolCall;
import io.euhedral_execution.inference.api.openai.Usage;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.context.request.async.DeferredResult;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/// Runs chat completions entirely on the backend's workers; container threads only hand requests over.
///
/// A request becomes a `DeferredResult` at once. On the workers it is planned (rendered, encoded, validated),
/// admitted, and generated; its result is a JSON body or, for `stream=true`, an `SseEmitter` whose chunks the
/// workers write. One generation runs at a time and at most `maxQueuedGenerations` wait; a finished
/// generation starts the next on a worker. Validation and capacity failures reach the client as HTTP errors,
/// because a stream's emitter becomes the result only once its request was planned and admitted. Because this
/// service depends on the backend, Spring destroys it before the engine: waiting requests are refused and the
/// running generation is stopped before the engine closes.
@Service
public class ChatCompletionService implements DisposableBean {
    private static final Logger LOG = LoggerFactory.getLogger(ChatCompletionService.class);
    private static final MediaType JSON = MediaType.APPLICATION_JSON;

    private final InferenceBackend backend;
    private final ChatRequestMapper requestMapper;
    private final long requestTimeoutMillis;
    private final int maxQueued;
    private final ArrayDeque<Admission> waiting = new ArrayDeque<>();
    private Admission running;
    private boolean closed;

    public ChatCompletionService(InferenceBackend backend, ChatRequestMapper requestMapper, ApiProperties properties) {
        this.backend = backend;
        this.requestMapper = requestMapper;
        this.requestTimeoutMillis = properties.requestTimeout().toMillis();
        this.maxQueued = properties.maxQueuedGenerations();
    }

    /// One planned request waiting for, or holding, the generation slot.
    private final class Admission {
        final ChatCompletionPlan plan;
        final DeferredResult<Object> result;
        volatile boolean abandoned;
        volatile ChatGeneration generation;

        Admission(ChatCompletionPlan plan, DeferredResult<Object> result) {
            this.plan = plan;
            this.result = result;
        }

        void abandon() {
            this.abandoned = true;
            ChatGeneration job = this.generation;
            if (job != null) job.abandon();
        }
    }

    /// Hands the request to the workers and returns its deferred response: a JSON body, an SSE stream, or an
    /// OpenAI error.
    public DeferredResult<Object> complete(ChatCompletionRequest request) {
        var result = new DeferredResult<Object>(this.requestTimeoutMillis);
        var admission = new AtomicReference<Admission>();
        result.onTimeout(() -> {
            Admission admitted = admission.get();
            if (admitted != null) admitted.abandon();
            result.setErrorResult(OpenAiException.timeout());
        });
        result.onError(failure -> {
            Admission admitted = admission.get();
            if (admitted != null) admitted.abandon();
        });
        this.requestMapper.planAsync(request).whenComplete((plan, failure) -> {
            if (failure != null) {
                result.setErrorResult(openAi(failure));
                return;
            }
            Admission admitted = new Admission(plan, result);
            admission.set(admitted);
            admit(admitted);
        });
        return result;
    }

    private void admit(Admission admission) {
        synchronized (this) {
            if (this.closed) {
                admission.result.setErrorResult(OpenAiException.unavailable("The server is shutting down."));
                return;
            }
            if (this.running != null) {
                if (this.waiting.size() < this.maxQueued) this.waiting.add(admission);
                else
                    admission.result.setErrorResult(
                            OpenAiException.unavailable("The server is at generation capacity; retry later."));
                return;
            }
            this.running = admission;
        }
        start(admission);
    }

    /// Runs on a worker with the slot held; releasing the slot starts the next admission.
    private void start(Admission admission) {
        if (admission.abandoned || admission.result.isSetOrExpired()) {
            release(admission);
            return;
        }
        InferenceBackend.Generation generation;
        try {
            generation = open(admission.plan);
        } catch (RuntimeException | Error failure) {
            admission.result.setErrorResult(openAi(failure));
            release(admission);
            return;
        }
        ChatCompletionPlan plan = admission.plan;
        CompletionSink sink;
        if (plan.stream()) {
            var emitter = new SseEmitter(this.requestTimeoutMillis);
            sink = new StreamSink(plan, emitter);
            emitter.onTimeout(() -> {
                admission.abandon();
                sink.fail(OpenAiException.timeout());
            });
            emitter.onError(failure -> admission.abandon());
            // The stream's headers: no caching, and no proxy buffering (nginx), so chunks reach the client as
            // they are produced.
            admission.result.setResult(ResponseEntity.ok()
                    .header("Cache-Control", "no-cache")
                    .header("X-Accel-Buffering", "no")
                    .body(emitter));
        } else sink = new JsonSink(plan, admission.result);
        var job = new ChatGeneration(
                plan, generation, sink, new SerialTasks(this.backend.workers()), () -> release(admission));
        admission.generation = job;
        if (admission.abandoned) job.abandon();
        job.start();
    }

    private void release(Admission admission) {
        Admission next;
        synchronized (this) {
            if (this.running != admission) return;
            next = this.waiting.poll();
            this.running = next;
        }
        if (next != null) this.backend.workers().execute(() -> start(next));
    }

    private InferenceBackend.Generation open(ChatCompletionPlan plan) {
        InferenceBackend.ToolConstraint tools = plan.tools().parsesOutput()
                ? new InferenceBackend.ToolConstraint(
                        plan.tools().callable().stream().map(FunctionTool::name).toList(),
                        plan.tools().choice() != ToolCalling.Choice.AUTO,
                        plan.tools().parallel())
                : null;
        return this.backend.openGeneration(plan.sampling(), new InferenceBackend.OutputSpec(plan.reasoning(), tools));
    }

    /// The OpenAI error for a failed planning or start.
    private OpenAiException openAi(Throwable failure) {
        Throwable cause =
                failure instanceof CompletionException && failure.getCause() != null ? failure.getCause() : failure;
        if (cause instanceof OpenAiException openAi) return openAi;
        if (cause instanceof InferenceUnavailableException unavailable)
            return OpenAiException.unavailable("The inference engine is unavailable: " + unavailable.getMessage());
        LOG.error("Chat completion request failed", cause);
        return OpenAiException.serverError();
    }

    @Override
    public void destroy() throws InterruptedException {
        // Graceful web shutdown has already drained what it could; stop the rest before the engine closes.
        List<Admission> refused;
        Admission active;
        synchronized (this) {
            this.closed = true;
            refused = List.copyOf(this.waiting);
            this.waiting.clear();
            active = this.running;
        }
        for (Admission admission : refused)
            admission.result.setErrorResult(OpenAiException.unavailable("The server is shutting down."));
        ChatGeneration job = active == null ? null : active.generation;
        if (job == null) return;
        job.shutDown();
        try {
            job.responded().get(30, TimeUnit.SECONDS);
        } catch (TimeoutException | ExecutionException late) {
            LOG.warn("The running generation did not stop within 30s; engine shutdown will cancel its session");
        }
    }

    private static final class JsonSink implements CompletionSink {
        private final ChatCompletionPlan plan;
        private final DeferredResult<Object> result;
        private final StringBuilder reasoning = new StringBuilder();
        private final StringBuilder content = new StringBuilder();
        private final List<ToolCall> toolCalls = new ArrayList<>();

        private JsonSink(ChatCompletionPlan plan, DeferredResult<Object> result) {
            this.plan = plan;
            this.result = result;
        }

        @Override
        public void start() {}

        @Override
        public void reasoning(String delta) {
            this.reasoning.append(delta);
        }

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
                    this.reasoning.toString(),
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
        public void reasoning(String delta) throws IOException {
            send(ChatCompletionChunk.reasoning(this.plan.id(), this.plan.created(), this.plan.model(), delta));
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
