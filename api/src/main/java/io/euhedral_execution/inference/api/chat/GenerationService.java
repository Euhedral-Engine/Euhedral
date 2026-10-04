package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.engine.ApiProperties;
import io.euhedral_execution.inference.api.engine.InferenceBackend;
import java.util.ArrayDeque;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.context.request.async.DeferredResult;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/// Runs generations entirely on the backend's workers for every API surface; container threads only hand
/// requests over.
///
/// A request becomes a `DeferredResult` at once. On the workers it is planned (mapped, rendered, encoded,
/// validated), admitted, and generated; its result is a JSON body or, for a stream, an `SseEmitter` whose events
/// the workers write, both in the format of the surface's [Responder]. One generation runs at a time and at most
/// `maxQueuedGenerations` wait; a finished generation starts the next on a worker. Validation and capacity
/// failures reach the client as HTTP errors, because a stream's emitter becomes the result only once its request
/// was planned and admitted. Because this service depends on the backend, Spring destroys it before the engine:
/// waiting requests are refused and the running generation is stopped before the engine closes.
@Service
public class GenerationService implements DisposableBean {
    private static final Logger LOG = LoggerFactory.getLogger(GenerationService.class);

    private final InferenceBackend backend;
    private final long requestTimeoutMillis;
    private final int maxQueued;
    private final ArrayDeque<Admission> waiting = new ArrayDeque<>();
    private Admission running;
    private boolean closed;

    public GenerationService(InferenceBackend backend, ApiProperties properties) {
        this.backend = backend;
        this.requestTimeoutMillis = properties.requestTimeout().toMillis();
        this.maxQueued = properties.maxQueuedGenerations();
    }

    /// Writes one surface's responses: a JSON body or the events of a stream.
    public interface Responder {
        CompletionSink json(GenerationPlan plan, DeferredResult<Object> result);

        CompletionSink stream(GenerationPlan plan, SseEmitter emitter);
    }

    /// One planned request waiting for, or holding, the generation slot.
    private final class Admission {
        final ConversationPlanner.Planned planned;
        final DeferredResult<Object> result;
        volatile boolean abandoned;
        volatile ChatGeneration generation;

        Admission(ConversationPlanner.Planned planned, DeferredResult<Object> result) {
            this.planned = planned;
            this.result = result;
        }

        void abandon() {
            this.abandoned = true;
            ChatGeneration job = this.generation;
            if (job != null) job.abandon();
        }
    }

    /// Hands a request being planned to the workers and returns its deferred response: a body, a stream, or an
    /// [ApiException] for the surface's error handler.
    public DeferredResult<Object> submit(CompletableFuture<ConversationPlanner.Planned> planning) {
        var result = new DeferredResult<Object>(this.requestTimeoutMillis);
        var admission = new AtomicReference<Admission>();
        result.onTimeout(() -> {
            Admission admitted = admission.get();
            if (admitted != null) admitted.abandon();
            result.setErrorResult(ApiException.timeout());
        });
        result.onError(failure -> {
            Admission admitted = admission.get();
            if (admitted != null) admitted.abandon();
        });
        planning.whenComplete((planned, failure) -> {
            if (failure != null) {
                result.setErrorResult(apiException(failure));
                return;
            }
            Admission admitted = new Admission(planned, result);
            admission.set(admitted);
            admit(admitted);
        });
        return result;
    }

    private void admit(Admission admission) {
        synchronized (this) {
            if (this.closed) {
                admission.result.setErrorResult(ApiException.unavailable("The server is shutting down."));
                return;
            }
            if (this.running != null) {
                if (this.waiting.size() < this.maxQueued) this.waiting.add(admission);
                else
                    admission.result.setErrorResult(
                            ApiException.unavailable("The server is at generation capacity; retry later."));
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
        GenerationPlan plan = admission.planned.plan();
        InferenceBackend.Generation generation;
        try {
            generation = this.backend.openGeneration(
                    plan.sampling(),
                    new InferenceBackend.OutputSpec(plan.reasoning(), plan.reasoningBudget(), plan.grammar()));
        } catch (RuntimeException | Error failure) {
            admission.result.setErrorResult(apiException(failure));
            release(admission);
            return;
        }
        Responder responder = admission.planned.responder();
        CompletionSink sink;
        if (plan.stream()) {
            var emitter = new SseEmitter(this.requestTimeoutMillis);
            sink = responder.stream(plan, emitter);
            emitter.onTimeout(() -> {
                admission.abandon();
                sink.fail(ApiException.timeout());
            });
            emitter.onError(failure -> admission.abandon());
            // The stream's headers: no caching, and no proxy buffering (nginx), so events reach the client as
            // they are produced.
            admission.result.setResult(ResponseEntity.ok()
                    .header("Cache-Control", "no-cache")
                    .header("X-Accel-Buffering", "no")
                    .body(emitter));
        } else sink = responder.json(plan, admission.result);
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

    private static ApiException apiException(Throwable failure) {
        return ApiException.from(failure);
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
            admission.result.setErrorResult(ApiException.unavailable("The server is shutting down."));
        ChatGeneration job = active == null ? null : active.generation;
        if (job == null) return;
        job.shutDown();
        try {
            job.responded().get(30, TimeUnit.SECONDS);
        } catch (TimeoutException | ExecutionException late) {
            LOG.warn("The running generation did not stop within 30s; engine shutdown will cancel its session");
        }
    }
}
