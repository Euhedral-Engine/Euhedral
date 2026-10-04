package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.engine.ApiProperties;
import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.api.metrics.ServerMetrics;
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
import org.springframework.boot.web.server.context.WebServerApplicationContext;
import org.springframework.context.SmartLifecycle;
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
/// was planned and admitted.
///
/// At shutdown the web server first stops accepting connections and waits for the requests in flight, queued ones
/// included, up to `spring.lifecycle.timeout-per-shutdown-phase`. This service stops next, in a lifecycle phase
/// between that wait and the web server's stop, while responses can still be written: requests still waiting are
/// refused with 503 and the running generation is stopped and answered with an error. Spring destroys the engine
/// after that, because this service depends on it.
@Service
public class GenerationService implements SmartLifecycle, DisposableBean {
    /// After the web server's graceful shutdown, before it stops.
    private static final int PHASE = WebServerApplicationContext.GRACEFUL_SHUTDOWN_PHASE - 512;

    private static final Logger LOG = LoggerFactory.getLogger(GenerationService.class);

    private final InferenceBackend backend;
    private final long requestTimeoutMillis;
    private final int maxQueued;
    private final ArrayDeque<Admission> waiting = new ArrayDeque<>();
    private Admission running;
    private boolean closed;
    private boolean started;

    private final ServerMetrics metrics;

    public GenerationService(InferenceBackend backend, ApiProperties properties, ServerMetrics metrics) {
        this.backend = backend;
        this.requestTimeoutMillis = properties.requestTimeout().toMillis();
        this.maxQueued = properties.maxQueuedGenerations();
        this.metrics = metrics;
        metrics.bindQueue(this::activeCount, this::queuedCount);
    }

    private synchronized int activeCount() {
        return this.running == null ? 0 : 1;
    }

    private synchronized int queuedCount() {
        return this.waiting.size();
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
        final ClientLink link;
        final String api;
        final long arrivedNanos;
        volatile boolean abandoned;
        volatile ChatGeneration generation;
        java.util.concurrent.atomic.AtomicBoolean streaming;
        /// Completes once the container has finished the request's response.
        CompletableFuture<Void> responded;

        Admission(
                ConversationPlanner.Planned planned,
                DeferredResult<Object> result,
                ClientLink link,
                String api,
                long arrivedNanos) {
            this.planned = planned;
            this.result = result;
            this.link = link;
            this.api = api;
            this.arrivedNanos = arrivedNanos;
        }

        /// The client is gone: stop the generation, or give up the place in the queue.
        void abandon() {
            this.abandoned = true;
            ChatGeneration job = this.generation;
            if (job != null) job.abandon();
            else
                synchronized (GenerationService.this) {
                    GenerationService.this.waiting.remove(this);
                }
        }
    }

    /// Hands a request of the `api` surface being planned to the workers and returns its deferred response: a body,
    /// a stream, or an [ApiException] for the surface's error handler. The client is watched through `link` while
    /// the request waits and generates; the link is closed when the request completes.
    public DeferredResult<Object> submit(
            String api, CompletableFuture<ConversationPlanner.Planned> planning, ClientLink link) {
        long arrived = System.nanoTime();
        var result = new DeferredResult<Object>(this.requestTimeoutMillis);
        var admission = new AtomicReference<Admission>();
        // A stream's response continues past this result, as its emitter; its link closes with the emitter.
        var streaming = new java.util.concurrent.atomic.AtomicBoolean();
        var responded = new CompletableFuture<Void>();
        result.onCompletion(() -> {
            responded.complete(null);
            if (!streaming.get()) link.close();
        });
        result.onTimeout(() -> {
            Admission admitted = admission.get();
            if (admitted != null) admitted.abandon();
            if (admitted == null || admitted.generation == null)
                this.metrics.request(api, ServerMetrics.Outcome.CANCELLED);
            result.setErrorResult(ApiException.timeout());
        });
        result.onError(failure -> {
            Admission admitted = admission.get();
            if (admitted != null) admitted.abandon();
        });
        planning.whenComplete((planned, failure) -> {
            if (failure != null) {
                ApiException refused = apiException(failure);
                if ("unsupported_schema".equals(refused.code())) this.metrics.schemaRejected(api);
                this.metrics.request(
                        api,
                        refused.status().is4xxClientError()
                                ? ServerMetrics.Outcome.INVALID
                                : refused.status().value() == 503
                                        ? ServerMetrics.Outcome.REJECTED
                                        : ServerMetrics.Outcome.FAILED);
                result.setErrorResult(refused);
                return;
            }
            Admission admitted = new Admission(planned, result, link, api, arrived);
            admitted.streaming = streaming;
            admitted.responded = responded;
            admission.set(admitted);
            admit(admitted);
        });
        return result;
    }

    private void admit(Admission admission) {
        synchronized (this) {
            if (this.closed) {
                this.metrics.request(admission.api, ServerMetrics.Outcome.REJECTED);
                admission.result.setErrorResult(ApiException.unavailable("The server is shutting down."));
                return;
            }
            if (this.running != null) {
                if (this.waiting.size() < this.maxQueued) this.waiting.add(admission);
                else {
                    this.metrics.request(admission.api, ServerMetrics.Outcome.REJECTED);
                    admission.result.setErrorResult(
                            ApiException.unavailable("The server is at generation capacity; retry later."));
                }
                return;
            }
            this.running = admission;
        }
        start(admission);
    }

    /// Runs on a worker with the slot held; releasing the slot starts the next admission.
    private void start(Admission admission) {
        if (admission.abandoned || admission.result.isSetOrExpired()) {
            this.metrics.request(admission.api, ServerMetrics.Outcome.CANCELLED);
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
            ApiException refused = apiException(failure);
            this.metrics.request(
                    admission.api,
                    refused.status().value() == 503 ? ServerMetrics.Outcome.REJECTED : ServerMetrics.Outcome.FAILED);
            admission.result.setErrorResult(refused);
            release(admission);
            return;
        }
        Responder responder = admission.planned.responder();
        CompletionSink sink;
        Runnable clientGone;
        if (plan.stream()) {
            var emitter = new SseEmitter(this.requestTimeoutMillis);
            sink = responder.stream(plan, emitter);
            admission.streaming.set(true);
            emitter.onCompletion(admission.link::close);
            emitter.onTimeout(() -> {
                admission.abandon();
                sink.fail(ApiException.timeout());
            });
            emitter.onError(failure -> {
                admission.abandon();
                admission.link.close();
            });
            clientGone = () -> emitter.completeWithError(new java.io.IOException("the client closed the connection"));
            // The stream's headers: no caching, and no proxy buffering (nginx), so events reach the client as
            // they are produced.
            admission.result.setResult(ResponseEntity.ok()
                    .header("Cache-Control", "no-cache")
                    .header("X-Accel-Buffering", "no")
                    .body(emitter));
        } else {
            sink = responder.json(plan, admission.result);
            clientGone = () -> admission.result.setErrorResult(ApiException.clientClosed());
        }
        var job = new ChatGeneration(
                plan,
                generation,
                sink,
                new SerialTasks(this.backend.workers()),
                () -> release(admission),
                new ChatGeneration.Client(admission.link, clientGone, this::probeWaiting),
                new ChatGeneration.Telemetry(this.metrics, admission.api, admission.arrivedNanos));
        admission.generation = job;
        if (admission.abandoned) job.abandon();
        job.start();
    }

    /// Gives up the places of queued requests whose clients left; runs on the probe tasks of the running generation.
    private void probeWaiting() {
        List<Admission> gone = new java.util.ArrayList<>();
        synchronized (this) {
            for (Admission admission : this.waiting) if (admission.link.gone()) gone.add(admission);
            this.waiting.removeAll(gone);
        }
        for (Admission admission : gone) {
            admission.abandoned = true;
            this.metrics.request(admission.api, ServerMetrics.Outcome.CANCELLED);
            admission.result.setErrorResult(ApiException.clientClosed());
        }
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

    /// Also reopens a service stopped by a context that is started again.
    @Override
    public synchronized void start() {
        this.started = true;
        this.closed = false;
    }

    /// Pausing a context (Spring stops its lifecycle beans while it is idle) must not refuse requests.
    @Override
    public boolean isPauseable() {
        return false;
    }

    @Override
    public synchronized boolean isRunning() {
        return this.started && !this.closed;
    }

    @Override
    public int getPhase() {
        return PHASE;
    }

    /// Graceful web shutdown has drained what it could; refuse what waits and stop what runs.
    @Override
    public void stop() {
        List<Admission> refused;
        Admission active;
        synchronized (this) {
            if (this.closed) return;
            this.closed = true;
            refused = List.copyOf(this.waiting);
            this.waiting.clear();
            active = this.running;
        }
        for (Admission admission : refused) {
            this.metrics.request(admission.api, ServerMetrics.Outcome.REJECTED);
            try {
                admission.result.setErrorResult(ApiException.unavailable("The server is shutting down."));
            } catch (RuntimeException completed) {
                // The container already ended the request.
            }
        }
        // Setting a result only schedules its dispatch; the web server stops after this returns, and would close the
        // connections before the refusals are written.
        try {
            CompletableFuture.allOf(refused.stream()
                            .map(admission -> admission.responded)
                            .toArray(CompletableFuture[]::new))
                    .get(10, TimeUnit.SECONDS);
        } catch (TimeoutException | ExecutionException late) {
            LOG.warn("Refusals of queued requests were not written within 10s");
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            return;
        }
        ChatGeneration job = active == null ? null : active.generation;
        LOG.info(
                "Shutting down: {} waiting requests refused, {}",
                refused.size(),
                job == null ? "no generation running" : "stopping the running generation");
        if (job == null) return;
        job.shutDown();
        try {
            job.responded().get(30, TimeUnit.SECONDS);
        } catch (TimeoutException | ExecutionException late) {
            LOG.warn("The running generation did not stop within 30s; engine shutdown will cancel its session");
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    /// For a context that closes without having started its lifecycle.
    @Override
    public void destroy() {
        stop();
    }
}
