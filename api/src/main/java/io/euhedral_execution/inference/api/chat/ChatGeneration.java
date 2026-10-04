package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.api.metrics.ServerMetrics;
import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// One request's generation, driven entirely by the backend's workers.
///
/// [#start] (on a worker) asks the generation to run and returns. Each decoded text reaches [#onText] on the
/// worker that retired its quantum, before the next quantum is admitted: stop-sequence matching and tool-call
/// parsing run there, so a stop or an invalid call cancels the generation before another quantum starts. The
/// sink's network writes are tasks on the request's [SerialTasks] instead, so they run on workers one at a
/// time in output order and a write that blocks its worker never holds up the quanta. When the generation
/// ends, the completion closes the session, queues the response's last write and reports the end to
/// `finished`, so the next request starts while this one's last writes drain. Output that opens in the model's
/// think block is split at `</think>` first: reasoning goes to the sink as it is decoded, and only the answer
/// meets stop sequences and tool-call parsing. When the generation
/// `abandon` is the only cross-thread entry point: the container calls it on disconnect or timeout,
/// and it cancels the session so no further quantum starts.
final class ChatGeneration implements ToolCallParser.Output, ReasoningSplitter.Output {
    private static final Logger LOG = LoggerFactory.getLogger(ChatGeneration.class);
    private static final tools.jackson.databind.json.JsonMapper JSON = tools.jackson.databind.json.JsonMapper.builder()
            .enable(tools.jackson.core.StreamReadFeature.STRICT_DUPLICATE_DETECTION)
            .enable(tools.jackson.databind.DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
            .build();

    private final GenerationPlan plan;
    private final InferenceBackend.Generation generation;
    private final CompletionSink sink;
    private final SerialTasks delivery;
    private final Runnable finished;
    private final Client client;
    private final Telemetry telemetry;
    private long startedNanos;
    // Set by the first decoded text; read when the generation ends.
    private volatile long firstTokenNanos;
    // A probe of the client's connection is queued and has not run yet.
    private final AtomicBoolean probing = new AtomicBoolean();
    private final StopSequenceFilter stopFilter;
    // Null when the output does not open in the think block.
    private final ReasoningSplitter reasoning;
    // Null when the request offers no callable tools: output is then plain text, byte for byte.
    private final JsonToolCallParser toolParser;
    // The answer of a JSON response format without tools, checked once whole.
    private final StringBuilder jsonAnswer;
    // Set by container threads and failed writes; read everywhere.
    private final AtomicBoolean abandoned = new AtomicBoolean();
    private final AtomicBoolean shutdown = new AtomicBoolean();
    private final CompletableFuture<Void> responded = new CompletableFuture<>();
    // Confined to the generation's callbacks, which the generation orders one after another.
    private boolean delivered;
    private boolean sinkStarted;
    private int prefillQuanta;
    private ToolCallParser.MalformedToolCallException malformedToolCall;
    // Set when the single call allowed by parallel_tool_calls=false is complete.
    private boolean callLimitReached;

    ChatGeneration(
            GenerationPlan plan,
            InferenceBackend.Generation generation,
            CompletionSink sink,
            SerialTasks delivery,
            Runnable finished,
            Client client,
            Telemetry telemetry) {
        this.client = client;
        this.telemetry = telemetry;
        this.plan = plan;
        this.generation = generation;
        this.sink = sink;
        this.delivery = delivery;
        this.finished = finished;
        this.stopFilter = new StopSequenceFilter(plan.stops());
        this.reasoning = plan.reasoning() ? new ReasoningSplitter() : null;
        this.toolParser = plan.tools().parsesOutput()
                ? new JsonToolCallParser(plan.tools(), plan.format().json())
                : null;
        this.jsonAnswer = plan.format().json() && this.toolParser == null ? new StringBuilder() : null;
    }

    /// The client's connection, probed as the generation progresses: `gone` completes the request when the probe
    /// finds it closed, and `others` probes the queued requests' clients on the same task.
    record Client(ClientLink link, Runnable gone, Runnable others) {
        static final Client NONE = new Client(ClientLink.NONE, () -> {}, () -> {});
    }

    /// Where the generation reports how it ended: its surface `api` and when the request arrived. Null records
    /// nothing.
    record Telemetry(ServerMetrics metrics, String api, long arrivedNanos) {}

    ChatGeneration(
            GenerationPlan plan,
            InferenceBackend.Generation generation,
            CompletionSink sink,
            SerialTasks delivery,
            Runnable finished) {
        this(plan, generation, sink, delivery, finished, Client.NONE, null);
    }

    /// The client disconnected or timed out: stop generating. Safe from any thread and after completion.
    void abandon() {
        if (this.abandoned.compareAndSet(false, true)) this.generation.cancel();
    }

    /// The server is shutting down: stop generating and report it to the client.
    void shutDown() {
        this.shutdown.set(true);
        this.generation.cancel();
    }

    boolean isAbandoned() {
        return this.abandoned.get();
    }

    /// Completes after the generation ended and the response's last write ran.
    CompletableFuture<Void> responded() {
        return this.responded;
    }

    /// Starts the generation on the workers and returns.
    void start() {
        this.startedNanos = System.nanoTime();
        CompletableFuture<InferenceBackend.Result> result;
        try {
            result = this.abandoned.get()
                    ? CompletableFuture.completedFuture(null)
                    : this.generation.generate(
                            this.plan.prompt(), this.plan.maxTokens(), this::onText, this::prefilled);
        } catch (RuntimeException | Error failure) {
            result = CompletableFuture.failedFuture(failure);
        }
        result.whenComplete(this::complete);
    }

    /// The generation ended: close the session, release the request's slot and queue the response's end.
    private void complete(InferenceBackend.Result result, Throwable failure) {
        try (InferenceBackend.Generation owned = this.generation) {
            if (this.abandoned.get()) {
                report(ServerMetrics.Outcome.CANCELLED, null);
                this.delivered = true;
                return;
            }
            if (failure != null) {
                Throwable cause =
                        failure instanceof java.util.concurrent.CompletionException ? failure.getCause() : failure;
                LOG.error("Generation {} failed", this.plan.id(), cause);
                deliverFailure(ApiException.serverError());
                return;
            }
            if (this.malformedToolCall != null) throw this.malformedToolCall;
            if (this.shutdown.get()) {
                deliverFailure(ApiException.unavailable("Generation was interrupted by server shutdown."));
                return;
            }
            // Nobody in this request cancelled, so the engine did: it is shutting down. Sampled before the
            // flush below, which may itself match a stop sequence and cancel.
            boolean endedByRequest = this.stopFilter.matched() || this.callLimitReached;
            boolean engineCancelled = !endedByRequest && owned.isCancelled();
            if (this.reasoning != null && !engineCancelled) this.reasoning.finish(this);
            if (!this.callLimitReached && !engineCancelled) {
                String remaining = this.stopFilter.finish();
                if (this.toolParser == null) content(remaining);
                else {
                    this.toolParser.accept(remaining, this);
                    this.toolParser.finish(result.stopTokenReached() && !this.stopFilter.matched(), this);
                }
            }
            if (engineCancelled) {
                deliverFailure(ApiException.unavailable("Generation was interrupted by engine shutdown."));
                return;
            }
            // The grammar admits only valid documents; a finished one that does not parse is never returned.
            if (this.jsonAnswer != null && result.stopTokenReached() && !isJson(this.jsonAnswer.toString())) {
                LOG.error("Generation {} produced invalid JSON under its response format", this.plan.id());
                deliverFailure(ApiException.invalidStructuredOutput());
                return;
            }
            var usage = new TokenUsage(
                    this.plan.promptTokens(),
                    result.cachedPromptTokens(),
                    result.completionTokens(),
                    this.plan.reasoning() ? result.reasoningTokens() : null);
            Finish.Reason reason = finishReason(result);
            var finish = new Finish(
                    reason, reason == Finish.Reason.STOP_SEQUENCE ? this.stopFilter.matchedStop() : null, usage);
            this.delivered = true;
            startSink();
            write(() -> this.sink.finish(finish));
            report(ServerMetrics.Outcome.SUCCESS, null);
            if (this.telemetry != null) {
                this.telemetry
                        .metrics()
                        .generation(
                                this.telemetry.api(),
                                finish,
                                this.telemetry.arrivedNanos(),
                                this.startedNanos,
                                this.firstTokenNanos,
                                System.nanoTime());
                if (this.toolParser != null)
                    this.telemetry.metrics().toolCalls(this.telemetry.api(), this.toolParser.calls());
            }
        } catch (ToolCallParser.MalformedToolCallException malformed) {
            LOG.warn("Generation {} failed: {}", this.plan.id(), malformed.getMessage());
            deliverFailure(ApiException.invalidToolCall(malformed.getMessage()));
        } catch (RuntimeException | Error closeFailure) {
            if (!this.abandoned.get()) LOG.error("Generation {} failed", this.plan.id(), closeFailure);
            deliverFailure(ApiException.serverError());
        } finally {
            this.delivery.execute(() -> this.responded.complete(null));
            this.finished.run();
        }
    }

    private Finish.Reason finishReason(InferenceBackend.Result result) {
        if (this.stopFilter.matched()) return Finish.Reason.STOP_SEQUENCE;
        if (this.callLimitReached) return Finish.Reason.TOOL_CALLS;
        if (!result.stopTokenReached()) return Finish.Reason.LENGTH;
        return this.toolParser != null && this.toolParser.calls() > 0 ? Finish.Reason.TOOL_CALLS : Finish.Reason.END;
    }

    /// One decoded text, on the worker that retired its quantum and before the next quantum is admitted.
    private void onText(String text) {
        if (this.firstTokenNanos == 0) this.firstTokenNanos = System.nanoTime();
        probe();
        if (this.abandoned.get() || this.malformedToolCall != null || this.stopFilter.matched()) return;
        startSink();
        if (this.reasoning == null) answer(text);
        else this.reasoning.accept(text, this);
    }

    /// Reasoning text, which stop sequences and tool-call parsing do not see.
    @Override
    public void reasoning(String text) {
        write(() -> this.sink.reasoning(text));
    }

    /// Answer text: everything without a think block, else what follows `</think>`.
    @Override
    public void answer(String text) {
        if (this.malformedToolCall != null || this.stopFilter.matched()) return;
        try {
            // Match stops on raw output before parsing a JSON tool-call envelope.
            String visible = this.stopFilter.accept(text);
            if (this.stopFilter.matched()) this.generation.cancel();
            if (this.toolParser == null) content(visible);
            else this.toolParser.accept(visible, this);
        } catch (ToolCallParser.MalformedToolCallException malformed) {
            // Fail closed: nothing after an invalid call can be returned, so stop decoding.
            this.malformedToolCall = malformed;
            this.generation.cancel();
        }
    }

    /// Assistant text after raw output has passed stop-sequence filtering.
    @Override
    public void content(String text) {
        if (text.isEmpty()) return;
        if (this.jsonAnswer != null) this.jsonAnswer.append(text);
        write(() -> this.sink.text(text));
    }

    private static boolean isJson(String text) {
        try {
            JSON.readTree(text);
            return true;
        } catch (tools.jackson.core.JacksonException invalid) {
            return false;
        }
    }

    /// A complete call after raw output has passed stop-sequence filtering.
    @Override
    public void toolCall(String name, String arguments) {
        int index = this.toolParser.calls() - 1;
        var call = new GeneratedCall(name, arguments);
        write(() -> this.sink.toolCall(index, call));
        if (!this.plan.tools().parallel()) {
            this.callLimitReached = true;
            this.generation.cancel();
        }
    }

    /// After each prefill quantum: a stream writes a keep-alive, which fails once its client left; the connection is
    /// probed as well. The first quantum writes none: a prompt of one quantum produces its first token next, which
    /// writes anyway.
    private void prefilled() {
        if (this.plan.stream() && this.prefillQuanta++ > 0 && !this.sinkStarted && !this.abandoned.get())
            write(this.sink::keepAlive);
        probe();
    }

    /// Queues one probe of the client's connection behind the response's writes, unless one is already queued. A
    /// client found gone abandons the generation, which stops before its next quantum, and completes the request.
    /// Runs after every prefill quantum and every decoded text: a client that leaves during a long prompt is
    /// noticed within a chunk, without a thread to watch it.
    private void probe() {
        if (this.client.link() == ClientLink.NONE || !this.probing.compareAndSet(false, true)) return;
        this.delivery.execute(() -> {
            this.probing.set(false);
            // A stream is watched by its writes; its committed connection reads nothing.
            if (!this.plan.stream()
                    && !this.abandoned.get()
                    && this.client.link().gone()) {
                abandon();
                this.client.gone().run();
            }
            this.client.others().run();
        });
    }

    /// Starts the response at the first token, when the prefix cache's restore is known.
    private void startSink() {
        if (this.sinkStarted) return;
        this.sinkStarted = true;
        int cached = this.generation.cachedPromptTokens();
        write(() -> this.sink.start(cached));
    }

    private void report(ServerMetrics.Outcome outcome, String constraintFailure) {
        if (this.telemetry == null) return;
        this.telemetry.metrics().request(this.telemetry.api(), outcome);
        if (constraintFailure != null) this.telemetry.metrics().constrainedFailure(constraintFailure);
    }

    private void deliverFailure(ApiException error) {
        if (this.abandoned.get()) {
            if (!this.delivered) report(ServerMetrics.Outcome.CANCELLED, null);
            this.delivered = true;
            return;
        }
        if (this.delivered) return;
        String code = error.code();
        report(
                error.status().value() == 503 ? ServerMetrics.Outcome.REJECTED : ServerMetrics.Outcome.FAILED,
                "invalid_tool_call".equals(code) || "invalid_structured_output".equals(code) ? code : null);
        this.delivered = true;
        this.delivery.execute(() -> {
            if (!this.abandoned.get()) this.sink.fail(error);
        });
    }

    /// Queues one network write behind the request's earlier writes; a client gone abandons the request.
    private void write(Write write) {
        this.delivery.execute(() -> {
            if (this.abandoned.get()) return;
            try {
                write.run();
            } catch (IOException clientGone) {
                abandon();
            }
        });
    }

    @FunctionalInterface
    private interface Write {
        void run() throws IOException;
    }
}
