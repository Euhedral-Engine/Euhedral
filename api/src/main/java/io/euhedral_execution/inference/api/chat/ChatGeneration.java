package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.api.openai.OpenAiException;
import io.euhedral_execution.inference.api.openai.ToolCall;
import io.euhedral_execution.inference.api.openai.Usage;
import java.io.IOException;
import java.util.UUID;
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
/// `finished`, so the next request starts while this one's last writes drain. `abandon` is the only cross-thread entry
/// point: the container calls it on disconnect or timeout,
/// and it cancels the session so no further quantum starts.
final class ChatGeneration implements ToolCallParser.Output {
    private static final Logger LOG = LoggerFactory.getLogger(ChatGeneration.class);

    private final ChatCompletionPlan plan;
    private final InferenceBackend.Generation generation;
    private final CompletionSink sink;
    private final SerialTasks delivery;
    private final Runnable finished;
    private final StopSequenceFilter stopFilter;
    // Null when the request offers no callable tools: output is then plain text, byte for byte.
    private final JsonToolCallParser toolParser;
    // Set by container threads and failed writes; read everywhere.
    private final AtomicBoolean abandoned = new AtomicBoolean();
    private final AtomicBoolean shutdown = new AtomicBoolean();
    private final CompletableFuture<Void> responded = new CompletableFuture<>();
    // Confined to the generation's callbacks, which the generation orders one after another.
    private boolean delivered;
    private ToolCallParser.MalformedToolCallException malformedToolCall;
    // Set when the single call allowed by parallel_tool_calls=false is complete.
    private boolean callLimitReached;

    ChatGeneration(
            ChatCompletionPlan plan,
            InferenceBackend.Generation generation,
            CompletionSink sink,
            SerialTasks delivery,
            Runnable finished) {
        this.plan = plan;
        this.generation = generation;
        this.sink = sink;
        this.delivery = delivery;
        this.finished = finished;
        this.stopFilter = new StopSequenceFilter(plan.stops());
        this.toolParser = plan.tools().parsesOutput() ? new JsonToolCallParser(plan.tools()) : null;
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
        write(this.sink::start);
        CompletableFuture<InferenceBackend.Result> result;
        try {
            result = this.abandoned.get()
                    ? CompletableFuture.completedFuture(null)
                    : this.generation.generate(this.plan.prompt(), this.plan.maxTokens(), this::onText);
        } catch (RuntimeException | Error failure) {
            result = CompletableFuture.failedFuture(failure);
        }
        result.whenComplete(this::complete);
    }

    /// The generation ended: close the session, release the request's slot and queue the response's end.
    private void complete(InferenceBackend.Result result, Throwable failure) {
        try (InferenceBackend.Generation owned = this.generation) {
            if (this.abandoned.get()) return;
            if (failure != null) {
                Throwable cause =
                        failure instanceof java.util.concurrent.CompletionException ? failure.getCause() : failure;
                LOG.error("Chat completion {} failed", this.plan.id(), cause);
                deliverFailure(OpenAiException.serverError());
                return;
            }
            if (this.malformedToolCall != null) throw this.malformedToolCall;
            if (this.shutdown.get()) {
                deliverFailure(OpenAiException.unavailable("Generation was interrupted by server shutdown."));
                return;
            }
            // Nobody in this request cancelled, so the engine did: it is shutting down. Sampled before the
            // flush below, which may itself match a stop sequence and cancel.
            boolean endedByRequest = this.stopFilter.matched() || this.callLimitReached;
            boolean engineCancelled = !endedByRequest && owned.isCancelled();
            if (!this.callLimitReached && !engineCancelled) {
                String remaining = this.stopFilter.finish();
                if (this.toolParser == null) content(remaining);
                else {
                    this.toolParser.accept(remaining, this);
                    this.toolParser.finish(result.stopTokenReached() && !this.stopFilter.matched(), this);
                }
            }
            if (engineCancelled) {
                deliverFailure(OpenAiException.unavailable("Generation was interrupted by engine shutdown."));
                return;
            }
            String finishReason = finishReason(result);
            Usage usage = Usage.of(this.plan.promptTokens(), result.completionTokens());
            this.delivered = true;
            write(() -> this.sink.finish(finishReason, usage));
        } catch (ToolCallParser.MalformedToolCallException malformed) {
            LOG.warn("Chat completion {} failed: {}", this.plan.id(), malformed.getMessage());
            deliverFailure(OpenAiException.invalidToolCall(malformed.getMessage()));
        } catch (RuntimeException | Error closeFailure) {
            if (!this.abandoned.get()) LOG.error("Chat completion {} failed", this.plan.id(), closeFailure);
            deliverFailure(OpenAiException.serverError());
        } finally {
            this.delivery.execute(() -> this.responded.complete(null));
            this.finished.run();
        }
    }

    private String finishReason(InferenceBackend.Result result) {
        if (this.stopFilter.matched()) return "stop";
        if (this.callLimitReached) return "tool_calls";
        if (!result.stopTokenReached()) return "length";
        return this.toolParser != null && this.toolParser.calls() > 0 ? "tool_calls" : "stop";
    }

    /// One decoded text, on the worker that retired its quantum and before the next quantum is admitted.
    private void onText(String text) {
        if (this.abandoned.get() || this.malformedToolCall != null || this.stopFilter.matched()) return;
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
        if (!text.isEmpty()) write(() -> this.sink.text(text));
    }

    /// A complete call after raw output has passed stop-sequence filtering.
    @Override
    public void toolCall(String name, String arguments) {
        String id = "call_" + UUID.randomUUID().toString().replace("-", "");
        int index = this.toolParser.calls() - 1;
        write(() -> this.sink.toolCall(index, ToolCall.function(id, name, arguments)));
        if (!this.plan.tools().parallel()) {
            this.callLimitReached = true;
            this.generation.cancel();
        }
    }

    private void deliverFailure(OpenAiException error) {
        if (this.abandoned.get() || this.delivered) return;
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
