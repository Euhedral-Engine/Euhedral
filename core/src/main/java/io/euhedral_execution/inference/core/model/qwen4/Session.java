package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.generation.Generation;
import io.euhedral_execution.inference.core.generation.GenerationFrames;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.GenerationTimingListener;
import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsSampler;
import io.euhedral_execution.inference.core.generation.StepPort;
import io.euhedral_execution.inference.core.generation.TokenRecord;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.HostTasks;
import io.euhedral_execution.inference.core.runtime.PromptSink;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import io.euhedral_execution.inference.core.runtime.graph.FrameLake;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.tokenizer.IncrementalDecoder;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.core.tokenizer.TokenConstraint;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/// One persistent Flash-Next sequence being generated from: prefill chunks, then one decode step per token, with the
/// engine's sampler, constraint and text decoding around the model. Every token is the target model's own.
///
/// A generation is a chain of frames (docs/FRAME_MODEL.md): each step's quantum throws a Select frame when it
/// retires, which samples, emits text and throws the next step's Admit. While a step waits for the device or an
/// expert the generation occupies no thread, and generations of different sessions interleave step by step. The text
/// callback runs on whichever worker runs a Select, one call at a time and in order. The session hands token ids to
/// the plan and takes a logits row back; it never knows the model is sparse.
public final class Session implements GenerationSession {

    private final QwenTokenizer tokenizer;
    private final ExecutionPlan plan;
    private final HostTasks host;
    private final GenerationFrames frames;
    private final int prefillChunkTokens;
    private final LogitsSampler sampler;
    private final HostLogits hostLogits;
    private final ExecutionPlan.LogitsSink sink;
    /// Every token the session generated, across its prompts.
    private final TokenRecord generated = new TokenRecord();

    private final AtomicBoolean generationActive = new AtomicBoolean();
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicBoolean closed = new AtomicBoolean();
    private volatile Call active;
    private volatile ExecutionPlan.Handle activeStep;
    private Consumer<? super Session> closeListener;

    private Sequence sequence;
    private IncrementalDecoder decoder;
    private boolean decoderFinished;
    private boolean promptPrefilled;

    /// A session over `plan`. Text prompts tokenize on `host`; the generation's frames are thrown into `lake`.
    public Session(
            QwenTokenizer tokenizer,
            ExecutionPlan plan,
            HostTasks host,
            FrameLake lake,
            ExecutionGpu gpu,
            GenerationConfig config,
            int prefillChunkTokens,
            Consumer<? super Session> closeListener) {
        if (prefillChunkTokens <= 0 || prefillChunkTokens > plan.maxRows())
            throw new IllegalArgumentException("prefillChunkTokens must be in 1.." + plan.maxRows());
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.host = Objects.requireNonNull(host, "host");
        this.frames = new GenerationFrames(Objects.requireNonNull(lake, "lake"));
        this.prefillChunkTokens = prefillChunkTokens;
        this.closeListener = Objects.requireNonNull(closeListener, "closeListener");
        Objects.requireNonNull(config, "config");
        this.sampler = new LogitsSampler(config, plan.vocabularySize());
        this.hostLogits = new HostLogits(Objects.requireNonNull(gpu, "gpu"), plan.vocabularySize());
        this.sink = address -> this.hostLogits.queueFinalRow(address, 1);
        this.decoder = tokenizer.newIncrementalDecoder();
    }

    @Override
    public boolean expectsFirstPrompt() {
        return !this.promptPrefilled;
    }

    @Override
    public CompletableFuture<List<Integer>> generateAsync(
            String prompt,
            int maxNewTokens,
            Consumer<String> text,
            TokenConstraint constraint,
            GenerationTimingListener timing) {
        Objects.requireNonNull(prompt, "prompt");
        Call call = begin(maxNewTokens, text, constraint, timing);
        try {
            this.host.tokenize(this.tokenizer, prompt, !this.promptPrefilled, call);
        } catch (RuntimeException | Error refused) {
            call.failed(refused);
        }
        return call.result();
    }

    @Override
    public CompletableFuture<List<Integer>> generateAsync(
            int[] promptTokenIds,
            int maxNewTokens,
            Consumer<String> text,
            TokenConstraint constraint,
            GenerationTimingListener timing) {
        Objects.requireNonNull(promptTokenIds, "promptTokenIds");
        if (promptTokenIds.length == 0) throw new IllegalArgumentException("prompt must encode to at least one token");
        Call call = begin(maxNewTokens, text, constraint, timing);
        call.encoded(promptTokenIds.clone());
        return call.result();
    }

    private Call begin(
            int maxNewTokens, Consumer<String> text, TokenConstraint constraint, GenerationTimingListener timing) {
        Objects.requireNonNull(text, "text");
        if (maxNewTokens < 0) throw new IllegalArgumentException("maxNewTokens must not be negative");
        if (!this.generationActive.compareAndSet(false, true))
            throw new IllegalStateException("a generation is already active for this Flash-Next session");
        try {
            ensureUsable();
            Call call = new Call(maxNewTokens, text, constraint, timing);
            this.active = call;
            return call;
        } catch (RuntimeException | Error failure) {
            this.generationActive.set(false);
            throw failure;
        }
    }

    /// One call: prefill chunks, then one decode step per token, in the order of the sequential loop.
    private final class Call extends Generation implements PromptSink {
        private final int maxNewTokens;
        private final Consumer<String> text;
        private final TokenConstraint constraint;
        private final GenerationTimingListener timing;
        private final List<Integer> callTokens = new ArrayList<>();
        private final int[] decodeToken = new int[1];
        private int[] prompt;
        private int offset;
        private int next = -1;
        private int generated;
        private boolean endedNormally;

        // The step in flight.
        private boolean samples;
        private int rows;
        private long started;

        Call(int maxNewTokens, Consumer<String> text, TokenConstraint constraint, GenerationTimingListener timing) {
            super(Session.this.frames);
            this.maxNewTokens = maxNewTokens;
            this.text = text;
            this.constraint = constraint;
            this.timing = timing;
        }

        @Override
        public void encoded(int[] ids) {
            if (ids.length == 0) {
                failed(new IllegalArgumentException("prompt must encode to at least one token"));
                return;
            }
            this.prompt = ids;
            start(this.begin);
        }

        @Override
        public void failed(Throwable failure) {
            fail(failure);
            finishNow();
        }

        /// The call's start, on a worker: the caller never waits for the sequence's allocation or the first step.
        private final StepPort begin = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                if (!isStopRequested()) {
                    if (Session.this.decoderFinished) {
                        Session.this.decoder = Session.this.tokenizer.newIncrementalDecoder();
                        Session.this.decoderFinished = false;
                    }
                    if (Session.this.sequence == null) Session.this.sequence = Session.this.plan.newSequence();
                    if (timing != null) timing.promptEncoded(System.nanoTime(), prompt.length);
                    // A greedy, unconstrained call selects each token on the device and reads back only its ID.
                    Session.this.hostLogits.selectOnDevice(Session.this.sampler.greedy() && constraint == null);
                }
                skip(select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) {
                if (isStopRequested()) {
                    complete(List.of());
                    return null;
                }
                return Call.this.chunk;
            }
        };

        /// The rest of the prompt: one quantum, a graph of its chunks; the last chunk samples the first token.
        private final StepPort chunk = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                rows = prompt.length - offset;
                samples = maxNewTokens > 0;
                started = System.nanoTime();
                startPromptStep(prompt, offset, rows, samples, select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) throws Exception {
                if (!succeeded(step, List.of())) return null;
                long executed = System.nanoTime();
                if (timing != null) timing.prefillQuantum(started, executed, rows);
                if (samples) {
                    next = select(constraint);
                    if (timing != null) timing.firstTokenSelected(System.nanoTime(), next);
                }
                offset += rows;
                return prefillNext();
            }
        };

        /// One decode token; it samples the next unless the budget is spent.
        private final StepPort decode = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                started = System.nanoTime();
                startStep(decodeToken, 0, 1, samples, select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) throws Exception {
                if (!succeeded(step, List.copyOf(callTokens))) return null;
                long executed = System.nanoTime();
                if (samples) {
                    next = select(constraint);
                    if (timing != null) timing.decodeQuantum(started, executed, System.nanoTime(), true, next);
                } else {
                    endedNormally = !isStopRequested();
                    if (timing != null) timing.decodeQuantum(started, executed, executed, false, -1);
                }
                generated++;
                return decodeNext();
            }
        };

        /// Reads the step's outcome: true when it committed. A cancelled step of a stopped generation ends the call
        /// with `stopped`; any other failure fails it.
        private boolean succeeded(AbstractQuantum step, List<Integer> stopped) throws Exception {
            Session.this.activeStep = null;
            Throwable failure = step.terminalFailure();
            if (samples) Session.this.hostLogits.retired(failure == null);
            if (failure == null) return true;
            if (failure instanceof CancellationException && isStopRequested()) {
                complete(stopped);
                return false;
            }
            if (failure instanceof Exception exception) throw exception;
            throw (Error) failure;
        }

        private StepPort prefillNext() {
            if (offset >= prompt.length) return afterPrefill();
            if (isStopRequested()) {
                complete(List.of());
                return null;
            }
            return chunk;
        }

        private StepPort afterPrefill() {
            Session.this.promptPrefilled = true;
            if (maxNewTokens == 0) {
                finishDecoder(text);
                complete(List.of());
                return null;
            }
            return decodeNext();
        }

        /// One iteration of the decode loop: commit the selected token, and sample the next unless the budget is
        /// spent.
        private StepPort decodeNext() {
            if (generated >= maxNewTokens || isStopRequested()) return finishDecoding();
            int tokenId = next;
            if (Session.this.tokenizer.isGenerationEosToken(tokenId)) {
                record(tokenId);
                endedNormally = true;
                return finishDecoding();
            }
            record(tokenId);
            emit(text, Session.this.decoder.append(tokenId));
            if (isStopRequested()) return finishDecoding();
            samples = generated + 1 < maxNewTokens;
            decodeToken[0] = tokenId;
            return decode;
        }

        private void record(int tokenId) {
            callTokens.add(tokenId);
            Session.this.generated.add(tokenId);
        }

        private StepPort finishDecoding() {
            if (endedNormally && !isStopRequested()) finishDecoder(text);
            complete(List.copyOf(callTokens));
            return null;
        }

        /// A failed generation leaves the session cancelled; the session accepts another generation before the
        /// caller's result completes.
        @Override
        protected void ended(List<Integer> tokens, Throwable failure) {
            if (failure != null) requestCancellation();
            Session.this.generationActive.set(false);
            if (Session.this.closed.get()) completeClose();
        }
    }

    private void startPromptStep(int[] tokens, int from, int count, boolean wantsLogits, AbstractFrame select) {
        if (count <= this.prefillChunkTokens) {
            startStep(tokens, from, count, wantsLogits, select);
            return;
        }
        ExecutionPlan.Handle step = this.plan.startPrompt(
                this.sequence, tokens, from, count, this.prefillChunkTokens, wantsLogits ? this.sink : null, select);
        this.activeStep = step;
        if (isStopRequested()) step.cancel();
    }

    private void startStep(int[] tokens, int from, int count, boolean wantsLogits, AbstractFrame select) {
        ExecutionPlan.Handle step =
                this.plan.start(this.sequence, tokens, from, count, wantsLogits ? this.sink : null, select);
        this.activeStep = step;
        // A cancellation that arrived before the handle existed is applied now.
        if (isStopRequested()) step.cancel();
    }

    private int select(TokenConstraint constraint) {
        int selected = this.sampler.selectToken(this.hostLogits, constraint);
        if (constraint != null) constraint.accept(selected);
        return selected;
    }

    private void finishDecoder(Consumer<String> text) {
        if (this.decoderFinished) return;
        String remaining = this.decoder.finish();
        this.decoderFinished = true;
        emit(text, remaining);
    }

    private static void emit(Consumer<String> text, String decoded) {
        if (!decoded.isEmpty()) text.accept(decoded);
    }

    @Override
    public void cancel() {
        if (this.closed.get()) return;
        requestCancellation();
    }

    @Override
    public boolean isCancelled() {
        return this.cancelled.get();
    }

    @Override
    public boolean isClosed() {
        return this.closed.get();
    }

    @Override
    public long currentTokenPosition() {
        Sequence current = this.sequence;
        return current == null ? 0 : current.position();
    }

    @Override
    public List<Integer> generatedTokenIds() {
        return this.generated.snapshot();
    }

    /// The model keeps no prefix cache: every prompt is prefilled whole.
    @Override
    public int restoredPromptTokens() {
        return 0;
    }

    /// Releases the sequence. During a generation it cancels it and waits for it to end, so it must not be called
    /// from the generation's text callback.
    @Override
    public void close() {
        boolean firstClose = this.closed.compareAndSet(false, true);
        if (this.generationActive.get()) {
            if (firstClose) requestCancellation();
            Call call = this.active;
            if (call != null) call.result().handle((tokens, failure) -> null).join();
        }
        completeClose();
    }

    private synchronized void completeClose() {
        if (this.generationActive.get()) return;
        Sequence held = this.sequence;
        this.sequence = null;
        try {
            if (held != null) held.close();
        } finally {
            this.hostLogits.close();
            if (this.closeListener != null) {
                this.closeListener.accept(this);
                this.closeListener = null;
            }
        }
    }

    private void ensureUsable() {
        if (this.closed.get()) throw new IllegalStateException("Flash-Next generation session is closed");
        if (this.cancelled.get()) throw new IllegalStateException("Flash-Next generation session is cancelled");
    }

    private boolean isStopRequested() {
        return this.cancelled.get() || this.closed.get();
    }

    private void requestCancellation() {
        this.cancelled.set(true);
        ExecutionPlan.Handle step = this.activeStep;
        if (step != null) step.cancel();
    }
}
