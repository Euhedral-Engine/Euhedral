package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.GenerationTimingListener;
import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsSampler;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.HostFrames;
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

/// One persistent Flash-Next sequence being generated from: prefill chunks, then one decode step
/// per token, with the engine's sampler, constraint and text decoding around the model. The model
/// is ordinary autoregressive text generation: every token is the target model's own.
///
/// A generation is a chain of continuations, not a loop on a thread. Each step is started on the
/// plan and ends by calling the chain on the worker that retired it; the chain samples, emits text,
/// and starts the next step, then returns. While a step waits for the device or for an expert the
/// generation occupies no thread. Generations of different sessions interleave step by step. The
/// text callback runs on whichever worker retired a step, one call at a time and in order. The
/// session never knows the model is sparse: it hands token ids to the plan and takes a logits row
/// back.
public final class Session implements GenerationSession {

    private final QwenTokenizer tokenizer;
    private final ExecutionPlan plan;
    private final HostFrames frames;
    private final int prefillChunkTokens;
    private final LogitsSampler sampler;
    private final HostLogits hostLogits;
    private final ExecutionPlan.LogitsSink sink;
    /// The generated tokens: appended by the generation chain, one frame at a time, and read by any
    /// thread. The array only grows, with its earlier entries copied before it is published, so a
    /// reader that takes the count and then the array sees that many tokens.
    private volatile int[] generatedIds = new int[64];

    private volatile int generatedCount;
    private final AtomicBoolean generationActive = new AtomicBoolean();
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicBoolean closed = new AtomicBoolean();
    private volatile CompletableFuture<List<Integer>> activeGeneration;
    private volatile ExecutionPlan.Handle activeStep;
    private Consumer<? super Session> closeListener;

    private Sequence sequence;
    private IncrementalDecoder decoder;
    private boolean decoderFinished;
    private boolean promptPrefilled;

    public Session(
            QwenTokenizer tokenizer,
            ExecutionPlan plan,
            HostFrames frames,
            ExecutionGpu gpu,
            GenerationConfig config,
            int prefillChunkTokens,
            Consumer<? super Session> closeListener) {
        if (prefillChunkTokens <= 0 || prefillChunkTokens > plan.maxRows())
            throw new IllegalArgumentException("prefillChunkTokens must be in 1.." + plan.maxRows());
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.frames = Objects.requireNonNull(frames, "frames");
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
        CompletableFuture<List<Integer>> result = new CompletableFuture<>();
        boolean first = !this.promptPrefilled;
        this.frames.run(HostFrames.of(
                () -> {
                    int[] ids = first
                            ? this.tokenizer.encodeWithModelSpecialTokens(prompt)
                            : this.tokenizer.encodeText(prompt);
                    generateAsync(ids, maxNewTokens, text, constraint, timing).whenComplete((tokens, failure) -> {
                        if (failure != null) result.completeExceptionally(failure);
                        else result.complete(tokens);
                    });
                },
                result::completeExceptionally));
        return result;
    }

    @Override
    public CompletableFuture<List<Integer>> generateAsync(
            int[] promptTokenIds,
            int maxNewTokens,
            Consumer<String> text,
            TokenConstraint constraint,
            GenerationTimingListener timing) {
        Objects.requireNonNull(promptTokenIds, "promptTokenIds");
        Objects.requireNonNull(text, "text");
        if (maxNewTokens < 0) throw new IllegalArgumentException("maxNewTokens must not be negative");
        if (promptTokenIds.length == 0) throw new IllegalArgumentException("prompt must encode to at least one token");
        if (!this.generationActive.compareAndSet(false, true))
            throw new IllegalStateException("a generation is already active for this Flash-Next session");
        CompletableFuture<List<Integer>> finished = new CompletableFuture<>();
        this.activeGeneration = finished;
        try {
            ensureUsable();
            Chain chain = new Chain(promptTokenIds.clone(), maxNewTokens, text, constraint, timing, finished);
            // The chain starts as host work: the caller never waits for the sequence's allocation or the first step.
            this.frames.run(new HostFrames.Task() {
                @Override
                public void run() {
                    chain.start();
                }

                @Override
                public void failed(Throwable cause) {
                    chain.fail(cause);
                }
            });
        } catch (RuntimeException | Error failure) {
            this.generationActive.set(false);
            finished.completeExceptionally(failure);
            throw failure;
        }
        return finished;
    }

    /// The generation as continuations: prefill chunks, then one decode step per token, each
    /// started by the previous step's end. The steps are the blocking loop's, in its order.
    private final class Chain implements ExecutionPlan.Listener {
        private final int[] prompt;
        private final int maxNewTokens;
        private final Consumer<String> text;
        private final TokenConstraint constraint;
        private final GenerationTimingListener timing;
        private final CompletableFuture<List<Integer>> finished;
        private final List<Integer> callTokens = new ArrayList<>();
        private final int[] decodeToken = new int[1];

        private int offset;
        private int next = -1;
        private int generated;
        private boolean endedNormally;

        // The step in flight.
        private boolean decoding;
        private boolean samples;
        private int rows;
        private long started;

        Chain(
                int[] prompt,
                int maxNewTokens,
                Consumer<String> text,
                TokenConstraint constraint,
                GenerationTimingListener timing,
                CompletableFuture<List<Integer>> finished) {
            this.prompt = prompt;
            this.maxNewTokens = maxNewTokens;
            this.text = text;
            this.constraint = constraint;
            this.timing = timing;
            this.finished = finished;
        }

        void start() {
            try {
                if (isStopRequested()) {
                    complete(List.of(), null);
                    return;
                }
                if (Session.this.decoderFinished) {
                    Session.this.decoder = Session.this.tokenizer.newIncrementalDecoder();
                    Session.this.decoderFinished = false;
                }
                if (Session.this.sequence == null) Session.this.sequence = Session.this.plan.newSequence();
                if (this.timing != null) this.timing.promptEncoded(System.nanoTime(), this.prompt.length);
                // A greedy, unconstrained call selects each token on the device and reads back only its ID.
                Session.this.hostLogits.selectOnDevice(Session.this.sampler.greedy() && this.constraint == null);
                prefillNext();
            } catch (Throwable failure) {
                fail(failure);
            }
        }

        private void prefillNext() {
            if (this.offset >= this.prompt.length) {
                afterPrefill();
                return;
            }
            if (isStopRequested()) {
                complete(List.of(), null);
                return;
            }
            this.rows = Math.min(Session.this.prefillChunkTokens, this.prompt.length - this.offset);
            this.samples = this.offset + this.rows == this.prompt.length && this.maxNewTokens > 0;
            this.decoding = false;
            this.started = System.nanoTime();
            startStep(this.prompt, this.offset, this.rows, this.samples);
        }

        private void afterPrefill() {
            Session.this.promptPrefilled = true;
            if (this.maxNewTokens == 0) {
                finishDecoder(this.text);
                complete(List.of(), null);
                return;
            }
            decodeNext();
        }

        /// One iteration of the decode loop: commit the selected token, and sample the next unless
        /// the budget is spent.
        private void decodeNext() {
            if (this.generated >= this.maxNewTokens || isStopRequested()) {
                finishDecoding();
                return;
            }
            int tokenId = this.next;
            if (Session.this.tokenizer.isGenerationEosToken(tokenId)) {
                record(tokenId);
                this.endedNormally = true;
                finishDecoding();
                return;
            }
            record(tokenId);
            emit(this.text, Session.this.decoder.append(tokenId));
            if (isStopRequested()) {
                finishDecoding();
                return;
            }
            this.samples = this.generated + 1 < this.maxNewTokens;
            this.decoding = true;
            this.decodeToken[0] = tokenId;
            this.started = System.nanoTime();
            startStep(this.decodeToken, 0, 1, this.samples);
        }

        private void startStep(int[] tokens, int from, int count, boolean wantsLogits) {
            Session.this.activeStep = Session.this.plan.start(
                    Session.this.sequence, tokens, from, count, wantsLogits ? Session.this.sink : null, this);
            // A cancellation that arrived before the handle existed is applied now.
            if (isStopRequested()) Session.this.activeStep.cancel();
        }

        /// The step ended, on the worker that retired it.
        @Override
        public void finished(Throwable failure) {
            Session.this.activeStep = null;
            try {
                if (this.samples) Session.this.hostLogits.retired(failure == null);
                if (failure != null) {
                    if (failure instanceof CancellationException && isStopRequested()) {
                        complete(this.decoding ? List.copyOf(this.callTokens) : List.of(), null);
                    } else fail(failure);
                    return;
                }
                if (!this.decoding) afterPrefillStep();
                else afterDecodeStep();
            } catch (Throwable thrown) {
                fail(thrown);
            }
        }

        private void afterPrefillStep() {
            long executed = System.nanoTime();
            if (this.timing != null) this.timing.prefillQuantum(this.started, executed, this.rows);
            if (this.samples) {
                this.next = select(this.constraint);
                if (this.timing != null) this.timing.firstTokenSelected(System.nanoTime(), this.next);
            }
            this.offset += this.rows;
            prefillNext();
        }

        private void afterDecodeStep() {
            long executed = System.nanoTime();
            if (this.samples) {
                this.next = select(this.constraint);
                if (this.timing != null)
                    this.timing.decodeQuantum(this.started, executed, System.nanoTime(), true, this.next);
            } else {
                this.endedNormally = !isStopRequested();
                if (this.timing != null) this.timing.decodeQuantum(this.started, executed, executed, false, -1);
            }
            this.generated++;
            decodeNext();
        }

        private void record(int tokenId) {
            this.callTokens.add(tokenId);
            Session session = Session.this;
            int count = session.generatedCount;
            int[] ids = session.generatedIds;
            if (count == ids.length) {
                ids = java.util.Arrays.copyOf(ids, count * 2);
                session.generatedIds = ids;
            }
            ids[count] = tokenId;
            session.generatedCount = count + 1;
        }

        private void finishDecoding() {
            if (this.endedNormally && !isStopRequested()) finishDecoder(this.text);
            complete(List.copyOf(this.callTokens), null);
        }

        /// Ends the generation with the tokens it produced or the failure that stopped it.
        void fail(Throwable failure) {
            requestCancellation();
            complete(null, failure);
        }

        private void complete(List<Integer> tokens, Throwable failure) {
            Session.this.generationActive.set(false);
            if (Session.this.closed.get()) completeClose();
            if (failure != null) this.finished.completeExceptionally(failure);
            else this.finished.complete(tokens);
        }
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
        ExecutionPlan.Handle step = this.activeStep;
        if (step != null) step.cancel();
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
        int count = this.generatedCount;
        int[] ids = this.generatedIds;
        List<Integer> copy = new ArrayList<>(count);
        for (int i = 0; i < count; i++) copy.add(ids[i]);
        return List.copyOf(copy);
    }

    /// The model keeps no prefix cache: every prompt is prefilled whole.
    @Override
    public int restoredPromptTokens() {
        return 0;
    }

    @Override
    public void close() {
        boolean firstClose = this.closed.compareAndSet(false, true);
        if (this.generationActive.get()) {
            if (firstClose) requestCancellation();
            CompletableFuture<List<Integer>> active = this.activeGeneration;
            if (active != null) active.handle((tokens, failure) -> null).join();
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
