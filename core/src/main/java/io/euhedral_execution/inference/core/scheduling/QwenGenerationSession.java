package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.tokenizer.IncrementalDecoder;
import io.euhedral_execution.inference.core.tokenizer.JsonEnvelopeConstraint;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;

/// Coordinates prompt and decode quanta for one persistent Qwen sequence.
///
/// The tokenizer, plan, runtime, and GPU are borrowed. The session owns its sequence, one sampler, and
/// the pinned host row into which each sampling quantum copies its final logits before it retires; each
/// completed prompt-to-output stream is flushed before the decoder is replaced for a later prompt.
public final class QwenGenerationSession implements AutoCloseable {

    /// Default prompt tokens per prefill quantum. Bounds per-quantum GPU workspace while the
    /// persistent sequence retains KV and GDN state.
    public static final int DEFAULT_PREFILL_CHUNK_TOKENS = 512;

    private final int prefillChunkTokens;

    private final QwenTokenizer tokenizer;
    private final QwenExecutionPlan plan;
    private final EuhedralInferenceRuntime runtime;
    private final ExecutionGpu gpu;
    private final QwenSequenceState sequence;
    private final QwenLogitsSampler sampler;
    private final QwenHostLogits hostLogits;
    private final List<Integer> generatedTokenIds = new ArrayList<>();
    private final AtomicBoolean generationActive = new AtomicBoolean();
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicBoolean closed = new AtomicBoolean();
    private final ReentrantLock generationLock = new ReentrantLock();
    private Consumer<? super QwenGenerationSession> closeListener;

    private IncrementalDecoder decoder;
    private boolean decoderFinished;
    private boolean promptPrefilled;
    /// MTP draft depth for greedy unconstrained generation from a fresh sequence; 0 disables it.
    private int speculativeDepth;
    private QwenSpeculativeDecoder speculative;

    /// Creates a session with a new persistent sequence owned by this instance.
    /// The plan, runtime, GPU, and tokenizer are borrowed and must remain usable until the session is closed.
    public QwenGenerationSession(
            QwenTokenizer tokenizer,
            QwenExecutionPlan plan,
            EuhedralInferenceRuntime runtime,
            ExecutionGpu gpu,
            long sequenceId,
            GenerationConfig config) {
        this(tokenizer, plan, runtime, gpu, sequenceId, config, ignored -> {});
    }

    /// Creates an owner-tracked session. The listener runs after successful cleanup with generation
    /// stopped, under the session lifecycle lock; it must not block or invoke session operations.
    /// Failed cleanup retains the listener for retry. Successful notification releases its reference.
    public QwenGenerationSession(
            QwenTokenizer tokenizer,
            QwenExecutionPlan plan,
            EuhedralInferenceRuntime runtime,
            ExecutionGpu gpu,
            long sequenceId,
            GenerationConfig config,
            Consumer<? super QwenGenerationSession> closeListener) {
        this(tokenizer, plan, runtime, gpu, sequenceId, config, DEFAULT_PREFILL_CHUNK_TOKENS, closeListener);
    }

    /// Creates an owner-tracked session that submits at most `prefillChunkTokens` prompt tokens per
    /// prefill quantum. Listener semantics match the constructor without a chunk size.
    public QwenGenerationSession(
            QwenTokenizer tokenizer,
            QwenExecutionPlan plan,
            EuhedralInferenceRuntime runtime,
            ExecutionGpu gpu,
            long sequenceId,
            GenerationConfig config,
            int prefillChunkTokens,
            Consumer<? super QwenGenerationSession> closeListener) {
        if (prefillChunkTokens <= 0) throw new IllegalArgumentException("prefillChunkTokens must be positive");
        this.prefillChunkTokens = prefillChunkTokens;
        this.closeListener = Objects.requireNonNull(closeListener, "closeListener");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.runtime = Objects.requireNonNull(runtime, "runtime");
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        Objects.requireNonNull(config, "config");
        this.sequence = new QwenSequenceState(sequenceId);
        this.sampler = new QwenLogitsSampler(config, plan.weights().config().vocabSize());
        this.hostLogits = new QwenHostLogits(gpu, plan.weights().config().vocabSize());
        this.decoder = tokenizer.newIncrementalDecoder();
    }

    /// Generates greedy, unconstrained calls from a fresh sequence with MTP speculative decoding of
    /// `depth` drafts per verification: the same tokens and state as one-row greedy decode, in fewer
    /// sequential steps (docs/MTP_CONTRACT.md). Requires a plan with a loaded MTP layer and draft head.
    public void enableSpeculativeDecoding(int depth) {
        if (depth < 0 || depth > 7) throw new IllegalArgumentException("depth must be 0 to 7");
        if (depth > 0 && !this.plan.drafts()) throw new IllegalStateException("the model has no MTP draft view");
        this.speculativeDepth = depth;
    }

    /// Prefills a prompt at the current sequence position and returns the IDs sampled by this call.
    /// The first prompt uses configured model special tokens; continuation prompts encode only their text.
    /// The output callback receives only newly decoded text and is never called for empty chunks.
    /// A sampled generation terminator is included in the returned IDs but is not sent through decode.
    public List<Integer> generate(
            String prompt, int maxNewTokens, Consumer<String> output, JsonEnvelopeConstraint constraint)
            throws InterruptedException, ExecutionException {
        return generate(prompt, maxNewTokens, output, constraint, null);
    }

    /// Generates as [#generate(String, int, Consumer, JsonEnvelopeConstraint)] while reporting
    /// execution boundaries to an optional timing listener. A null listener records nothing.
    public List<Integer> generate(
            String prompt,
            int maxNewTokens,
            Consumer<String> output,
            JsonEnvelopeConstraint constraint,
            GenerationTimingListener timing)
            throws InterruptedException, ExecutionException {
        Objects.requireNonNull(prompt, "prompt");
        return generate(prompt, null, maxNewTokens, output, constraint, timing);
    }

    /// Generates as [#generate(String, int, Consumer, JsonEnvelopeConstraint)] from a prompt already encoded
    /// as that method would encode it: with the model special tokens for the session's first prompt
    /// ([QwenTokenizer#encodeWithModelSpecialTokens]), as plain text ([QwenTokenizer#encodeText]) after it.
    public List<Integer> generate(
            int[] promptTokenIds, int maxNewTokens, Consumer<String> output, JsonEnvelopeConstraint constraint)
            throws InterruptedException, ExecutionException {
        Objects.requireNonNull(promptTokenIds, "promptTokenIds");
        return generate(null, promptTokenIds.clone(), maxNewTokens, output, constraint, null);
    }

    /// Whether the session's next prompt is its first, which carries the model special tokens.
    public boolean expectsFirstPrompt() {
        return !this.promptPrefilled;
    }

    private List<Integer> generate(
            String prompt,
            int[] encoded,
            int maxNewTokens,
            Consumer<String> output,
            JsonEnvelopeConstraint constraint,
            GenerationTimingListener timing)
            throws InterruptedException, ExecutionException {
        Objects.requireNonNull(output, "output");
        if (maxNewTokens < 0) throw new IllegalArgumentException("maxNewTokens must not be negative");
        if (!this.generationActive.compareAndSet(false, true)) {
            throw new IllegalStateException("a generation is already active for this Qwen session");
        }
        if (!this.generationLock.tryLock()) {
            this.generationActive.set(false);
            throw new IllegalStateException("the Qwen session is changing lifecycle state");
        }

        try {
            ensureUsable();
            if (this.decoderFinished) {
                this.decoder = this.tokenizer.newIncrementalDecoder();
                this.decoderFinished = false;
            }
            int[] promptTokenIds = encoded != null ? encoded : tokenize(prompt);
            if (promptTokenIds.length == 0) {
                throw new IllegalArgumentException("prompt must encode to at least one token");
            }
            if (timing != null) timing.promptEncoded(System.nanoTime(), promptTokenIds.length);

            try {
                return generateLocked(promptTokenIds, maxNewTokens, output, constraint, timing);
            } catch (InterruptedException | ExecutionException | RuntimeException | Error failure) {
                if (this.sequence.terminalState() == QwenSequenceState.TerminalState.ACTIVE) {
                    try {
                        requestCancellation();
                    } catch (RuntimeException | Error cleanupFailure) {
                        if (cleanupFailure != failure) failure.addSuppressed(cleanupFailure);
                    }
                }
                throw failure;
            }
        } finally {
            this.generationActive.set(false);
            try {
                if (this.closed.get()) completeClose();
            } finally {
                this.generationLock.unlock();
            }
        }
    }

    /// Encodes the prompt on the lattice's workers (EuhedralInferenceRuntime#tokenize).
    private int[] tokenize(String prompt) throws InterruptedException {
        try {
            return this.runtime
                    .tokenize(this.tokenizer, prompt, !this.promptPrefilled)
                    .get();
        } catch (ExecutionException failure) {
            Throwable cause = failure.getCause();
            if (cause instanceof RuntimeException runtimeFailure) throw runtimeFailure;
            if (cause instanceof Error error) throw error;
            throw new IllegalStateException("prompt tokenization failed", cause);
        }
    }

    public List<Integer> generate(String prompt, int maxNewTokens, Consumer<String> output)
            throws InterruptedException, ExecutionException {
        return generate(prompt, maxNewTokens, output, null);
    }

    /// Requests cancellation of the current quantum or prevents the next one from starting.
    /// A cancelled session cannot accept another prompt and remains owned until closed.
    public void cancel() {
        if (this.closed.get()) return;
        requestCancellation();
    }

    /// Returns the authoritative token position retained by the session's sequence state.
    public long currentTokenPosition() {
        return this.sequence.currentTokenPosition();
    }

    /// Returns an immutable snapshot of tokens sampled by all prompts in this session.
    public List<Integer> generatedTokenIds() {
        synchronized (this.generatedTokenIds) {
            return List.copyOf(this.generatedTokenIds);
        }
    }

    /// Reports whether cancellation has been requested for this session.
    public boolean isCancelled() {
        return this.cancelled.get();
    }

    /// Reports whether the caller has closed this session.
    public boolean isClosed() {
        return this.closed.get();
    }

    /// Allows the owning engine to reject reentrant shutdown from an output callback.
    public boolean isGeneratingOnCurrentThread() {
        return this.generationLock.isHeldByCurrentThread();
    }

    QwenSequenceState sequenceState() {
        return this.sequence;
    }

    /// Completes the sequence, releasing its persistent KV and GDN state.
    /// Closing during generation requests cancellation and waits for the in-flight quantum to detach.
    @Override
    public void close() {
        boolean firstClose = this.closed.compareAndSet(false, true);
        if (firstClose && this.generationActive.get()) requestCancellation();
        if (this.generationActive.get() && this.generationLock.isHeldByCurrentThread()) return;
        this.generationLock.lock();
        try {
            completeClose();
        } finally {
            this.generationLock.unlock();
        }
    }

    private void completeClose() {
        this.sequence.complete();
        // The sequence completes only once no quantum holds its lease, and a quantum releases the lease
        // after its retirement boundary, so no copy into the host row can still be queued.
        this.hostLogits.close();
        if (this.speculative != null) this.speculative.close();
        if (this.closeListener != null) {
            this.closeListener.accept(this);
            this.closeListener = null;
        }
    }

    /// Runs one call as a chain of continuations on the lattice's workers and emits its text on the calling
    /// thread. Each quantum's outcome callback runs on the worker whose retirement frame completes it: it
    /// selects the next token and admits the next quantum, so the calling thread is never woken between
    /// quanta. Decoded text goes to a nonblocking queue that the calling thread drains into `output`, so a
    /// slow client never holds a worker. The call returns only after the chain finished, whatever happened.
    private List<Integer> generateLocked(
            int[] promptTokenIds,
            int maxNewTokens,
            Consumer<String> output,
            JsonEnvelopeConstraint constraint,
            GenerationTimingListener timing)
            throws InterruptedException, ExecutionException {
        Emission emission = new Emission();
        CompletableFuture<List<Integer>> done;
        try {
            done = startGeneration(promptTokenIds, maxNewTokens, constraint, timing, emission);
        } catch (RuntimeException | Error failure) {
            done = CompletableFuture.failedFuture(failure);
        }
        done.whenComplete((tokens, failure) -> emission.end());
        return emission.drain(output, done, this::requestCancellation);
    }

    private CompletableFuture<List<Integer>> startGeneration(
            int[] promptTokenIds,
            int maxNewTokens,
            JsonEnvelopeConstraint constraint,
            GenerationTimingListener timing,
            Emission emission) {
        if (isStopRequested()) return CompletableFuture.completedFuture(List.of());
        // A greedy, unconstrained call selects each token on the device and reads back only its ID.
        this.hostLogits.selectOnDevice(this.sampler.greedy() && constraint == null);
        if (this.speculativeDepth > 0
                && this.sampler.greedy()
                && constraint == null
                && !this.promptPrefilled
                && this.sequence.currentTokenPosition() == 0
                && maxNewTokens > 0) {
            return generateSpeculative(promptTokenIds, maxNewTokens, emission, timing);
        }
        Chain chain = new Chain(promptTokenIds, maxNewTokens, constraint, timing, emission);
        chain.prefillNext();
        return chain.result;
    }

    /// One-row decode as continuations: the prefill chunks, then one decode quantum per token, each started
    /// by the previous quantum's outcome. The steps are the blocking loop's, in its order.
    private final class Chain {
        private final int[] promptTokenIds;
        private final int maxNewTokens;
        private final JsonEnvelopeConstraint constraint;
        private final GenerationTimingListener timing;
        private final Emission emission;
        private final List<Integer> callTokenIds = new ArrayList<>();
        private final CompletableFuture<List<Integer>> result = new CompletableFuture<>();
        private int offset;
        private OptionalInt nextToken = OptionalInt.empty();
        private int generated;
        private boolean endedNormally;

        Chain(
                int[] promptTokenIds,
                int maxNewTokens,
                JsonEnvelopeConstraint constraint,
                GenerationTimingListener timing,
                Emission emission) {
            this.promptTokenIds = promptTokenIds;
            this.maxNewTokens = maxNewTokens;
            this.constraint = constraint;
            this.timing = timing;
            this.emission = emission;
        }

        void prefillNext() {
            if (this.offset >= this.promptTokenIds.length) {
                afterPrefill();
                return;
            }
            if (isStopRequested()) {
                this.result.complete(List.of());
                return;
            }
            int end = Math.min(this.offset + QwenGenerationSession.this.prefillChunkTokens, this.promptTokenIds.length);
            long started = this.timing == null ? 0L : System.nanoTime();
            boolean samples = end == this.promptTokenIds.length && this.maxNewTokens > 0;
            QwenExecutionContext prefill = new QwenExecutionContext(
                    QwenGenerationSession.this.plan,
                    QwenGenerationSession.this.sequence,
                    QwenExecutionContext.ExecutionKind.PREFILL,
                    QwenGenerationSession.this.sequence.currentTokenPosition(),
                    Arrays.copyOfRange(this.promptTokenIds, this.offset, end),
                    samples ? QwenLogitsRequirement.LAST_TOKEN : QwenLogitsRequirement.NONE,
                    samples ? QwenGenerationSession.this.hostLogits : null);
            executeAndSelect(prefill, samples, this.constraint, this.timing, started, true, this.result, next -> {
                this.nextToken = next;
                if (isStopRequested()) {
                    this.result.complete(List.of());
                    return;
                }
                this.offset = end;
                prefillNext();
            });
        }

        private void afterPrefill() {
            QwenGenerationSession.this.promptPrefilled = true;
            if (this.maxNewTokens == 0) {
                finishDecoder(this.emission);
                this.result.complete(List.of());
                return;
            }
            decodeNext();
        }

        /// One iteration of the decode loop: commit the selected token, and sample the next unless the
        /// budget is spent.
        private void decodeNext() {
            if (this.generated >= this.maxNewTokens || isStopRequested() || this.nextToken.isEmpty()) {
                finish();
                return;
            }
            int tokenId = this.nextToken.getAsInt();
            if (QwenGenerationSession.this.tokenizer.isGenerationEosToken(tokenId)) {
                record(tokenId);
                // EOS terminates the generation and is not a model input quantum.
                this.endedNormally = true;
                finish();
                return;
            }
            if (isStopRequested()) {
                finish();
                return;
            }
            boolean anotherTokenAllowed = this.generated + 1 < this.maxNewTokens;
            record(tokenId);
            this.emission.text(QwenGenerationSession.this.decoder.append(tokenId));
            // Cancellation makes the session terminal; do not admit another quantum to preserve history.
            if (isStopRequested()) {
                finish();
                return;
            }
            long started = this.timing == null ? 0L : System.nanoTime();
            QwenExecutionContext decode = new QwenExecutionContext(
                    QwenGenerationSession.this.plan,
                    QwenGenerationSession.this.sequence,
                    QwenExecutionContext.ExecutionKind.DECODE,
                    QwenGenerationSession.this.sequence.currentTokenPosition(),
                    new int[] {tokenId},
                    anotherTokenAllowed ? QwenLogitsRequirement.LAST_TOKEN : QwenLogitsRequirement.NONE,
                    anotherTokenAllowed ? QwenGenerationSession.this.hostLogits : null);
            // Commit the final non-terminal token for continuation without sampling beyond the limit.
            executeAndSelect(
                    decode, anotherTokenAllowed, this.constraint, this.timing, started, false, this.result, next -> {
                        this.nextToken = next;
                        if (!anotherTokenAllowed) this.endedNormally = !isStopRequested();
                        if (isStopRequested() || !anotherTokenAllowed) {
                            finish();
                            return;
                        }
                        this.generated++;
                        decodeNext();
                    });
        }

        private void record(int tokenId) {
            this.callTokenIds.add(tokenId);
            synchronized (QwenGenerationSession.this.generatedTokenIds) {
                QwenGenerationSession.this.generatedTokenIds.add(tokenId);
            }
        }

        private void finish() {
            if (this.endedNormally && !isStopRequested()) finishDecoder(this.emission);
            this.result.complete(List.copyOf(this.callTokenIds));
        }
    }

    private CompletableFuture<List<Integer>> generateSpeculative(
            int[] promptTokenIds, int maxNewTokens, Emission emission, GenerationTimingListener timing) {
        if (this.speculative == null)
            this.speculative = new QwenSpeculativeDecoder(
                    this.runtime,
                    this.plan,
                    this.gpu,
                    this.sequence,
                    this.tokenizer::isGenerationEosToken,
                    this.speculativeDepth,
                    this.prefillChunkTokens);
        return this.speculative
                .generateAsync(
                        promptTokenIds,
                        maxNewTokens,
                        token -> {
                            synchronized (this.generatedTokenIds) {
                                this.generatedTokenIds.add(token);
                            }
                            // As in ordinary decode, a generation terminator is returned but never decoded into text.
                            if (!this.tokenizer.isGenerationEosToken(token)) emission.text(this.decoder.append(token));
                        },
                        timing)
                .thenApply(tokens -> {
                    this.promptPrefilled = true;
                    if (!isStopRequested()) finishDecoder(emission);
                    return tokens;
                });
    }

    /// Admits `context` and, once its outcome is published, selects its next token (when `selectToken`) and
    /// passes it to `next` on the worker that retired the quantum. A failure completes `result` instead.
    private void executeAndSelect(
            QwenExecutionContext context,
            boolean selectToken,
            JsonEnvelopeConstraint constraint,
            GenerationTimingListener timing,
            long startedNanos,
            boolean prefill,
            CompletableFuture<List<Integer>> result,
            Consumer<OptionalInt> next) {
        CompletableFuture<QwenExecutionContext.Outcome> outcome;
        try {
            outcome = this.runtime.submit(context);
        } catch (RuntimeException | Error failure) {
            result.completeExceptionally(failure);
            return;
        }
        outcome.whenComplete((completed, failure) -> {
            OptionalInt selected;
            try {
                selected = select(context, completed, failure, selectToken, constraint, timing, startedNanos, prefill);
            } catch (Throwable selectionFailure) {
                result.completeExceptionally(selectionFailure);
                return;
            }
            try {
                next.accept(selected);
            } catch (Throwable continuationFailure) {
                result.completeExceptionally(continuationFailure);
            }
        });
    }

    private OptionalInt select(
            QwenExecutionContext context,
            QwenExecutionContext.Outcome outcome,
            Throwable outcomeFailure,
            boolean selectToken,
            JsonEnvelopeConstraint constraint,
            GenerationTimingListener timing,
            long startedNanos,
            boolean prefill)
            throws ExecutionException {
        QwenDeviceLogits logits = null;
        Throwable executionFailure = null;
        try {
            if (outcomeFailure != null) throw new ExecutionException(outcomeFailure);
            if (outcome.status() == QwenExecutionContext.Status.CANCELLED) {
                this.cancelled.set(true);
                return OptionalInt.empty();
            }
            if (outcome.status() == QwenExecutionContext.Status.FAILED) {
                throw new IllegalStateException("Qwen execution quantum failed", outcome.failure());
            }
            if (outcome.status() != QwenExecutionContext.Status.SUCCESS) {
                throw new IllegalStateException("runtime returned an unknown quantum outcome");
            }
            long executedNanos = timing == null ? 0L : System.nanoTime();
            if (!selectToken || isStopRequested()) {
                if (timing != null)
                    reportQuantum(timing, context, prefill, startedNanos, executedNanos, false, executedNanos, -1);
                return OptionalInt.empty();
            }
            // The quantum copied its final row into the host logits before it retired.
            int selected = this.sampler.selectToken(this.hostLogits, constraint);
            if (constraint != null) constraint.accept(selected);
            if (timing != null)
                reportQuantum(timing, context, prefill, startedNanos, executedNanos, true, System.nanoTime(), selected);
            return OptionalInt.of(selected);
        } catch (ExecutionException | RuntimeException | Error failure) {
            executionFailure = failure;
            throw failure;
        } finally {
            if (logits == null) logits = context.logitsOutput().orElse(null);
            if (logits != null) {
                try {
                    logits.close();
                } catch (RuntimeException | Error cleanupFailure) {
                    if (executionFailure != null) executionFailure.addSuppressed(cleanupFailure);
                    else throw cleanupFailure;
                }
            }
        }
    }

    /// Decoded text on its way from the generating workers to the calling thread. Workers never block on it.
    private static final class Emission {
        private static final Object END = new Object();
        private final LinkedBlockingQueue<Object> queue = new LinkedBlockingQueue<>();

        void text(String text) {
            if (!text.isEmpty()) this.queue.add(text);
        }

        void end() {
            this.queue.add(END);
        }

        /// Hands every text to `output` until the generation ended, then returns its tokens. When `output`
        /// throws or the calling thread is interrupted, `cancel` stops the generation, which is still awaited.
        List<Integer> drain(Consumer<String> output, CompletableFuture<List<Integer>> done, Runnable cancel)
                throws InterruptedException, ExecutionException {
            Throwable outputFailure = null;
            boolean interrupted = false;
            while (true) {
                Object item;
                try {
                    item = this.queue.take();
                } catch (InterruptedException interruption) {
                    if (!interrupted) cancel.run();
                    interrupted = true;
                    continue;
                }
                if (item == END) break;
                if (outputFailure != null || interrupted) continue;
                try {
                    output.accept((String) item);
                } catch (RuntimeException | Error failure) {
                    outputFailure = failure;
                    cancel.run();
                }
            }
            if (interrupted) {
                Thread.currentThread().interrupt();
                throw new InterruptedException("generation was interrupted");
            }
            if (outputFailure instanceof RuntimeException runtimeFailure) throw runtimeFailure;
            if (outputFailure instanceof Error error) throw error;
            try {
                return done.join();
            } catch (java.util.concurrent.CompletionException failure) {
                Throwable cause = failure.getCause();
                if (cause instanceof ExecutionException executionFailure) throw executionFailure;
                if (cause instanceof RuntimeException runtimeFailure) throw runtimeFailure;
                if (cause instanceof Error error) throw error;
                throw new ExecutionException(cause);
            }
        }
    }

    private static void reportQuantum(
            GenerationTimingListener timing,
            QwenExecutionContext context,
            boolean prefill,
            long startedNanos,
            long executedNanos,
            boolean sampled,
            long selectedNanos,
            int selectedTokenId) {
        if (prefill) {
            timing.prefillQuantum(startedNanos, executedNanos, context.inputTokenCount());
            if (sampled) timing.firstTokenSelected(selectedNanos, selectedTokenId);
        } else {
            timing.decodeQuantum(startedNanos, executedNanos, selectedNanos, sampled, selectedTokenId);
        }
    }

    private void finishDecoder(Emission emission) {
        if (this.decoderFinished) return;
        String remaining = this.decoder.finish();
        this.decoderFinished = true;
        emission.text(remaining);
    }

    private void ensureUsable() {
        if (this.closed.get()) throw new IllegalStateException("Qwen generation session is closed");
        if (this.cancelled.get()) throw new IllegalStateException("Qwen generation session is cancelled");
        if (this.sequence.terminalState() != QwenSequenceState.TerminalState.ACTIVE) {
            throw new IllegalStateException("Qwen sequence is terminal: " + this.sequence.terminalState());
        }
    }

    private boolean isStopRequested() {
        return this.cancelled.get() || this.closed.get() || this.sequence.cancellationRequested();
    }

    private void requestCancellation() {
        if (this.cancelled.compareAndSet(false, true)) this.sequence.cancel();
    }
}
