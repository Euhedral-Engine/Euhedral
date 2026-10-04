package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.prefix.PrefixNode;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.tokenizer.IncrementalDecoder;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.core.tokenizer.TokenConstraint;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
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
    /// The current or latest generation; completes after its cleanup.
    private volatile CompletableFuture<List<Integer>> activeGeneration;
    /// The thread draining the blocking form's text, while it does.
    private volatile Thread drainingThread;
    private Consumer<? super QwenGenerationSession> closeListener;

    private IncrementalDecoder decoder;
    private boolean decoderFinished;
    private boolean promptPrefilled;
    /// Prompt tokens the last generation took from the prefix cache instead of prefilling them.
    private volatile int restoredPromptTokens;
    /// MTP draft depth for greedy unconstrained generation from a fresh sequence; 0 disables it.
    private int speculativeDepth;
    private QwenSpeculativeDecoder speculative;
    private PrefixCache prefixCache;

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

    /// Reuses and stores the state of this session's first prompt in `cache`; null stops using one. Set before
    /// the first prompt. The cache's checkpoints sit on the prefill chunk grid, so the session must prefill in
    /// that chunk size.
    public void usePrefixCache(PrefixCache cache) {
        if (cache != null && this.prefillChunkTokens != PrefixCache.CHUNK_TOKENS)
            throw new IllegalStateException(
                    "the prefix cache needs prefill chunks of " + PrefixCache.CHUNK_TOKENS + " tokens");
        this.prefixCache = cache;
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
    public List<Integer> generate(String prompt, int maxNewTokens, Consumer<String> output, TokenConstraint constraint)
            throws InterruptedException, ExecutionException {
        return generate(prompt, maxNewTokens, output, constraint, null);
    }

    /// Generates as [#generate(String, int, Consumer, TokenConstraint)] while reporting
    /// execution boundaries to an optional timing listener. A null listener records nothing.
    public List<Integer> generate(
            String prompt,
            int maxNewTokens,
            Consumer<String> output,
            TokenConstraint constraint,
            GenerationTimingListener timing)
            throws InterruptedException, ExecutionException {
        Objects.requireNonNull(prompt, "prompt");
        return generate(prompt, null, maxNewTokens, output, constraint, timing);
    }

    /// Generates as [#generate(String, int, Consumer, TokenConstraint)] from a prompt already encoded
    /// as that method would encode it: with the model special tokens for the session's first prompt
    /// ([QwenTokenizer#encodeWithModelSpecialTokens]), as plain text ([QwenTokenizer#encodeText]) after it.
    public List<Integer> generate(
            int[] promptTokenIds, int maxNewTokens, Consumer<String> output, TokenConstraint constraint)
            throws InterruptedException, ExecutionException {
        Objects.requireNonNull(promptTokenIds, "promptTokenIds");
        return generate(null, promptTokenIds.clone(), maxNewTokens, output, constraint, null);
    }

    /// Whether the session's next prompt is its first, which carries the model special tokens.
    public boolean expectsFirstPrompt() {
        return !this.promptPrefilled;
    }

    /// Starts a generation without blocking: the future completes, on a lattice worker, with the IDs this
    /// call sampled. Encoding (for a text prompt), every quantum and token selection run on the lattice's
    /// workers as continuations of quantum retirement. `text` receives newly decoded, non-empty text on those
    /// workers, one call at a time and in order, before the next quantum is admitted, so a [#cancel] from it
    /// stops the generation before another quantum runs. It must not block: queue blocking work (a network
    /// write) as further work for the workers. The session runs one generation at a time; a failed or cancelled
    /// generation leaves it cancelled.
    public CompletableFuture<List<Integer>> generateAsync(
            String prompt,
            int maxNewTokens,
            Consumer<String> text,
            TokenConstraint constraint,
            GenerationTimingListener timing) {
        Objects.requireNonNull(prompt, "prompt");
        return begin(prompt, null, maxNewTokens, text, constraint, timing, null);
    }

    /// As [#generateAsync(String, int, Consumer, TokenConstraint, GenerationTimingListener)] from a prompt
    /// already encoded as that method would encode it: with the model special tokens for the session's first
    /// prompt ([QwenTokenizer#encodeWithModelSpecialTokens]), as plain text ([QwenTokenizer#encodeText]) after it.
    public CompletableFuture<List<Integer>> generateAsync(
            int[] promptTokenIds, int maxNewTokens, Consumer<String> text, TokenConstraint constraint) {
        Objects.requireNonNull(promptTokenIds, "promptTokenIds");
        return begin(null, promptTokenIds.clone(), maxNewTokens, text, constraint, null, null);
    }

    private CompletableFuture<List<Integer>> begin(
            String prompt,
            int[] encoded,
            int maxNewTokens,
            Consumer<String> text,
            TokenConstraint constraint,
            GenerationTimingListener timing,
            Emission heldBy) {
        Objects.requireNonNull(text, "text");
        if (maxNewTokens < 0) throw new IllegalArgumentException("maxNewTokens must not be negative");
        if (!this.generationActive.compareAndSet(false, true)) {
            throw new IllegalStateException("a generation is already active for this Qwen session");
        }
        CompletableFuture<List<Integer>> finished = new CompletableFuture<>();
        this.activeGeneration = finished;
        CompletableFuture<List<Integer>> done;
        try {
            ensureUsable();
            if (this.decoderFinished) {
                this.decoder = this.tokenizer.newIncrementalDecoder();
                this.decoderFinished = false;
            }
            CompletableFuture<int[]> promptTokenIds = encoded != null
                    ? CompletableFuture.completedFuture(encoded)
                    : this.runtime.tokenize(this.tokenizer, prompt, !this.promptPrefilled);
            done = promptTokenIds.thenCompose(ids -> {
                if (ids.length == 0) throw new IllegalArgumentException("prompt must encode to at least one token");
                if (timing != null) timing.promptEncoded(System.nanoTime(), ids.length);
                return startGeneration(ids, maxNewTokens, constraint, timing, text);
            });
        } catch (RuntimeException | Error failure) {
            this.generationActive.set(false);
            finished.completeExceptionally(failure);
            throw failure;
        }
        done.whenComplete((tokens, failure) -> {
            if (failure != null && this.sequence.terminalState() == QwenSequenceState.TerminalState.ACTIVE)
                requestCancellation();
            if (heldBy == null) end(finished, tokens, failure);
            else {
                // The blocking form's caller still hands out text: the generation stays active until it drained.
                // The end is registered before the caller can see the last text, so the caller's own completion
                // of `drained` ends the generation and `generate` returns only once another may start.
                heldBy.drained.whenComplete((ignored, unused) -> end(finished, tokens, failure));
                heldBy.end(tokens, failure);
            }
        });
        return finished;
    }

    private void end(CompletableFuture<List<Integer>> finished, List<Integer> tokens, Throwable failure) {
        try {
            this.generationActive.set(false);
            if (this.closed.get()) completeClose();
        } finally {
            if (failure != null) finished.completeExceptionally(failure);
            else finished.complete(tokens);
        }
    }

    /// The blocking form: generates on the lattice's workers as [#generateAsync] does, and hands the text to
    /// `output` on the calling thread, concurrently with later quanta. Returns after the generation finished.
    private List<Integer> generate(
            String prompt,
            int[] encoded,
            int maxNewTokens,
            Consumer<String> output,
            TokenConstraint constraint,
            GenerationTimingListener timing)
            throws InterruptedException, ExecutionException {
        Objects.requireNonNull(output, "output");
        Emission emission = new Emission();
        begin(prompt, encoded, maxNewTokens, emission::text, constraint, timing, emission);
        this.drainingThread = Thread.currentThread();
        try {
            return emission.drain(output, this::requestCancellation);
        } finally {
            this.drainingThread = null;
            // Ends the generation on this thread, so it returns once another may start.
            emission.drained.complete(null);
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

    /// Prompt tokens the session's last generation restored from the prefix cache; 0 when it prefilled all of them.
    public int restoredPromptTokens() {
        return this.restoredPromptTokens;
    }

    /// Reports whether cancellation has been requested for this session.
    public boolean isCancelled() {
        return this.cancelled.get();
    }

    /// Reports whether the caller has closed this session.
    public boolean isClosed() {
        return this.closed.get();
    }

    /// Allows the owning engine to reject reentrant shutdown from the blocking form's output callback.
    public boolean isGeneratingOnCurrentThread() {
        return this.generationActive.get() && Thread.currentThread() == this.drainingThread;
    }

    QwenSequenceState sequenceState() {
        return this.sequence;
    }

    /// Completes the sequence, releasing its persistent KV and GDN state.
    /// Closing during generation requests cancellation and waits until the generation ended, so it must not
    /// be called from that generation's text callback. From the blocking form's output callback it returns at
    /// once instead, and the generation completes the close when it ends.
    @Override
    public void close() {
        boolean firstClose = this.closed.compareAndSet(false, true);
        if (this.generationActive.get()) {
            if (firstClose) requestCancellation();
            if (Thread.currentThread() == this.drainingThread) return;
            CompletableFuture<List<Integer>> active = this.activeGeneration;
            if (active != null) active.handle((tokens, failure) -> null).join();
        }
        completeClose();
    }

    /// Releases the sequence and host state; runs again after a failed attempt, and once a generation that
    /// was active at close ends.
    private synchronized void completeClose() {
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

    private CompletableFuture<List<Integer>> startGeneration(
            int[] promptTokenIds,
            int maxNewTokens,
            TokenConstraint constraint,
            GenerationTimingListener timing,
            Consumer<String> text) {
        if (isStopRequested()) return CompletableFuture.completedFuture(List.of());
        // A greedy, unconstrained call selects each token on the device and reads back only its ID.
        this.hostLogits.selectOnDevice(this.sampler.greedy() && constraint == null);
        if (this.speculativeDepth > 0
                && this.sampler.greedy()
                && constraint == null
                && !this.promptPrefilled
                && this.sequence.currentTokenPosition() == 0
                && maxNewTokens > 0) {
            return generateSpeculative(promptTokenIds, maxNewTokens, text, timing);
        }
        Chain chain = new Chain(promptTokenIds, maxNewTokens, constraint, timing, text);
        PrefixCache cache = this.prefixCache;
        if (cache == null || this.promptPrefilled || this.sequence.currentTokenPosition() != 0) chain.prefillNext();
        else chain.startFromCache(cache);
        return chain.result;
    }

    /// One-row decode as continuations: the prefill chunks, then one decode quantum per token, each started
    /// by the previous quantum's outcome. The steps are the blocking loop's, in its order.
    private final class Chain {
        private final int[] promptTokenIds;
        private final int maxNewTokens;
        private final TokenConstraint constraint;
        private final GenerationTimingListener timing;
        private final Consumer<String> text;
        private final List<Integer> callTokenIds = new ArrayList<>();
        private final CompletableFuture<List<Integer>> result = new CompletableFuture<>();
        private int offset;
        private OptionalInt nextToken = OptionalInt.empty();
        private int generated;
        private boolean endedNormally;
        /// Where this prompt's next checkpoint attaches: the deepest stored node of its prefix. Null unless the
        /// prompt started a fresh sequence through the cache: the offsets of a continuation prompt are not
        /// positions of the sequence, so nothing it holds may be stored under its tokens.
        private PrefixNode cursor;

        Chain(
                int[] promptTokenIds,
                int maxNewTokens,
                TokenConstraint constraint,
                GenerationTimingListener timing,
                Consumer<String> text) {
            this.promptTokenIds = promptTokenIds;
            this.maxNewTokens = maxNewTokens;
            this.constraint = constraint;
            this.timing = timing;
            this.text = text;
        }

        /// Restores the longest stored prefix of the prompt, then prefills what is left.
        void startFromCache(PrefixCache cache) {
            PrefixCache.Hit hit = cache.lookup(this.promptTokenIds, false);
            this.cursor = hit == null ? cache.root() : hit.cursor();
            if (hit == null) {
                prefillNext();
                return;
            }
            long started = System.nanoTime();
            cache.restore(
                            QwenGenerationSession.this.runtime.frames(),
                            QwenGenerationSession.this.plan,
                            QwenGenerationSession.this.sequence,
                            hit)
                    .whenComplete((restored, failure) -> {
                        cache.release(hit);
                        try {
                            if (failure != null) {
                                this.result.completeExceptionally(failure);
                                return;
                            }
                            if (!restored || isStopRequested()) {
                                this.result.complete(List.of());
                                return;
                            }
                            if (this.timing != null)
                                this.timing.prefixRestored(hit.position(), System.nanoTime() - started);
                            this.offset = hit.position();
                            QwenGenerationSession.this.restoredPromptTokens = hit.position();
                            prefillNext();
                        } catch (Throwable continuationFailure) {
                            this.result.completeExceptionally(continuationFailure);
                        }
                    });
        }

        /// After a prefill chunk that ended at `end`: store a checkpoint when the cache wants one, then go on.
        private void checkpointThen(int end, Runnable next) {
            PrefixCache cache = QwenGenerationSession.this.prefixCache;
            if (cache == null
                    || this.cursor == null
                    || end <= this.cursor.position()
                    || !cache.wantsCheckpoint(end, this.promptTokenIds.length)) {
                next.run();
                return;
            }
            cache.capture(
                            QwenGenerationSession.this.runtime.frames(),
                            QwenGenerationSession.this.sequence,
                            this.cursor,
                            this.promptTokenIds,
                            end)
                    .whenComplete((node, failure) -> {
                        try {
                            if (failure == null) this.cursor = node;
                            next.run();
                        } catch (Throwable continuationFailure) {
                            this.result.completeExceptionally(continuationFailure);
                        }
                    });
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
                checkpointThen(end, this::prefillNext);
            });
        }

        private void afterPrefill() {
            QwenGenerationSession.this.promptPrefilled = true;
            if (this.maxNewTokens == 0) {
                finishDecoder(this.text);
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
            emit(this.text, QwenGenerationSession.this.decoder.append(tokenId));
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
            if (this.endedNormally && !isStopRequested()) finishDecoder(this.text);
            this.result.complete(List.copyOf(this.callTokenIds));
        }
    }

    private CompletableFuture<List<Integer>> generateSpeculative(
            int[] promptTokenIds, int maxNewTokens, Consumer<String> text, GenerationTimingListener timing) {
        if (this.speculative == null)
            this.speculative = new QwenSpeculativeDecoder(
                    this.runtime,
                    this.plan,
                    this.gpu,
                    this.sequence,
                    this.tokenizer::isGenerationEosToken,
                    this.speculativeDepth,
                    this.prefillChunkTokens);
        PrefixCache cache = this.prefixCache;
        if (cache == null || !cache.supportsMtp())
            return runSpeculative(promptTokenIds, maxNewTokens, text, timing, null, 0);
        // A speculative prompt restores only through checkpoints that hold MTP state, and stores them.
        PrefixCache.Hit hit = cache.lookup(promptTokenIds, true);
        AtomicReference<PrefixNode> cursor = new AtomicReference<>(hit == null ? cache.root() : hit.cursor());
        QwenSpeculativeDecoder.PrefixHooks hooks = (end, seedRow) -> {
            PrefixNode attachedTo = cursor.get();
            if (end <= attachedTo.position() || !cache.wantsCheckpoint(end, promptTokenIds.length))
                return CompletableFuture.completedFuture(null);
            return cache.capture(this.runtime.frames(), this.sequence, attachedTo, promptTokenIds, end, seedRow)
                    .thenAccept(cursor::set);
        };
        if (hit == null) return runSpeculative(promptTokenIds, maxNewTokens, text, timing, hooks, 0);
        long started = System.nanoTime();
        return cache.restore(this.runtime.frames(), this.plan, this.sequence, hit, true)
                .whenComplete((restored, failure) -> cache.release(hit))
                .thenCompose(restored -> {
                    if (!restored || isStopRequested()) return CompletableFuture.completedFuture(List.<Integer>of());
                    if (timing != null) timing.prefixRestored(hit.position(), System.nanoTime() - started);
                    this.restoredPromptTokens = hit.position();
                    return runSpeculative(promptTokenIds, maxNewTokens, text, timing, hooks, hit.position());
                });
    }

    private CompletableFuture<List<Integer>> runSpeculative(
            int[] promptTokenIds,
            int maxNewTokens,
            Consumer<String> text,
            GenerationTimingListener timing,
            QwenSpeculativeDecoder.PrefixHooks hooks,
            int startPosition) {
        return this.speculative
                .generateAsync(
                        promptTokenIds,
                        maxNewTokens,
                        token -> {
                            synchronized (this.generatedTokenIds) {
                                this.generatedTokenIds.add(token);
                            }
                            // As in ordinary decode, a generation terminator is returned but never decoded into text.
                            if (!this.tokenizer.isGenerationEosToken(token)) emit(text, this.decoder.append(token));
                        },
                        timing,
                        hooks,
                        startPosition)
                .thenApply(tokens -> {
                    this.promptPrefilled = true;
                    if (!isStopRequested()) finishDecoder(text);
                    return tokens;
                });
    }

    /// Admits `context` and, once its outcome is published, selects its next token (when `selectToken`) and
    /// passes it to `next` on the worker that retired the quantum. A failure completes `result` instead.
    private void executeAndSelect(
            QwenExecutionContext context,
            boolean selectToken,
            TokenConstraint constraint,
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
            TokenConstraint constraint,
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
        /// Completed by the caller after the last text; the generation ends then.
        final CompletableFuture<Void> drained = new CompletableFuture<>();
        private List<Integer> tokens;
        private Throwable failure;

        void text(String text) {
            if (!text.isEmpty()) this.queue.add(text);
        }

        /// The quanta are done; the queue publishes the result to the draining thread.
        void end(List<Integer> tokens, Throwable failure) {
            this.tokens = tokens;
            this.failure = failure;
            this.queue.add(END);
        }

        /// Hands every text to `output` until the generation ended, then returns its tokens. When `output`
        /// throws or the calling thread is interrupted, `cancel` stops the generation, which is still awaited.
        List<Integer> drain(Consumer<String> output, Runnable cancel) throws InterruptedException, ExecutionException {
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
            if (this.failure == null) return this.tokens;
            Throwable cause = this.failure instanceof java.util.concurrent.CompletionException wrapped
                            && wrapped.getCause() != null
                    ? wrapped.getCause()
                    : this.failure;
            if (cause instanceof ExecutionException executionFailure) throw executionFailure;
            if (cause instanceof RuntimeException runtimeFailure) throw runtimeFailure;
            if (cause instanceof Error error) throw error;
            throw new ExecutionException(cause);
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

    private void finishDecoder(Consumer<String> text) {
        if (this.decoderFinished) return;
        String remaining = this.decoder.finish();
        this.decoderFinished = true;
        emit(text, remaining);
    }

    private static void emit(Consumer<String> text, String decoded) {
        if (!decoded.isEmpty()) text.accept(decoded);
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
