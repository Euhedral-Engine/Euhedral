package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.Generation;
import io.euhedral_execution.inference.core.generation.GenerationFrames;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.GenerationTimingListener;
import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.generation.LogitsSampler;
import io.euhedral_execution.inference.core.generation.StepPort;
import io.euhedral_execution.inference.core.generation.TokenRecord;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.prefix.PrefixCache;
import io.euhedral_execution.inference.core.model.qwen38.speculative.MtpDecoder;
import io.euhedral_execution.inference.core.model.qwen38.speculative.SpeculativeCheckpoint;
import io.euhedral_execution.inference.core.model.qwen38.speculative.SpeculativeDecoding;
import io.euhedral_execution.inference.core.prefix.PrefixNode;
import io.euhedral_execution.inference.core.runtime.PromptSink;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
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
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/// Coordinates prompt and decode quanta for one persistent Qwen sequence.
///
/// The tokenizer, plan, runtime, and GPU are borrowed. The session owns its sequence, one sampler, and
/// the pinned host row into which each sampling quantum copies its final logits before it retires; each
/// completed prompt-to-output stream is flushed before the decoder is replaced for a later prompt.
///
/// A generation is a chain of frames (docs/FRAME_MODEL.md): each step's quantum throws a Select frame when it
/// retires, which samples, emits text and throws the next step's Admit. No thread waits for a quantum.
public final class Session implements GenerationSession {

    /// Default prompt tokens per prefill quantum. Bounds per-quantum GPU workspace while the
    /// persistent sequence retains KV and GDN state.
    public static final int DEFAULT_PREFILL_CHUNK_TOKENS = InferenceConfig.PREFILL_CHUNK_TOKENS;

    private final int prefillChunkTokens;

    private final QwenTokenizer tokenizer;
    private final ExecutionPlan plan;
    private final Execution runtime;
    private final ExecutionGpu gpu;
    private final Sequence sequence;
    private final LogitsSampler sampler;
    private final HostLogits hostLogits;
    private final GenerationFrames frames;
    /// Every token the session generated, across its prompts.
    private final TokenRecord generatedTokenIds = new TokenRecord();
    private final AtomicBoolean generationActive = new AtomicBoolean();
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicBoolean closed = new AtomicBoolean();
    /// The current or latest generation.
    private volatile Call active;
    /// The thread draining the blocking form's text, while it does.
    private volatile Thread drainingThread;
    private Consumer<? super Session> closeListener;

    private IncrementalDecoder decoder;
    private boolean decoderFinished;
    private boolean promptPrefilled;
    /// Prompt tokens the last generation took from the prefix cache instead of prefilling them.
    private volatile int restoredPromptTokens;
    /// The speculative strategy for greedy unconstrained generation from a fresh sequence; null disables it.
    private SpeculativeDecoding.Factory speculation;
    private SpeculativeDecoding speculative;
    private PrefixCache prefixCache;

    /// Creates a session with a new persistent sequence owned by this instance.
    /// The plan, runtime, GPU, and tokenizer are borrowed and must remain usable until the session is closed.
    public Session(
            QwenTokenizer tokenizer,
            ExecutionPlan plan,
            Execution runtime,
            ExecutionGpu gpu,
            long sequenceId,
            GenerationConfig config) {
        this(tokenizer, plan, runtime, gpu, sequenceId, config, ignored -> {});
    }

    /// Creates an owner-tracked session. The listener runs after successful cleanup with generation
    /// stopped, under the session lifecycle lock; it must not block or invoke session operations.
    /// Failed cleanup retains the listener for retry. Successful notification releases its reference.
    public Session(
            QwenTokenizer tokenizer,
            ExecutionPlan plan,
            Execution runtime,
            ExecutionGpu gpu,
            long sequenceId,
            GenerationConfig config,
            Consumer<? super Session> closeListener) {
        this(tokenizer, plan, runtime, gpu, sequenceId, config, DEFAULT_PREFILL_CHUNK_TOKENS, closeListener);
    }

    /// Creates an owner-tracked session that submits at most `prefillChunkTokens` prompt tokens per
    /// prefill quantum. Listener semantics match the constructor without a chunk size.
    public Session(
            QwenTokenizer tokenizer,
            ExecutionPlan plan,
            Execution runtime,
            ExecutionGpu gpu,
            long sequenceId,
            GenerationConfig config,
            int prefillChunkTokens,
            Consumer<? super Session> closeListener) {
        if (prefillChunkTokens <= 0) throw new IllegalArgumentException("prefillChunkTokens must be positive");
        this.prefillChunkTokens = prefillChunkTokens;
        this.closeListener = Objects.requireNonNull(closeListener, "closeListener");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.runtime = Objects.requireNonNull(runtime, "runtime");
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        Objects.requireNonNull(config, "config");
        this.sequence = new Sequence(sequenceId);
        this.sampler = new LogitsSampler(config, plan.weights().config().vocabSize());
        this.hostLogits = new HostLogits(gpu, plan.weights().config().vocabSize());
        this.frames = new GenerationFrames(runtime.lake());
        this.decoder = tokenizer.newIncrementalDecoder();
    }

    /// Reuses and stores the state of this session's first prompt in `cache`; null stops using one. Set before
    /// the first prompt. The cache's checkpoints sit on the prefill chunk grid, so the session must prefill in
    /// that chunk size.
    public void usePrefixCache(PrefixCache cache) {
        if (cache != null && this.prefillChunkTokens != InferenceConfig.PREFILL_CHUNK_TOKENS)
            throw new IllegalStateException(
                    "the prefix cache needs prefill chunks of " + InferenceConfig.PREFILL_CHUNK_TOKENS + " tokens");
        this.prefixCache = cache;
    }

    /// Generates greedy, unconstrained calls from a fresh sequence with MTP speculative decoding of
    /// `depth` drafts per verification: the same tokens and state as one-row greedy decode, in fewer
    /// sequential steps (docs/MTP_CONTRACT.md). Requires a plan with a loaded MTP layer and draft head.
    public void enableSpeculativeDecoding(int depth) {
        if (depth < 0 || depth > 7) throw new IllegalArgumentException("depth must be 0 to 7");
        if (depth > 0 && !this.plan.drafts()) throw new IllegalStateException("the model has no MTP draft view");
        useSpeculativeDecoding(depth == 0 ? null : MtpDecoder.factory(depth));
    }

    /// Generates greedy, unconstrained calls from a fresh sequence with the speculative strategy `factory`
    /// opens (null: none). Set before the first prompt.
    public void useSpeculativeDecoding(SpeculativeDecoding.Factory factory) {
        if (this.speculative != null) throw new IllegalStateException("the session already speculates");
        this.speculation = factory;
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

    public List<Integer> generate(String prompt, int maxNewTokens, Consumer<String> output)
            throws InterruptedException, ExecutionException {
        return generate(prompt, maxNewTokens, output, null);
    }

    /// Whether the session's next prompt is its first, which carries the model special tokens.
    @Override
    public boolean expectsFirstPrompt() {
        return !this.promptPrefilled;
    }

    /// Starts a generation without blocking: the future completes, on a lattice worker, with the IDs this
    /// call sampled. Encoding (for a text prompt), every quantum and token selection run on the lattice's
    /// workers as frames. `text` receives newly decoded, non-empty text on those workers, one call at a time
    /// and in order, before the next quantum is admitted, so a [#cancel] from it stops the generation before
    /// another quantum runs. It must not block: queue blocking work (a network write) as further work for the
    /// workers. The session runs one generation at a time; a failed or cancelled generation leaves it cancelled.
    @Override
    public CompletableFuture<List<Integer>> generateAsync(
            String prompt,
            int maxNewTokens,
            Consumer<String> text,
            TokenConstraint constraint,
            GenerationTimingListener timing) {
        Objects.requireNonNull(prompt, "prompt");
        return begin(prompt, null, maxNewTokens, text, constraint, timing, null).result();
    }

    /// As [#generateAsync(String, int, Consumer, TokenConstraint, GenerationTimingListener)] from a prompt
    /// already encoded as that method would encode it: with the model special tokens for the session's first
    /// prompt ([QwenTokenizer#encodeWithModelSpecialTokens]), as plain text ([QwenTokenizer#encodeText]) after it.
    public CompletableFuture<List<Integer>> generateAsync(
            int[] promptTokenIds, int maxNewTokens, Consumer<String> text, TokenConstraint constraint) {
        return generateAsync(promptTokenIds, maxNewTokens, text, constraint, null);
    }

    /// As [#generateAsync(int[], int, Consumer, TokenConstraint)], reporting execution boundaries to an optional
    /// timing listener on the workers that retire them.
    @Override
    public CompletableFuture<List<Integer>> generateAsync(
            int[] promptTokenIds,
            int maxNewTokens,
            Consumer<String> text,
            TokenConstraint constraint,
            GenerationTimingListener timing) {
        Objects.requireNonNull(promptTokenIds, "promptTokenIds");
        return begin(null, promptTokenIds.clone(), maxNewTokens, text, constraint, timing, null)
                .result();
    }

    private Call begin(
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
        Call call;
        try {
            ensureUsable();
            if (this.decoderFinished) {
                this.decoder = this.tokenizer.newIncrementalDecoder();
                this.decoderFinished = false;
            }
            call = new Call(maxNewTokens, text, constraint, timing, heldBy);
        } catch (RuntimeException | Error failure) {
            this.generationActive.set(false);
            throw failure;
        }
        this.active = call;
        if (encoded != null) call.encoded(encoded);
        else {
            try {
                this.runtime.tokenize(this.tokenizer, prompt, !this.promptPrefilled, call);
            } catch (RuntimeException | Error refused) {
                call.failed(refused);
            }
        }
        return call;
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
        Call call = begin(prompt, encoded, maxNewTokens, emission::text, constraint, timing, emission);
        this.drainingThread = Thread.currentThread();
        try {
            return emission.drain(output, this::requestCancellation);
        } finally {
            this.drainingThread = null;
            // Ends the generation on this thread, so it returns once another may start.
            call.settle();
        }
    }

    /// Requests cancellation of the current quantum or prevents the next one from starting.
    /// A cancelled session cannot accept another prompt and remains owned until closed.
    @Override
    public void cancel() {
        if (this.closed.get()) return;
        requestCancellation();
    }

    /// Returns the authoritative token position retained by the session's sequence state.
    @Override
    public long currentTokenPosition() {
        return this.sequence.currentTokenPosition();
    }

    /// Returns an immutable snapshot of tokens sampled by all prompts in this session.
    @Override
    public List<Integer> generatedTokenIds() {
        return this.generatedTokenIds.snapshot();
    }

    /// Prompt tokens the session's last generation restored from the prefix cache; 0 when it prefilled all of them.
    @Override
    public int restoredPromptTokens() {
        return this.restoredPromptTokens;
    }

    /// Reports whether cancellation has been requested for this session.
    @Override
    public boolean isCancelled() {
        return this.cancelled.get();
    }

    /// Reports whether the caller has closed this session.
    @Override
    public boolean isClosed() {
        return this.closed.get();
    }

    /// Allows the owning engine to reject reentrant shutdown from the blocking form's output callback.
    @Override
    public boolean isGeneratingOnCurrentThread() {
        return this.generationActive.get() && Thread.currentThread() == this.drainingThread;
    }

    Sequence sequenceState() {
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
            Call call = this.active;
            if (call != null) call.settled.join();
        }
        completeClose();
    }

    /// Releases the sequence and host state; runs again after a failed attempt, and once a generation that
    /// was active at close ends.
    private synchronized void completeClose() {
        this.sequence.complete();
        // The sequence completes only once no quantum is in flight, and a quantum retires from the sequence
        // after its retirement boundary, so no copy into the host row can still be queued.
        this.hostLogits.close();
        if (this.speculative != null) this.speculative.close();
        if (this.closeListener != null) {
            this.closeListener.accept(this);
            this.closeListener = null;
        }
    }

    /// One call: prefix restore, prefill chunks with their checkpoints, then one decode quantum per token (or a
    /// speculative strategy's steps), in the order of the sequential loop.
    private final class Call extends Generation implements PromptSink {
        private final int maxNewTokens;
        private final Consumer<String> text;
        private final TokenConstraint constraint;
        private final GenerationTimingListener timing;
        private final Emission heldBy;
        /// Completes once the session accepts another generation: for the blocking form, after the caller drained.
        final CompletableFuture<Void> settled = new CompletableFuture<>();

        private final List<Integer> callTokenIds = new ArrayList<>();
        private int[] promptTokenIds;
        private int offset;
        private int end;
        private boolean samples;
        private long started;
        private OptionalInt nextToken = OptionalInt.empty();
        private int generated;
        private int decodeToken;
        private boolean anotherTokenAllowed;
        private boolean endedNormally;
        /// Where this prompt's next checkpoint attaches: the deepest stored node of its prefix. Null unless the
        /// prompt started a fresh sequence through the cache: the offsets of a continuation prompt are not
        /// positions of the sequence, so nothing it holds may be stored under its tokens.
        private volatile PrefixNode cursor;

        private PrefixCache.Hit hit;
        private SpeculativeCheckpoint speculativeState;
        private long restoreStarted;
        // Written by a copy's completion, then read by the Select it throws.
        private volatile boolean restored;
        private volatile Throwable copyFailure;
        private volatile PrefixNode captured;

        Call(
                int maxNewTokens,
                Consumer<String> text,
                TokenConstraint constraint,
                GenerationTimingListener timing,
                Emission heldBy) {
            super(Session.this.frames);
            this.maxNewTokens = maxNewTokens;
            this.text = text;
            this.constraint = constraint;
            this.timing = timing;
            this.heldBy = heldBy;
        }

        @Override
        public void encoded(int[] ids) {
            try {
                if (ids.length == 0) throw new IllegalArgumentException("prompt must encode to at least one token");
                if (this.timing != null) this.timing.promptEncoded(System.nanoTime(), ids.length);
                this.promptTokenIds = ids;
                startGeneration();
            } catch (Throwable failure) {
                failed(failure);
            }
        }

        @Override
        public void failed(Throwable failure) {
            fail(failure);
            finishNow();
        }

        /// Chooses the first step; its last statement starts the chain or finishes it.
        private void startGeneration() {
            if (isStopRequested()) {
                complete(List.of());
                finishNow();
                return;
            }
            // A greedy, unconstrained call selects each token on the device and reads back only its ID.
            hostLogits.selectOnDevice(sampler.greedy() && this.constraint == null);
            if (speculation != null
                    && sampler.greedy()
                    && this.constraint == null
                    && !promptPrefilled
                    && sequence.currentTokenPosition() == 0
                    && this.maxNewTokens > 0) {
                startSpeculative();
                return;
            }
            PrefixCache cache = prefixCache;
            if (cache == null || promptPrefilled || sequence.currentTokenPosition() != 0) {
                start(this.prompt);
                return;
            }
            this.hit = cache.lookup(this.promptTokenIds);
            this.cursor = this.hit == null ? cache.root() : this.hit.cursor();
            start(this.hit == null ? this.prompt : this.restore);
        }

        /// Restores the longest stored prefix of the prompt; the copy's completion throws the Select.
        private final StepPort restore = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                PrefixCache cache = prefixCache;
                PrefixCache.Hit stored = hit;
                restoreStarted = System.nanoTime();
                cache.restore(runtime.frames(), plan, sequence, stored).whenComplete((done, failure) -> {
                    cache.release(stored);
                    restored = Boolean.TRUE.equals(done);
                    copyFailure = failure;
                    skip(select);
                });
            }

            @Override
            public StepPort retired(AbstractQuantum step) throws Exception {
                if (copyFailure != null) throw rethrown(copyFailure);
                if (!restored || isStopRequested()) {
                    complete(List.of());
                    return null;
                }
                if (timing != null) timing.prefixRestored(hit.position(), System.nanoTime() - restoreStarted);
                offset = hit.position();
                restoredPromptTokens = hit.position();
                return prefillNext();
            }
        };

        /// A prefill chunk; the last one samples the first token.
        private final StepPort prompt = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                end = Math.min(offset + prefillChunkTokens, promptTokenIds.length);
                started = timing == null ? 0L : System.nanoTime();
                samples = end == promptTokenIds.length && maxNewTokens > 0;
                runtime.admit(
                        new Quantum(
                                plan,
                                sequence,
                                Quantum.ExecutionKind.PREFILL,
                                sequence.currentTokenPosition(),
                                Arrays.copyOfRange(promptTokenIds, offset, end),
                                samples ? LogitsRequirement.LAST_TOKEN : LogitsRequirement.NONE,
                                samples ? hostLogits : null),
                        select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) throws Exception {
                nextToken = select((Quantum) step, samples, true);
                if (isStopRequested()) {
                    complete(List.of());
                    return null;
                }
                offset = end;
                return checkpointThen(end);
            }
        };

        /// Stores the state after a chunk when the cache wants a checkpoint there; a failed capture is ignored.
        private final StepPort capture = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                prefixCache
                        .capture(runtime.frames(), sequence, cursor, promptTokenIds, end)
                        .whenComplete((node, failure) -> {
                            captured = failure == null ? node : null;
                            skip(select);
                        });
            }

            @Override
            public StepPort retired(AbstractQuantum step) {
                if (captured != null) cursor = captured;
                captured = null;
                return prefillNext();
            }
        };

        /// One decode token; it samples the next unless the budget is spent.
        private final StepPort decode = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                started = timing == null ? 0L : System.nanoTime();
                runtime.admit(
                        new Quantum(
                                plan,
                                sequence,
                                Quantum.ExecutionKind.DECODE,
                                sequence.currentTokenPosition(),
                                new int[] {decodeToken},
                                anotherTokenAllowed ? LogitsRequirement.LAST_TOKEN : LogitsRequirement.NONE,
                                anotherTokenAllowed ? hostLogits : null),
                        select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) throws Exception {
                // Commit the final non-terminal token for continuation without sampling beyond the limit.
                nextToken = select((Quantum) step, anotherTokenAllowed, false);
                if (!anotherTokenAllowed) endedNormally = !isStopRequested();
                if (isStopRequested() || !anotherTokenAllowed) return finish();
                generated++;
                return decodeNext();
            }
        };

        private StepPort checkpointThen(int end) {
            PrefixCache cache = prefixCache;
            if (cache == null
                    || this.cursor == null
                    || end <= this.cursor.position()
                    || !cache.wantsCheckpoint(end, this.promptTokenIds.length)) return prefillNext();
            return this.capture;
        }

        private StepPort prefillNext() {
            if (this.offset >= this.promptTokenIds.length) return afterPrefill();
            if (isStopRequested()) {
                complete(List.of());
                return null;
            }
            return this.prompt;
        }

        private StepPort afterPrefill() {
            promptPrefilled = true;
            if (this.maxNewTokens == 0) {
                finishDecoder(this.text);
                complete(List.of());
                return null;
            }
            return decodeNext();
        }

        /// One iteration of the decode loop: commit the selected token, and sample the next unless the budget is
        /// spent.
        private StepPort decodeNext() {
            if (this.generated >= this.maxNewTokens || isStopRequested() || this.nextToken.isEmpty()) return finish();
            int tokenId = this.nextToken.getAsInt();
            if (tokenizer.isGenerationEosToken(tokenId)) {
                record(tokenId);
                // EOS terminates the generation and is not a model input quantum.
                this.endedNormally = true;
                return finish();
            }
            if (isStopRequested()) return finish();
            this.anotherTokenAllowed = this.generated + 1 < this.maxNewTokens;
            record(tokenId);
            emit(this.text, decoder.append(tokenId));
            // Cancellation makes the session terminal; do not admit another quantum to preserve history.
            if (isStopRequested()) return finish();
            this.decodeToken = tokenId;
            return this.decode;
        }

        private void record(int tokenId) {
            this.callTokenIds.add(tokenId);
            generatedTokenIds.add(tokenId);
        }

        private StepPort finish() {
            if (this.endedNormally && !isStopRequested()) finishDecoder(this.text);
            complete(List.copyOf(this.callTokenIds));
            return null;
        }

        /// Reads a retired quantum's published outcome and selects its next token when `selectToken`.
        private OptionalInt select(Quantum context, boolean selectToken, boolean prefill) throws ExecutionException {
            Quantum.Outcome outcome = context.conclusion();
            DeviceLogits logits = null;
            Throwable executionFailure = null;
            try {
                if (outcome == null) throw new IllegalStateException("the quantum's outcome was not published");
                if (outcome.status() == Quantum.Status.CANCELLED) {
                    cancelled.set(true);
                    return OptionalInt.empty();
                }
                if (outcome.status() == Quantum.Status.FAILED) {
                    throw new IllegalStateException("Qwen execution quantum failed", outcome.failure());
                }
                if (outcome.status() != Quantum.Status.SUCCESS) {
                    throw new IllegalStateException("runtime returned an unknown quantum outcome");
                }
                long executedNanos = this.timing == null ? 0L : System.nanoTime();
                if (!selectToken || isStopRequested()) {
                    if (this.timing != null)
                        reportQuantum(
                                this.timing, context, prefill, this.started, executedNanos, false, executedNanos, -1);
                    return OptionalInt.empty();
                }
                // The quantum copied its final row into the host logits before it retired.
                int selected = sampler.selectToken(hostLogits, this.constraint);
                if (this.constraint != null) this.constraint.accept(selected);
                if (this.timing != null)
                    reportQuantum(
                            this.timing,
                            context,
                            prefill,
                            this.started,
                            executedNanos,
                            true,
                            System.nanoTime(),
                            selected);
                return OptionalInt.of(selected);
            } catch (RuntimeException | Error failure) {
                executionFailure = failure;
                throw failure;
            } finally {
                logits = context.logitsOutput().orElse(null);
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

        // ---------------------------------------------------------------- speculation

        private void startSpeculative() {
            if (speculative == null)
                speculative = speculation.open(
                        runtime, plan, gpu, sequence, tokenizer::isGenerationEosToken, prefillChunkTokens);
            PrefixCache cache = prefixCache;
            if (cache == null) {
                start(speculative.start(
                        this.promptTokenIds,
                        this.maxNewTokens,
                        this::speculativeToken,
                        this.timing,
                        null,
                        0,
                        this::speculativeEnded));
                return;
            }
            // A speculative prompt restores only through checkpoints that hold its strategy's state, and stores them.
            this.speculativeState = speculative.checkpoint();
            this.hit = cache.lookup(this.promptTokenIds, this.speculativeState);
            this.cursor = this.hit == null ? cache.root() : this.hit.cursor();
            if (this.hit == null) {
                start(speculative.start(
                        this.promptTokenIds,
                        this.maxNewTokens,
                        this::speculativeToken,
                        this.timing,
                        this::checkpoint,
                        0,
                        this::speculativeEnded));
                return;
            }
            start(this.speculativeRestore);
        }

        private final StepPort speculativeRestore = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                PrefixCache cache = prefixCache;
                PrefixCache.Hit stored = hit;
                restoreStarted = System.nanoTime();
                cache.restore(runtime.frames(), plan, sequence, stored, speculativeState)
                        .whenComplete((done, failure) -> {
                            cache.release(stored);
                            restored = Boolean.TRUE.equals(done);
                            copyFailure = failure;
                            skip(select);
                        });
            }

            @Override
            public StepPort retired(AbstractQuantum step) throws Exception {
                if (copyFailure != null) throw rethrown(copyFailure);
                if (!restored || isStopRequested()) {
                    complete(List.of());
                    return null;
                }
                if (timing != null) timing.prefixRestored(hit.position(), System.nanoTime() - restoreStarted);
                restoredPromptTokens = hit.position();
                return speculative.start(
                        promptTokenIds,
                        maxNewTokens,
                        Call.this::speculativeToken,
                        timing,
                        Call.this::checkpoint,
                        hit.position(),
                        Call.this::speculativeEnded);
            }
        };

        /// The prefix cache's hook after a speculative chunk: stores a checkpoint when the cache wants one there.
        private void checkpoint(int end, Consumer<Throwable> done) {
            PrefixCache cache = prefixCache;
            PrefixNode attachedTo = this.cursor;
            if (end <= attachedTo.position() || !cache.wantsCheckpoint(end, this.promptTokenIds.length)) {
                done.accept(null);
                return;
            }
            cache.capture(runtime.frames(), sequence, attachedTo, this.promptTokenIds, end, this.speculativeState)
                    .whenComplete((node, failure) -> {
                        if (failure == null) this.cursor = node;
                        done.accept(failure);
                    });
        }

        private void speculativeToken(int token) {
            generatedTokenIds.add(token);
            // As in ordinary decode, a generation terminator is returned but never decoded into text.
            if (!tokenizer.isGenerationEosToken(token)) emit(this.text, decoder.append(token));
        }

        private void speculativeEnded(List<Integer> tokens) {
            promptPrefilled = true;
            if (!isStopRequested()) finishDecoder(this.text);
            complete(tokens);
        }

        // ---------------------------------------------------------------- the end

        /// A failure leaves the sequence cancelled. The session accepts another generation before the caller's
        /// result completes, or, for the blocking form, once its caller drained the text.
        @Override
        protected void ended(List<Integer> tokens, Throwable failure) {
            if (failure != null && sequence.terminalState() == Sequence.TerminalState.ACTIVE) requestCancellation();
            if (this.heldBy == null) settle();
            else this.heldBy.end(tokens, failure);
        }

        /// The session may accept another generation.
        void settle() {
            try {
                generationActive.set(false);
                if (closed.get()) completeClose();
            } finally {
                this.settled.complete(null);
            }
        }
    }

    private static Exception rethrown(Throwable failure) {
        Throwable cause = failure instanceof CompletionException wrapped && wrapped.getCause() != null
                ? wrapped.getCause()
                : failure;
        if (cause instanceof Error error) throw error;
        return cause instanceof Exception exception ? exception : new ExecutionException(cause);
    }

    private static void reportQuantum(
            GenerationTimingListener timing,
            Quantum context,
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
        if (this.sequence.terminalState() != Sequence.TerminalState.ACTIVE) {
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
