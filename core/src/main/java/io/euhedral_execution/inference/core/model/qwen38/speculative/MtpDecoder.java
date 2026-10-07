package io.euhedral_execution.inference.core.model.qwen38.speculative;

import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.generation.GenerationTimingListener;
import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.AttentionStates;
import io.euhedral_execution.inference.core.model.qwen38.Execution;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;

/// Greedy MTP speculative decoding for one sequence (docs/MTP_CONTRACT.md §4). Every output token is
/// the token ordinary greedy decode produces, and the committed state is the state it leaves: drafts
/// only decide how many tokens one verification commits.
///
/// Each step is a chain of quanta, each on the existing runtime:
/// 1. A VERIFY quantum over `[t₀, d₁ .. d_n]`: row-exact, with greedy selections and acceptance on retirement.
/// 2. A DRAFT catch-up over the committed outputs, seeded by the verified rows' hidden states. It commits
///    those MTP cache rows, and its last row drafts d′₁.
/// 3. n − 1 recursive DRAFT rows, each seeded by the previous MTP hidden.
///
/// The prompt is prefilled in chunks that seed drafting, each followed by its catch-up. Not thread-safe;
/// one decoder per sequence.
public final class MtpDecoder implements SpeculativeDecoding {

    /// Per-generation measurements. `acceptedDrafts[a]` counts verifications that accepted a drafts.
    public static final class Statistics {
        public final long[] acceptedDrafts;
        public long verifications;
        public long outputTokens;
        public long verifyNanos;
        /// MTP catch-up after verification steps; the prompt's, over every prompt token, is in
        /// `promptCatchUpNanos` (time to first token, like `prefillNanos`).
        public long catchUpNanos;
        public long promptCatchUpNanos;
        public long recursionNanos;
        public long prefillNanos;
        /// Rejections whose base token was in the draft shortlist, and outside it.
        public long rejectionsInShortlist;
        public long rejectionsOutsideShortlist;

        Statistics(int depth) {
            this.acceptedDrafts = new long[depth + 1];
        }

        public double meanAcceptedDrafts() {
            long sum = 0;
            for (int a = 0; a < this.acceptedDrafts.length; a++) sum += a * this.acceptedDrafts[a];
            return this.verifications == 0 ? 0 : (double) sum / this.verifications;
        }

        @Override
        public String toString() {
            return "verifications " + this.verifications + ", output tokens " + this.outputTokens + ", accepted drafts "
                    + Arrays.toString(this.acceptedDrafts)
                    + ", shortlist rejections in/out " + this.rejectionsInShortlist + "/"
                    + this.rejectionsOutsideShortlist
                    + String.format(
                            java.util.Locale.ROOT,
                            ", mean %.3f, verify %.2f ms, catch-up %.2f ms, recursion %.2f ms (per verification)",
                            meanAcceptedDrafts(),
                            perStep(this.verifyNanos),
                            perStep(this.catchUpNanos),
                            perStep(this.recursionNanos));
        }

        private double perStep(long nanos) {
            return this.verifications == 0 ? 0 : nanos / 1e6 / this.verifications;
        }
    }

    /// Largest MTP catch-up quantum (prompt chunks are split).
    static final int CATCH_UP_ROWS = 128;

    private final Execution runtime;
    private final ExecutionPlan plan;
    private final Sequence sequence;
    private final IntPredicate endOfGeneration;
    private final int depth;
    private final int prefillChunk;
    private final int hidden;
    private final HostLogits baseLogits;
    private final HostLogits draftLogits;
    private final int[] draftTokens;
    /// Whether each vocabulary token is in the draft head's shortlist.
    private final boolean[] inShortlist;
    private final MtpCheckpoint checkpoint;
    private Statistics statistics;
    /// The current generation's timing listener, or null.
    private GenerationTimingListener timing;

    /// The MTP strategy at `depth` drafts per verification.
    public static SpeculativeDecoding.Factory factory(int depth) {
        if (depth < 1 || depth > 7) throw new IllegalArgumentException("depth must be 1 to 7");
        return (runtime, plan, gpu, sequence, endOfGeneration, prefillChunk) ->
                new MtpDecoder(runtime, plan, gpu, sequence, endOfGeneration, depth, prefillChunk);
    }

    public MtpDecoder(
            Execution runtime,
            ExecutionPlan plan,
            ExecutionGpu gpu,
            Sequence sequence,
            IntPredicate endOfGeneration,
            int depth,
            int prefillChunk) {
        this.runtime = Objects.requireNonNull(runtime, "runtime");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.sequence = Objects.requireNonNull(sequence, "sequence");
        this.endOfGeneration = Objects.requireNonNull(endOfGeneration, "endOfGeneration");
        if (!plan.drafts()) throw new IllegalArgumentException("the plan has no MTP draft view");
        if (depth < 1 || depth > 7) throw new IllegalArgumentException("depth must be 1 to 7");
        if (prefillChunk <= 0) throw new IllegalArgumentException("prefillChunk must be positive");
        this.depth = depth;
        this.prefillChunk = prefillChunk;
        this.hidden = plan.weights().config().hiddenSize();
        this.baseLogits = new HostLogits(gpu, plan.weights().config().vocabSize());
        this.draftLogits = new HostLogits(gpu, plan.draftVocabularySize());
        this.baseLogits.selectOnDevice(true);
        this.draftLogits.selectOnDevice(true);
        this.draftTokens = draftTokenIds(gpu, plan);
        this.inShortlist = new boolean[plan.weights().config().vocabSize()];
        for (int token : this.draftTokens)
            if (token >= 0 && token < this.inShortlist.length) this.inShortlist[token] = true;
        this.checkpoint = new MtpCheckpoint(plan.weights().config());
    }

    /// MTP state in a checkpoint at `p`: the MTP cache's rows below `p - 1` and the base hidden row of `p - 1`.
    @Override
    public SpeculativeCheckpoint checkpoint() {
        return this.checkpoint;
    }

    /// The draft head's token for each of its rows (`text/draft_head_token_ids`).
    private static int[] draftTokenIds(ExecutionGpu gpu, ExecutionPlan plan) {
        TensorHandle ids = plan.weights().runtimeObjects().get("text/draft_head_token_ids");
        if (ids == null) throw new IllegalArgumentException("the model has no draft head token ids");
        int count = Math.toIntExact(ids.shape()[0]);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate((long) count * Integer.BYTES, Integer.BYTES);
            gpu.copyDeviceToHost(host, ids.deviceAddress(), host.byteSize());
            return host.toArray(ValueLayout.JAVA_INT);
        }
    }

    public Statistics statistics() {
        return this.statistics;
    }

    /// Prefills `prompt` and generates up to `maxNewTokens` tokens, exactly as greedy decode would,
    /// reporting each token as it is committed.
    public List<Integer> generate(int[] prompt, int maxNewTokens, IntConsumer onToken)
            throws InterruptedException, ExecutionException {
        return generate(prompt, maxNewTokens, onToken, null);
    }

    /// As [#generate(int[], int, IntConsumer)], reporting prefill chunks, the first token, every
    /// verification step and a final commit-only quantum to `timing` (when not null). Blocks the caller
    /// until [#generateAsync] completes.
    public List<Integer> generate(int[] prompt, int maxNewTokens, IntConsumer onToken, GenerationTimingListener timing)
            throws InterruptedException, ExecutionException {
        try {
            return generateAsync(prompt, maxNewTokens, onToken, timing).get();
        } catch (ExecutionException failure) {
            if (failure.getCause() instanceof RuntimeException runtimeFailure) throw runtimeFailure;
            if (failure.getCause() instanceof Error error) throw error;
            throw failure;
        }
    }

    /// Generates as [#generate(int[], int, IntConsumer, GenerationTimingListener)] as a chain of
    /// continuations: each quantum's outcome, on the worker that retired it, admits the next quantum.
    /// `onToken` and `timing` run on those workers, one call at a time, in generation order.
    public CompletableFuture<List<Integer>> generateAsync(
            int[] prompt, int maxNewTokens, IntConsumer onToken, GenerationTimingListener timing) {
        return generateAsync(prompt, maxNewTokens, onToken, timing, null, 0);
    }

    /// As [#generateAsync(int[], int, IntConsumer, GenerationTimingListener)] for a sequence restored from the
    /// prefix cache at `startPosition` (0 for a fresh one): its base state holds `[0, startPosition)`, its MTP
    /// cache `[0, startPosition - 1)`, and its draft seed buffer the hidden row of position `startPosition - 1`.
    /// `hooks`, when not null, is called after each prefill chunk and its MTP catch-up, before the next chunk
    /// overwrites the draft seed rows.
    @Override
    public CompletableFuture<List<Integer>> generateAsync(
            int[] prompt,
            int maxNewTokens,
            IntConsumer onToken,
            GenerationTimingListener timing,
            PrefixHooks hooks,
            int startPosition) {
        if (prompt.length == 0 || maxNewTokens <= 0) throw new IllegalArgumentException("empty generation");
        if (startPosition < 0 || startPosition >= prompt.length)
            throw new IllegalArgumentException("startPosition must lie within the prompt");
        this.statistics = new Statistics(this.depth);
        this.timing = timing;
        return new Run(prompt, maxNewTokens, onToken, timing, hooks, startPosition).start();
    }

    /// One generation's continuations. The steps are those of the sequential algorithm, in its order.
    private final class Run {
        private final int[] prompt;
        private final int maxNewTokens;
        private final IntConsumer onToken;
        private final GenerationTimingListener timing;
        private final PrefixHooks hooks;
        private final int startPosition;
        private final List<Integer> output = new ArrayList<>();
        private int first = -1;
        private int[] drafts;

        Run(
                int[] prompt,
                int maxNewTokens,
                IntConsumer onToken,
                GenerationTimingListener timing,
                PrefixHooks hooks,
                int startPosition) {
            this.prompt = prompt;
            this.maxNewTokens = maxNewTokens;
            this.onToken = onToken;
            this.timing = timing;
            this.hooks = hooks;
            this.startPosition = startPosition;
        }

        /// A restored sequence's MTP cache stops one row short of its position; its seed buffer holds that
        /// row's base hidden. Pair it with the token that follows the stored prefix, as the catch-up after
        /// that chunk would have, then prefill the rest.
        CompletableFuture<List<Integer>> start() {
            if (this.startPosition == 0) return prefill(0);
            int boundary = this.startPosition;
            long seeds = states().draftSeedRows(1, MtpDecoder.this.hidden);
            return catchUpPiece(boundary - 1, new int[] {this.prompt[boundary]}, false, seeds, 0)
                    .thenCompose(ignored -> prefill(boundary));
        }

        /// Gives the prefix cache the state after a chunk, then goes on.
        private CompletableFuture<Void> afterChunk(int offset, int end) {
            if (this.hooks == null) return CompletableFuture.completedFuture(null);
            int hidden = MtpDecoder.this.hidden;
            long seeds = states().draftSeedRows(end - offset, hidden);
            MtpDecoder.this.checkpoint.seedRow(seeds + (long) (end - offset - 1) * hidden * Short.BYTES);
            return this.hooks.afterChunk(end);
        }

        /// Prompt: prefill chunks that seed drafting, each followed by its MTP catch-up.
        CompletableFuture<List<Integer>> prefill(int offset) {
            if (offset >= this.prompt.length) return afterPrompt();
            int end = Math.min(this.prompt.length, offset + MtpDecoder.this.prefillChunk);
            boolean last = end == this.prompt.length;
            long started = System.nanoTime();
            return execute(new Quantum(
                                    MtpDecoder.this.plan,
                                    MtpDecoder.this.sequence,
                                    Quantum.ExecutionKind.PREFILL,
                                    offset,
                                    Arrays.copyOfRange(this.prompt, offset, end),
                                    last ? LogitsRequirement.LAST_TOKEN : LogitsRequirement.NONE,
                                    last ? MtpDecoder.this.baseLogits : null)
                            .seedingDraft())
                    .thenCompose(ignored -> {
                        long executed = System.nanoTime();
                        MtpDecoder.this.statistics.prefillNanos += executed - started;
                        if (this.timing != null) this.timing.prefillQuantum(started, executed, end - offset);
                        int[] next = new int[end - offset];
                        System.arraycopy(this.prompt, offset + 1, next, 0, end - offset - 1);
                        if (last) {
                            this.first = MtpDecoder.this.baseLogits.selectedToken();
                            if (this.timing != null) this.timing.firstTokenSelected(System.nanoTime(), this.first);
                            next[next.length - 1] = this.first;
                        } else next[next.length - 1] = this.prompt[end];
                        long catchingUp = System.nanoTime();
                        return catchUp(offset, next, last).thenCompose(chunkDrafts -> {
                            MtpDecoder.this.statistics.promptCatchUpNanos += System.nanoTime() - catchingUp;
                            if (last) this.drafts = chunkDrafts;
                            return afterChunk(offset, end).thenCompose(stored -> prefill(end));
                        });
                    });
        }

        private CompletableFuture<List<Integer>> afterPrompt() {
            this.output.add(this.first);
            this.onToken.accept(this.first);
            MtpDecoder.this.statistics.outputTokens++;
            if (MtpDecoder.this.endOfGeneration.test(this.first)) return CompletableFuture.completedFuture(this.output);
            if (this.maxNewTokens == 1)
                return feedFinal(this.first, this.timing).thenApply(ignored -> this.output);
            return verify(this.first);
        }

        /// One verification of `current` and the drafts, then the catch-up and drafting of the next step.
        private CompletableFuture<List<Integer>> verify(int current) {
            int[] rows = new int[MtpDecoder.this.depth + 1];
            rows[0] = current;
            System.arraycopy(this.drafts, 0, rows, 1, MtpDecoder.this.depth);
            long position = MtpDecoder.this.sequence.currentTokenPosition();
            var acceptance = new SpeculativeAcceptance(
                    rows, MtpDecoder.this.endOfGeneration, this.maxNewTokens - this.output.size());
            long started = System.nanoTime();
            return execute(new Quantum(
                                    MtpDecoder.this.plan,
                                    MtpDecoder.this.sequence,
                                    Quantum.ExecutionKind.VERIFY,
                                    position,
                                    rows,
                                    LogitsRequirement.ALL_TOKENS,
                                    MtpDecoder.this.baseLogits)
                            .withAcceptance(acceptance)
                            .seedingDraft())
                    .thenCompose(ignored -> {
                        Statistics statistics = MtpDecoder.this.statistics;
                        long executed = System.nanoTime();
                        statistics.verifyNanos += executed - started;
                        statistics.verifications++;
                        statistics.acceptedDrafts[acceptance.acceptedDrafts()]++;
                        int[] committed = acceptance.outputs();
                        int rejected = acceptance.rejectedBaseToken();
                        boolean[] shortlist = MtpDecoder.this.inShortlist;
                        int rejection = rejected < 0 ? -1 : rejected < shortlist.length && shortlist[rejected] ? 0 : 1;
                        if (rejection == 0) statistics.rejectionsInShortlist++;
                        if (rejection == 1) statistics.rejectionsOutsideShortlist++;
                        if (this.timing != null)
                            this.timing.speculativeStep(
                                    started, executed, committed.length, acceptance.acceptedDrafts(), rejection);
                        for (int token : committed) {
                            this.output.add(token);
                            this.onToken.accept(token);
                        }
                        statistics.outputTokens += committed.length;
                        int next = committed[committed.length - 1];
                        if (MtpDecoder.this.endOfGeneration.test(next))
                            return CompletableFuture.completedFuture(this.output);
                        if (this.output.size() >= this.maxNewTokens)
                            return feedFinal(next, this.timing).thenApply(done -> this.output);
                        // Discard the previous step's recursive draft rows, then catch the MTP cache up.
                        mtpCache().truncate(Math.toIntExact(position));
                        return catchUp(position, committed, true).thenCompose(drafts -> {
                            this.drafts = drafts;
                            return verify(next);
                        });
                    });
        }
    }

    /// MTP catch-up over the base hidden rows just seeded, paired with `tokens` (the token each row
    /// predicted), at MTP positions from `position`. When `draft` is set, its last row drafts d₁ and the
    /// recursive rows draft the rest; the future then holds the drafts, else null.
    private CompletableFuture<int[]> catchUp(long position, int[] tokens, boolean draft) {
        AttentionStates states = states();
        long seeds = states.draftSeedRows(tokens.length, this.hidden);
        long started = System.nanoTime();
        boolean prompt = this.statistics.outputTokens == 0;
        return catchUpPiece(position, tokens, draft, seeds, 0).thenCompose(ignored -> {
            long caughtUp = System.nanoTime();
            if (!prompt) this.statistics.catchUpNanos += caughtUp - started;
            if (this.timing != null)
                this.timing.draftQuantum(prompt ? "prompt-catch-up" : "catch-up", started, caughtUp);
            if (!draft) return CompletableFuture.completedFuture(null);
            int[] drafts = new int[this.depth];
            drafts[0] = this.draftTokens[this.draftLogits.selectedToken()];
            long recursion = System.nanoTime();
            return recurse(position + tokens.length, drafts, 1, states).thenApply(done -> {
                long recursed = System.nanoTime();
                if (!prompt) this.statistics.recursionNanos += recursed - recursion;
                if (this.timing != null && this.depth > 1) this.timing.draftQuantum("recursion", recursion, recursed);
                return drafts;
            });
        });
    }

    /// Pieces of at most CATCH_UP_ROWS rows: the draft view's workspace is retained at the largest quantum it
    /// ran, and MTP cache appends are contiguous, so the pieces equal one catch-up.
    private CompletableFuture<Void> catchUpPiece(long position, int[] tokens, boolean draft, long seeds, int first) {
        if (first >= tokens.length) return CompletableFuture.completedFuture(null);
        int count = Math.min(CATCH_UP_ROWS, tokens.length - first);
        boolean last = first + count == tokens.length;
        return execute(new Quantum(
                                this.plan,
                                this.sequence,
                                Quantum.ExecutionKind.DRAFT,
                                position + first,
                                Arrays.copyOfRange(tokens, first, first + count),
                                draft && last ? LogitsRequirement.LAST_TOKEN : LogitsRequirement.NONE,
                                draft && last ? this.draftLogits : null)
                        .withDraftSeed(seeds + (long) first * this.hidden * Short.BYTES, count))
                .thenCompose(ignored -> catchUpPiece(position, tokens, draft, seeds, first + count));
    }

    /// Recursive draft rows `index` and on, each seeded by the MTP's own hidden of the previous row.
    private CompletableFuture<Void> recurse(long base, int[] drafts, int index, AttentionStates states) {
        if (index >= this.depth) return CompletableFuture.completedFuture(null);
        return execute(new Quantum(
                                this.plan,
                                this.sequence,
                                Quantum.ExecutionKind.DRAFT,
                                base + index - 1,
                                new int[] {drafts[index - 1]},
                                LogitsRequirement.LAST_TOKEN,
                                this.draftLogits)
                        .withDraftSeed(states.draftRecursionHidden(this.hidden), 1))
                .thenCompose(ignored -> {
                    drafts[index] = this.draftTokens[this.draftLogits.selectedToken()];
                    return recurse(base, drafts, index + 1, states);
                });
    }

    /// Ordinary decode feeds the last allowed token without sampling; so does the decoder, so both leave
    /// the same state.
    private CompletableFuture<Void> feedFinal(int token, GenerationTimingListener timing) {
        long started = System.nanoTime();
        return execute(new Quantum(
                        this.plan,
                        this.sequence,
                        Quantum.ExecutionKind.DECODE,
                        this.sequence.currentTokenPosition(),
                        new int[] {token},
                        LogitsRequirement.NONE))
                .thenApply(ignored -> {
                    long executed = System.nanoTime();
                    if (timing != null) timing.decodeQuantum(started, executed, executed, false, -1);
                    return null;
                });
    }

    private AttentionStates states() {
        return (AttentionStates) this.sequence.kvCacheState();
    }

    private AttentionKvState mtpCache() {
        return states().forLayer(this.plan.weights().config().numHiddenLayers());
    }

    /// Admits `context`; the future completes on the worker that retired it, failing unless it succeeded.
    private CompletableFuture<Void> execute(Quantum context) {
        CompletableFuture<Quantum.Outcome> outcome;
        try {
            outcome = this.runtime.submit(context);
        } catch (RuntimeException | Error failure) {
            return CompletableFuture.failedFuture(failure);
        }
        return outcome.thenApply(completed -> {
            if (completed.status() != Quantum.Status.SUCCESS)
                throw new IllegalStateException("speculative quantum " + completed.status(), completed.failure());
            return null;
        });
    }

    @Override
    public void close() {
        this.baseLogits.close();
        this.draftLogits.close();
    }
}
