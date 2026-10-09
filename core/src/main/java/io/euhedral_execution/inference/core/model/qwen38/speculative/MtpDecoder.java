package io.euhedral_execution.inference.core.model.qwen38.speculative;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.generation.Generation;
import io.euhedral_execution.inference.core.generation.GenerationFrames;
import io.euhedral_execution.inference.core.generation.GenerationTimingListener;
import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.generation.StepPort;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.AttentionStates;
import io.euhedral_execution.inference.core.model.qwen38.Execution;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;

/// Greedy MTP speculative decoding for one sequence (docs/MTP_CONTRACT.md §4). Every output token is
/// the token ordinary greedy decode produces, and the committed state is the state it leaves: drafts
/// only decide how many tokens one verification commits.
///
/// Each step is a sequence of quanta, run as generation frames through the decoder's step ports:
/// 1. A VERIFY quantum over `[t₀, d₁ .. d_n]`: row-exact, with greedy selections and acceptance on retirement.
/// 2. A DRAFT catch-up over the committed outputs, seeded by the verified rows' hidden states. It commits
///    those MTP cache rows, and its last row drafts d′₁.
/// 3. n − 1 recursive DRAFT rows, each seeded by the previous MTP hidden.
///
/// The prompt is prefilled in chunks that seed drafting, each followed by its catch-up. The chunks before the last
/// run ahead ([Run#ahead]): they are admitted with their catch-ups at once, up to the next prefix checkpoint, and
/// the sequence's carried state (the seed rows, the MTP cache) orders them. Not thread-safe; one decoder per
/// sequence.
public final class MtpDecoder implements SpeculativeDecoding {

    /// Per-generation measurements. `acceptedDrafts[a]` counts verifications that accepted a drafts.
    public static final class Statistics {
        public final long[] acceptedDrafts;
        /// `draftLengths[n]` counts verifications that checked n drafts.
        public final long[] draftLengths;
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
            this.draftLengths = new long[depth + 1];
        }

        public double meanDraftLength() {
            long sum = 0;
            for (int n = 0; n < this.draftLengths.length; n++) sum += n * this.draftLengths[n];
            return this.verifications == 0 ? 0 : (double) sum / this.verifications;
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
                    + ", draft lengths "
                    + Arrays.toString(this.draftLengths)
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

    /// Sees each verified step (tests and drafting-quality measurements): the position of the step's first row, its
    /// token, every draft the step drafted with the natural log of the probability the draft head gave it (over the
    /// head's shortlist), and the drafts the verification accepted.
    public interface StepListener {
        void verified(long position, int current, int[] drafts, float[] draftLogProbabilities, int acceptedDrafts);
    }

    /// Most drafts a step drafts; at most [#MAX_VERIFIED] of them are verified.
    public static final int MAX_DRAFTS = 16;

    /// Most drafts one verification checks.
    public static final int MAX_VERIFIED = 7;

    /// Largest MTP catch-up quantum (prompt chunks are split).
    static final int CATCH_UP_ROWS = 128;

    private final Execution runtime;
    private final ExecutionPlan plan;
    private final Sequence sequence;
    private final IntPredicate endOfGeneration;
    private final DraftLength length;
    /// Drafts each verification checks at most: the first `verified` of the step's drafts.
    private final int verified;
    private final int prefillChunk;
    private final int hidden;
    private final HostLogits baseLogits;
    private final HostLogits draftLogits;
    private final int[] draftTokens;
    /// Whether each vocabulary token is in the draft head's shortlist.
    private final boolean[] inShortlist;
    private final MtpCheckpoint checkpoint;
    private Statistics statistics;
    private StepListener steps;
    /// Whether each step commits one token, so that every output position starts a step.
    private boolean everyPosition;
    /// The latest draft's log-probability, when a listener reads the draft rows on the host.
    private float draftLogProbability;
    /// The current generation's timing listener, or null.
    private GenerationTimingListener timing;

    /// The MTP strategy drafting `length` tokens per verification, all of them verified.
    public static SpeculativeDecoding.Factory factory(DraftLength length) {
        if (length.most() > MAX_VERIFIED)
            throw new IllegalArgumentException("a verification checks at most " + MAX_VERIFIED + " drafts");
        return (runtime, plan, gpu, sequence, endOfGeneration, prefillChunk) ->
                new MtpDecoder(runtime, plan, gpu, sequence, endOfGeneration, length, prefillChunk, length.most());
    }

    public MtpDecoder(
            Execution runtime,
            ExecutionPlan plan,
            ExecutionGpu gpu,
            Sequence sequence,
            IntPredicate endOfGeneration,
            int depth,
            int prefillChunk) {
        this(runtime, plan, gpu, sequence, endOfGeneration, DraftLength.fixed(depth), prefillChunk, depth);
    }

    /// As the public constructor, drafting `length` tokens per step (up to [#MAX_DRAFTS]) and verifying only the first
    /// `verified` of them (up to [#MAX_VERIFIED]); drafts past them reach only the step listener.
    public MtpDecoder(
            Execution runtime,
            ExecutionPlan plan,
            ExecutionGpu gpu,
            Sequence sequence,
            IntPredicate endOfGeneration,
            DraftLength length,
            int prefillChunk,
            int verified) {
        this.runtime = Objects.requireNonNull(runtime, "runtime");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.sequence = Objects.requireNonNull(sequence, "sequence");
        this.endOfGeneration = Objects.requireNonNull(endOfGeneration, "endOfGeneration");
        if (!plan.drafts()) throw new IllegalArgumentException("the plan has no MTP draft view");
        this.length = Objects.requireNonNull(length, "length");
        if (verified < 1 || verified > Math.min(length.most(), MAX_VERIFIED))
            throw new IllegalArgumentException("verified must be 1 to min(most drafts, " + MAX_VERIFIED + ")");
        if (prefillChunk <= 0) throw new IllegalArgumentException("prefillChunk must be positive");
        this.verified = verified;
        this.prefillChunk = prefillChunk;
        this.hidden = plan.weights().config().hiddenSize();
        this.baseLogits = new HostLogits(gpu, plan.weights().config().vocabSize());
        this.draftLogits = new HostLogits(gpu, plan.draftVocabularySize());
        this.baseLogits.selectOnDevice(true);
        this.draftLogits.selectOnDevice(true);
        this.draftLogits.scoreOnDevice(length.gated());
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

    /// Reports every verified step to `listener` (null stops it). The draft rows then come back to the host, which
    /// selects each draft as the device would and scores it. With `everyPosition`, each step commits only its first
    /// output, so every output position starts a step: a measurement of the drafts from each position, at the
    /// cost of one verification per token. The output is greedy decode's either way. Set between generations.
    public void observe(StepListener listener, boolean everyPosition) {
        this.steps = listener;
        this.everyPosition = listener != null && everyPosition;
        this.draftLogits.selectOnDevice(listener == null);
    }

    /// The draft head's row of the draft quantum that just retired: its greedy choice, as the device argmax makes it
    /// (the lowest row among equal maxima). Its log-softmax at that row is kept in `draftLogProbability`: from the
    /// device when the draft length is gated (NaN when unscored), or from the host row with a listener.
    private int draftRow() {
        if (this.draftLogits.hasSelection()) {
            this.draftLogProbability = this.draftLogits.selectedLogProbability();
            return this.draftLogits.selectedToken();
        }
        MemorySegment row = this.draftLogits.row();
        int count = this.draftLogits.vocabularySize();
        int best = -1;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) {
            float value = bf16(row.getAtIndex(ValueLayout.JAVA_SHORT, i));
            if (value > max) {
                max = value;
                best = i;
            }
        }
        if (best < 0) throw new IllegalArgumentException("draft logit row has no selectable token");
        double sum = 0;
        for (int i = 0; i < count; i++) sum += Math.exp(bf16(row.getAtIndex(ValueLayout.JAVA_SHORT, i)) - max);
        this.draftLogProbability = (float) -Math.log(sum);
        return best;
    }

    private static float bf16(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }

    /// Prefills `prompt` and generates up to `maxNewTokens` tokens, exactly as greedy decode would,
    /// reporting each token as it is committed.
    public List<Integer> generate(int[] prompt, int maxNewTokens, IntConsumer onToken)
            throws InterruptedException, ExecutionException {
        return generate(prompt, maxNewTokens, onToken, null);
    }

    /// As [#generate(int[], int, IntConsumer)], reporting prefill chunks, the first token, every
    /// verification step and a final commit-only quantum to `timing` (when not null). The steps run as
    /// generation frames on the lattice; the calling thread waits for them (tests and tools).
    public List<Integer> generate(int[] prompt, int maxNewTokens, IntConsumer onToken, GenerationTimingListener timing)
            throws InterruptedException, ExecutionException {
        Generation generation = new Generation(new GenerationFrames(this.runtime.lake()));
        StepPort first;
        try {
            first = start(prompt, maxNewTokens, onToken, timing, null, 0, generation::complete);
        } catch (RuntimeException | Error refused) {
            generation.fail(refused);
            generation.finishNow();
            throw refused;
        }
        generation.start(first);
        try {
            return generation.result().get();
        } catch (ExecutionException failure) {
            if (failure.getCause() instanceof RuntimeException runtimeFailure) throw runtimeFailure;
            if (failure.getCause() instanceof Error error) throw error;
            throw failure;
        }
    }

    /// The first step of a generation for a sequence restored from the prefix cache at `startPosition` (0 for a
    /// fresh one): its base state holds `[0, startPosition)`, its MTP cache `[0, startPosition - 1)`, and its draft
    /// seed buffer the hidden row of position `startPosition - 1`. `hooks`, when not null, is called after each
    /// prefill chunk and its MTP catch-up, before the next chunk overwrites the draft seed rows.
    @Override
    public StepPort start(
            int[] prompt,
            int maxNewTokens,
            IntConsumer onToken,
            GenerationTimingListener timing,
            PrefixHooks hooks,
            int startPosition,
            Consumer<List<Integer>> ended) {
        if (prompt.length == 0 || maxNewTokens <= 0) throw new IllegalArgumentException("empty generation");
        if (startPosition < 0 || startPosition >= prompt.length)
            throw new IllegalArgumentException("startPosition must lie within the prompt");
        this.statistics = new Statistics(this.verified);
        this.timing = timing;
        return new Run(prompt, maxNewTokens, onToken, timing, hooks, startPosition, ended).first();
    }

    /// One generation's steps, in the order of the sequential algorithm: each port admits one quantum and, when it
    /// retired, names the next.
    private final class Run {
        /// What a finished catch-up goes on to.
        private enum Then {
            PREFILL_AT_BOUNDARY,
            AFTER_CHUNK,
            VERIFY
        }

        private final int[] prompt;
        private final int maxNewTokens;
        private final IntConsumer onToken;
        private final GenerationTimingListener timing;
        private final PrefixHooks hooks;
        private final int startPosition;
        private final Consumer<List<Integer>> ended;
        private final List<Integer> output = new ArrayList<>();
        private int first = -1;
        private int[] drafts;
        private float[] draftLogProbabilities;

        // The prefill chunk in flight.
        private int offset;
        private int end;
        private boolean last;
        private long started;
        private long promptCatchUpStarted;

        // The run-ahead in flight: chunks [offset, aheadEnd) with their catch-ups.
        private int aheadEnd;
        private Throwable aheadRefusal;
        private Throwable drainFailure;

        // The catch-up in flight.
        private long catchUpPosition;
        private int[] catchUpTokens;
        private boolean catchUpDraft;
        private long catchUpSeeds;
        private int catchUpFirst;
        private int catchUpCount;
        private boolean catchUpTimed;
        private boolean catchUpOfPrompt;
        private long catchUpStarted;
        private Then then;

        // The recursion in flight.
        private int[] recursionDrafts;
        private float[] recursionLogProbabilities;
        private int recursionIndex;
        private double recursionScore;
        private long recursionBase;
        private long recursionStarted;
        private AttentionStates recursionStates;

        // The verification in flight.
        private int current;
        private int[] rows;
        private long position;
        private SpeculativeAcceptance acceptance;
        private int finalToken;
        private volatile Throwable captureFailure;

        Run(
                int[] prompt,
                int maxNewTokens,
                IntConsumer onToken,
                GenerationTimingListener timing,
                PrefixHooks hooks,
                int startPosition,
                Consumer<List<Integer>> ended) {
            this.prompt = prompt;
            this.maxNewTokens = maxNewTokens;
            this.onToken = onToken;
            this.timing = timing;
            this.hooks = hooks;
            this.startPosition = startPosition;
            this.ended = ended;
        }

        /// A restored sequence's MTP cache stops one row short of its position; its seed buffer holds that row's
        /// base hidden. Pair it with the token that follows the stored prefix, as the catch-up after that chunk
        /// would have, then prefill the rest.
        StepPort first() {
            if (this.startPosition == 0) return prefillFrom(0);
            this.offset = this.startPosition;
            long seeds = states().draftSeedRows(1, hidden);
            return catchUp(
                    this.startPosition - 1,
                    new int[] {this.prompt[this.startPosition]},
                    false,
                    seeds,
                    false,
                    Then.PREFILL_AT_BOUNDARY);
        }

        private StepPort prefillFrom(int from) {
            this.offset = from;
            if (this.offset >= this.prompt.length) return afterPrompt();
            int lastStart = from + (this.prompt.length - from - 1) / prefillChunk * prefillChunk;
            if (from == lastStart) return this.prefill;
            // Up to the last chunk, whose prefill fills the host row, or to the first checkpoint the cache wants.
            int stop = from + prefillChunk;
            while (stop < lastStart && (this.hooks == null || !this.hooks.wants(stop))) stop += prefillChunk;
            this.aheadEnd = stop;
            return this.ahead;
        }

        /// Prompt chunks before the last, each with its catch-up, admitted at once. A chunk's seed-row
        /// writer follows the previous catch-up's readers through the carried seed rows, and the rest of
        /// the chunk runs ahead; a catch-up finds its seed rows when its stem runs. Only the last
        /// catch-up's conclusion comes back: the sequence concludes the quanta in admission order, and
        /// one that failed blocks the rest.
        private final StepPort ahead = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                started = System.nanoTime();
                aheadRefusal = null;
                for (int at = offset; at < aheadEnd; at += prefillChunk) {
                    int to = at + prefillChunk;
                    refused(runtime.admitAhead(new Quantum(
                                    plan,
                                    sequence,
                                    Quantum.ExecutionKind.PREFILL,
                                    at,
                                    Arrays.copyOfRange(prompt, at, to),
                                    LogitsRequirement.NONE)
                            .seedingDraft()));
                    // Each row pairs with the token that follows it.
                    for (int first = 0; first < prefillChunk; first += CATCH_UP_ROWS) {
                        int count = Math.min(CATCH_UP_ROWS, prefillChunk - first);
                        Quantum catchUp = new Quantum(
                                        plan,
                                        sequence,
                                        Quantum.ExecutionKind.DRAFT,
                                        at + first,
                                        Arrays.copyOfRange(prompt, at + first + 1, at + first + 1 + count),
                                        LogitsRequirement.NONE)
                                .withSeedRows(first, count);
                        if (to == aheadEnd && first + count == prefillChunk) runtime.admit(catchUp, select);
                        else refused(runtime.admitAhead(catchUp));
                    }
                }
            }

            @Override
            public StepPort retired(AbstractQuantum step) {
                // A refused last catch-up leaves the earlier quanta in flight: the run ends after them.
                if (sequence.inFlight()) {
                    drainFailure = ((Quantum) step).conclusion().failure();
                    return drain;
                }
                if (aheadRefusal != null) {
                    // The refusal is the cause; what the later quanta then failed on follows from it.
                    var refused = new IllegalStateException("a prompt chunk was refused", aheadRefusal);
                    Throwable later = ((Quantum) step).conclusion().failure();
                    if (later != null) refused.addSuppressed(later);
                    throw refused;
                }
                // A failure of an earlier quantum failed the sequence first; the later ones failed after it.
                Throwable first = sequence.terminalFailure();
                Quantum.Outcome outcome = ((Quantum) step).conclusion();
                if (outcome.status() != Quantum.Status.SUCCESS && first != null && first != outcome.failure())
                    throw new IllegalStateException("a prompt chunk failed", first);
                succeeded(step);
                if (sequence.committedFrontier() != aheadEnd || mtpCache().length() != aheadEnd)
                    throw new IllegalStateException("the prompt's chunks stopped short of " + aheadEnd);
                long executed = System.nanoTime();
                statistics.prefillNanos += executed - started;
                if (timing != null) timing.prefillQuantum(started, executed, aheadEnd - offset);
                end = aheadEnd;
                if (hooks == null || !hooks.wants(end)) return prefillFrom(end);
                checkpoint.seedRow(states().seedRows() + (long) (prefillChunk - 1) * hidden * Short.BYTES);
                return capture;
            }
        };

        private void refused(Throwable refusal) {
            if (refusal == null) return;
            if (this.aheadRefusal == null) this.aheadRefusal = refusal;
            else if (this.aheadRefusal != refusal) this.aheadRefusal.addSuppressed(refusal);
        }

        /// Waits, one thrown frame at a time, for quanta a refused step left in flight, then fails the run.
        private final StepPort drain = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                runtime.lake().publishOrRun(select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) {
                if (sequence.inFlight()) return this;
                throw new IllegalStateException("a speculative quantum was refused", drainFailure);
            }
        };

        /// Prompt: a prefill chunk that seeds drafting; its MTP catch-up follows.
        private final StepPort prefill = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                end = Math.min(prompt.length, offset + prefillChunk);
                last = end == prompt.length;
                started = System.nanoTime();
                runtime.admit(
                        new Quantum(
                                        plan,
                                        sequence,
                                        Quantum.ExecutionKind.PREFILL,
                                        offset,
                                        Arrays.copyOfRange(prompt, offset, end),
                                        last ? LogitsRequirement.LAST_TOKEN : LogitsRequirement.NONE,
                                        last ? baseLogits : null)
                                .seedingDraft(),
                        select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) {
                succeeded(step);
                long executed = System.nanoTime();
                statistics.prefillNanos += executed - started;
                if (timing != null) timing.prefillQuantum(started, executed, end - offset);
                int[] next = new int[end - offset];
                System.arraycopy(prompt, offset + 1, next, 0, end - offset - 1);
                if (last) {
                    first = baseLogits.selectedToken();
                    if (timing != null) timing.firstTokenSelected(System.nanoTime(), first);
                    next[next.length - 1] = first;
                } else next[next.length - 1] = prompt[end];
                promptCatchUpStarted = System.nanoTime();
                return catchUp(offset, next, last, states().draftSeedRows(next.length, hidden), true, Then.AFTER_CHUNK);
            }
        };

        /// MTP catch-up over the base hidden rows just seeded, paired with `tokens` (the token each row
        /// predicted), at MTP positions from `position`, in pieces of at most CATCH_UP_ROWS rows: the
        /// draft view's workspace is retained at the largest quantum it ran, and MTP cache appends are
        /// contiguous, so the pieces equal one catch-up. With `draft`, its last row drafts d₁ and the
        /// recursive rows draft the rest.
        private StepPort catchUp(long position, int[] tokens, boolean draft, long seeds, boolean timed, Then then) {
            this.catchUpPosition = position;
            this.catchUpTokens = tokens;
            this.catchUpDraft = draft;
            this.catchUpSeeds = seeds;
            this.catchUpFirst = 0;
            this.catchUpTimed = timed;
            this.catchUpOfPrompt = statistics.outputTokens == 0;
            this.catchUpStarted = System.nanoTime();
            this.then = then;
            return this.catchUpPiece;
        }

        private final StepPort catchUpPiece = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                catchUpCount = Math.min(CATCH_UP_ROWS, catchUpTokens.length - catchUpFirst);
                boolean lastPiece = catchUpFirst + catchUpCount == catchUpTokens.length;
                runtime.admit(
                        new Quantum(
                                        plan,
                                        sequence,
                                        Quantum.ExecutionKind.DRAFT,
                                        catchUpPosition + catchUpFirst,
                                        Arrays.copyOfRange(catchUpTokens, catchUpFirst, catchUpFirst + catchUpCount),
                                        catchUpDraft && lastPiece
                                                ? LogitsRequirement.LAST_TOKEN
                                                : LogitsRequirement.NONE,
                                        catchUpDraft && lastPiece ? draftLogits : null)
                                .withDraftSeed(catchUpSeeds + (long) catchUpFirst * hidden * Short.BYTES, catchUpCount),
                        select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) {
                succeeded(step);
                catchUpFirst += catchUpCount;
                return catchUpFirst < catchUpTokens.length ? this : caughtUp();
            }
        };

        private StepPort caughtUp() {
            if (!this.catchUpTimed) return afterCatchUp(null);
            long caughtUpAt = System.nanoTime();
            if (!this.catchUpOfPrompt) statistics.catchUpNanos += caughtUpAt - this.catchUpStarted;
            if (this.timing != null)
                this.timing.draftQuantum(
                        this.catchUpOfPrompt ? "prompt-catch-up" : "catch-up", this.catchUpStarted, caughtUpAt);
            if (!this.catchUpDraft) return afterCatchUp(null);
            this.recursionDrafts = new int[length.most()];
            this.recursionLogProbabilities = new float[length.most()];
            this.recursionDrafts[0] = draftTokens[draftRow()];
            this.recursionLogProbabilities[0] = draftLogProbability;
            this.recursionScore = draftLogProbability;
            this.recursionIndex = 1;
            this.recursionBase = this.catchUpPosition + this.catchUpTokens.length;
            this.recursionStates = states();
            this.recursionStarted = System.nanoTime();
            return length.continues(this.recursionIndex, this.recursionScore) ? this.recurse : recursed();
        }

        /// Recursive draft rows, each seeded by the MTP's own hidden of the previous row.
        private final StepPort recurse = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                runtime.admit(
                        new Quantum(
                                        plan,
                                        sequence,
                                        Quantum.ExecutionKind.DRAFT,
                                        recursionBase + recursionIndex - 1,
                                        new int[] {recursionDrafts[recursionIndex - 1]},
                                        LogitsRequirement.LAST_TOKEN,
                                        draftLogits)
                                .withDraftSeed(recursionStates.draftRecursionHidden(hidden), 1),
                        select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) {
                succeeded(step);
                recursionDrafts[recursionIndex] = draftTokens[draftRow()];
                recursionLogProbabilities[recursionIndex] = draftLogProbability;
                recursionScore += draftLogProbability;
                recursionIndex++;
                return length.continues(recursionIndex, recursionScore) ? this : recursed();
            }
        };

        private StepPort recursed() {
            long recursedAt = System.nanoTime();
            if (!this.catchUpOfPrompt) statistics.recursionNanos += recursedAt - this.recursionStarted;
            if (this.timing != null && this.recursionIndex > 1)
                this.timing.draftQuantum("recursion", this.recursionStarted, recursedAt);
            this.recursionLogProbabilities = Arrays.copyOf(this.recursionLogProbabilities, this.recursionIndex);
            return afterCatchUp(Arrays.copyOf(this.recursionDrafts, this.recursionIndex));
        }

        private StepPort afterCatchUp(int[] newDrafts) {
            switch (this.then) {
                case PREFILL_AT_BOUNDARY:
                    return prefillFrom(this.offset);
                case AFTER_CHUNK:
                    statistics.promptCatchUpNanos += System.nanoTime() - this.promptCatchUpStarted;
                    if (this.last) {
                        this.drafts = newDrafts;
                        this.draftLogProbabilities = this.recursionLogProbabilities;
                    }
                    if (this.hooks == null || !this.hooks.wants(this.end)) return prefillFrom(this.end);
                    // Gives the prefix cache the state after the chunk, before the next chunk overwrites the seeds.
                    long seeds = states().draftSeedRows(this.end - this.offset, hidden);
                    checkpoint.seedRow(seeds + (long) (this.end - this.offset - 1) * hidden * Short.BYTES);
                    return this.capture;
                default:
                    this.drafts = newDrafts;
                    this.draftLogProbabilities = this.recursionLogProbabilities;
                    return this.verify;
            }
        }

        private final StepPort capture = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                hooks.afterChunk(end, failure -> captureFailure = failure, select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) {
                Throwable failure = captureFailure;
                captureFailure = null;
                if (failure != null) throw new IllegalStateException("the prefix checkpoint failed", failure);
                return prefillFrom(end);
            }
        };

        private StepPort afterPrompt() {
            this.output.add(this.first);
            this.onToken.accept(this.first);
            statistics.outputTokens++;
            if (endOfGeneration.test(this.first)) return done();
            if (this.maxNewTokens == 1) {
                this.finalToken = this.first;
                return this.feedFinal;
            }
            this.current = this.first;
            return this.verify;
        }

        /// One verification of `current` and the drafts, then the catch-up and drafting of the next step.
        private final StepPort verify = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                int checked = Math.min(drafts.length, verified);
                rows = new int[checked + 1];
                rows[0] = current;
                System.arraycopy(drafts, 0, rows, 1, checked);
                position = sequence.currentTokenPosition();
                int budget = maxNewTokens - output.size();
                acceptance = new SpeculativeAcceptance(rows, endOfGeneration, everyPosition ? 1 : budget);
                started = System.nanoTime();
                runtime.admit(
                        new Quantum(
                                        plan,
                                        sequence,
                                        Quantum.ExecutionKind.VERIFY,
                                        position,
                                        rows,
                                        LogitsRequirement.ALL_TOKENS,
                                        baseLogits)
                                .withAcceptance(acceptance)
                                .seedingDraft(),
                        select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) {
                succeeded(step);
                Statistics statistics = MtpDecoder.this.statistics;
                long executed = System.nanoTime();
                statistics.verifyNanos += executed - started;
                statistics.verifications++;
                statistics.acceptedDrafts[acceptance.acceptedDrafts()]++;
                statistics.draftLengths[rows.length - 1]++;
                int[] committed = acceptance.outputs();
                int rejected = acceptance.rejectedBaseToken();
                boolean[] shortlist = inShortlist;
                int rejection = rejected < 0 ? -1 : rejected < shortlist.length && shortlist[rejected] ? 0 : 1;
                if (rejection == 0) statistics.rejectionsInShortlist++;
                if (rejection == 1) statistics.rejectionsOutsideShortlist++;
                if (timing != null)
                    timing.speculativeStep(started, executed, committed.length, acceptance.acceptedDrafts(), rejection);
                if (steps != null)
                    steps.verified(
                            position,
                            current,
                            drafts.clone(),
                            draftLogProbabilities.clone(),
                            acceptance.acceptedDrafts());
                for (int token : committed) {
                    output.add(token);
                    onToken.accept(token);
                }
                statistics.outputTokens += committed.length;
                int next = committed[committed.length - 1];
                if (endOfGeneration.test(next)) return done();
                if (output.size() >= maxNewTokens) {
                    finalToken = next;
                    return feedFinal;
                }
                // Discard the previous step's recursive draft rows, then catch the MTP cache up.
                mtpCache().truncate(Math.toIntExact(position));
                current = next;
                return catchUp(
                        position, committed, true, states().draftSeedRows(committed.length, hidden), true, Then.VERIFY);
            }
        };

        /// Ordinary decode feeds the last allowed token without sampling; so does the decoder, so both leave the
        /// same state.
        private final StepPort feedFinal = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                started = System.nanoTime();
                runtime.admit(
                        new Quantum(
                                plan,
                                sequence,
                                Quantum.ExecutionKind.DECODE,
                                sequence.currentTokenPosition(),
                                new int[] {finalToken},
                                LogitsRequirement.NONE),
                        select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) {
                succeeded(step);
                long executed = System.nanoTime();
                if (timing != null) timing.decodeQuantum(started, executed, executed, false, -1);
                return done();
            }
        };

        private StepPort done() {
            this.ended.accept(this.output);
            return null;
        }
    }

    /// A speculative step must succeed: anything else ends the run.
    private static void succeeded(AbstractQuantum step) {
        Quantum.Outcome outcome = ((Quantum) step).conclusion();
        if (outcome.status() != Quantum.Status.SUCCESS)
            throw new IllegalStateException("speculative quantum " + outcome.status(), outcome.failure());
    }

    private AttentionStates states() {
        return (AttentionStates) this.sequence.kvCacheState();
    }

    private AttentionKvState mtpCache() {
        return states().forLayer(this.plan.weights().config().numHiddenLayers());
    }

    @Override
    public void close() {
        this.baseLogits.close();
        this.draftLogits.close();
    }
}
