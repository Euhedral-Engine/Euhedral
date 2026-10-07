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
        this.statistics = new Statistics(this.depth);
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

        // The prefill chunk in flight.
        private int offset;
        private int end;
        private boolean last;
        private long started;
        private long promptCatchUpStarted;

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
        private int recursionIndex;
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
            return this.offset >= this.prompt.length ? afterPrompt() : this.prefill;
        }

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
            this.recursionDrafts = new int[depth];
            this.recursionDrafts[0] = draftTokens[draftLogits.selectedToken()];
            this.recursionIndex = 1;
            this.recursionBase = this.catchUpPosition + this.catchUpTokens.length;
            this.recursionStates = states();
            this.recursionStarted = System.nanoTime();
            return this.recursionIndex < depth ? this.recurse : recursed();
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
                recursionDrafts[recursionIndex] = draftTokens[draftLogits.selectedToken()];
                recursionIndex++;
                return recursionIndex < depth ? this : recursed();
            }
        };

        private StepPort recursed() {
            long recursedAt = System.nanoTime();
            if (!this.catchUpOfPrompt) statistics.recursionNanos += recursedAt - this.recursionStarted;
            if (this.timing != null && depth > 1)
                this.timing.draftQuantum("recursion", this.recursionStarted, recursedAt);
            return afterCatchUp(this.recursionDrafts);
        }

        private StepPort afterCatchUp(int[] newDrafts) {
            switch (this.then) {
                case PREFILL_AT_BOUNDARY:
                    return prefillFrom(this.offset);
                case AFTER_CHUNK:
                    statistics.promptCatchUpNanos += System.nanoTime() - this.promptCatchUpStarted;
                    if (this.last) this.drafts = newDrafts;
                    if (this.hooks == null) return prefillFrom(this.end);
                    // Gives the prefix cache the state after the chunk, before the next chunk overwrites the seeds.
                    long seeds = states().draftSeedRows(this.end - this.offset, hidden);
                    checkpoint.seedRow(seeds + (long) (this.end - this.offset - 1) * hidden * Short.BYTES);
                    return this.capture;
                default:
                    this.drafts = newDrafts;
                    return this.verify;
            }
        }

        private final StepPort capture = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                hooks.afterChunk(end, failure -> {
                    captureFailure = failure;
                    runtime.lake().publish(select);
                });
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
                rows = new int[depth + 1];
                rows[0] = current;
                System.arraycopy(drafts, 0, rows, 1, depth);
                position = sequence.currentTokenPosition();
                acceptance = new SpeculativeAcceptance(rows, endOfGeneration, maxNewTokens - output.size());
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
                int[] committed = acceptance.outputs();
                int rejected = acceptance.rejectedBaseToken();
                boolean[] shortlist = inShortlist;
                int rejection = rejected < 0 ? -1 : rejected < shortlist.length && shortlist[rejected] ? 0 : 1;
                if (rejection == 0) statistics.rejectionsInShortlist++;
                if (rejection == 1) statistics.rejectionsOutsideShortlist++;
                if (timing != null)
                    timing.speculativeStep(started, executed, committed.length, acceptance.acceptedDrafts(), rejection);
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
