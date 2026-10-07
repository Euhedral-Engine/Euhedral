package io.euhedral_execution.inference.core.model.qwen38.speculative;

import io.euhedral_execution.core.frames.AbstractFrame;
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
import io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Config;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;

/// Greedy DFlash2 speculative decoding for one sequence (docs/DFLASH2.md). Every output token is the token
/// ordinary greedy decode produces, and the committed target state is the state it leaves: the drafter only
/// decides how many tokens one verification commits.
///
/// The drafter conditions on the target's hidden rows after five of its layers (the taps), projected into each
/// draft layer's keys and values of the committed positions (its context). Each step is a sequence of quanta,
/// run as generation frames through the decoder's step ports:
/// 1. a DRAFT block: the anchor (the last committed token, not yet fed to the target) and 7 mask tokens at its
///    position and the next 7, through the 5 draft layers together, the target's output head, the top 16
///    candidates per position and the selector's path, which is the proposal;
/// 2. a VERIFY of `[anchor, proposal]`: the target's exact row-by-row verification, which also taps every row;
/// 3. a DRAFT_CONTEXT over the committed rows' taps, so the next block sees them.
///
/// The prompt is prefilled in chunks that tap their rows, each followed by its context quantum. Not thread-safe;
/// one decoder per sequence.
public final class DFlash2Decoder implements SpeculativeDecoding {

    /// Per-generation measurements. `acceptedDrafts[a]` counts verifications that accepted `a` drafts.
    public static final class Statistics {
        public final long[] acceptedDrafts;
        public long verifications;
        public long outputTokens;
        public long verifyNanos;
        /// Draft blocks after the prompt; the first is in `firstBlockNanos` (time to the first verification).
        public long blockNanos;
        public long firstBlockNanos;
        /// Context quanta after verifications; the prompt's are in `promptContextNanos`.
        public long contextNanos;
        public long promptContextNanos;
        public long prefillNanos;

        Statistics(int drafts) {
            this.acceptedDrafts = new long[drafts + 1];
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
                    + String.format(
                            Locale.ROOT,
                            ", mean %.3f, verify %.2f ms, block %.2f ms, context %.2f ms (per verification)",
                            meanAcceptedDrafts(),
                            perStep(this.verifyNanos),
                            perStep(this.blockNanos),
                            perStep(this.contextNanos));
        }

        private double perStep(long nanos) {
            return this.verifications == 0 ? 0 : nanos / 1e6 / this.verifications;
        }
    }

    /// Sees each verified step (tests and quality measurements): the anchor's position, the anchor, the proposal,
    /// each proposal position's 16 candidates (logit descending, row-major), and the drafts the target accepted.
    public interface StepListener {
        void verified(long position, int anchor, int[] proposal, int[] candidates, int acceptedDrafts);
    }

    private final Execution runtime;
    private final ExecutionPlan plan;
    private final Sequence sequence;
    private final IntPredicate endOfGeneration;
    private final int prefillChunk;
    private final DFlash2Config config;
    private final HostLogits baseLogits;
    private final DFlash2Proposal proposal;
    private final DFlash2Checkpoint checkpoint;
    /// Drafts each verification checks: the first `verified` of the block's proposal.
    private final int verified;
    private Statistics statistics;
    private StepListener steps;

    /// DFlash2 verifying the first `verified` drafts of each block.
    public static SpeculativeDecoding.Factory factory(int verified) {
        return (runtime, plan, gpu, sequence, end, chunk) ->
                new DFlash2Decoder(runtime, plan, gpu, sequence, end, chunk, verified);
    }

    public DFlash2Decoder(
            Execution runtime,
            ExecutionPlan plan,
            ExecutionGpu gpu,
            Sequence sequence,
            IntPredicate endOfGeneration,
            int prefillChunk) {
        this(runtime, plan, gpu, sequence, endOfGeneration, prefillChunk, Integer.MAX_VALUE);
    }

    /// As the public constructor, verifying only the first `verified` drafts of each block; the block always proposes
    /// all of them.
    public DFlash2Decoder(
            Execution runtime,
            ExecutionPlan plan,
            ExecutionGpu gpu,
            Sequence sequence,
            IntPredicate endOfGeneration,
            int prefillChunk,
            int verified) {
        this.runtime = Objects.requireNonNull(runtime, "runtime");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.sequence = Objects.requireNonNull(sequence, "sequence");
        this.endOfGeneration = Objects.requireNonNull(endOfGeneration, "endOfGeneration");
        if (!plan.draftsWithDFlash2()) throw new IllegalArgumentException("the plan has no DFlash2 drafter");
        if (prefillChunk <= 0) throw new IllegalArgumentException("prefillChunk must be positive");
        this.prefillChunk = prefillChunk;
        this.config = plan.weights().dflash2().config();
        this.baseLogits = new HostLogits(gpu, plan.weights().config().vocabSize());
        this.baseLogits.selectOnDevice(true);
        this.proposal = new DFlash2Proposal(gpu, drafts(), this.config.selectorTopK());
        this.checkpoint = new DFlash2Checkpoint(this.config);
        if (verified < 1) throw new IllegalArgumentException("a verification checks at least one draft");
        this.verified = Math.min(verified, drafts());
    }

    /// Drafts per verification: the block less its anchor.
    public int drafts() {
        return this.config.blockSize() - 1;
    }

    public Statistics statistics() {
        return this.statistics;
    }

    public void observe(StepListener listener) {
        this.steps = listener;
    }

    @Override
    public SpeculativeCheckpoint checkpoint() {
        return this.checkpoint;
    }

    /// Prefills `prompt` and generates up to `maxNewTokens` tokens, exactly as greedy decode would. The steps run
    /// as generation frames on the lattice; the calling thread waits for them (tests and tools).
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

    /// A sequence restored at `startPosition` holds the target state and the drafter's context for `[0,
    /// startPosition)`; prefilling resumes there.
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
        this.statistics = new Statistics(drafts());
        return new Run(prompt, maxNewTokens, onToken, timing, hooks, ended).prefillFrom(startPosition);
    }

    /// One generation's steps, in the order of the sequential algorithm: each port admits one quantum and, when it
    /// retired, names the next.
    private final class Run {
        /// What a finished context quantum goes on to.
        private enum Context {
            PROMPT,
            AFTER_VERIFY
        }

        private final int[] prompt;
        private final int maxNewTokens;
        private final IntConsumer onToken;
        private final GenerationTimingListener timing;
        private final PrefixHooks hooks;
        private final Consumer<List<Integer>> ended;
        private final List<Integer> output = new ArrayList<>();
        private boolean firstBlock = true;

        // The prefill chunk in flight.
        private int offset;
        private int end;
        private boolean last;
        private long started;
        private int[] chunk;

        // The context quantum in flight.
        private long contextPosition;
        private int[] contextTokens;
        private long contextStarted;
        private Context context;

        // The block and verification in flight.
        private int anchor;
        private long position;
        private long drafting;
        private int[] rows;
        private int[] candidates;
        private SpeculativeAcceptance acceptance;
        private int finalToken;
        private volatile Throwable captureFailure;

        Run(
                int[] prompt,
                int maxNewTokens,
                IntConsumer onToken,
                GenerationTimingListener timing,
                PrefixHooks hooks,
                Consumer<List<Integer>> ended) {
            this.prompt = prompt;
            this.maxNewTokens = maxNewTokens;
            this.onToken = onToken;
            this.timing = timing;
            this.hooks = hooks;
            this.ended = ended;
        }

        StepPort prefillFrom(int from) {
            this.offset = from;
            return this.offset >= this.prompt.length ? afterPrompt() : this.prefill;
        }

        /// Prompt: a prefill chunk that taps its rows; its context quantum follows.
        private final StepPort prefill = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                end = Math.min(prompt.length, offset + prefillChunk);
                last = end == prompt.length;
                started = System.nanoTime();
                chunk = Arrays.copyOfRange(prompt, offset, end);
                runtime.admit(
                        new Quantum(
                                        plan,
                                        sequence,
                                        Quantum.ExecutionKind.PREFILL,
                                        offset,
                                        chunk,
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
                return contextOf(offset, chunk, Context.PROMPT);
            }
        };

        private StepPort contextOf(long at, int[] tokens, Context then) {
            this.contextPosition = at;
            this.contextTokens = tokens;
            this.context = then;
            this.contextStarted = System.nanoTime();
            return this.contextStep;
        }

        /// The drafter's keys and values of the committed rows, from the tap rows the latest target quantum left.
        private final StepPort contextStep = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                runtime.admit(
                        new Quantum(
                                plan,
                                sequence,
                                Quantum.ExecutionKind.DRAFT_CONTEXT,
                                contextPosition,
                                contextTokens,
                                LogitsRequirement.NONE),
                        select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) {
                succeeded(step);
                long contextDone = System.nanoTime();
                if (context == Context.PROMPT) {
                    statistics.promptContextNanos += contextDone - contextStarted;
                    if (timing != null) timing.draftQuantum("prompt-context", contextStarted, contextDone);
                    return hooks == null ? prefillFrom(end) : capture;
                }
                statistics.contextNanos += contextDone - contextStarted;
                if (timing != null) timing.draftQuantum("context", contextStarted, contextDone);
                return blockFrom(anchor);
            }
        };

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
            int first = baseLogits.selectedToken();
            if (this.timing != null) this.timing.firstTokenSelected(System.nanoTime(), first);
            this.output.add(first);
            this.onToken.accept(first);
            statistics.outputTokens++;
            if (endOfGeneration.test(first)) return done();
            if (this.maxNewTokens == 1) {
                this.finalToken = first;
                return this.feedFinal;
            }
            return blockFrom(first);
        }

        /// One draft block from `anchor`; its verification and the context of what it committed follow.
        private StepPort blockFrom(int anchor) {
            this.anchor = anchor;
            this.position = sequence.currentTokenPosition();
            DFlash2State state = state();
            if (state == null || state.contextLength() != this.position)
                throw new IllegalStateException("the drafter's context does not reach the anchor at " + this.position);
            return this.blockStep;
        }

        private final StepPort blockStep = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                int[] block = new int[config.blockSize()];
                Arrays.fill(block, config.maskToken());
                block[0] = anchor;
                drafting = System.nanoTime();
                runtime.admit(
                        new Quantum(
                                        plan,
                                        sequence,
                                        Quantum.ExecutionKind.DRAFT,
                                        position,
                                        block,
                                        LogitsRequirement.ALL_TOKENS)
                                .withProposal(proposal),
                        select);
            }

            @Override
            public StepPort retired(AbstractQuantum step) {
                succeeded(step);
                long draftedAt = System.nanoTime();
                long drafted = draftedAt - drafting;
                if (timing != null) timing.draftQuantum("block", drafting, draftedAt);
                if (firstBlock) statistics.firstBlockNanos += drafted;
                else statistics.blockNanos += drafted;
                firstBlock = false;
                rows = new int[verified + 1];
                rows[0] = anchor;
                System.arraycopy(proposal.tokens(), 0, rows, 1, rows.length - 1);
                candidates = steps == null ? null : proposal.candidates();
                return verify;
            }
        };

        private final StepPort verify = new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
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
                Statistics statistics = DFlash2Decoder.this.statistics;
                long executed = System.nanoTime();
                statistics.verifyNanos += executed - started;
                statistics.verifications++;
                statistics.acceptedDrafts[acceptance.acceptedDrafts()]++;
                StepListener listener = steps;
                if (listener != null)
                    listener.verified(
                            position,
                            rows[0],
                            Arrays.copyOfRange(rows, 1, rows.length),
                            candidates,
                            acceptance.acceptedDrafts());
                int[] committed = acceptance.outputs();
                if (timing != null)
                    timing.speculativeStep(started, executed, committed.length, acceptance.acceptedDrafts());
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
                // The committed rows' taps are the verified rows 0 .. committed - 1: the anchor and the accepted
                // drafts, at the positions the target just committed.
                anchor = next;
                return contextOf(position, Arrays.copyOf(rows, acceptance.committedRows()), Context.AFTER_VERIFY);
            }
        };

        /// Ordinary decode feeds the last allowed token without sampling; so does the decoder, so both leave the
        /// same target state.
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

    private DFlash2State state() {
        return this.sequence.kvCacheState() instanceof AttentionStates attention ? attention.dflash2() : null;
    }

    @Override
    public void close() {
        this.baseLogits.close();
        this.proposal.close();
    }
}
