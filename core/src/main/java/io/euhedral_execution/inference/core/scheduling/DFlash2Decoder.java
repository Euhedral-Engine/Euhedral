package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.config.DFlash2Config;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;

/// Greedy DFlash2 speculative decoding for one sequence (docs/DFLASH2.md). Every output token is the token
/// ordinary greedy decode produces, and the committed target state is the state it leaves: the drafter only
/// decides how many tokens one verification commits.
///
/// The drafter conditions on the target's hidden rows after five of its layers (the taps), projected into each
/// draft layer's keys and values of the committed positions (its context). Each step is a chain of quanta:
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

    private final EuhedralInferenceRuntime runtime;
    private final QwenExecutionPlan plan;
    private final QwenSequenceState sequence;
    private final IntPredicate endOfGeneration;
    private final int prefillChunk;
    private final DFlash2Config config;
    private final QwenHostLogits baseLogits;
    private final DFlash2Proposal proposal;
    private final DFlash2Checkpoint checkpoint;
    private Statistics statistics;
    private StepListener steps;

    public static SpeculativeDecoding.Factory factory() {
        return DFlash2Decoder::new;
    }

    public DFlash2Decoder(
            EuhedralInferenceRuntime runtime,
            QwenExecutionPlan plan,
            ExecutionGpu gpu,
            QwenSequenceState sequence,
            IntPredicate endOfGeneration,
            int prefillChunk) {
        this.runtime = Objects.requireNonNull(runtime, "runtime");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.sequence = Objects.requireNonNull(sequence, "sequence");
        this.endOfGeneration = Objects.requireNonNull(endOfGeneration, "endOfGeneration");
        if (!plan.draftsWithDFlash2()) throw new IllegalArgumentException("the plan has no DFlash2 drafter");
        if (prefillChunk <= 0) throw new IllegalArgumentException("prefillChunk must be positive");
        this.prefillChunk = prefillChunk;
        this.config = plan.weights().dflash2().config();
        this.baseLogits = new QwenHostLogits(gpu, plan.weights().config().vocabSize());
        this.baseLogits.selectOnDevice(true);
        this.proposal = new DFlash2Proposal(gpu, drafts(), this.config.selectorTopK());
        this.checkpoint = new DFlash2Checkpoint(this.config);
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

    /// Prefills `prompt` and generates up to `maxNewTokens` tokens, exactly as greedy decode would, blocking until
    /// done (tests and tools).
    public List<Integer> generate(int[] prompt, int maxNewTokens, IntConsumer onToken, GenerationTimingListener timing)
            throws InterruptedException, ExecutionException {
        try {
            return generateAsync(prompt, maxNewTokens, onToken, timing, null, 0).get();
        } catch (ExecutionException failure) {
            if (failure.getCause() instanceof RuntimeException runtimeFailure) throw runtimeFailure;
            if (failure.getCause() instanceof Error error) throw error;
            throw failure;
        }
    }

    /// A sequence restored at `startPosition` holds the target state and the drafter's context for `[0,
    /// startPosition)`; prefilling resumes there.
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
        this.statistics = new Statistics(drafts());
        return new Run(prompt, maxNewTokens, onToken, timing, hooks).prefill(startPosition);
    }

    /// One generation's continuations, in the order of the sequential algorithm.
    private final class Run {
        private final int[] prompt;
        private final int maxNewTokens;
        private final IntConsumer onToken;
        private final GenerationTimingListener timing;
        private final PrefixHooks hooks;
        private final List<Integer> output = new ArrayList<>();
        private boolean firstBlock = true;

        Run(int[] prompt, int maxNewTokens, IntConsumer onToken, GenerationTimingListener timing, PrefixHooks hooks) {
            this.prompt = prompt;
            this.maxNewTokens = maxNewTokens;
            this.onToken = onToken;
            this.timing = timing;
            this.hooks = hooks;
        }

        /// Prompt: prefill chunks that tap their rows, each followed by its context quantum.
        CompletableFuture<List<Integer>> prefill(int offset) {
            if (offset >= this.prompt.length) return afterPrompt();
            int end = Math.min(this.prompt.length, offset + DFlash2Decoder.this.prefillChunk);
            boolean last = end == this.prompt.length;
            long started = System.nanoTime();
            int[] chunk = Arrays.copyOfRange(this.prompt, offset, end);
            return execute(new QwenExecutionContext(
                                    DFlash2Decoder.this.plan,
                                    DFlash2Decoder.this.sequence,
                                    QwenExecutionContext.ExecutionKind.PREFILL,
                                    offset,
                                    chunk,
                                    last ? QwenLogitsRequirement.LAST_TOKEN : QwenLogitsRequirement.NONE,
                                    last ? DFlash2Decoder.this.baseLogits : null)
                            .seedingDraft())
                    .thenCompose(ignored -> {
                        long executed = System.nanoTime();
                        DFlash2Decoder.this.statistics.prefillNanos += executed - started;
                        if (this.timing != null) this.timing.prefillQuantum(started, executed, end - offset);
                        long contextStarted = System.nanoTime();
                        return context(offset, chunk).thenCompose(done -> {
                            long contextDone = System.nanoTime();
                            DFlash2Decoder.this.statistics.promptContextNanos += contextDone - contextStarted;
                            if (this.timing != null)
                                this.timing.draftQuantum("prompt-context", contextStarted, contextDone);
                            CompletableFuture<Void> stored = this.hooks == null
                                    ? CompletableFuture.completedFuture(null)
                                    : this.hooks.afterChunk(end);
                            return stored.thenCompose(ignoredStore -> prefill(end));
                        });
                    });
        }

        private CompletableFuture<List<Integer>> afterPrompt() {
            int first = DFlash2Decoder.this.baseLogits.selectedToken();
            if (this.timing != null) this.timing.firstTokenSelected(System.nanoTime(), first);
            this.output.add(first);
            this.onToken.accept(first);
            DFlash2Decoder.this.statistics.outputTokens++;
            if (DFlash2Decoder.this.endOfGeneration.test(first)) return CompletableFuture.completedFuture(this.output);
            if (this.maxNewTokens == 1) return feedFinal(first).thenApply(ignored -> this.output);
            return step(first);
        }

        /// One draft block from `anchor`, its verification, and the context of what it committed.
        private CompletableFuture<List<Integer>> step(int anchor) {
            long position = DFlash2Decoder.this.sequence.currentTokenPosition();
            DFlash2SequenceState state = state();
            if (state == null || state.contextLength() != position)
                return CompletableFuture.failedFuture(
                        new IllegalStateException("the drafter's context does not reach the anchor at " + position));
            int[] block = new int[DFlash2Decoder.this.config.blockSize()];
            Arrays.fill(block, DFlash2Decoder.this.config.maskToken());
            block[0] = anchor;
            long drafting = System.nanoTime();
            return execute(new QwenExecutionContext(
                                    DFlash2Decoder.this.plan,
                                    DFlash2Decoder.this.sequence,
                                    QwenExecutionContext.ExecutionKind.DRAFT,
                                    position,
                                    block,
                                    QwenLogitsRequirement.ALL_TOKENS)
                            .withProposal(DFlash2Decoder.this.proposal))
                    .thenCompose(ignored -> {
                        long draftedAt = System.nanoTime();
                        long drafted = draftedAt - drafting;
                        if (this.timing != null) this.timing.draftQuantum("block", drafting, draftedAt);
                        if (this.firstBlock) DFlash2Decoder.this.statistics.firstBlockNanos += drafted;
                        else DFlash2Decoder.this.statistics.blockNanos += drafted;
                        this.firstBlock = false;
                        int[] rows = new int[block.length];
                        rows[0] = anchor;
                        System.arraycopy(DFlash2Decoder.this.proposal.tokens(), 0, rows, 1, drafts());
                        int[] candidates =
                                DFlash2Decoder.this.steps == null ? null : DFlash2Decoder.this.proposal.candidates();
                        return verify(position, rows, candidates);
                    });
        }

        private CompletableFuture<List<Integer>> verify(long position, int[] rows, int[] candidates) {
            var acceptance = new SpeculativeAcceptance(
                    rows, DFlash2Decoder.this.endOfGeneration, this.maxNewTokens - this.output.size());
            long started = System.nanoTime();
            return execute(new QwenExecutionContext(
                                    DFlash2Decoder.this.plan,
                                    DFlash2Decoder.this.sequence,
                                    QwenExecutionContext.ExecutionKind.VERIFY,
                                    position,
                                    rows,
                                    QwenLogitsRequirement.ALL_TOKENS,
                                    DFlash2Decoder.this.baseLogits)
                            .withAcceptance(acceptance)
                            .seedingDraft())
                    .thenCompose(ignored -> {
                        Statistics statistics = DFlash2Decoder.this.statistics;
                        long executed = System.nanoTime();
                        statistics.verifyNanos += executed - started;
                        statistics.verifications++;
                        statistics.acceptedDrafts[acceptance.acceptedDrafts()]++;
                        StepListener listener = DFlash2Decoder.this.steps;
                        if (listener != null)
                            listener.verified(
                                    position,
                                    rows[0],
                                    Arrays.copyOfRange(rows, 1, rows.length),
                                    candidates,
                                    acceptance.acceptedDrafts());
                        int[] committed = acceptance.outputs();
                        if (this.timing != null)
                            this.timing.speculativeStep(
                                    started, executed, committed.length, acceptance.acceptedDrafts());
                        for (int token : committed) {
                            this.output.add(token);
                            this.onToken.accept(token);
                        }
                        statistics.outputTokens += committed.length;
                        int next = committed[committed.length - 1];
                        if (DFlash2Decoder.this.endOfGeneration.test(next))
                            return CompletableFuture.completedFuture(this.output);
                        if (this.output.size() >= this.maxNewTokens)
                            return feedFinal(next).thenApply(done -> this.output);
                        // The committed rows' taps are the verified rows 0 .. committed - 1: the anchor and the
                        // accepted drafts, at the positions the target just committed.
                        int[] contextRows = Arrays.copyOf(rows, acceptance.committedRows());
                        long contextStarted = System.nanoTime();
                        return context(position, contextRows).thenCompose(done -> {
                            long contextDone = System.nanoTime();
                            DFlash2Decoder.this.statistics.contextNanos += contextDone - contextStarted;
                            if (this.timing != null) this.timing.draftQuantum("context", contextStarted, contextDone);
                            return step(next);
                        });
                    });
        }

        /// Ordinary decode feeds the last allowed token without sampling; so does the decoder, so both leave the
        /// same target state.
        private CompletableFuture<Void> feedFinal(int token) {
            long started = System.nanoTime();
            return execute(new QwenExecutionContext(
                            DFlash2Decoder.this.plan,
                            DFlash2Decoder.this.sequence,
                            QwenExecutionContext.ExecutionKind.DECODE,
                            DFlash2Decoder.this.sequence.currentTokenPosition(),
                            new int[] {token},
                            QwenLogitsRequirement.NONE))
                    .thenApply(ignored -> {
                        long executed = System.nanoTime();
                        if (this.timing != null) this.timing.decodeQuantum(started, executed, executed, false, -1);
                        return null;
                    });
        }
    }

    /// The drafter's keys and values of the committed rows at `position`, from the tap rows the latest target
    /// quantum left.
    private CompletableFuture<Void> context(long position, int[] tokens) {
        return execute(new QwenExecutionContext(
                this.plan,
                this.sequence,
                QwenExecutionContext.ExecutionKind.DRAFT_CONTEXT,
                position,
                tokens,
                QwenLogitsRequirement.NONE));
    }

    private DFlash2SequenceState state() {
        return this.sequence.kvCacheState() instanceof AttentionSequenceStates attention ? attention.dflash2() : null;
    }

    /// Admits `context`; the future completes on the worker that retired it, failing unless it succeeded.
    private CompletableFuture<Void> execute(QwenExecutionContext context) {
        CompletableFuture<QwenExecutionContext.Outcome> outcome;
        try {
            outcome = this.runtime.submit(context);
        } catch (RuntimeException | Error failure) {
            return CompletableFuture.failedFuture(failure);
        }
        return outcome.thenApply(completed -> {
            if (completed.status() != QwenExecutionContext.Status.SUCCESS)
                throw new IllegalStateException("speculative quantum " + completed.status(), completed.failure());
            return null;
        });
    }

    @Override
    public void close() {
        this.baseLogits.close();
        this.proposal.close();
    }
}
