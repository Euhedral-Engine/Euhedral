package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
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
public final class QwenSpeculativeDecoder implements AutoCloseable {

    /// Per-generation measurements. `acceptedDrafts[a]` counts verifications that accepted a drafts.
    public static final class Statistics {
        public final long[] acceptedDrafts;
        public long verifications;
        public long outputTokens;
        public long verifyNanos;
        public long catchUpNanos;
        public long recursionNanos;
        public long prefillNanos;

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

    private final EuhedralInferenceRuntime runtime;
    private final QwenExecutionPlan plan;
    private final QwenSequenceState sequence;
    private final IntPredicate endOfGeneration;
    private final int depth;
    private final int prefillChunk;
    private final int hidden;
    private final QwenHostLogits baseLogits;
    private final QwenHostLogits draftLogits;
    private final int[] draftTokens;
    private Statistics statistics;

    public QwenSpeculativeDecoder(
            EuhedralInferenceRuntime runtime,
            QwenExecutionPlan plan,
            ExecutionGpu gpu,
            QwenSequenceState sequence,
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
        this.baseLogits = new QwenHostLogits(gpu, plan.weights().config().vocabSize());
        this.draftLogits = new QwenHostLogits(gpu, plan.draftVocabularySize());
        this.baseLogits.selectOnDevice(true);
        this.draftLogits.selectOnDevice(true);
        this.draftTokens = draftTokenIds(gpu, plan);
    }

    /// The draft head's token for each of its rows (`text/draft_head_token_ids`).
    private static int[] draftTokenIds(ExecutionGpu gpu, QwenExecutionPlan plan) {
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
        if (prompt.length == 0 || maxNewTokens <= 0) throw new IllegalArgumentException("empty generation");
        this.statistics = new Statistics(this.depth);
        List<Integer> output = new ArrayList<>();
        // Prompt: prefill chunks that seed drafting, each followed by its MTP catch-up.
        int first = -1;
        int[] drafts = null;
        for (int offset = 0; offset < prompt.length; offset += this.prefillChunk) {
            int end = Math.min(prompt.length, offset + this.prefillChunk);
            boolean last = end == prompt.length;
            long started = System.nanoTime();
            execute(new QwenExecutionContext(
                            this.plan,
                            this.sequence,
                            QwenExecutionContext.ExecutionKind.PREFILL,
                            offset,
                            Arrays.copyOfRange(prompt, offset, end),
                            last ? QwenLogitsRequirement.LAST_TOKEN : QwenLogitsRequirement.NONE,
                            last ? this.baseLogits : null)
                    .seedingDraft());
            this.statistics.prefillNanos += System.nanoTime() - started;
            int[] next = new int[end - offset];
            System.arraycopy(prompt, offset + 1, next, 0, end - offset - 1);
            if (last) {
                first = this.baseLogits.selectedToken();
                next[next.length - 1] = first;
            } else next[next.length - 1] = prompt[end];
            int[] chunkDrafts = catchUp(offset, next, last);
            if (last) drafts = chunkDrafts;
        }
        output.add(first);
        onToken.accept(first);
        this.statistics.outputTokens++;
        if (this.endOfGeneration.test(first)) return output;
        if (maxNewTokens == 1) {
            feedFinal(first);
            return output;
        }
        int current = first;
        while (true) {
            int[] rows = new int[this.depth + 1];
            rows[0] = current;
            System.arraycopy(drafts, 0, rows, 1, this.depth);
            long position = this.sequence.currentTokenPosition();
            var acceptance = new SpeculativeAcceptance(rows, this.endOfGeneration, maxNewTokens - output.size());
            long started = System.nanoTime();
            execute(new QwenExecutionContext(
                            this.plan,
                            this.sequence,
                            QwenExecutionContext.ExecutionKind.VERIFY,
                            position,
                            rows,
                            QwenLogitsRequirement.ALL_TOKENS,
                            this.baseLogits)
                    .withAcceptance(acceptance)
                    .seedingDraft());
            this.statistics.verifyNanos += System.nanoTime() - started;
            this.statistics.verifications++;
            this.statistics.acceptedDrafts[acceptance.acceptedDrafts()]++;
            int[] committed = acceptance.outputs();
            for (int token : committed) {
                output.add(token);
                onToken.accept(token);
            }
            this.statistics.outputTokens += committed.length;
            current = committed[committed.length - 1];
            if (this.endOfGeneration.test(current)) return output;
            if (output.size() >= maxNewTokens) {
                feedFinal(current);
                return output;
            }
            // Discard the previous step's recursive draft rows, then catch the MTP cache up.
            mtpCache().truncate(Math.toIntExact(position));
            drafts = catchUp(position, committed, true);
        }
    }

    /// MTP catch-up over the base hidden rows just seeded, paired with `tokens` (the token each row
    /// predicted), at MTP positions from `position`. When `draft` is set, its last row drafts d₁ and the
    /// recursive rows draft the rest.
    private int[] catchUp(long position, int[] tokens, boolean draft) throws InterruptedException, ExecutionException {
        AttentionSequenceStates states = states();
        long seeds = states.draftSeedRows(tokens.length, this.hidden);
        long started = System.nanoTime();
        execute(new QwenExecutionContext(
                        this.plan,
                        this.sequence,
                        QwenExecutionContext.ExecutionKind.DRAFT,
                        position,
                        tokens,
                        draft ? QwenLogitsRequirement.LAST_TOKEN : QwenLogitsRequirement.NONE,
                        draft ? this.draftLogits : null)
                .withDraftSeed(seeds, tokens.length));
        this.statistics.catchUpNanos += System.nanoTime() - started;
        if (!draft) return null;
        int[] drafts = new int[this.depth];
        drafts[0] = this.draftTokens[this.draftLogits.selectedToken()];
        started = System.nanoTime();
        for (int i = 1; i < this.depth; i++) {
            execute(new QwenExecutionContext(
                            this.plan,
                            this.sequence,
                            QwenExecutionContext.ExecutionKind.DRAFT,
                            position + tokens.length + i - 1,
                            new int[] {drafts[i - 1]},
                            QwenLogitsRequirement.LAST_TOKEN,
                            this.draftLogits)
                    .withDraftSeed(states.draftRecursionHidden(this.hidden), 1));
            drafts[i] = this.draftTokens[this.draftLogits.selectedToken()];
        }
        this.statistics.recursionNanos += System.nanoTime() - started;
        return drafts;
    }

    /// Ordinary decode feeds the last allowed token without sampling; so does the decoder, so both leave
    /// the same state.
    private void feedFinal(int token) throws InterruptedException, ExecutionException {
        execute(new QwenExecutionContext(
                this.plan,
                this.sequence,
                QwenExecutionContext.ExecutionKind.DECODE,
                this.sequence.currentTokenPosition(),
                new int[] {token},
                QwenLogitsRequirement.NONE));
    }

    private AttentionSequenceStates states() {
        return (AttentionSequenceStates) this.sequence.kvCacheState();
    }

    private AttentionKvState mtpCache() {
        return states().forLayer(this.plan.weights().config().numHiddenLayers());
    }

    private void execute(QwenExecutionContext context) throws InterruptedException, ExecutionException {
        var outcome = this.runtime.execute(List.of(context)).getFirst();
        if (outcome.status() != QwenExecutionContext.Status.SUCCESS)
            throw new IllegalStateException("speculative quantum " + outcome.status(), outcome.failure());
    }

    @Override
    public void close() {
        this.baseLogits.close();
        this.draftLogits.close();
    }
}
