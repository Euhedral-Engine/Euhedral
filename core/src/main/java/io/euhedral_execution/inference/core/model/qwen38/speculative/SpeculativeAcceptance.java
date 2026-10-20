package io.euhedral_execution.inference.core.model.qwen38.speculative;

import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsSampler;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.IntPredicate;
import java.util.function.Supplier;

/// Exact-match acceptance of one speculative verification (docs/speculation/MTP_CONTRACT.md §4).
///
/// The verified rows are `[t₀, d₁ .. d_n]`: the last committed token, then n drafts. Row j's selected
/// token g_j (greedy, or sampled at its position) is base decode's next token after rows 0..j. Drafts are accepted
/// while
/// `d_{j+1} == g_j`. The outputs are the accepted drafts followed by the first g that no draft
/// matched, which is exactly the tokens ordinary decode produces. They are cut after the first
/// end-of-generation token and at `maxOutputs`.
///
/// Committed rows equal the output count: row 0 carries t₀ and rows 1..k−1 the first k−1 outputs, which
/// one-row decode would have fed. The last output (the next t₀, or the end token) is not fed yet.
public final class SpeculativeAcceptance {
    /// Reads each verified row's selected token from the rows a verification copied to the host.
    @FunctionalInterface
    public interface Selector {
        int[] select(HostLogits rows);

        /// The device's greedy choice of each row (rows queued with [HostLogits#queueRowSelections]).
        Selector GREEDY = HostLogits::selectedTokens;

        /// The sampler's token for each row, drawn at the position that token occupies: row j of a verification
        /// whose first row sits at `firstPosition - 1` is selected at `firstPosition + j`. A draft is
        /// accepted only when it equals that token, so the committed sequence is the one ordinary sampled
        /// decode produces.
        static Selector sampled(LogitsSampler sampler, long firstPosition) {
            Objects.requireNonNull(sampler, "sampler");
            return rows -> sampler.selectRows(rows, firstPosition);
        }
    }

    private int[] rows;
    /// Supplies the rows at resolution when they were not known at construction.
    private final Supplier<int[]> lateRows;
    private final int rowCount;
    private final IntPredicate endOfGeneration;
    private final int maxOutputs;
    private int[] outputs;
    private int accepted;
    private int rejectedBaseToken = -1;

    /// `rows` are the verified input tokens; `maxOutputs` caps the outputs this step may commit.
    public SpeculativeAcceptance(int[] rows, IntPredicate endOfGeneration, int maxOutputs) {
        if (Objects.requireNonNull(rows, "rows").length == 0) throw new IllegalArgumentException("no verified rows");
        if (maxOutputs <= 0) throw new IllegalArgumentException("maxOutputs must be positive");
        this.rows = rows.clone();
        this.lateRows = null;
        this.rowCount = rows.length;
        this.endOfGeneration = Objects.requireNonNull(endOfGeneration, "endOfGeneration");
        this.maxOutputs = maxOutputs;
    }

    /// As above for `rowCount` rows that are known only when the verification resolves: drafts the device passed
    /// to the verification, which `rows` reads back once they retired.
    public SpeculativeAcceptance(int rowCount, Supplier<int[]> rows, IntPredicate endOfGeneration, int maxOutputs) {
        if (rowCount <= 0) throw new IllegalArgumentException("no verified rows");
        if (maxOutputs <= 0) throw new IllegalArgumentException("maxOutputs must be positive");
        this.lateRows = Objects.requireNonNull(rows, "rows");
        this.rowCount = rowCount;
        this.endOfGeneration = Objects.requireNonNull(endOfGeneration, "endOfGeneration");
        this.maxOutputs = maxOutputs;
    }

    /// Resolves the step from each row's selected token. Idempotent for the same selections.
    public void resolve(int[] selected) {
        if (selected.length != this.rowCount) throw new IllegalArgumentException("one selection per verified row");
        if (this.rows == null) {
            int[] late = this.lateRows.get();
            if (late.length != this.rowCount)
                throw new IllegalStateException("the late rows are not the verified rows");
            this.rows = late.clone();
        }
        int accepted = 0;
        while (accepted + 1 < this.rows.length && this.rows[accepted + 1] == selected[accepted]) accepted++;
        int[] outputs = new int[accepted + 1];
        System.arraycopy(this.rows, 1, outputs, 0, accepted);
        outputs[accepted] = selected[accepted];
        int count = Math.min(outputs.length, this.maxOutputs);
        for (int i = 0; i < count; i++) {
            if (this.endOfGeneration.test(outputs[i])) {
                count = i + 1;
                break;
            }
        }
        this.accepted = accepted;
        this.rejectedBaseToken = accepted + 1 < this.rows.length ? selected[accepted] : -1;
        this.outputs = Arrays.copyOf(outputs, count);
    }

    public boolean resolved() {
        return this.outputs != null;
    }

    /// Drafts the verifier confirmed (0..n), before any end-token or budget cut.
    public int acceptedDrafts() {
        requireResolved();
        return this.accepted;
    }

    /// The base model's token where a draft was rejected (the token the first unconfirmed draft should
    /// have been), or -1 when every draft was accepted.
    public int rejectedBaseToken() {
        requireResolved();
        return this.rejectedBaseToken;
    }

    /// The tokens this step commits to the output, in order.
    public int[] outputs() {
        requireResolved();
        return this.outputs.clone();
    }

    /// Verified rows whose state is committed.
    public int committedRows() {
        requireResolved();
        return this.outputs.length;
    }

    public int verifiedRows() {
        return this.rowCount;
    }

    /// The verified rows: the current token, then the drafts (known once resolved when they came late).
    public int[] rows() {
        if (this.rows == null) throw new IllegalStateException("the verified rows are known once it resolves");
        return this.rows.clone();
    }

    private void requireResolved() {
        if (this.outputs == null) throw new IllegalStateException("verification is not resolved yet");
    }
}
