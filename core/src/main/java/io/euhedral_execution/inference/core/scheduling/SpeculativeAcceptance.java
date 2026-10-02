package io.euhedral_execution.inference.core.scheduling;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.IntPredicate;

/// Greedy acceptance of one speculative verification (docs/MTP_CONTRACT.md §4).
///
/// The verified rows are `[t₀, d₁ .. d_n]`: the last committed token, then n drafts. Row j's greedy
/// token g_j is base greedy decode's next token after rows 0..j. Drafts are accepted while
/// `d_{j+1} == g_j`. The outputs are the accepted drafts followed by the first g that no draft
/// matched, which is exactly the tokens ordinary greedy decode produces. They are cut after the first
/// end-of-generation token and at `maxOutputs`.
///
/// Committed rows equal the output count: row 0 carries t₀ and rows 1..k−1 the first k−1 outputs, which
/// one-row decode would have fed. The last output (the next t₀, or the end token) is not fed yet.
public final class SpeculativeAcceptance {
    private final int[] rows;
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
        this.endOfGeneration = Objects.requireNonNull(endOfGeneration, "endOfGeneration");
        this.maxOutputs = maxOutputs;
    }

    /// Resolves the step from each row's greedy token. Idempotent for the same selections.
    public void resolve(int[] selected) {
        if (selected.length != this.rows.length) throw new IllegalArgumentException("one selection per verified row");
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
        return this.rows.length;
    }

    private void requireResolved() {
        if (this.outputs == null) throw new IllegalStateException("verification is not resolved yet");
    }
}
