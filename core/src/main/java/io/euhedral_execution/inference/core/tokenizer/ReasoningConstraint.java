package io.euhedral_execution.inference.core.tokenizer;

import java.util.Objects;

/// Output that opens inside the model's think block: free reasoning until `</think>`, then an answer that may
/// follow a constraint of its own.
///
/// The reasoning may end with `</think>` at any point, and must once `budget` reasoning tokens were generated.
/// With an answer constraint a terminator cannot end the reasoning, since no answer would exist; the answer
/// constraint decides everything after `</think>` (including the whitespace that may separate the two). Without
/// one the answer is free.
public final class ReasoningConstraint implements TokenConstraint {
    private final QwenTokenizer tokenizer;
    private final int thinkEndId;
    private final int budget;
    private final TokenConstraint answer;
    private boolean answering;
    private int reasoningTokens;

    /// `budget` caps the reasoning tokens; `Integer.MAX_VALUE` leaves them uncapped. A null `answer` is free.
    public ReasoningConstraint(QwenTokenizer tokenizer, int budget, TokenConstraint answer) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.thinkEndId = tokenizer
                .controlTokenId("</think>")
                .orElseThrow(() -> new IllegalArgumentException("the tokenizer has no </think> token"));
        if (budget < 0) throw new IllegalArgumentException("budget must not be negative");
        this.budget = budget;
        this.answer = answer;
    }

    @Override
    public boolean allows(int tokenId) {
        if (this.answering) return this.answer == null || this.answer.allows(tokenId);
        if (tokenId == this.thinkEndId) return true;
        if (this.reasoningTokens >= this.budget) return false;
        return this.answer == null || !this.tokenizer.isGenerationEosToken(tokenId);
    }

    @Override
    public void accept(int tokenId) {
        if (!allows(tokenId)) throw new IllegalArgumentException("token violates the reasoning constraint");
        if (this.answering) {
            if (this.answer != null) this.answer.accept(tokenId);
        } else if (tokenId == this.thinkEndId) this.answering = true;
        else this.reasoningTokens++;
    }

    @Override
    public void maskDisallowed(float[] logits) {
        if (this.answering) {
            if (this.answer != null) this.answer.maskDisallowed(logits);
            return;
        }
        if (this.reasoningTokens >= this.budget) {
            for (int tokenId = 0; tokenId < logits.length; tokenId++)
                if (tokenId != this.thinkEndId) logits[tokenId] = Float.NEGATIVE_INFINITY;
            return;
        }
        if (this.answer == null) return;
        for (int tokenId : this.tokenizer.generationEosTokenIds())
            if (tokenId < logits.length) logits[tokenId] = Float.NEGATIVE_INFINITY;
    }

    /// Reasoning tokens committed so far, `</think>` excluded.
    public int reasoningTokens() {
        return this.reasoningTokens;
    }

    @Override
    public void close() {
        if (this.answer != null) this.answer.close();
    }
}
