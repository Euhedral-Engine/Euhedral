package io.euhedral_execution.inference.core.tokenizer;

import java.util.Objects;

/// Output that opens inside the model's think block: free reasoning until `</think>`, then an answer that may
/// follow a byte grammar.
///
/// The reasoning may end with `</think>` at any point, and must once `budget` reasoning tokens were generated.
/// With an answer grammar a terminator cannot end the reasoning, the answer may begin with a few whitespace
/// bytes (the template separates it from `</think>` by a blank line), and the grammar decides the rest.
/// Without one the answer is free.
public final class ReasoningConstraint implements TokenConstraint {
    /// Whitespace allowed between `</think>` and an answer grammar's first byte.
    static final int MAX_GAP_BYTES = 8;

    private final QwenTokenizer tokenizer;
    private final int thinkEndId;
    private final int budget;
    private final ByteGrammar answer;
    private boolean answering;
    private int reasoningTokens;
    private int gapBytes;

    /// `budget` caps the reasoning tokens; `Integer.MAX_VALUE` leaves them uncapped. A null `answer` is free text.
    public ReasoningConstraint(QwenTokenizer tokenizer, int budget, ByteGrammar answer) {
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
        if (!this.answering) {
            if (tokenId == this.thinkEndId) return true;
            if (this.reasoningTokens >= this.budget) return false;
            return this.answer == null || !this.tokenizer.isGenerationEosToken(tokenId);
        }
        if (this.answer == null) return true;
        if (this.tokenizer.isGenerationEosToken(tokenId)) return this.answer.complete();
        byte[] bytes = this.tokenizer.generationTokenBytes(tokenId);
        if (bytes == null || bytes.length == 0) return false;
        int from = gapEnd(bytes);
        return from >= 0 && (from == bytes.length || this.answer.allows(bytes, from));
    }

    @Override
    public void accept(int tokenId) {
        if (!allows(tokenId)) throw new IllegalArgumentException("token violates the reasoning constraint");
        if (!this.answering) {
            if (tokenId == this.thinkEndId) this.answering = true;
            else this.reasoningTokens++;
            return;
        }
        if (this.answer == null || this.tokenizer.isGenerationEosToken(tokenId)) return;
        byte[] bytes = this.tokenizer.generationTokenBytes(tokenId);
        int from = gapEnd(bytes);
        if (from < bytes.length) {
            this.gapBytes = MAX_GAP_BYTES + 1;
            this.answer.accept(bytes, from);
        } else this.gapBytes += bytes.length;
    }

    /// Reasoning tokens committed so far, `</think>` excluded.
    public int reasoningTokens() {
        return this.reasoningTokens;
    }

    /// Index of the first byte past the leading gap whitespace, or -1 when the gap grows too long. Once the
    /// grammar has begun there is no gap.
    private int gapEnd(byte[] bytes) {
        if (this.gapBytes > MAX_GAP_BYTES) return 0;
        int index = 0;
        while (index < bytes.length && isSpace(bytes[index])) index++;
        return this.gapBytes + index > MAX_GAP_BYTES ? -1 : index;
    }

    private static boolean isSpace(byte value) {
        return value == ' ' || value == '\n' || value == '\t' || value == '\r';
    }
}
