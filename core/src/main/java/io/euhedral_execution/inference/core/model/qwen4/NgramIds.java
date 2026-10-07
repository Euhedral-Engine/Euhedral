package io.euhedral_execution.inference.core.model.qwen4;

/// The hashed n-gram row ids of Qwen4ExpTextNGramEmbedding.
///
/// For token position `p`, the 2-gram and 3-gram heads hash the token and the one or two tokens before it, never
/// reaching back across an end-of-sequence token (a missing predecessor reads as the end-of-sequence token):
///
/// ```
/// shifted[s][p] = token[p - s]  if no EOS lies in token[p - s .. p - 1] and p - s >= 0, else EOS
/// mixed(n)      = XOR over s < n of shifted[s] * multiplier[s]        (64-bit)
/// id(head h of n-gram n) = mixed(n) mod vocab[h] + offset[h]
/// ```
///
/// Heads `0 .. headsPerNgram` belong to the 2-gram and the next `headsPerNgram` to the 3-gram. The two tokens before
/// a chunk are the state a sequence carries between calls ([Context]); a new sequence starts with two EOS tokens,
/// as upstream's cache does.
public final class NgramIds {

    /// The tokens before the next position: what survives between prefill chunks and decode steps.
    public static final class Context {
        private final int[] previous;

        public Context(int length, int endOfSequence) {
            this.previous = new int[length];
            reset(endOfSequence);
        }

        public void reset(int endOfSequence) {
            java.util.Arrays.fill(this.previous, endOfSequence);
        }

        public int length() {
            return this.previous.length;
        }

        public int token(int index) {
            return this.previous[index];
        }

        public Context copy() {
            Context copy = new Context(this.previous.length, 0);
            System.arraycopy(this.previous, 0, copy.previous, 0, this.previous.length);
            return copy;
        }
    }

    private final int ngramSize;
    private final int headsPerNgram;
    private final long[] multipliers;
    private final long[] headVocabSizes;
    private final long[] headOffsets;
    private final int endOfSequence;

    public NgramIds(
            int ngramSize,
            int headsPerNgram,
            long[] multipliers,
            long[] headVocabSizes,
            long[] headOffsets,
            int endOfSequence) {
        if (multipliers.length != ngramSize) throw new IllegalArgumentException("one multiplier per n-gram position");
        int heads = (ngramSize - 1) * headsPerNgram;
        if (headVocabSizes.length != heads || headOffsets.length != heads)
            throw new IllegalArgumentException("one vocabulary size and offset per head");
        this.ngramSize = ngramSize;
        this.headsPerNgram = headsPerNgram;
        this.multipliers = multipliers.clone();
        this.headVocabSizes = headVocabSizes.clone();
        this.headOffsets = headOffsets.clone();
        this.endOfSequence = endOfSequence;
    }

    public int heads() {
        return (this.ngramSize - 1) * this.headsPerNgram;
    }

    /// Tokens of context a sequence carries.
    public int contextLength() {
        return this.ngramSize - 1;
    }

    public Context newContext() {
        return new Context(contextLength(), this.endOfSequence);
    }

    /// Writes the `heads()` ids of each of `count` tokens to `ids[i * heads() + h]` and advances `context` past them.
    public void compute(Context context, int[] tokens, int offset, int count, long[] ids) {
        int heads = heads();
        if (context.length() != contextLength()) throw new IllegalArgumentException("context length");
        if (ids.length < (long) count * heads) throw new IllegalArgumentException("ids is too small");
        int history = contextLength();
        for (int position = 0; position < count; position++) {
            long mixed = tokenAt(context, tokens, offset, position, 0) * this.multipliers[0];
            int column = 0;
            for (int ngram = 2; ngram <= this.ngramSize; ngram++) {
                // The mix of an n-gram extends the previous one by the token n - 1 positions back.
                mixed ^= tokenAt(context, tokens, offset, position, ngram - 1) * this.multipliers[ngram - 1];
                for (int head = 0; head < this.headsPerNgram; head++, column++)
                    ids[position * heads + column] =
                            Math.floorMod(mixed, this.headVocabSizes[column]) + this.headOffsets[column];
            }
        }
        // The new context is entries count .. count + history - 1 of (context ++ tokens).
        int[] next = new int[history];
        for (int i = 0; i < history; i++) {
            int combined = count + i;
            next[i] = combined >= history ? tokens[offset + combined - history] : context.previous[combined];
        }
        System.arraycopy(next, 0, context.previous, 0, history);
    }

    /// The token `shift` positions before `position` when no end-of-sequence token lies between them, else EOS.
    private long tokenAt(Context context, int[] tokens, int offset, int position, int shift) {
        if (shift == 0) return tokens[offset + position];
        for (int back = 1; back <= shift; back++) {
            int token = tokenBefore(context, tokens, offset, position, back);
            if (token == this.endOfSequence) return this.endOfSequence;
        }
        return tokenBefore(context, tokens, offset, position, shift);
    }

    private int tokenBefore(Context context, int[] tokens, int offset, int position, int back) {
        int index = position - back;
        return index >= 0 ? tokens[offset + index] : context.previous[context.previous.length + index];
    }
}
