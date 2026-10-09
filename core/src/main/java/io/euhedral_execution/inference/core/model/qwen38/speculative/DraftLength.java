package io.euhedral_execution.inference.core.model.qwen38.speculative;

/// How many tokens an MTP step drafts before its verification (docs/MTP_DRAFT_CONFIDENCE.md): at least `least`, at
/// most `most`, and past `least` in blocks of `step` drafts, each block only while the running sum of the drafts'
/// log-probabilities under the draft head is at or above `threshold`. The verification checks every draft, so the
/// output is greedy decode's whatever the length; the length only decides how many tokens one verification can
/// commit. A step drafts `least + k * step` tokens (or `most`), so a coarse step keeps the verification's row counts,
/// and with them its captured graphs, few.
public record DraftLength(int least, int most, int step, float threshold) {

    public DraftLength {
        if (least < 1 || most < least || most > MtpDecoder.MAX_DRAFTS)
            throw new IllegalArgumentException(
                    "draft length must satisfy 1 <= least <= most <= " + MtpDecoder.MAX_DRAFTS);
        if (step < 1) throw new IllegalArgumentException("step must be positive");
        if (Float.isNaN(threshold)) throw new IllegalArgumentException("threshold must be a number");
    }

    /// Exactly `drafts` drafts per step.
    public static DraftLength fixed(int drafts) {
        return new DraftLength(drafts, drafts, 1, Float.NEGATIVE_INFINITY);
    }

    /// Whether the length depends on the drafts' log-probabilities.
    public boolean gated() {
        return this.least < this.most;
    }

    /// Whether a step that has drafted `drafted` tokens whose log-probabilities sum to `logProbability` drafts
    /// another. The sum is checked at `least` and at every block boundary after it; a sum that is not a number (an
    /// unscored draft) stops the step there.
    public boolean continues(int drafted, double logProbability) {
        if (drafted >= this.most) return false;
        if (drafted < this.least || (drafted - this.least) % this.step != 0) return true;
        return logProbability >= this.threshold;
    }
}
