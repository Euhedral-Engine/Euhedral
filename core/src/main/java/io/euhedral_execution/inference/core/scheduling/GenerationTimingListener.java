package io.euhedral_execution.inference.core.scheduling;

/// Opt-in observer of the execution boundaries of one [QwenGenerationSession#generate] call.
///
/// Every timestamp is `System.nanoTime()`. Callbacks run between quanta on the lattice worker that advances
/// the generation (the one that retired the previous quantum), one at a time and in the order below; the
/// generating call returns after the last one. They must be cheap and must not call session operations.
/// Only successfully executed quanta are reported. Without a listener the session records nothing.
///
/// Boundaries, in call order:
/// - [#promptEncoded]: after prompt tokenization and validation, before the first prefill quantum.
/// - [#prefillQuantum]: once per prefill chunk. `startNanos` is taken before the quantum's context
///   is built; `executedNanos` after the Euhedral/CUDA runtime reports success and before sampling.
/// - [#firstTokenSelected]: after the first generated token is sampled from the final prefill
///   quantum's logits, including any JSON constraint acceptance. Not called when `maxNewTokens` is
///   zero or the call stops before sampling. Output callbacks and text decoding happen later.
/// - [#decodeQuantum]: once per decode quantum. Each decode quantum commits the previously
///   selected token; if another token is allowed it also samples the next one. The last quantum of a
///   call that reaches `maxNewTokens` only commits (`sampled` is false). A sampled generation
///   terminator is never submitted as a decode quantum.
///
/// Incremental text decoding for a token runs after its selection and before the decode quantum that
/// commits it; the output callback receives the text on the calling thread, concurrently with later quanta.
/// The final decoder flush runs after the last quantum.
public interface GenerationTimingListener {
    void promptEncoded(long nanos, int promptTokens);

    void prefillQuantum(long startNanos, long executedNanos, int tokens);

    void firstTokenSelected(long nanos, int tokenId);

    /// `selectedNanos` equals `executedNanos` and `selectedTokenId` is -1 when `sampled` is false.
    void decodeQuantum(long startNanos, long executedNanos, long selectedNanos, boolean sampled, int selectedTokenId);

    /// A cached prefix of `tokens` tokens was restored in `nanos` before the first prefill quantum.
    default void prefixRestored(int tokens, long nanos) {}

    /// Speculative decoding: one verification, started at `startNanos` (after the previous step's
    /// drafting) and resolved at `executedNanos`, committed `outputs` tokens, accepting `acceptedDrafts`
    /// drafts. It replaces decode quanta, except a final commit-only quantum at the token budget.
    default void speculativeStep(long startNanos, long executedNanos, int outputs, int acceptedDrafts) {}

    /// As [#speculativeStep(long, long, int, int)], also classifying a rejection: `rejection` is -1 when
    /// every draft was accepted, 0 when the base token the rejected draft should have been was in the
    /// draft head's shortlist (the MTP chose another), and 1 when it was not (no draft could match it).
    default void speculativeStep(long startNanos, long executedNanos, int outputs, int acceptedDrafts, int rejection) {
        speculativeStep(startNanos, executedNanos, outputs, acceptedDrafts);
    }

    /// Speculative decoding: one drafting quantum, admitted at `startNanos` and retired at `executedNanos`. `phase`
    /// names its part of the strategy: for DFlash2 `block` (the draft block: draft layers, output head, top-k and
    /// selector), `context` (the drafter's keys and values of a verification's committed rows) and `prompt-context`
    /// (of a prefill chunk's rows); for MTP `catch-up`, `prompt-catch-up` and `recursion`.
    default void draftQuantum(String phase, long startNanos, long executedNanos) {}
}
