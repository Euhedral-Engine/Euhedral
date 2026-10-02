package io.euhedral_execution.inference.core.scheduling;

/// Opt-in observer of the execution boundaries of one [QwenGenerationSession#generate] call.
///
/// Every timestamp is `System.nanoTime()` taken on the generating thread. Callbacks run
/// synchronously between quanta, so they must be cheap and must not call session operations.
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
/// The output callback and incremental text decoding for a token run after its selection and
/// before the decode quantum that commits it, so they fall between reported quanta. The final
/// decoder flush runs after the last quantum.
public interface GenerationTimingListener {
    void promptEncoded(long nanos, int promptTokens);

    void prefillQuantum(long startNanos, long executedNanos, int tokens);

    void firstTokenSelected(long nanos, int tokenId);

    /// `selectedNanos` equals `executedNanos` and `selectedTokenId` is -1 when `sampled` is false.
    void decodeQuantum(long startNanos, long executedNanos, long selectedNanos, boolean sampled, int selectedTokenId);

    /// Speculative decoding: one verification, started at `startNanos` (after the previous step's
    /// drafting) and resolved at `executedNanos`, committed `outputs` tokens, accepting `acceptedDrafts`
    /// drafts. It replaces decode quanta, except a final commit-only quantum at the token budget.
    default void speculativeStep(long startNanos, long executedNanos, int outputs, int acceptedDrafts) {}
}
