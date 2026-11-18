//! What a caller asks of a generation, and what it is told when the generation ends.

use std::fmt::{self, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use thiserror::Error;

use super::constraint::TokenConstraint;
use super::listener::GenerationListener;
use crate::sampling::SamplingConfig;
use crate::stages::QuantumKind;

/// Whether the generation drafts tokens with the artifact's draft layer and verifies them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speculation {
    /// Draft with the artifact's MTP layer, as deep as the artifact states, and verify: the tokens are the ones
    /// `Off` selects. Refused for an artifact without the layer.
    Artifact,
    /// One row at a time.
    Off,
}

/// A prompt to continue.
pub struct GenerationRequest {
    /// The prompt's token ids.
    pub prompt: Vec<u32>,
    /// The most tokens to generate; a terminator counts, and is the last.
    pub max_tokens: u32,
    /// The ids that end a generation (the tokenizer's generation terminators). A selected terminator is reported and
    /// returned like any token and is not fed back to the model.
    pub stop_tokens: Vec<u32>,
    /// Speculation is exact, so it never changes the tokens; a request that samples does not speculate yet, and runs as
    /// if this were [`Speculation::Off`].
    pub speculation: Speculation,
    /// How a token is chosen from the logits. A greedy configuration (the flag set or a temperature of zero or less)
    /// selects on the device, as a request that never asked for sampling, unless it has a `constraint`; any other copies
    /// the logits row to the host and selects there, one token at a time, with the draw of the seed and the token's
    /// position.
    pub sampling: SamplingConfig,
    /// Which tokens each selection may choose. A constrained request selects on the host even when its configuration is
    /// greedy (the best allowed token), one token at a time, and never speculates: `speculation` is read as
    /// [`Speculation::Off`]. A constraint that fails ends the request with [`FinishReason::Failed`].
    pub constraint: Option<Box<dyn TokenConstraint>>,
    pub listener: Box<dyn GenerationListener>,
}

impl fmt::Debug for GenerationRequest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationRequest")
            .field("prompt_tokens", &self.prompt.len())
            .field("max_tokens", &self.max_tokens)
            .field("speculation", &self.speculation)
            .field("sampling", &self.sampling)
            .field("constrained", &self.constraint.is_some())
            .finish_non_exhaustive()
    }
}

/// The cancel of one generation. The caller makes it and passes it with the request
/// ([`EngineRuntime::generate_cancellable`](crate::EngineRuntime::generate_cancellable)), so a cancel asked for before or
/// while the generation starts is never lost: a handle already cancelled ends the generation before its first quantum.
/// One handle belongs to one generation. Clones share it. [`cancel`](CancelHandle::cancel) may be called from any thread, from a
/// listener's methods too, at any time and any number of times: it only sets a flag, never waits, and the generation's
/// owner observes it at its next turn, which is a quantum boundary (nothing is torn down mid-quantum). The generation then
/// ends through its listener, once, as [`FinishReason::Cancelled`], and the sequence is free for the next request. A
/// cancel after the generation ended does nothing.
#[derive(Clone, Debug, Default)]
pub struct CancelHandle {
    flag: Arc<AtomicBool>,
}

impl CancelHandle {
    /// Asks the generation to end at its next quantum boundary.
    pub fn cancel(&self) {
        // Relaxed: the flag carries no data. The owner only needs to see it at some later turn, and the turns that
        // follow are ordered after the retirements that start them.
        self.flag.store(true, Ordering::Relaxed);
    }

    /// Whether a cancel was requested (it stays so after the generation ended).
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

/// Why a request was refused before anything ran.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AdmissionError {
    #[error("the engine's sequence is running another generation")]
    Busy,
    #[error("the prompt has no tokens")]
    EmptyPrompt,
    #[error("the prompt has {tokens} tokens and the context holds {capacity}")]
    PromptTooLong { tokens: usize, capacity: usize },
    #[error("token {id} at position {index} is outside the vocabulary of {vocabulary}")]
    TokenOutsideVocabulary { id: u32, index: usize, vocabulary: usize },
    #[error("the artifact has no draft layer to speculate with")]
    SpeculationUnavailable,
    #[error("{0}")]
    Engine(String),
}

/// Why a generation ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// `max_tokens` tokens were generated.
    Length,
    /// A stop token was selected.
    Stop,
    /// The context was full before `max_tokens` tokens were generated.
    ContextFull,
    /// The generation was cancelled: it ends with the tokens reported before the cancel was observed.
    Cancelled,
    /// A quantum failed or the engine could not run one; the text says why.
    Failed(String),
}

/// How a generation ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationOutcome {
    pub reason: FinishReason,
    pub prompt_tokens: usize,
    /// Tokens selected and reported, a stop token included.
    pub generated: usize,
    /// From the start of the request to its end.
    pub elapsed: Duration,
}

/// A quantum that ran for the generation: what it was, and when it entered the engine and left it.
#[derive(Clone, Copy, Debug)]
pub struct QuantumEvent {
    pub kind: QuantumKind,
    pub rows: usize,
    pub start_position: u64,
    /// When the owner admitted it to its graph.
    pub admitted: Instant,
    /// When the owner found it retired.
    pub retired: Instant,
    /// Whether it ran as one launch of a recorded graph.
    pub replayed: bool,
    /// Whether its launches were recorded into a graph as it ran.
    pub recorded: bool,
}

/// One verification of a speculative step: what the step drafted, how much of it held, and what it cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Verification {
    /// Drafts the verifier confirmed, before any end-token or budget cut: 0 to `drafted`.
    pub accepted: usize,
    /// Drafts the verification checked.
    pub drafted: usize,
    /// Tokens the step committed to the output: the confirmed drafts and the verifier's token, cut after an end token
    /// and at the budget.
    pub committed: usize,
    /// From the retirement of the step's last draft quantum to the verification's: the verification's own time (the
    /// whole step when it drafted nothing).
    pub verify_nanos: u64,
    /// From the admission of the step to the retirement of its last draft quantum: the catch-up and the recursive
    /// drafts.
    pub draft_nanos: u64,
}

#[cfg(test)]
mod tests {
    use super::CancelHandle;

    #[test]
    fn a_cancel_is_idempotent_shared_by_clones_and_seen_from_any_thread() {
        let handle = CancelHandle::default();
        assert!(!handle.is_cancelled());
        let clone = handle.clone();
        clone.cancel();
        clone.cancel();
        assert!(handle.is_cancelled());
        handle.cancel();
        assert!(handle.is_cancelled());
        assert!(
            !CancelHandle::default().is_cancelled(),
            "a generation's handle is its own"
        );
    }
}
