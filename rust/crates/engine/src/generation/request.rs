//! What a caller asks of a generation, and what it is told when the generation ends.

use std::fmt::{self, Formatter};
use std::time::{Duration, Instant};

use thiserror::Error;

use super::listener::GenerationListener;
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
    pub speculation: Speculation,
    pub listener: Box<dyn GenerationListener>,
}

impl fmt::Debug for GenerationRequest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationRequest")
            .field("prompt_tokens", &self.prompt.len())
            .field("max_tokens", &self.max_tokens)
            .field("speculation", &self.speculation)
            .finish_non_exhaustive()
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
    /// Whether the launches of its stages were recorded into a graph beside it.
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
