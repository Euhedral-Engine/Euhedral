//! The run that writes the drafter's training corpus: the artifact answers chat prompts, and each answer is appended
//! to a [`corpus`](crate::corpus) with the post-final-norm hidden of every row the model ran and the base model's
//! greedy choice at each position.
//!
//! - [`prompt`]: a prompts file's lines rendered with the chat template and tokenized.
//! - [`run`]: the run: up to `batch` prompts answered at once through the engine's hidden requests; a finished answer
//!   is appended from a host lake frame and starts the next prompt, so nothing waits.
//!
//! A prompt is answered greedily or sampled by a hash of its id, so a prompt gets the same kind of answer in every
//! run and the corpus keeps a fixed share of each: a greedy answer's labels are its tokens (what the drafter must
//! predict), a sampled answer's labels are the greedy choices the sampled path passed over.

pub mod prompt;
pub mod run;

use std::io;

use thiserror::Error;

pub use run::{Answerer, Options, Summary, start};

/// The longest prompt, in tokens: a longer one is refused, so the context holds the prompt and any answer.
pub const PROMPT_TOKENS: usize = 4096;

/// A rendered prompt: its id and its tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    pub id: String,
    pub tokens: Vec<u32>,
}

/// The seed of a prompt: the 64-bit FNV-1a hash of its id's UTF-8 bytes. It seeds the draw that picks the kind of
/// answer and, for a sampled answer, the sampler.
pub fn seed(id: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    id.bytes()
        .fold(OFFSET, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(PRIME))
}

/// Whether the prompt with this `seed` is answered greedily: the top 53 bits of the seed as a fraction of one, below
/// `fraction`. The seed is unsigned, so the shift is a logical one.
pub fn answers_greedily(seed: u64, fraction: f64) -> bool {
    const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
    ((seed >> 11) as f64) * SCALE < fraction
}

/// Why a run did not complete.
#[derive(Debug, Error)]
pub enum GenerateError {
    /// The engine refused a prompt, or an answer failed or was cut short.
    #[error("{0}")]
    Engine(String),
    /// The corpus could not be written.
    #[error("the corpus: {0}")]
    Corpus(String),
    /// What the engine reported does not add up: a row out of order, a count that differs.
    #[error("inconsistent answer: {0}")]
    Inconsistent(String),
    /// The run was dropped before it ended.
    #[error("the run was dropped before it ended")]
    Dropped,
}

impl From<io::Error> for GenerateError {
    fn from(error: io::Error) -> Self {
        GenerateError::Corpus(error.to_string())
    }
}
