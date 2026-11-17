//! The settings of one generation's sampling, and the ways a selection is refused.

use thiserror::Error;

/// Why a configuration or a selection was refused.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SamplingError {
    /// The temperature is NaN or infinite.
    #[error("temperature must be finite")]
    Temperature,

    /// The top-p is NaN, infinite, not above zero or above one.
    #[error("top_p must be finite and in the range (0, 1]")]
    TopP,

    /// A sampler needs a vocabulary of at least one token.
    #[error("the vocabulary must hold at least one token")]
    EmptyVocabulary,

    /// The row is not as wide as the vocabulary.
    #[error("expected {expected} vocabulary logits, got {got}")]
    RowWidth { expected: usize, got: usize },

    /// The row holds no token that is neither NaN nor negative infinity.
    #[error("logit row has no selectable token")]
    NoSelectableToken,

    /// The selectable scores leave no positive, finite total weight to normalise by.
    #[error("logit row has no normalizable probability mass")]
    NoNormalizableMass,

    /// The retained candidates carry no positive, finite probability.
    #[error("sampling filters removed all probability mass")]
    FiltersRemovedAllMass,
}

/// Sampling settings for one generation. A temperature at or below zero, or the greedy flag, selects the argmax;
/// a top-k of zero and a top-p of one disable those filters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingConfig {
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
    greedy: bool,
}

impl SamplingConfig {
    /// A configuration, refused when the temperature is not finite or the top-p is not finite and in (0, 1].
    pub fn new(temperature: f32, top_k: u32, top_p: f32, seed: u64, greedy: bool) -> Result<Self, SamplingError> {
        if !temperature.is_finite() {
            return Err(SamplingError::Temperature);
        }
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(SamplingError::TopP);
        }
        Ok(Self {
            temperature,
            top_k,
            top_p,
            seed,
            greedy,
        })
    }

    /// Argmax selection: temperature zero, no filters, the flag set.
    pub fn greedy(seed: u64) -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed,
            greedy: true,
        }
    }

    /// Whether every selection is the argmax of the row: the flag is set or the temperature is not above zero.
    pub fn is_greedy(&self) -> bool {
        self.greedy || self.temperature <= 0.0
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// How many of the best tokens are kept; zero keeps every selectable token.
    pub fn top_k(&self) -> u32 {
        self.top_k
    }

    pub fn top_p(&self) -> f32 {
        self.top_p
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }
}
