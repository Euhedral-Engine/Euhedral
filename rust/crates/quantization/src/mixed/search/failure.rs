use thiserror::Error;

use super::super::error::MixedError;
use super::super::search_state::SearchError;

/// Why a search stopped.
#[derive(Debug, Error)]
pub enum SearchFailure {
    /// A setting or an input the search cannot honour; the message names the value and the limit.
    #[error("{0}")]
    Refused(String),
    /// Not even the smallest allocation fits the device at the target context.
    #[error(
        "even the smallest allocation does not fit at a context of {context} tokens with {free_mib} MiB free on the \
         device"
    )]
    NothingFits { context: u32, free_mib: i64 },
    /// The seeds measured nothing, so no model can be fitted.
    #[error("no candidate was measured: every seed is infeasible or was refused, so there is nothing to fit")]
    NothingMeasured,
    #[error("{0}")]
    Mixed(#[from] MixedError),
    #[error("{0}")]
    State(#[from] SearchError),
    /// The scorer failed on a candidate.
    #[error("scoring failed: {0}")]
    Scoring(String),
    /// A file of the front could not be written.
    #[error("writing the outputs failed: {0}")]
    Output(String),
    /// Host work did not come back (a frame was lost).
    #[error("host work was lost: {0}")]
    Host(String),
}

impl SearchFailure {
    pub(super) fn refused(message: impl Into<String>) -> SearchFailure {
        SearchFailure::Refused(message.into())
    }
}
