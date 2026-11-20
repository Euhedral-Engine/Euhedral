use thiserror::Error;

/// Why a piece of the mixed allocation search refused its input.
#[derive(Debug, Error, PartialEq)]
pub enum MixedError {
    /// An argument or a file is not what the search accepts; the message names what.
    #[error("{0}")]
    Invalid(String),
    /// The budget is below what the cheapest choices cost.
    #[error("the budget of {0} bytes is below the cheapest choices")]
    BudgetTooSmall(i64),
    /// The residency plan could not be made for a reason other than memory.
    #[error("the residency plan failed: {0}")]
    Plan(String),
}

impl MixedError {
    pub(crate) fn invalid(message: impl Into<String>) -> MixedError {
        MixedError::Invalid(message.into())
    }
}
