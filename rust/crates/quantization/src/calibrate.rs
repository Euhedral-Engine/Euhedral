//! The calibration of a BF16 checkpoint: the layers run in order over the calibration tokens, each linear's input
//! gathered into a Hessian, the layer's linears rounded under it and written, and the layer's output passed on as the
//! next layer's input.
//!
//! - [`plan`]: what a run covers (the layers, when it checkpoints) and how it resumes: the resume key, the decision,
//!   and the order the linears of a layer are rounded in.
//! - [`hessians`]: the Hessians of a layer's linears, which linears that read the same rows share, and the sum over the
//!   batches of a pass.
//! - [`driver`]: the layer loop, generic over the device's calls ([`Stages`]): the next layer is read while this one
//!   runs, and every step ends in the continuation of the call before it, never in a wait.
//! - [`device`]: the calls on the card: the layer pipeline, the GPTQ lanes and the files.
//!
//! The rounding arm is the GPTQ one. A format whose rounding differs (trellis) plugs in at [`Stages::layer`].

pub mod device;
pub mod driver;
pub mod hessians;
pub mod plan;
pub mod rounding;

use std::fmt::Display;
use thiserror::Error;

pub use driver::{Stages, Then, drive};
pub use hessians::HessianSet;
pub use plan::{Plan, round_order};

/// Why a calibration stopped.
#[derive(Debug, Error)]
pub enum CalibrateError {
    /// The run is set up wrongly: a value the calibration cannot use.
    #[error("{0}")]
    Invalid(String),
    /// A call on the card, a read or a write failed.
    #[error("{0}")]
    Failed(String),
}

impl CalibrateError {
    /// A failure of `what`, with the error's own text.
    pub fn failed(what: &str, error: impl Display) -> Self {
        CalibrateError::Failed(format!("{what}: {error}"))
    }
}
