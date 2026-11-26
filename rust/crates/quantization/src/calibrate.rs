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
//! - [`schedule`]: which lane takes which linear of a layer, and the return of the lanes and the decoded weights.
//! - [`files`], [`embedding`], [`manifest`]: a file saved whole and the resume states in chunks; the checked rows of
//!   the embedding table; the manifest of how far a run got, and the output directory of a run of another key cleared.
//!
//! - [`heads`]: what a run of the whole model writes after its layers (the MTP block, the output head, the draft head)
//!   and the order they and the dropping of the resume file happen in; [`head_state`]: what the heads keep on the card,
//!   and the bookkeeping around a call that holds whatever the card does.
//!
//! A run of the whole model checkpoints its last layer's states, rounds the heads and drops the resume file once the
//! heads are listed; a head that fails leaves the checkpoint, and the next run starts at the heads.
//!
//! Two rounding arms share the driver: GPTQ, and trellis (each layer rounded through the trellis library, the heads with
//! the trellis recipe, the checkpoint written as exllamav3). The trellis arm plugs in at [`Stages::layer`].

pub mod allocation;
pub mod device;
pub mod draft_ids;
pub mod driver;
pub mod embedding;
pub mod files;
pub mod head_state;
pub mod heads;
pub mod hessians;
pub mod manifest;
pub mod plan;
pub mod rounding;
pub mod schedule;
pub mod trellis_plan;

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
