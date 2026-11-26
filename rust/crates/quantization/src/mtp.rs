//! The multi-token-prediction drafter (the decoder's draft head) as a trainable FP32 model: its forward over a window
//! of context rows and draft depths, and the gradients of its parameters.
//!
//! The model is one decoder layer over the concatenation of a token's embedding and the previous hidden state, whose
//! projections run NVFP4-rounded (the artifact's rounding) with straight-through gradients into FP32 master weights,
//! unrolled over `depths` drafts of a window of rows. Row `r` of a window is a lineage: its depth-1 row reads the
//! base model's hidden at position `a + r` and the token at `a + r + 1`; its depth-`k` row reads the depth-`k - 1`
//! row's output and the token at `a + r + k`, sits at position `a + r + k - 1`, as the decoder's recursion does.
//! Attention keys: the depth-1 keys and values of every position up to the row's lineage position (earlier windows of
//! a sequence included, from `first_key`), and the deeper keys and values of its own lineage, which is what the
//! drafter's cache holds when the decoder drafts.
//!
//! - [`Drafter`]: the parameters, one device arena for everything, and the planners of the forward of a depth, the
//!   backward of a depth and the optimiser step. Nothing here touches a device: each call plans [`Work`] that the
//!   caller enqueues in order on one lane, and the arena belongs to the caller, who allocates [`Drafter::arena_bytes`]
//!   and gives its address.
//! - [`DrafterKernels`]: the resolved kernels of the training and drafter modules, and the enqueue of [`Work`].
//! - the planners of the drafter module's kernels (`plan_*`), like those of [`training`](crate::training).

mod drafter;
mod kernels;
mod layout;
mod plan;

use thiserror::Error;

use crate::training::{Gemm, Step, TrainError};

pub use drafter::{Drafter, Forward, Window};
pub use kernels::DrafterKernels;
pub use layout::{Config, Parameter, Shape};
pub use plan::{
    Lineage, Nvfp4Codes, Softmax, plan_absolute_maximum, plan_copy_columns, plan_cross_entropy, plan_lineage_dot,
    plan_lineage_gather, plan_lineage_spread, plan_nvfp4_round, plan_softmax, plan_softmax_backward,
};

/// One piece of work on a lane: kernel launches of one module, or one FP32 product, in the order planned.
#[derive(Clone, Debug)]
pub enum Work {
    /// Kernel launches, all of one module.
    Steps(Vec<Step>),
    /// An FP32 product on the lane's cuBLAS handle.
    Product(Gemm),
}

/// Why the drafter cannot be built, planned or run.
#[derive(Debug, Error)]
pub enum DrafterError {
    /// A dimension or setting of the configuration is not one the model can run with.
    #[error("{what} {value} is not valid for the drafter")]
    Config { what: &'static str, value: u64 },
    /// The arena address is null or not aligned to the device's allocations.
    #[error("the arena address {0:#x} is null or not 256-byte aligned")]
    Arena(u64),
    /// A depth outside `1..=depths`.
    #[error("depth {depth} is outside 1..={depths}")]
    Depth { depth: u32, depths: u32 },
    /// A window of no rows or of more rows than the drafter was built for.
    #[error("a window of {count} rows, at most {rows} fit")]
    Rows { count: u32, rows: u32 },
    /// A window whose first key lies after its position.
    #[error("the first key {first_key} lies after the window's position {position}")]
    FirstKey { first_key: u64, position: u64 },
    /// A window that reaches more keys than the drafter was built for.
    #[error("a window reaches {keys} keys, at most {max} fit")]
    Keys { keys: u64, max: u32 },
    /// A window that reaches past the positions the drafter was built for.
    #[error("position {position} is too far, at most {max} fit")]
    Position { position: u64, max: u32 },
    /// A pass that needs the forward pass of depth `.0` over the same window and rounding, which was not planned (or was
    /// planned over another window, with the other rounding, or has since been dropped by a shallower depth's pass).
    #[error("depth {0} has no forward pass over this window and rounding")]
    NotForwarded(u32),
    /// A backward pass or a step before the gradients and moments were zeroed with `initialise`.
    #[error("the gradients and moments were not zeroed: plan `initialise` first")]
    NotInitialised,
    /// A rounded forward pass before `measure`: every rounded weight would decode to zero.
    #[error("the projections' global scales were not measured: plan `measure` first")]
    NotMeasured,
    /// A floating-point setting that is not finite and positive.
    #[error("{what} {value} is not valid for the drafter")]
    Setting { what: &'static str, value: f64 },
    /// A training operation refused its shape or an address.
    #[error(transparent)]
    Train(#[from] TrainError),
}

#[cfg(test)]
mod tests;
