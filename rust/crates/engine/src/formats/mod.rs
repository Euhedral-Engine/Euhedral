//! The interface the linear formats implement: how one weight multiplies a block of rows.
//!
//! This module declares the contract only. The formats (trellis, NVFP4, Q3 embedding, BF16) implement it, and the
//! stages that run a linear consume it through the format registry.

use euhedral_inference_cuda::{CudaError, DevicePtr, Submission};

use crate::model::WeightRef;

/// A region of the workspace's expansion scratch a linear takes for one launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchRegion {
    pub ptr: DevicePtr,
    pub bytes: usize,
}

/// How a launch is shaped by what is running it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinearMode {
    /// Every row computes as a one-row launch does (a speculative verification).
    pub row_exact: bool,
    /// The launch belongs to a prefill chunk, not to decode.
    pub prefill: bool,
}

/// One format's way of running `y = W x` over `rows` rows.
pub trait LinearFormat: Send + Sync {
    /// Queues the launches of one linear on `submission`'s lane: `weight` multiplies the `rows` rows at `x`, the
    /// result lands at `y`, and `scratch` holds what the route needs (at least [`LinearFormat::scratch_bytes`]).
    /// Nothing waits.
    #[allow(
        clippy::too_many_arguments,
        reason = "the signature the plan fixes for the formats and their consumers"
    )]
    fn submit(
        &self,
        submission: &Submission<'_>,
        weight: &WeightRef,
        x: DevicePtr,
        y: DevicePtr,
        rows: usize,
        scratch: ScratchRegion,
        mode: LinearMode,
    ) -> Result<(), CudaError>;

    /// The scratch bytes a launch of up to `rows` rows needs.
    fn scratch_bytes(&self, rows: usize) -> usize;
}
