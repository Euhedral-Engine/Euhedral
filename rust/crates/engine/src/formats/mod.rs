//! The linear formats: how one weight multiplies a block of rows.
//!
//! [`LinearFormat`] is the contract the stages consume. [`TrellisLinear`] and [`Nvfp4Linear`] implement it with the
//! kernel choice, grid and K split of the Java formats, and [`Q3Embedding`] looks up token rows of a Q3 table.
//! [`ShapeRecorder`] lists the launches a format queued, for the tests that compare them with Java's.

mod embedding;
mod nvfp4;
pub mod shapes;
mod trellis;

use euhedral_inference_cuda::{CudaError, DevicePtr, Submission};

use crate::model::WeightRef;

pub use embedding::Q3Embedding;
pub use nvfp4::{Nvfp4Linear, Nvfp4Route, nvfp4_native_scratch_bytes};
pub use shapes::{KernelArguments, Launch, LinearKernel, ShapeRecorder, Table};
pub use trellis::{ROW_EXACT_KERNELS, TrellisLinear, TrellisOperands, TrellisShape};

/// A region of the workspace's expansion scratch a linear takes for one launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchRegion {
    pub ptr: DevicePtr,
    pub bytes: usize,
}

impl ScratchRegion {
    /// No scratch: the launch of a route that takes none.
    pub const NONE: ScratchRegion = ScratchRegion {
        ptr: DevicePtr::NULL,
        bytes: 0,
    };
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

    /// The scratch bytes a launch of up to `rows` rows over `weight` needs. The route and its scratch are a function
    /// of the weight's shape, so the scratch of a format is asked per weight. A launch that is not `mode.prefill`
    /// never takes the prefill route (`LinearFormat.decodeScratchBytes`), so it needs less. 0 for a weight this
    /// format does not run (its submission refuses it).
    fn scratch_bytes(&self, weight: &WeightRef, rows: usize, mode: LinearMode) -> usize;
}
