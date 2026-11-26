//! The drafter's training operations: what each launch of `train.cu` and each FP32 product is for the library, as
//! plain values.
//!
//! - the planners (`plan_*`): from a shape and device addresses to the [`Step`]s that run one kernel. A planner
//!   touches no device and refuses every shape a launch would fault on, so geometry is tested without a card.
//! - [`Gemm`]: one FP32 product on row-major matrices, which the cuBLAS handle of the lane runs on the stream the
//!   lane lends.
//! - [`TrainKernels`]: the resolved kernels of the training module, and the enqueue of planned steps on a lane.
//!
//! Tensors are FP32 and row-major; a linear's weight is `[out, in]`. Backward kernels add into their gradients (`+=`)
//! unless their name says otherwise, so a tensor used twice sums both contributions. Every address is a device
//! address.

mod gemm;
mod plan;
mod run;

use euhedral_gpu::GpuError;
use thiserror::Error;

pub use crate::kernels::{Step, StepError};
pub use gemm::{
    Gemm, Matrices, plan_linear, plan_linear_input_gradient, plan_product, plan_product_left_transposed,
    plan_product_transposed,
};
pub use plan::{
    AdamSettings, Heads, plan_adam, plan_bf16_to_float, plan_copy, plan_fill, plan_gate_backward, plan_gate_forward,
    plan_qk_norm_rope_backward, plan_qk_norm_rope_forward, plan_rms_norm_backward, plan_rms_norm_forward,
    plan_squared_error, plan_swiglu_backward, plan_swiglu_forward, plan_zero,
};
pub(crate) use plan::{THREADS, address, count64, dimension, elementwise, out_of_range};
pub use run::TrainKernels;

/// Why an operation cannot be planned or run.
#[derive(Debug, Error)]
pub enum TrainError {
    /// A device address is null.
    #[error("{0} is a null address")]
    NullAddress(&'static str),
    /// A dimension or count is zero, negative or beyond what the kernel's parameter holds.
    #[error("{name} {value} is out of range")]
    OutOfRange { name: &'static str, value: i64 },
    /// A leading dimension is shorter than the row it strides over.
    #[error("{name} {leading} is shorter than a row of {row}")]
    ShortLeading { name: &'static str, leading: i32, row: i32 },
    /// The head dimension is not one the kernel's block can be: a multiple of 32, at most 512.
    #[error("head dimension {0} is not a multiple of 32 up to 512")]
    HeadDimension(u32),
    /// The rotary width is odd or wider than the head.
    #[error("rotary width {rotary} is odd or wider than the head {head}")]
    Rotary { rotary: u32, head: u32 },
    /// The module has no kernel of this name.
    #[error("the training module has no kernel {0}")]
    NoKernel(String),
    /// A step could not be enqueued.
    #[error(transparent)]
    Step(#[from] StepError),
    /// The device, the driver or a library failed.
    #[error("device failure: {0}")]
    Gpu(String),
}

impl From<GpuError> for TrainError {
    fn from(error: GpuError) -> Self {
        Self::Gpu(error.to_string())
    }
}

#[cfg(test)]
mod tests;
