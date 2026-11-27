//! The device operations of a stage as data: the order a stage queues them in, which a test reads without a device.
//!
//! Most are launches of the Flash-Next table ([`ops`](super::ops)); the NVFP4 linears, the GDN recurrence and the
//! residual add are the shared kernels Java reaches through `ExecutionGpu`, and the staging copy is the upload of the
//! gathered n-gram records.

use euhedral_inference_cuda::{CudaError, DevicePtr, KernelOps, PinnedBuffer};
use smallvec::SmallVec;

use crate::formats::shapes::Launch;

/// One device operation.
#[allow(
    clippy::large_enum_variant,
    reason = "an operation list is built per quantum into inline storage; boxing the launch would allocate for each"
)]
#[derive(Debug)]
pub enum Op<'a> {
    /// A kernel of the Flash-Next table.
    Launch(Launch),
    /// `output = input * weights^T` over NVFP4 weights (`ExecutionGpu.linearNvfp4Bf16`).
    Nvfp4Linear {
        input: DevicePtr,
        weights: DevicePtr,
        output: DevicePtr,
        rows: u32,
        in_features: u32,
        out_features: u32,
        weights_bytes: u64,
    },
    /// The gated delta rule over the convolved rows (`ExecutionGpu.gdnRecurrenceBf16`).
    GdnRecurrence {
        convolved: DevicePtr,
        alpha: DevicePtr,
        beta: DevicePtr,
        recurrent_state: DevicePtr,
        output: DevicePtr,
        rows: u32,
        key_heads: u32,
        value_heads: u32,
        key_head_dim: u32,
        value_head_dim: u32,
        output_scale: f32,
    },
    /// `output = residual + delta` (`ExecutionGpu.residualAddBf16`).
    ResidualAdd {
        residual: DevicePtr,
        delta: DevicePtr,
        output: DevicePtr,
        rows: u32,
        width: u32,
    },
    /// The copy of `bytes` bytes of a pinned buffer, at `offset`, to the device.
    Upload {
        destination: DevicePtr,
        source: &'a PinnedBuffer,
        offset: usize,
        bytes: usize,
    },
}

/// The operations of one stage, in the order they are queued.
pub type OpList<'a> = SmallVec<[Op<'a>; 16]>;

impl Op<'_> {
    /// Queues the operation on `kernels`.
    ///
    /// An [`Op::Upload`]'s source must stay alive, unmodified, until the lane's work has retired: the stage that builds
    /// it holds the buffer in the quantum.
    pub fn queue(&self, kernels: &mut dyn KernelOps) -> Result<(), CudaError> {
        match *self {
            Op::Launch(ref launch) => launch.queue(kernels),
            Op::Nvfp4Linear {
                input,
                weights,
                output,
                rows,
                in_features,
                out_features,
                weights_bytes,
            } => kernels.linear_nvfp4_bf16(input, weights, output, rows, in_features, out_features, weights_bytes),
            Op::GdnRecurrence {
                convolved,
                alpha,
                beta,
                recurrent_state,
                output,
                rows,
                key_heads,
                value_heads,
                key_head_dim,
                value_head_dim,
                output_scale,
            } => kernels.gdn_recurrence_bf16(
                convolved,
                alpha,
                beta,
                recurrent_state,
                output,
                rows,
                key_heads,
                value_heads,
                key_head_dim,
                value_head_dim,
                output_scale,
            ),
            Op::ResidualAdd {
                residual,
                delta,
                output,
                rows,
                width,
            } => kernels.residual_add_bf16(residual, delta, output, rows, width),
            Op::Upload {
                destination,
                source,
                offset,
                bytes,
            } => {
                // SAFETY: the stage that built the operation keeps the pinned buffer in the quantum, which is dropped
                // after the quantum's work retired, and does not write it again.
                unsafe { kernels.copy_upload_to_device(destination, source, offset, bytes) }
            }
        }
    }

    /// The operation as a line: a launch in the recorder's format, the others by name and arguments.
    pub fn line(&self) -> String {
        let address = |pointer: DevicePtr| format!("p{:x}", pointer.get());
        match *self {
            Op::Launch(ref launch) => launch.line(),
            Op::Nvfp4Linear {
                input,
                weights,
                output,
                rows,
                in_features,
                out_features,
                weights_bytes,
            } => format!(
                "nvfp4_linear {},{},{},i{rows},i{in_features},i{out_features},i{weights_bytes}",
                address(input),
                address(weights),
                address(output)
            ),
            Op::GdnRecurrence {
                convolved,
                alpha,
                beta,
                recurrent_state,
                output,
                rows,
                key_heads,
                value_heads,
                key_head_dim,
                value_head_dim,
                output_scale,
            } => format!(
                "gdn_recurrence {},{},{},{},{},i{rows},i{key_heads},i{value_heads},i{key_head_dim},i{value_head_dim},i{}",
                address(convolved),
                address(alpha),
                address(beta),
                address(recurrent_state),
                address(output),
                output_scale.to_bits()
            ),
            Op::ResidualAdd {
                residual,
                delta,
                output,
                rows,
                width,
            } => format!(
                "residual_add {},{},{},i{rows},i{width}",
                address(residual),
                address(delta),
                address(output)
            ),
            Op::Upload {
                destination,
                offset,
                bytes,
                ..
            } => format!("upload {} offset={offset} bytes={bytes}", address(destination)),
        }
    }
}

/// Queues `ops` in order on `kernels`.
pub fn queue_all(ops: &[Op<'_>], kernels: &mut dyn KernelOps) -> Result<(), CudaError> {
    ops.iter().try_for_each(|op| op.queue(kernels))
}

/// The lines of `ops`, in order.
pub fn lines(ops: &[Op<'_>]) -> Vec<String> {
    ops.iter().map(Op::line).collect()
}
