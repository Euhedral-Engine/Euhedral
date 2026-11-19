//! Grouped linears: Q3, Q4 and Q5 values in 64-value groups with FP16 scales, and Q3 entropy coded as P2E2 (the routes
//! of `GroupedLinearFormat` and `P2e2Format`, which the native entry points in `q3_linear_bf16.c`, `q45_linear.c` and
//! `q3_mx.c` serve).
//!
//! The route is a function of the rows, the shape and the layout:
//!
//! - **9 rows and more** on a device with block-scaled FP8 tensor cores (and not for a verification, which computes
//!   each row as a one-row launch does): the FP8 route, which quantizes the activations into the scratch. It declines
//!   shapes it has no kernel for and runs under no exact numerics, and the launch then takes the route below.
//! - **Anything else**: the decode kernels (up to 8 rows contiguous, up to 32 on the token-row twins) or, for a shape
//!   or an alignment they do not take and under exact numerics, the scalar reference.
//! - **A P2E2 tensor** of up to 8 rows runs on its fused decode kernels where they take the launch; every other launch
//!   expands the tensor into the scratch first and runs the row-split route on the expansion, so the outputs are the
//!   same bits in every case.

use euhedral_inference_cuda::launch::q3::{
    Bits, DECODE_MAX_ROWS, MX_MIN_ROWS, P2E2_EXPAND_LIMIT, linear_scratch_bytes, mx_scratch_bytes, q3_byte_size,
};
use euhedral_inference_cuda::{CudaError, CudaLibrary, DevicePtr, KernelOps};
use euhedral_inference_edrl::{WeightFormat, WeightLayout};

use crate::formats::{LinearFormat, LinearMode, ScratchRegion};
use crate::model::WeightRef;

/// The library's status for a route that declines a launch (`EUHEDRAL_CUDA_ROUTE_UNAVAILABLE`).
const ROUTE_UNAVAILABLE: i32 = -5;

/// The Q3, Q4 and Q5 linears of a device.
#[derive(Debug)]
pub struct Q3Linear {
    mx_available: bool,
    sm_count: u32,
}

impl Q3Linear {
    /// Linears on `library`'s device: the FP8 route runs where the device has it.
    pub fn new(library: &CudaLibrary) -> Self {
        Self::with_mx_route(library.q3_mx_available(), library.device().sm_count)
    }

    /// Linears on a device of `sm_count` streaming multiprocessors that has the FP8 route or does not (for tests).
    pub fn with_mx_route(mx_available: bool, sm_count: u32) -> Self {
        Self { mx_available, sm_count }
    }

    /// Runs a row-split tensor of `bits` at `weights`: the FP8 route where it runs and `activations` holds its scratch,
    /// else the decode kernels.
    #[allow(clippy::too_many_arguments, reason = "one launch's operands")]
    fn run_row_split(
        &self,
        kernels: &mut dyn KernelOps,
        bits: Bits,
        shape: Shape,
        x: DevicePtr,
        weights: DevicePtr,
        weights_bytes: u64,
        y: DevicePtr,
        rows: u32,
        activations: ScratchRegion,
        row_exact: bool,
    ) -> Result<(), CudaError> {
        let Shape { out, input } = shape;
        let tiled = input.is_multiple_of(128) && out.is_multiple_of(128);
        if rows >= MX_MIN_ROWS && self.mx_available && !row_exact && tiled {
            let needed = mx_scratch_bytes(rows, input, out, self.sm_count, 0);
            if (activations.bytes as u64) < needed {
                return Err(CudaError::Argument(format!(
                    "a {}-bit linear of {rows} rows needs {needed} bytes of scratch and has {}",
                    bits.number(),
                    activations.bytes
                )));
            }
            let mx = kernels.linear_q3_mx_bf16(
                bits.number(),
                x,
                weights,
                y,
                activations.ptr,
                rows,
                input,
                out,
                weights_bytes,
                activations.bytes as u64,
                0,
            );
            match mx {
                // The FP8 route declines under exact numerics and for alignments it has no kernel for.
                Err(CudaError::Call {
                    status: ROUTE_UNAVAILABLE,
                    ..
                }) => {}
                other => return other,
            }
        }
        match bits {
            Bits::Q3 => kernels.linear_q3_bf16(x, weights, y, rows, input, out, weights_bytes),
            Bits::Q4 | Bits::Q5 => {
                kernels.linear_quantized_bf16(x, weights, y, rows, input, out, weights_bytes, bits.number())
            }
        }
    }
}

impl LinearFormat for Q3Linear {
    fn submit(
        &self,
        kernels: &mut dyn KernelOps,
        weight: &WeightRef,
        x: DevicePtr,
        y: DevicePtr,
        rows: usize,
        scratch: ScratchRegion,
        mode: LinearMode,
    ) -> Result<(), CudaError> {
        let (bits, shape, p2e2) = describe(weight)?;
        let rows = u32::try_from(rows)
            .ok()
            .filter(|rows| *rows > 0)
            .ok_or_else(|| CudaError::Argument(format!("a grouped linear takes at least one row, not {rows}")))?;
        // A verification may be selected on the lane as well as asked for by the stage.
        let row_exact = mode.row_exact || kernels.row_exact();
        if !p2e2 {
            return self.run_row_split(
                kernels,
                bits,
                shape,
                x,
                weight.ptr,
                weight.bytes,
                y,
                rows,
                scratch,
                row_exact,
            );
        }
        let Shape { out, input } = shape;
        if rows <= DECODE_MAX_ROWS {
            let decode = kernels.linear_q3_p2e2_decode_bf16(x, weight.ptr, y, rows, input, out, weight.bytes);
            match decode {
                Err(CudaError::Call {
                    status: ROUTE_UNAVAILABLE,
                    ..
                }) => {}
                other => return other,
            }
        }
        // Every other launch expands the tensor into the scratch and runs the row-split route on the expansion, with
        // the activations of the FP8 route behind it.
        let expanded = q3_byte_size(out, input);
        if expanded > P2E2_EXPAND_LIMIT {
            return Err(CudaError::Argument(format!(
                "{} expands to {expanded} bytes, more than the {P2E2_EXPAND_LIMIT} the scratch takes whole",
                weight.name
            )));
        }
        let offset = expanded.next_multiple_of(256);
        let reserve = if rows >= MX_MIN_ROWS && self.mx_available && !row_exact && input.is_multiple_of(128) {
            mx_scratch_bytes(rows, input, out, self.sm_count, 0)
        } else {
            0
        };
        if (scratch.bytes as u64) < offset + reserve {
            return Err(CudaError::Argument(format!(
                "a launch of {rows} rows over {} needs {} bytes of scratch and has {}",
                weight.name,
                offset + reserve,
                scratch.bytes
            )));
        }
        kernels.q3_p2e2_expand(weight.ptr, weight.bytes, out, input, 0, out, scratch.ptr, expanded)?;
        let activations = ScratchRegion {
            ptr: scratch.ptr.offset(offset),
            bytes: reserve as usize,
        };
        self.run_row_split(
            kernels,
            Bits::Q3,
            shape,
            x,
            scratch.ptr,
            expanded,
            y,
            rows,
            activations,
            row_exact,
        )
    }

    fn scratch_bytes(&self, weight: &WeightRef, rows: usize, _mode: LinearMode) -> usize {
        let Ok((_, Shape { out, input }, p2e2)) = describe(weight) else {
            return 0;
        };
        let Ok(rows) = u32::try_from(rows) else {
            return usize::MAX;
        };
        // A tensor too large to expand whole is refused at submission, which needs no scratch.
        let bytes = linear_scratch_bytes(p2e2, rows, input, out, self.mx_available, Some(self.sm_count)).unwrap_or(0);
        usize::try_from(bytes).unwrap_or(usize::MAX)
    }
}

/// `out_features` and `in_features` of a grouped tensor.
#[derive(Clone, Copy, Debug)]
struct Shape {
    out: u32,
    input: u32,
}

/// The bits, shape and compression of a grouped linear, or the refusal of a weight that is none.
fn describe(weight: &WeightRef) -> Result<(Bits, Shape, bool), CudaError> {
    let refuse = || CudaError::Argument(format!("{} is not a two-dimensional Q3, Q4 or Q5 object", weight.name));
    let bits = match weight.format {
        WeightFormat::Q3G64Fp16 => Bits::Q3,
        WeightFormat::Q4G64Fp16 => Bits::Q4,
        WeightFormat::Q5G64Fp16 => Bits::Q5,
        _ => return Err(refuse()),
    };
    let p2e2 = match (bits, weight.layout) {
        (_, WeightLayout::RowSplitK128V1) => false,
        (Bits::Q3, WeightLayout::RowSplitP2e2V1) => true,
        _ => return Err(refuse()),
    };
    match weight.shape.as_slice() {
        [out, input] => Ok((
            bits,
            Shape {
                out: u32::try_from(*out).map_err(|_| refuse())?,
                input: u32::try_from(*input).map_err(|_| refuse())?,
            },
            p2e2,
        )),
        _ => Err(refuse()),
    }
}
