//! NVFP4 linears: the routes of `Nvfp4Format.linear` and `euhedral_cuda_linear_nvfp4_bf16`
//! (native/src/host/nvfp4_linear.c).
//!
//! The route is a function of the rows, the shape and the layout (plain or SD4 scale tables), whether the launch is a
//! verification, and whether the device has the native Blackwell route:
//!
//! - **1 to 8 rows**: the tensor-core decode kernels on BF16 activations, one kernel per row count; a launch of M rows
//!   computes each row bit for bit as a one-row launch.
//! - **9 to 32 rows** of a verification (or where the native route is not there): the grouped twins of those kernels,
//!   in groups of 8 rows, row-exact as well.
//! - **9 rows and more** otherwise: the native route, which quantizes the activations to two NVFP4 terms into the
//!   scratch and multiplies with block-scaled FP4 tensor cores; a skinny kernel up to 64 rows, split over K to occupy
//!   the device, a tile kernel beyond. The native route declines shapes it has no kernel for.
//! - **Anything else**, and every launch under exact numerics: the scalar reference.

use euhedral_inference_cuda::launch::nvfp4::{NATIVE_MIN_ROWS, SKINNY_MAX_ROWS};
pub use euhedral_inference_cuda::launch::nvfp4::{Nvfp4Route, nvfp4_native_scratch_bytes, route};
use euhedral_inference_cuda::{CudaError, CudaLibrary, DevicePtr, KernelOps};
use euhedral_inference_edrl::{WeightFormat, WeightLayout};

use crate::formats::shapes::ShapeRecorder;
use crate::formats::{LinearFormat, LinearMode, ScratchRegion};
use crate::model::WeightRef;

/// The library's status for a route that declines a launch (`EUHEDRAL_CUDA_ROUTE_UNAVAILABLE`).
const ROUTE_UNAVAILABLE: i32 = -5;

/// The NVFP4 linears of a device.
#[derive(Debug)]
pub struct Nvfp4Linear {
    native_available: bool,
    record: bool,
}

impl Nvfp4Linear {
    /// Linears on `library`'s device: the native route runs where the device has it.
    pub fn new(library: &CudaLibrary) -> Self {
        Self {
            native_available: library.nvfp4_native_available(),
            record: false,
        }
    }

    /// Linears on a device that has the native route or does not (for tests).
    pub fn with_native_route(native_available: bool) -> Self {
        Self {
            native_available,
            record: false,
        }
    }

    /// Linears that list each launch they queue on [`ShapeRecorder`] (for tests).
    pub fn with_shape_recording(mut self) -> Self {
        self.record = true;
        self
    }

    /// The route of a launch of `rows` rows over `weight`, or `None` for a weight that is not a two-dimensional NVFP4
    /// object. `exact_numerics` is the library's process-wide oracle selection.
    pub fn route_of(
        &self,
        weight: &WeightRef,
        rows: usize,
        row_exact: bool,
        exact_numerics: bool,
    ) -> Option<Nvfp4Route> {
        let (out_features, in_features) = matrix(weight).ok()?;
        Some(route(
            u32::try_from(rows).ok()?,
            in_features,
            out_features,
            weight.layout == WeightLayout::RowSplitK128Sd4V1,
            row_exact,
            exact_numerics,
            self.native_available,
        ))
    }
}

impl LinearFormat for Nvfp4Linear {
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
        let (out_features, in_features) = matrix(weight)?;
        let rows = u32::try_from(rows)
            .ok()
            .filter(|rows| *rows > 0)
            .ok_or_else(|| CudaError::Argument(format!("an NVFP4 launch takes at least one row, not {rows}")))?;
        // A verification may be selected on the lane as well as asked for by the stage.
        let row_exact = mode.row_exact || kernels.row_exact();
        let sd4 = weight.layout == WeightLayout::RowSplitK128Sd4V1;
        // Exact numerics are the library's selection: its native route declines under them and the dispatch below
        // runs the reference, so the route is predicted without them.
        let route = route(
            rows,
            in_features,
            out_features,
            sd4,
            row_exact,
            false,
            self.native_available,
        );
        if self.record {
            for line in route.lines(rows, in_features, out_features, sd4) {
                ShapeRecorder::record(line);
            }
        }
        if matches!(route, Nvfp4Route::Native { .. }) {
            let native = kernels.linear_nvfp4_native_bf16(
                x,
                weight.ptr,
                y,
                scratch.ptr,
                rows,
                in_features,
                out_features,
                weight.bytes,
                scratch.bytes as u64,
            );
            match native {
                // The native route declines under exact numerics and for shapes it has no kernel for.
                Err(CudaError::Call {
                    status: ROUTE_UNAVAILABLE,
                    ..
                }) => {}
                other => return other,
            }
        }
        kernels.linear_nvfp4_bf16(x, weight.ptr, y, rows, in_features, out_features, weight.bytes)
    }

    /// Enough for any launch of up to `rows` rows. The skinny kernels split over K and keep FP32 partials, which grow
    /// with the rows up to 64 and are gone above it, so the tile route's scratch at 65 rows is smaller than the
    /// skinny route's at 64: the answer is the larger of the two routes' at `rows`, and of the skinny route's at its
    /// most rows. Below 9 rows no route takes scratch.
    fn scratch_bytes(&self, weight: &WeightRef, rows: usize, _mode: LinearMode) -> usize {
        let Ok((out_features, in_features)) = matrix(weight) else {
            return 0;
        };
        if !self.native_available || rows < NATIVE_MIN_ROWS as usize {
            return 0;
        }
        let (inputs, outputs) = (u64::from(in_features), u64::from(out_features));
        let rows = rows as u64;
        let bytes = nvfp4_native_scratch_bytes(rows, inputs, outputs).max(nvfp4_native_scratch_bytes(
            rows.min(u64::from(SKINNY_MAX_ROWS)),
            inputs,
            outputs,
        ));
        usize::try_from(bytes).unwrap_or(usize::MAX)
    }
}

/// `(out_features, in_features)` of an NVFP4 object.
fn matrix(weight: &WeightRef) -> Result<(u32, u32), CudaError> {
    let refuse = || CudaError::Argument(format!("{} is not a two-dimensional NVFP4 object", weight.name));
    if weight.format != WeightFormat::Nvfp4 {
        return Err(refuse());
    }
    match weight.shape.as_slice() {
        [out, input] => Ok((
            u32::try_from(*out).map_err(|_| refuse())?,
            u32::try_from(*input).map_err(|_| refuse())?,
        )),
        _ => Err(refuse()),
    }
}
