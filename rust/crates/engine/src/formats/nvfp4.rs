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

use euhedral_inference_cuda::{CudaError, CudaLibrary, DevicePtr, Submission, ops};
use euhedral_inference_edrl::{WeightFormat, WeightLayout};

use crate::formats::shapes::ShapeRecorder;
use crate::formats::{LinearFormat, LinearMode, ScratchRegion};
use crate::model::WeightRef;

/// The most rows of the decode kernels (`EUHEDRAL_DECODE_MAX_ROWS`).
const DECODE_MAX_ROWS: u32 = 8;

/// The most groups of 8 rows of the grouped twins (`DECODE_GROUPS_MAX`).
const DECODE_GROUPS_MAX: u32 = 4;

/// Rows from which an NVFP4 linear runs on the native route (`CudaGpuMemory.NVFP4_NATIVE_MIN_ROWS`).
const NATIVE_MIN_ROWS: u32 = 9;

/// The most rows of the native skinny kernels (`SKINNY_MAX_ROWS`).
const SKINNY_MAX_ROWS: u32 = 64;

/// Output columns of a block of the skinny kernels (`SKINNY_COLUMNS`).
const SKINNY_COLUMNS: u64 = 64;

/// The blocks the native library splits a skinny launch over K to reach (`SKINNY_TARGET_CTAS`, its own constant).
const SKINNY_TARGET_CTAS: u64 = 140;

/// Dynamic shared bytes of a tile kernel (`NATIVE_SHARED_BYTES`) and its threads (`NATIVE_TILE_THREADS`).
const NATIVE_SHARED_BYTES: u32 = 85040;
const NATIVE_TILE_THREADS: u32 = 288;

/// Activation terms of the native route: the value and its quantized residual (`NATIVE_TERMS`).
const NATIVE_TERMS: u64 = 2;

/// The library's status for a route that declines a launch (`EUHEDRAL_CUDA_ROUTE_UNAVAILABLE`).
const ROUTE_UNAVAILABLE: i32 = -5;

/// Where an NVFP4 linear runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Route {
    /// `rows` (1 to 8) rows on the decode kernels.
    DecodeRows { rows: u32 },
    /// Up to `groups * 8` rows (9 to 32) on the grouped twins.
    DecodeGroups { groups: u32 },
    /// The native Blackwell route: its skinny kernel (up to 64 rows) or its tile kernel.
    Native { skinny: bool },
    /// The scalar reference.
    Reference,
}

/// The route of a launch of `rows` rows over a weight of `out_features` by `in_features`.
///
/// `native_available` is whether the device has the native route; the native route is taken from 9 rows, never by a
/// verification (`row_exact`) and never under `exact_numerics`.
pub fn route(
    rows: u32,
    in_features: u32,
    out_features: u32,
    sd4: bool,
    row_exact: bool,
    exact_numerics: bool,
    native_available: bool,
) -> Nvfp4Route {
    let native = rows >= NATIVE_MIN_ROWS
        && !row_exact
        && !exact_numerics
        && in_features.is_multiple_of(128)
        && native_available
        && native_accepts(rows, in_features, sd4);
    if native {
        return Nvfp4Route::Native {
            skinny: skinny_route(rows, in_features),
        };
    }
    let qualified = rows <= 8 * DECODE_GROUPS_MAX && in_features.is_multiple_of(128) && out_features.is_multiple_of(16);
    if exact_numerics || !qualified {
        Nvfp4Route::Reference
    } else if rows > DECODE_MAX_ROWS {
        Nvfp4Route::DecodeGroups {
            groups: rows.div_ceil(8),
        }
    } else {
        Nvfp4Route::DecodeRows { rows }
    }
}

/// Whether the native route takes the shape: the skinny kernels any multiple of 256 inputs; the tile kernels' tensor
/// maps need whole scale rows (`launch_native`).
fn native_accepts(rows: u32, in_features: u32, sd4: bool) -> bool {
    skinny_route(rows, in_features) || in_features.is_multiple_of(if sd4 { 512 } else { 256 })
}

fn skinny_route(rows: u32, in_features: u32) -> bool {
    rows <= SKINNY_MAX_ROWS && in_features.is_multiple_of(256)
}

/// The decode tile of a shape: the 4-warp 32-row tile above 5120 inputs when the outputs allow it, else the 8-warp
/// 16-row tile. A function of the shape alone, so one-row and multi-row calls share their summation order.
fn decode_tile(in_features: u32, out_features: u32) -> bool {
    in_features > 5120 && out_features.is_multiple_of(32)
}

/// K splits of the skinny kernel: enough blocks to keep every streaming multiprocessor reading weights.
fn skinny_splits(in_features: u64, out_features: u64) -> u64 {
    let tiles = out_features.div_ceil(SKINNY_COLUMNS);
    let k_tiles = in_features / 256;
    let splits = SKINNY_TARGET_CTAS.div_ceil(tiles).min(k_tiles / 2);
    splits.max(1)
}

fn align256(value: u64) -> u64 {
    value.div_ceil(256) * 256
}

/// Bytes of the quantized activations of `rows` rows (`activation_bytes`): the codes, the tile-major scales and one
/// FP32 global per row.
fn activation_bytes(rows: u64, in_features: u64) -> u64 {
    let k = in_features.div_ceil(128) * 128;
    let planes = rows * NATIVE_TERMS;
    let pad = rows.div_ceil(128) * 128;
    let scales = align256(planes * k / 2);
    align256(scales + k / 128 * NATIVE_TERMS * pad * 8) + 4 * rows
}

/// The scratch one native linear needs (`euhedral_cuda_nvfp4_native_scratch_bytes`): the activations, then, for a
/// skinny launch split over K, FP32 partials.
pub fn nvfp4_native_scratch_bytes(rows: u64, in_features: u64, out_features: u64) -> u64 {
    let mut bytes = activation_bytes(rows, in_features);
    if skinny_route(rows as u32, in_features as u32) && skinny_splits(in_features, out_features) > 1 {
        bytes = align256(bytes) + 4 * skinny_splits(in_features, out_features) * rows * out_features;
    }
    bytes
}

/// Dynamic shared bytes of a skinny kernel of `rows` rows (`skinny_shared_bytes`).
fn skinny_shared_bytes(terms: u64, rows: u64) -> u64 {
    let stage = (terms * rows + SKINNY_COLUMNS) * 160;
    (101_376 / stage).min(4) * stage
}

impl Nvfp4Route {
    /// The launches of the route as lines of the recorder's format, without arguments: the library queues them
    /// itself, so the geometry is what it computes for the shape.
    pub fn lines(self, rows: u32, in_features: u32, out_features: u32, sd4: bool) -> Vec<String> {
        let suffix = if sd4 { "_sd4" } else { "" };
        let rows = u64::from(rows);
        let (inputs, outputs) = (u64::from(in_features), u64::from(out_features));
        match self {
            Nvfp4Route::DecodeRows { rows: count } => {
                let tile = decode_tile(in_features, out_features);
                vec![format!(
                    "euhedral_nvfp4_decode_rows{count}_{}{suffix} grid={},1,1 block={},1,1 smem=0",
                    if tile { "w4" } else { "w8" },
                    outputs / if tile { 32 } else { 16 },
                    if tile { 128 } else { 256 },
                )]
            }
            Nvfp4Route::DecodeGroups { groups } => {
                let tile = decode_tile(in_features, out_features);
                vec![format!(
                    "euhedral_nvfp4_decode_groups{groups}_{}{suffix} grid={},1,1 block={},1,1 smem=0",
                    if tile { "w4" } else { "w8" },
                    outputs / if tile { 32 } else { 16 },
                    if tile { 128 } else { 256 },
                )]
            }
            Nvfp4Route::Reference => {
                let grid = (rows * outputs).min(2_147_483_647);
                vec![format!(
                    "euhedral_nvfp4_reference{suffix} grid={grid},1,1 block=128,1,1 smem=0"
                )]
            }
            Nvfp4Route::Native { skinny } => {
                let mut lines = vec![format!(
                    "euhedral_nvfp4n_quantize_rows grid={rows},1,1 block=128,1,1 smem=0"
                )];
                if skinny {
                    let index = if rows <= 16 {
                        0
                    } else if rows <= 32 {
                        1
                    } else {
                        2
                    };
                    let splits = skinny_splits(inputs, outputs);
                    lines.push(format!(
                        "euhedral_nvfp4n_skinny_{}{suffix} grid={},{splits},1 block=128,1,1 smem={}",
                        16 << index,
                        outputs.div_ceil(SKINNY_COLUMNS),
                        skinny_shared_bytes(NATIVE_TERMS, 16 << index),
                    ));
                    if splits > 1 {
                        lines.push(format!(
                            "euhedral_nvfp4n_skinny_finish grid={},1,1 block=256,1,1 smem=0",
                            (rows * outputs).div_ceil(256)
                        ));
                    }
                } else {
                    lines.push(format!(
                        "euhedral_nvfp4n_linear_128x128{suffix} grid={},1,1 block={NATIVE_TILE_THREADS},1,1 \
                         smem={NATIVE_SHARED_BYTES}",
                        rows.div_ceil(128) * outputs.div_ceil(128)
                    ));
                }
                lines
            }
        }
    }
}

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
        submission: &Submission<'_>,
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
        let row_exact = mode.row_exact || submission.mode().row_exact;
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
            let native = ops::linear_nvfp4_native_bf16(
                submission,
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
        ops::linear_nvfp4_bf16(
            submission,
            x,
            weight.ptr,
            y,
            rows,
            in_features,
            out_features,
            weight.bytes,
        )
    }

    fn scratch_bytes(&self, weight: &WeightRef, rows: usize, _mode: LinearMode) -> usize {
        let Ok((out_features, in_features)) = matrix(weight) else {
            return 0;
        };
        if self.native_available && rows >= NATIVE_MIN_ROWS as usize {
            nvfp4_native_scratch_bytes(rows as u64, u64::from(in_features), u64::from(out_features)) as usize
        } else {
            0
        }
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
