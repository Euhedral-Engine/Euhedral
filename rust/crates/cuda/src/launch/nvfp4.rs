//! Plans the launches of the NVFP4 linear kernels (decode, reference and native), as the native library's host code
//! (`native/src/host/nvfp4_linear.c`, `reference.c`) issues them: the same kernels in the same order with the same grid,
//! block, dynamic shared bytes and argument words.
//!
//! The functions are pure: shapes and addresses in, a [`Nvfp4Plan`] (at most three launches, no allocation) or a
//! [`PlanError`] out. A refusal is a status the C entry point returns before it launches anything.
//!
//! The native tile kernel takes three tensor maps by value as its leading arguments. A plan carries the parameters
//! each map is built from ([`TensorMapPlan`], what the C code passes to `cuTensorMapEncodeTiled`) and the launch's own
//! arguments start after the maps; the launcher encodes the maps and puts them in front.

use crate::DevicePtr;
use crate::launch::PlanError;
use crate::launch::catalog::{Arguments, KernelId, Launch, find_const};

/// The most rows of the decode kernels (`EUHEDRAL_DECODE_MAX_ROWS`, `decode_shapes.h`).
pub const DECODE_MAX_ROWS: u32 = 8;

/// The most groups of 8 rows of the grouped twins (`DECODE_GROUPS_MAX`, `nvfp4_linear.c`).
pub const DECODE_GROUPS_MAX: u32 = 4;

/// Rows from which an NVFP4 linear runs on the native route (`CudaGpuMemory.NVFP4_NATIVE_MIN_ROWS`, the engine's
/// choice; the native entry point itself takes any row count).
pub const NATIVE_MIN_ROWS: u32 = 9;

/// The most rows of the native skinny kernels (`SKINNY_MAX_ROWS`, `nvfp4_linear.c`).
pub const SKINNY_MAX_ROWS: u32 = 64;

/// Output columns of a block of the skinny kernels (`SKINNY_COLUMNS`, `nvfp4_linear.c`).
const SKINNY_COLUMNS: u64 = 64;

/// The blocks the native library splits a skinny launch over K to reach: two per streaming multiprocessor, a machine
/// measurement (`SKINNY_TARGET_CTAS`, `nvfp4_linear.c`), ported as it is.
const SKINNY_TARGET_CTAS: u64 = 140;

/// Dynamic shared bytes of a tile kernel (`NATIVE_SHARED_BYTES`, `nvfp4_linear.c`).
pub const NATIVE_SHARED_BYTES: u32 = 85040;

/// Threads of a tile kernel (`NATIVE_TILE_THREADS`, `nvfp4_linear.c`).
pub const NATIVE_TILE_THREADS: u32 = 288;

/// Activation terms of the native route: the value and its quantized residual (`NATIVE_TERMS`, `nvfp4_linear.c`).
const NATIVE_TERMS: u64 = 2;

/// The largest grid-x of a launch (`2147483647`, `reference.c` and `nvfp4_linear.c`).
const MAX_GRID: u64 = 2_147_483_647;

/// How an NVFP4 weight stores its block scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Layout {
    /// One E4M3 scale per 16 values, then the FP32 global.
    Plain,
    /// A 4-bit scale index per 16 values, then the 16-code table and the global (`row-split-k128-sd4-v1`).
    Sd4,
}

impl Nvfp4Layout {
    const fn index(self) -> usize {
        matches!(self, Nvfp4Layout::Sd4) as usize
    }
}

/// The shape of one linear and the addresses it works on.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4Operands {
    pub input: DevicePtr,
    pub weights: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub in_features: u32,
    pub out_features: u32,
    /// The weight's byte size; the layout is told from it.
    pub weights_byte_size: u64,
}

/// The parameters of one tensor map, as the C code passes them to `cuTensorMapEncodeTiled` (`byte_map`,
/// `nvfp4_linear.c`). Every map is over bytes (`UINT8`) with unit element strides, no interleave, 128-byte L2
/// promotion and no out-of-bounds fill; those do not vary and are not fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorMapPlan {
    /// The global address (`CUdeviceptr`).
    pub address: u64,
    /// The rank (2 or 3): how many of `dims` and `box_dims`, and one fewer of `strides`, are used.
    pub rank: u32,
    /// Extents in elements (bytes), innermost first.
    pub dims: [u64; 3],
    /// Byte strides of dimensions 1 and up.
    pub strides: [u64; 2],
    /// Box extents in elements.
    pub box_dims: [u32; 3],
    pub swizzle: TensorMapSwizzle,
}

/// The swizzle of a tensor map (`CUtensorMapSwizzle`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorMapSwizzle {
    None,
    B64,
}

impl TensorMapPlan {
    /// The extents in use.
    pub fn dims(&self) -> &[u64] {
        &self.dims[..self.rank as usize]
    }

    /// The strides in use.
    pub fn strides(&self) -> &[u64] {
        &self.strides[..self.rank as usize - 1]
    }

    /// The box extents in use.
    pub fn box_dims(&self) -> &[u32] {
        &self.box_dims[..self.rank as usize]
    }
}

/// The launches of one linear, in order (at most three), and the tensor maps of the last one when it is a tile launch.
#[derive(Clone, Debug)]
pub struct Nvfp4Plan {
    launches: [Launch; 3],
    len: usize,
    maps: Option<[TensorMapPlan; 3]>,
}

/// Plans are equal when their launches in use and their maps are; the unused slots are filler.
impl PartialEq for Nvfp4Plan {
    fn eq(&self, other: &Self) -> bool {
        self.launches() == other.launches() && self.maps == other.maps
    }
}

impl Eq for Nvfp4Plan {}

impl Nvfp4Plan {
    fn single(launch: Launch) -> Self {
        Self {
            launches: [launch; 3],
            len: 1,
            maps: None,
        }
    }

    fn push(&mut self, launch: Launch) {
        self.launches[self.len] = launch;
        self.len += 1;
    }

    /// The launches, in the order to issue them. The tile kernel's launch takes the maps of [`Self::tensor_maps`] as
    /// its leading arguments: they must be placed in front of the arguments the launch carries.
    pub fn launches(&self) -> &[Launch] {
        &self.launches[..self.len]
    }

    /// The tensor maps (activation codes, weight codes, weight scales) the last launch takes as its three leading
    /// arguments, before the arguments it carries; `None` for every launch but the tile kernel's.
    pub fn tensor_maps(&self) -> Option<&[TensorMapPlan; 3]> {
        self.maps.as_ref()
    }
}

// The decode kernels by [layout][tile][rows - 1] and [layout][tile][groups - 2]; tile 0 is `_w8`, 1 is `_w4`.
macro_rules! rows_kernels {
    ($tile:literal, $suffix:literal) => {
        [
            find_const(concat!("euhedral_nvfp4_decode_rows1_", $tile, $suffix)),
            find_const(concat!("euhedral_nvfp4_decode_rows2_", $tile, $suffix)),
            find_const(concat!("euhedral_nvfp4_decode_rows3_", $tile, $suffix)),
            find_const(concat!("euhedral_nvfp4_decode_rows4_", $tile, $suffix)),
            find_const(concat!("euhedral_nvfp4_decode_rows5_", $tile, $suffix)),
            find_const(concat!("euhedral_nvfp4_decode_rows6_", $tile, $suffix)),
            find_const(concat!("euhedral_nvfp4_decode_rows7_", $tile, $suffix)),
            find_const(concat!("euhedral_nvfp4_decode_rows8_", $tile, $suffix)),
        ]
    };
}

macro_rules! groups_kernels {
    ($tile:literal, $suffix:literal) => {
        [
            find_const(concat!("euhedral_nvfp4_decode_groups2_", $tile, $suffix)),
            find_const(concat!("euhedral_nvfp4_decode_groups3_", $tile, $suffix)),
            find_const(concat!("euhedral_nvfp4_decode_groups4_", $tile, $suffix)),
        ]
    };
}

const DECODE_ROWS: [[[KernelId; 8]; 2]; 2] = [
    [rows_kernels!("w8", ""), rows_kernels!("w4", "")],
    [rows_kernels!("w8", "_sd4"), rows_kernels!("w4", "_sd4")],
];

const DECODE_GROUPS: [[[KernelId; 3]; 2]; 2] = [
    [groups_kernels!("w8", ""), groups_kernels!("w4", "")],
    [groups_kernels!("w8", "_sd4"), groups_kernels!("w4", "_sd4")],
];

const REFERENCE: [KernelId; 2] = [
    find_const("euhedral_nvfp4_reference"),
    find_const("euhedral_nvfp4_reference_sd4"),
];

const TILE: [KernelId; 2] = [
    find_const("euhedral_nvfp4n_linear_128x128"),
    find_const("euhedral_nvfp4n_linear_128x128_sd4"),
];

/// By [layout][rows up to 16, 32 and 64].
const SKINNY: [[KernelId; 3]; 2] = [
    [
        find_const("euhedral_nvfp4n_skinny_16"),
        find_const("euhedral_nvfp4n_skinny_32"),
        find_const("euhedral_nvfp4n_skinny_64"),
    ],
    [
        find_const("euhedral_nvfp4n_skinny_16_sd4"),
        find_const("euhedral_nvfp4n_skinny_32_sd4"),
        find_const("euhedral_nvfp4n_skinny_64_sd4"),
    ],
];

const QUANTIZE: KernelId = find_const("euhedral_nvfp4n_quantize_rows");
const FINISH: KernelId = find_const("euhedral_nvfp4n_skinny_finish");

fn align256(value: u64) -> u64 {
    value.div_ceil(256) * 256
}

/// Byte size of an NVFP4 tensor of `rows` rows of `in_features` values (`nvfp4_size`, `nvfp4_linear.c`).
pub fn nvfp4_byte_size(rows: u32, in_features: u32, layout: Nvfp4Layout) -> u64 {
    let sd4 = layout == Nvfp4Layout::Sd4;
    let k = u64::from(in_features).div_ceil(128) * 128;
    let rows = u64::from(rows);
    let scales = align256(rows * k / 2);
    align256(scales + rows * k / if sd4 { 32 } else { 16 }) + if sd4 { 20 } else { 4 }
}

/// The layout a byte size says (`nvfp4_layout`, `nvfp4_linear.c`); `None` when it matches neither.
pub fn nvfp4_layout(rows: u32, in_features: u32, byte_size: u64) -> Option<Nvfp4Layout> {
    [Nvfp4Layout::Plain, Nvfp4Layout::Sd4]
        .into_iter()
        .find(|layout| byte_size == nvfp4_byte_size(rows, in_features, *layout))
}

/// Whether the decode kernels take 128-multiple inputs and 16-multiple outputs up to 32 rows (`qualified`,
/// `nvfp4_linear.c`).
fn decode_qualified(rows: u32, in_features: u32, out_features: u32) -> bool {
    rows <= 8 * DECODE_GROUPS_MAX && in_features.is_multiple_of(128) && out_features.is_multiple_of(16)
}

/// The decode tile of a shape: the 4-warp 32-row tile (`true`) above 5120 inputs when the outputs allow it, else the
/// 8-warp 16-row tile (`decode_tile`, `nvfp4_linear.c`).
pub fn decode_tile(in_features: u32, out_features: u32) -> bool {
    in_features > 5120 && out_features.is_multiple_of(32)
}

/// Whether the skinny kernels take the shape (`skinny_route`, `nvfp4_linear.c`).
pub fn skinny_route(rows: u64, in_features: u64) -> bool {
    rows <= u64::from(SKINNY_MAX_ROWS) && in_features.is_multiple_of(256)
}

/// K splits of the skinny kernel: enough blocks to keep every streaming multiprocessor reading weights
/// (`skinny_splits`, `nvfp4_linear.c`).
pub fn skinny_splits(in_features: u64, out_features: u64) -> u64 {
    let tiles = out_features.div_ceil(SKINNY_COLUMNS);
    let k_tiles = in_features / 256;
    let splits = SKINNY_TARGET_CTAS.div_ceil(tiles).min(k_tiles / 2);
    splits.max(1)
}

/// Dynamic shared bytes of a skinny kernel of `rows` rows with `terms` activation terms (`skinny_shared_bytes`,
/// `nvfp4_linear.c`).
pub fn skinny_shared_bytes(terms: u64, rows: u64) -> u64 {
    let stage = (terms * rows + SKINNY_COLUMNS) * 160;
    (101_376 / stage).min(4) * stage
}

/// Bytes of the quantized activations of `rows` rows (`activation_bytes`, `nvfp4_linear.c`): the codes, the
/// tile-major scales and one FP32 global per row.
fn activation_bytes(rows: u64, in_features: u64) -> u64 {
    let k = in_features.div_ceil(128) * 128;
    let planes = rows * NATIVE_TERMS;
    let pad = rows.div_ceil(128) * 128;
    let scales = align256(planes * k / 2);
    align256(scales + k / 128 * NATIVE_TERMS * pad * 8) + 4 * rows
}

/// The scratch one native linear needs (`euhedral_cuda_nvfp4_native_scratch_bytes`, `nvfp4_linear.c`): the
/// activations, then, for a skinny launch split over K, FP32 partials.
pub fn nvfp4_native_scratch_bytes(rows: u64, in_features: u64, out_features: u64) -> u64 {
    let mut bytes = activation_bytes(rows, in_features);
    if skinny_route(rows, in_features) && skinny_splits(in_features, out_features) > 1 {
        bytes = align256(bytes) + 4 * skinny_splits(in_features, out_features) * rows * out_features;
    }
    bytes
}

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
/// verification (`row_exact`) and never under `exact_numerics`. The decode and reference part mirrors the router of the
/// C function `euhedral_cuda_linear_nvfp4_bf16`; the native part is the engine's policy over the C native entry point's
/// own checks (`launch_native`).
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
            skinny: skinny_route(u64::from(rows), u64::from(in_features)),
        };
    }
    if exact_numerics || !decode_qualified(rows, in_features, out_features) {
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
/// maps need whole scale rows (`launch_native`, `nvfp4_linear.c`).
fn native_accepts(rows: u32, in_features: u32, sd4: bool) -> bool {
    skinny_route(u64::from(rows), u64::from(in_features)) || in_features.is_multiple_of(if sd4 { 512 } else { 256 })
}

/// The operands of the unchecked planners.
struct Pointers {
    input: DevicePtr,
    weights: DevicePtr,
    output: DevicePtr,
    scratch: DevicePtr,
}

fn decode_launch(rows: u32, in_features: u32, out_features: u32, layout: Nvfp4Layout, at: &Pointers) -> Launch {
    let tile = decode_tile(in_features, out_features);
    let (grid, block) = if tile {
        (out_features / 32, 128)
    } else {
        (out_features / 16, 256)
    };
    let args = Arguments::new()
        .pointer(at.input)
        .pointer(at.weights)
        .pointer(at.output)
        .u32(in_features)
        .u32(out_features);
    if rows > DECODE_MAX_ROWS {
        // Groups of 8 rows (`nvfp4_linear.c`), the rows as a last argument.
        let kernel = DECODE_GROUPS[layout.index()][tile as usize][(rows.div_ceil(8) - 2) as usize];
        Launch::new(kernel, [grid, 1, 1], [block, 1, 1], 0, args.u32(rows))
    } else {
        let kernel = DECODE_ROWS[layout.index()][tile as usize][rows as usize - 1];
        Launch::new(kernel, [grid, 1, 1], [block, 1, 1], 0, args)
    }
}

fn reference_launch(rows: u32, in_features: u32, out_features: u32, layout: Nvfp4Layout, at: &Pointers) -> Launch {
    // `euhedral_reference_nvfp4` (`reference.c`): the grid is capped at the grid-x limit, the kernel strides.
    let grid = (u64::from(rows) * u64::from(out_features)).min(MAX_GRID) as u32;
    let args = Arguments::new()
        .pointer(at.input)
        .pointer(at.weights)
        .pointer(at.output)
        .u32(rows)
        .u32(in_features)
        .u32(out_features);
    Launch::new(REFERENCE[layout.index()], [grid, 1, 1], [128, 1, 1], 0, args)
}

/// The native launches (`launch_native`, `nvfp4_linear.c`) once the checks have passed.
fn native_launches(rows: u32, in_features: u32, out_features: u32, layout: Nvfp4Layout, at: &Pointers) -> Nvfp4Plan {
    let (rows64, inputs, outputs) = (u64::from(rows), u64::from(in_features), u64::from(out_features));
    let scratch = at.scratch;
    let quantize = Launch::new(
        QUANTIZE,
        [rows, 1, 1],
        [128, 1, 1],
        0,
        Arguments::new()
            .pointer(at.input)
            .pointer(scratch)
            .u32(rows)
            .u32(in_features),
    );
    let mut plan = Nvfp4Plan::single(quantize);
    if skinny_route(rows64, inputs) {
        // `index`: the 16, 32 or 64 row kernel (`nvfp4_linear.c`).
        let index = if rows <= 16 {
            0
        } else if rows <= 32 {
            1
        } else {
            2
        };
        let splits = skinny_splits(inputs, outputs);
        let partials = if splits > 1 {
            scratch.get() + align256(activation_bytes(rows64, inputs))
        } else {
            0
        };
        plan.push(Launch::new(
            SKINNY[layout.index()][index],
            [outputs.div_ceil(SKINNY_COLUMNS) as u32, splits as u32, 1],
            [128, 1, 1],
            skinny_shared_bytes(NATIVE_TERMS, 16 << index) as u32,
            Arguments::new()
                .pointer(scratch)
                .pointer(at.weights)
                .pointer(at.output)
                .u64(partials)
                .u32(rows)
                .u32(in_features)
                .u32(out_features),
        ));
        if splits > 1 {
            // `count` is a 32-bit value in C (`launch_native`), computed only here, with at least two splits: a
            // shape whose product wraps is as wrong in C as here, and with one split no count exists, so there is
            // no check to make. The wrap is kept as C has it.
            let count = rows.wrapping_mul(out_features);
            plan.push(Launch::new(
                FINISH,
                [count.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
                Arguments::new()
                    .u64(partials)
                    .pointer(at.output)
                    .u32(count)
                    .u32(splits as u32),
            ));
        }
        return plan;
    }
    // The tile kernel and its three maps: activation codes (K / 2 bytes, rows, terms), weight codes
    // (K / 2 bytes, weight rows) and weight scales (row_scales bytes, weight rows).
    let row_bytes = inputs / 2;
    let row_scales = inputs / if layout == Nvfp4Layout::Sd4 { 32 } else { 16 };
    let scale_offset = align256(outputs * row_bytes);
    let b_rows = 128;
    plan.maps = Some([
        TensorMapPlan {
            address: scratch.get(),
            rank: 3,
            dims: [row_bytes, rows64, NATIVE_TERMS],
            strides: [row_bytes, row_bytes * rows64],
            box_dims: [64, 128, 1],
            swizzle: TensorMapSwizzle::B64,
        },
        TensorMapPlan {
            address: at.weights.get(),
            rank: 2,
            dims: [row_bytes, outputs, 0],
            strides: [row_bytes, 0],
            box_dims: [64, b_rows, 0],
            swizzle: TensorMapSwizzle::B64,
        },
        TensorMapPlan {
            address: at.weights.get() + scale_offset,
            rank: 2,
            dims: [row_scales, outputs, 0],
            strides: [row_scales, 0],
            box_dims: [16, b_rows, 0],
            swizzle: TensorMapSwizzle::None,
        },
    ]);
    let grid = rows64.div_ceil(128) * outputs.div_ceil(128);
    plan.push(Launch::new(
        TILE[layout.index()],
        [grid as u32, 1, 1],
        [NATIVE_TILE_THREADS, 1, 1],
        NATIVE_SHARED_BYTES,
        Arguments::new()
            .pointer(scratch)
            .pointer(at.weights)
            .pointer(at.output)
            .u32(rows)
            .u32(in_features)
            .u32(out_features),
    ));
    plan
}

/// The checks every NVFP4 entry point makes first (`prepare`, `nvfp4_linear.c`), up to the layout.
fn prepare(operands: &Nvfp4Operands) -> Result<Nvfp4Layout, PlanError> {
    let Nvfp4Operands {
        input,
        weights,
        output,
        rows,
        in_features,
        out_features,
        weights_byte_size,
    } = *operands;
    if input == DevicePtr::NULL
        || weights == DevicePtr::NULL
        || output == DevicePtr::NULL
        || rows == 0
        || in_features == 0
        || out_features == 0
    {
        return Err(PlanError::InvalidArgument);
    }
    let layout = nvfp4_layout(out_features, in_features, weights_byte_size).ok_or(PlanError::FormatMismatch)?;
    if !in_features.is_multiple_of(32) || !input.get().is_multiple_of(16) || !weights.get().is_multiple_of(16) {
        return Err(PlanError::InvalidArgument);
    }
    Ok(layout)
}

/// The launch of `euhedral_cuda_linear_nvfp4_bf16` (`nvfp4_linear.c`): 1 to 8 rows on the decode kernels, 9
/// to 32 on their grouped twins, and under exact numerics or for a shape they do not take, the scalar reference.
pub fn plan_linear(operands: &Nvfp4Operands, exact_numerics: bool) -> Result<Nvfp4Plan, PlanError> {
    let layout = prepare(operands)?;
    let Nvfp4Operands {
        input,
        weights,
        output,
        rows,
        in_features,
        out_features,
        ..
    } = *operands;
    let at = Pointers {
        input,
        weights,
        output,
        scratch: DevicePtr::NULL,
    };
    let launch = if exact_numerics || !decode_qualified(rows, in_features, out_features) {
        reference_launch(rows, in_features, out_features, layout, &at)
    } else {
        decode_launch(rows, in_features, out_features, layout, &at)
    };
    Ok(Nvfp4Plan::single(launch))
}

/// The launches of `euhedral_cuda_linear_nvfp4_native_bf16` (`nvfp4_linear.c`): the quantization of the
/// activations into `scratch`, then the skinny kernel (and its finish when split over K) or the tile kernel.
///
/// `scratch_byte_size` is the scratch's size. Where the C entry point declines, the refusal is
/// [`PlanError::RouteUnavailable`] and the caller runs [`plan_linear`] instead.
pub fn plan_native(
    operands: &Nvfp4Operands,
    scratch: DevicePtr,
    scratch_byte_size: u64,
    exact_numerics: bool,
) -> Result<Nvfp4Plan, PlanError> {
    let Nvfp4Operands {
        input,
        weights,
        output,
        rows,
        in_features,
        out_features,
        ..
    } = *operands;
    // Activations are quantized, so exact numerics keep the scalar reference.
    if !in_features.is_multiple_of(128) || exact_numerics {
        return Err(PlanError::RouteUnavailable);
    }
    let layout = prepare(operands)?;
    let (rows64, inputs, outputs) = (u64::from(rows), u64::from(in_features), u64::from(out_features));
    let skinny = skinny_route(rows64, inputs);
    // The tile kernels' maps need 16-byte strides: the scale rows and an aligned scratch.
    let scale_unit = if layout == Nvfp4Layout::Sd4 { 512 } else { 256 };
    if !skinny && (!in_features.is_multiple_of(scale_unit) || !scratch.get().is_multiple_of(16)) {
        return Err(PlanError::RouteUnavailable);
    }
    let needed = if skinny {
        nvfp4_native_scratch_bytes(rows64, inputs, outputs)
    } else {
        activation_bytes(rows64, inputs)
    };
    if scratch == DevicePtr::NULL || scratch_byte_size < needed {
        return Err(PlanError::InvalidArgument);
    }
    if rows64.div_ceil(128) * outputs.div_ceil(128) > MAX_GRID {
        return Err(PlanError::SizeOverflow);
    }
    let at = Pointers {
        input,
        weights,
        output,
        scratch,
    };
    Ok(native_launches(rows, in_features, out_features, layout, &at))
}

impl Nvfp4Route {
    /// The launches of the route as lines of the shape recorder's format, without arguments: the library queues them
    /// itself, so the geometry is what it computes for the shape.
    pub fn lines(self, rows: u32, in_features: u32, out_features: u32, sd4: bool) -> Vec<String> {
        let layout = if sd4 { Nvfp4Layout::Sd4 } else { Nvfp4Layout::Plain };
        let at = Pointers {
            input: DevicePtr::NULL,
            weights: DevicePtr::NULL,
            output: DevicePtr::NULL,
            scratch: DevicePtr::NULL,
        };
        let plan = match self {
            Nvfp4Route::DecodeRows { rows: count } => {
                Nvfp4Plan::single(decode_launch(count, in_features, out_features, layout, &at))
            }
            Nvfp4Route::DecodeGroups { groups } => {
                Nvfp4Plan::single(decode_launch(groups * 8, in_features, out_features, layout, &at))
            }
            Nvfp4Route::Reference => Nvfp4Plan::single(reference_launch(rows, in_features, out_features, layout, &at)),
            Nvfp4Route::Native { .. } => native_launches(rows, in_features, out_features, layout, &at),
        };
        plan.launches()
            .iter()
            .map(|launch| {
                let [gx, gy, gz] = launch.grid;
                let [bx, by, bz] = launch.block;
                format!(
                    "{} grid={gx},{gy},{gz} block={bx},{by},{bz} smem={}",
                    launch.kernel.name(),
                    launch.shared_bytes
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
