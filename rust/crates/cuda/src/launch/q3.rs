//! Plans the launches of the grouped linears (Q3, Q4 and Q5 values in 64-value groups with FP16 scales) and of the P2E2
//! compressed Q3 tensors, as the native library's host code (`native/src/host/q3_linear_bf16.c`, `q45_linear.c`,
//! `q3_mx.c`, `reference.c`) issues them: the same kernels in the same order with the same grid, block, dynamic shared
//! bytes and argument words.
//!
//! The functions are pure: shapes and addresses in, a [`Q3Plan`] (at most three launches, no allocation) or a
//! [`PlanError`] out. A refusal is a status the C entry point returns before it launches anything. The routes are:
//!
//! - **1 to 8 rows** (9 to 32 on the token-row twins) on the contiguous decode kernels, every row bit for bit as a
//!   one-row launch;
//! - **the scalar reference** under exact numerics and for a shape or an alignment the decode kernels do not take;
//! - **the FP8 route** ([`plan_mx`]) from 9 rows on a device with block-scaled FP8 tensor cores: the activations are
//!   quantized into the scratch, multiplied against the weights, and a K-split result is reduced;
//! - **P2E2**: [`plan_p2e2_decode`] runs up to 8 rows straight off the compressed payload, and [`plan_p2e2_expand`]
//!   turns rows of it into a row-split tensor for every other route.

use crate::DevicePtr;
use crate::launch::catalog::{Arguments, KernelId, Launch, find_const};
use crate::launch::embedding::{align256, p2e2_geometry};
use crate::launch::{PlanError, advance};

/// The most rows of the contiguous decode kernels (`EUHEDRAL_DECODE_MAX_ROWS`, `decode_shapes.h`).
pub const DECODE_MAX_ROWS: u32 = 8;

/// The most rows of the token-row twins (`EUHEDRAL_DECODE_TOKENS_MAX`, `decode_shapes.h`).
pub const DECODE_TOKENS_MAX: u32 = 32;

/// Rows from which a grouped linear takes the FP8 route (`Q3_MX_MIN_ROWS`, `CudaGpuMemory`).
pub const MX_MIN_ROWS: u32 = DECODE_MAX_ROWS + 1;

/// The most rows of one P2E2 decode launch (`q3_linear_bf16.c`).
const P2E2_ROWS_PER_LAUNCH: u32 = 4;

/// Dynamic shared bytes of the FP8 linear (`Q3MX_SHARED_BYTES`, `q3_mx.c`): three stages of (A hi, A lo, expanded W,
/// 128 FP32 scales).
pub const MX_SHARED_BYTES: u32 = 3 * (3 * 128 * 64 + 128 * 4);

/// The largest grid-x of a launch.
const MAX_GRID: u64 = 2_147_483_647;

/// The widths of the grouped formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bits {
    Q3,
    Q4,
    Q5,
}

impl Bits {
    /// The bit width, as the native entry points take it.
    pub const fn number(self) -> u32 {
        match self {
            Bits::Q3 => 3,
            Bits::Q4 => 4,
            Bits::Q5 => 5,
        }
    }

    /// The width for a native `bits` argument.
    pub const fn from_number(bits: u32) -> Option<Bits> {
        match bits {
            3 => Some(Bits::Q3),
            4 => Some(Bits::Q4),
            5 => Some(Bits::Q5),
            _ => None,
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// The shape of one linear and the addresses it works on.
#[derive(Clone, Copy, Debug)]
pub struct Q3Operands {
    pub input: DevicePtr,
    pub weights: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub in_features: u32,
    pub out_features: u32,
    /// The weight's byte size.
    pub weights_byte_size: u64,
}

/// The launches of one linear, in order (at most three).
#[derive(Clone, Debug)]
pub struct Q3Plan {
    launches: [Launch; 3],
    len: usize,
}

/// Plans are equal when their launches in use are; the unused slots are filler.
impl PartialEq for Q3Plan {
    fn eq(&self, other: &Self) -> bool {
        self.launches() == other.launches()
    }
}

impl Eq for Q3Plan {}

impl Q3Plan {
    fn single(launch: Launch) -> Self {
        Self {
            launches: [launch; 3],
            len: 1,
        }
    }

    fn push(&mut self, launch: Launch) {
        self.launches[self.len] = launch;
        self.len += 1;
    }

    /// The launches, in the order to issue them.
    pub fn launches(&self) -> &[Launch] {
        &self.launches[..self.len]
    }
}

macro_rules! rows_kernels {
    ($bits:literal) => {
        [
            find_const(concat!("euhedral_q", $bits, "_decode_contiguous")),
            find_const(concat!("euhedral_q", $bits, "_decode_contiguous_rows2")),
            find_const(concat!("euhedral_q", $bits, "_decode_contiguous_rows3")),
            find_const(concat!("euhedral_q", $bits, "_decode_contiguous_rows4")),
            find_const(concat!("euhedral_q", $bits, "_decode_contiguous_rows5")),
            find_const(concat!("euhedral_q", $bits, "_decode_contiguous_rows6")),
            find_const(concat!("euhedral_q", $bits, "_decode_contiguous_rows7")),
            find_const(concat!("euhedral_q", $bits, "_decode_contiguous_rows8")),
        ]
    };
}

/// The decode kernels by [bits][rows - 1].
const DECODE_ROWS: [[KernelId; 8]; 3] = [rows_kernels!("3"), rows_kernels!("4"), rows_kernels!("5")];

/// The token-row twins by [bits].
const DECODE_TOKENS: [KernelId; 3] = [
    find_const("euhedral_q3_decode_contiguous_tokens"),
    find_const("euhedral_q4_decode_contiguous_tokens"),
    find_const("euhedral_q5_decode_contiguous_tokens"),
];

const P2E2_DECODE: [KernelId; 4] = [
    find_const("euhedral_q3_p2e2_decode"),
    find_const("euhedral_q3_p2e2_decode_rows2"),
    find_const("euhedral_q3_p2e2_decode_rows3"),
    find_const("euhedral_q3_p2e2_decode_rows4"),
];

const P2E2_EXPAND: KernelId = find_const("euhedral_q3_p2e2_expand");

const REFERENCE_Q3: KernelId = find_const("euhedral_q3_reference");
const REFERENCE_Q45: KernelId = find_const("euhedral_q45_reference");

const MX_QUANTIZE: KernelId = find_const("euhedral_q3mx_quantize");
const MX_LINEAR: [KernelId; 3] = [
    find_const("euhedral_q3mx_linear_128x128"),
    find_const("euhedral_q4mx_linear_128x128"),
    find_const("euhedral_q5mx_linear_128x128"),
];
const MX_REDUCE: KernelId = find_const("euhedral_q3mx_reduce");

/// The groups of 64 values in a row of `in_features` values, K padded to 128 (`groups`, `q3_linear_bf16.c`).
const fn q3_groups(in_features: u32) -> u64 {
    (in_features as u64).div_ceil(128) * 128 / 64
}

/// Byte offset of a Q3 tensor's scale plane: its 24-byte code groups, 256-aligned (`scale_offset`,
/// `q3_linear_bf16.c`).
pub const fn q3_scale_offset(out_features: u32, in_features: u32) -> u64 {
    align256(out_features as u64 * q3_groups(in_features) * 24)
}

/// Byte size of a row-split Q3 tensor of `rows` rows of `in_features` values (`euhedral_q3_row_split_size`,
/// `q3_p2e2_geometry.h`, which is the same size from the other side).
pub const fn q3_byte_size(rows: u32, in_features: u32) -> u64 {
    q3_scale_offset(rows, in_features) + rows as u64 * q3_groups(in_features) * 2
}

/// Byte size of a Q4 or Q5 tensor, `None` for an `in_features` that is not a multiple of 128 or a size that overflows
/// (`quantized_byte_size`, `q45_linear.c`): the nibble plane, for Q5 the fifth-bit plane, then the FP16 scales, each
/// plane 256-aligned.
pub fn quantized_byte_size(bits: Bits, rows: u32, in_features: u32) -> Option<u64> {
    if bits == Bits::Q3 {
        return Some(q3_byte_size(rows, in_features));
    }
    if in_features == 0 || !in_features.is_multiple_of(128) || rows == 0 {
        return None;
    }
    let groups = u64::from(in_features) / 64;
    let rows = u64::from(rows);
    let code_plane = align256(rows * groups * 32);
    let high_plane = if bits == Bits::Q5 {
        align256(rows * groups * 8)
    } else {
        0
    };
    Some(code_plane + high_plane + rows * groups * 2)
}

/// Whether the Q3 decode kernels take the shape (`euhedral_q3_decode_shape`, `decode_shapes.h`).
const fn q3_decode_shape(in_features: u32, out_features: u32) -> bool {
    in_features != 0 && in_features.is_multiple_of(1024) && out_features != 0 && out_features.is_multiple_of(16)
}

/// Whether the Q4 and Q5 decode kernels take the shape (`euhedral_q45_decode_shape`, `decode_shapes.h`).
const fn q45_decode_shape(in_features: u32, out_features: u32) -> bool {
    in_features != 0 && in_features.is_multiple_of(1024) && out_features != 0 && out_features.is_multiple_of(8)
}

/// Whether a decode launch of a Q3 tensor reaches a decode kernel: the shape, up to 32 rows, and the alignments they
/// need.
pub fn q3_decode_qualified(operands: &Q3Operands) -> bool {
    operands.rows <= DECODE_TOKENS_MAX
        && q3_decode_shape(operands.in_features, operands.out_features)
        && operands.input.get().is_multiple_of(16)
        && operands.weights.get().is_multiple_of(4)
}

/// The same for a Q4 or Q5 tensor.
pub fn q45_decode_qualified(operands: &Q3Operands) -> bool {
    operands.rows <= DECODE_TOKENS_MAX
        && q45_decode_shape(operands.in_features, operands.out_features)
        && (operands.weights.get() | operands.input.get()).is_multiple_of(16)
}

fn nulls(operands: &Q3Operands) -> bool {
    operands.input == DevicePtr::NULL
        || operands.weights == DevicePtr::NULL
        || operands.output == DevicePtr::NULL
        || operands.rows == 0
        || operands.out_features == 0
}

/// The launch of the Q3 scalar reference (`euhedral_reference_q3`, `reference.c`).
fn q3_reference(operands: &Q3Operands) -> Result<Q3Plan, PlanError> {
    let grid = u64::from(operands.rows) * u64::from(operands.out_features);
    if grid > MAX_GRID {
        return Err(PlanError::SizeOverflow);
    }
    let args = Arguments::new()
        .pointer(operands.input)
        .pointer(operands.weights)
        .pointer(operands.output)
        .u32(operands.rows)
        .u32(operands.in_features)
        .u32(operands.out_features)
        .u64(q3_scale_offset(operands.out_features, operands.in_features));
    Ok(Q3Plan::single(Launch::new(
        REFERENCE_Q3,
        [grid as u32, 1, 1],
        [128, 1, 1],
        0,
        args,
    )))
}

/// The launch of the Q4 and Q5 scalar reference (`euhedral_reference_q45`, `reference.c`): the grid is capped at the
/// grid-x limit, the kernel strides.
fn q45_reference(bits: Bits, operands: &Q3Operands) -> Q3Plan {
    let count = u64::from(operands.rows) * u64::from(operands.out_features);
    let args = Arguments::new()
        .pointer(operands.input)
        .pointer(operands.weights)
        .pointer(operands.output)
        .u32(operands.rows)
        .u32(operands.in_features)
        .u32(operands.out_features)
        .u32(bits.number());
    Q3Plan::single(Launch::new(
        REFERENCE_Q45,
        [count.min(MAX_GRID) as u32, 1, 1],
        [128, 1, 1],
        0,
        args,
    ))
}

/// The launch of a contiguous decode kernel: the rows on the one-row kernel or its 2 to 8 row twins with `rows_per_cta`
/// outputs a block, or 9 to 32 rows on the token-row twin, which runs 4 outputs a block.
fn decode_launch(bits: Bits, operands: &Q3Operands, outputs_per_block: u32) -> Launch {
    let (kernel, grid) = if operands.rows > DECODE_MAX_ROWS {
        (DECODE_TOKENS[bits.index()], operands.out_features / 4)
    } else {
        (
            DECODE_ROWS[bits.index()][operands.rows as usize - 1],
            operands.out_features / outputs_per_block,
        )
    };
    let mut args = Arguments::new()
        .pointer(operands.input)
        .pointer(operands.weights)
        .pointer(operands.output)
        .u32(operands.rows)
        .u32(operands.in_features)
        .u32(operands.out_features);
    if bits == Bits::Q3 {
        args = args.u64(q3_scale_offset(operands.out_features, operands.in_features));
    }
    Launch::new(kernel, [grid, 1, 1], [128, 1, 1], 0, args)
}

/// The launch of `euhedral_cuda_linear_q3_bf16` (`q3_linear_bf16.c`): 1 to 8 rows on the contiguous decode kernels, 9
/// to 32 on their token-row twin, and under exact numerics or for a shape or an alignment they do not take, the
/// scalar reference.
pub fn plan_linear_q3(operands: &Q3Operands, exact_numerics: bool) -> Result<Q3Plan, PlanError> {
    if nulls(operands) || operands.in_features == 0 || operands.weights_byte_size == 0 {
        return Err(PlanError::InvalidArgument);
    }
    if q3_byte_size(operands.out_features, operands.in_features) != operands.weights_byte_size {
        return Err(PlanError::FormatMismatch);
    }
    if exact_numerics || !q3_decode_qualified(operands) {
        return q3_reference(operands);
    }
    Ok(Q3Plan::single(decode_launch(Bits::Q3, operands, 16)))
}

/// The launch of `euhedral_cuda_linear_quantized_bf16` (`q45_linear.c`) for a Q4 or Q5 tensor.
pub fn plan_linear_q45(bits: Bits, operands: &Q3Operands, exact_numerics: bool) -> Result<Q3Plan, PlanError> {
    if bits == Bits::Q3 {
        return Err(PlanError::InvalidArgument);
    }
    if nulls(operands) {
        return Err(PlanError::InvalidArgument);
    }
    match quantized_byte_size(bits, operands.out_features, operands.in_features) {
        None => return Err(PlanError::FormatMismatch),
        Some(required) if required != operands.weights_byte_size => return Err(PlanError::FormatMismatch),
        Some(_) => {}
    }
    if u64::from(operands.rows) * u64::from(operands.out_features) > u64::from(u32::MAX) {
        return Err(PlanError::SizeOverflow);
    }
    if exact_numerics || !q45_decode_qualified(operands) {
        return Ok(q45_reference(bits, operands));
    }
    Ok(Q3Plan::single(decode_launch(bits, operands, 8)))
}

/// The launches of the P2E2 fused decode (`euhedral_cuda_linear_q3_p2e2_decode_bf16`, `q3_linear_bf16.c`): up to four
/// rows a launch, each row the one-row result whichever launch computes it. Where the kernels do not run (exact
/// numerics, more than 8 rows, a shape or alignment they do not take) the refusal is
/// [`PlanError::RouteUnavailable`] and the caller expands the tensor instead.
pub fn plan_p2e2_decode(operands: &Q3Operands, exact_numerics: bool) -> Result<Q3Plan, PlanError> {
    if operands.input == DevicePtr::NULL
        || operands.weights == DevicePtr::NULL
        || operands.output == DevicePtr::NULL
        || operands.rows == 0
        || operands.out_features == 0
    {
        return Err(PlanError::InvalidArgument);
    }
    p2e2_geometry(operands.out_features, operands.in_features, operands.weights_byte_size)?;
    if !p2e2_decode_runs(operands, exact_numerics) {
        return Err(PlanError::RouteUnavailable);
    }
    let mut plan: Option<Q3Plan> = None;
    let mut first = 0;
    while first < operands.rows {
        let count = (operands.rows - first).min(P2E2_ROWS_PER_LAUNCH);
        let args = Arguments::new()
            .pointer(advance(
                operands.input,
                u64::from(first) * u64::from(operands.in_features) * 2,
            ))
            .pointer(operands.weights)
            .pointer(advance(
                operands.output,
                u64::from(first) * u64::from(operands.out_features) * 2,
            ))
            .u32(operands.in_features)
            .u32(operands.out_features);
        let launch = Launch::new(
            P2E2_DECODE[count as usize - 1],
            [operands.out_features / 16, 1, 1],
            [128, 1, 1],
            0,
            args,
        );
        match plan.as_mut() {
            Some(plan) => plan.push(launch),
            None => plan = Some(Q3Plan::single(launch)),
        }
        first += count;
    }
    plan.ok_or(PlanError::InvalidArgument)
}

/// Whether the fused P2E2 decode runs for the launch: it reproduces the contiguous decode kernels, which serve 1 to 8
/// rows on 16-aligned operands of the decode shape.
pub fn p2e2_decode_runs(operands: &Q3Operands, exact_numerics: bool) -> bool {
    !exact_numerics
        && operands.rows <= DECODE_MAX_ROWS
        && operands.input.get().is_multiple_of(16)
        && operands.weights.get().is_multiple_of(16)
        && q3_decode_shape(operands.in_features, operands.out_features)
}

/// The launch of `euhedral_cuda_q3_p2e2_expand` (`q3_linear_bf16.c`): rows `first_row..first_row + row_count` of the
/// P2E2 tensor `weights` of `rows` rows become a row-split Q3 tensor of `row_count` rows at `destination`.
#[allow(clippy::too_many_arguments, reason = "the native entry point's parameter list")]
pub fn plan_p2e2_expand(
    weights: DevicePtr,
    weights_byte_size: u64,
    rows: u32,
    in_features: u32,
    first_row: u32,
    row_count: u32,
    destination: DevicePtr,
    destination_byte_size: u64,
) -> Result<Q3Plan, PlanError> {
    if weights == DevicePtr::NULL
        || destination == DevicePtr::NULL
        || row_count == 0
        || first_row >= rows
        || row_count > rows - first_row
    {
        return Err(PlanError::InvalidArgument);
    }
    p2e2_geometry(rows, in_features, weights_byte_size)?;
    if destination_byte_size < q3_byte_size(row_count, in_features) {
        return Err(PlanError::SizeOverflow);
    }
    if !weights.get().is_multiple_of(16) || !destination.get().is_multiple_of(16) {
        return Err(PlanError::InvalidArgument);
    }
    let args = Arguments::new()
        .pointer(weights)
        .pointer(destination)
        .u32(rows)
        .u32(in_features)
        .u32(first_row)
        .u32(row_count);
    Ok(Q3Plan::single(Launch::new(
        P2E2_EXPAND,
        [row_count.div_ceil(4), 1, 1],
        [128, 1, 1],
        0,
        args,
    )))
}

/// The split-K factor of an FP8 linear (`choose_splits`, `q3_mx.c`): one CTA per 128 x 128 output tile fills the
/// multiprocessors in whole waves, so a skinny grid leaves some idle in its last wave. Splitting K by `s` multiplies the
/// CTAs and divides each one's work; the cost is the waves over `s` plus four percent a split for the FP32 partials,
/// and a split is only taken when it is clearly better. `split_rows`, when not zero, names the weight rows of the whole
/// tensor the split is chosen from, so a tensor run in output-row chunks sums in the order the whole tensor does.
pub fn mx_splits(rows: u32, width: u32, weight_rows: u32, sm_count: u32, split_rows: u32) -> u32 {
    let weight_rows = if split_rows != 0 { split_rows } else { weight_rows };
    let sm_count = u64::from(sm_count.max(1));
    let tiles = u64::from(rows).div_ceil(128) * u64::from(weight_rows / 128);
    let groups = width / 64;
    let single = tiles.div_ceil(sm_count) as f64;
    let mut best = 1;
    let mut best_cost = single;
    let mut splits = 2;
    while splits <= 4 {
        if groups.is_multiple_of(splits) {
            let cost = (tiles * u64::from(splits)).div_ceil(sm_count) as f64 / f64::from(splits)
                * (1.0 + 0.04 * f64::from(splits - 1));
            if cost < best_cost && cost < 0.92 * single {
                best_cost = cost;
                best = splits;
            }
        }
        splits *= 2;
    }
    best
}

/// The scratch of an FP8 linear (`euhedral_cuda_q3_mx_scratch_bytes`, `q3_mx.c`): the hi plane, the lo plane (`rows` x
/// `width` bytes each) and the UE8M0 scales (`rows` x `width` / 32), each 256-aligned, then the split-K partials
/// (`splits` x `rows` x `weight_rows` floats) when the linear splits. `weight_rows` 0 names no weight and no partials.
pub fn mx_scratch_bytes(rows: u32, width: u32, weight_rows: u32, sm_count: u32, split_rows: u32) -> u64 {
    let splits = if weight_rows == 0 {
        1
    } else {
        mx_splits(rows, width, weight_rows, sm_count, split_rows)
    };
    mx_scratch_for_splits(rows, width, weight_rows, splits)
}

fn mx_scratch_for_splits(rows: u32, width: u32, weight_rows: u32, splits: u32) -> u64 {
    let plane = align256(u64::from(rows) * u64::from(width));
    let mut bytes = 2 * plane + align256(u64::from(rows) * u64::from(width / 32));
    if splits > 1 {
        bytes += align256(u64::from(splits) * u64::from(rows) * u64::from(weight_rows) * 4);
    }
    bytes
}

/// The most FP8 scratch any launch of `MX_MIN_ROWS` to `rows` rows takes. The scratch is not monotone in the rows: the
/// split of K follows the tile count, which steps with `rows` in units of 128, so each such unit is measured at its
/// last row, where its planes and partials are largest. 0 below [`MX_MIN_ROWS`]. Where the device is not known
/// (`sm_count` is `None`) the answer is the most any device takes: the split of K is at most 4 and divides the groups.
pub fn mx_scratch_bytes_upto(rows: u32, width: u32, weight_rows: u32, sm_count: Option<u32>) -> u64 {
    if rows < MX_MIN_ROWS {
        return 0;
    }
    let mut largest = 0;
    let mut top = 128.min(rows);
    loop {
        largest = largest.max(match sm_count {
            Some(sm_count) => mx_scratch_bytes(top, width, weight_rows, sm_count, 0),
            None => {
                let groups = width / 64;
                let splits = if groups.is_multiple_of(4) {
                    4
                } else if groups.is_multiple_of(2) {
                    2
                } else {
                    1
                };
                mx_scratch_for_splits(top, width, weight_rows, splits)
            }
        });
        if top >= rows {
            return largest;
        }
        top = (top + 128).min(rows);
    }
}

/// The largest tensor a P2E2 linear expands into the shared scratch whole (`Q3_SCRATCH_LIMIT`): a larger one runs in
/// chunks of output rows.
pub const P2E2_EXPAND_LIMIT: u64 = 128 << 20;

/// The scratch a Q3, Q4 or Q5 linear of up to `rows` rows over a tensor of `out_features` by `in_features` takes,
/// where `mx_available` says the device runs the FP8 route and `sm_count` is its multiprocessor count (`scratchBytes`
/// of the grouped formats). A plain tensor takes the FP8 route's scratch when it runs; a P2E2 tensor of more than 8
/// rows is expanded first, so it takes the expansion, 256-aligned, and the FP8 scratch behind it. A P2E2 tensor too
/// large to expand whole is run in chunks, which this does not size: its answer is `None`.
pub fn linear_scratch_bytes(
    p2e2: bool,
    rows: u32,
    in_features: u32,
    out_features: u32,
    mx_available: bool,
    sm_count: Option<u32>,
) -> Option<u64> {
    if rows < MX_MIN_ROWS {
        return Some(0);
    }
    let activations = |tiled_outputs: bool| {
        if mx_available && in_features.is_multiple_of(128) && (!tiled_outputs || out_features.is_multiple_of(128)) {
            mx_scratch_bytes_upto(rows, in_features, out_features, sm_count)
        } else {
            0
        }
    };
    if !p2e2 {
        return Some(activations(true));
    }
    let expanded = q3_byte_size(out_features, in_features);
    (expanded <= P2E2_EXPAND_LIMIT).then(|| align256(expanded) + activations(false))
}

/// The offsets of the planes of a Q4 or Q5 tensor that the FP8 kernel takes: the fifth-bit plane's and the scales'
/// (`run`, `q3_mx.c`). A Q3 tensor has its codes first.
fn mx_offsets(bits: Bits, weight_rows: u32, width: u32) -> (u64, u64) {
    let groups = u64::from(weight_rows) * u64::from(width / 64);
    match bits {
        Bits::Q3 => (0, align256(groups * 24)),
        Bits::Q4 => {
            let high = align256(groups * 32);
            (high, high)
        }
        Bits::Q5 => {
            let high = align256(groups * 32);
            (high, high + align256(groups * 8))
        }
    }
}

/// The launches of the FP8 linear (`euhedral_cuda_linear_q3_mx_bf16` and `euhedral_cuda_linear_q45_mx_bf16`,
/// `q3_mx.c`): the quantization of the activations into `scratch`, the tiled multiply and, when it splits K, the
/// reduction of the partials.
///
/// `sm_count` is the device's multiprocessor count and `split_rows` the whole tensor's weight rows for a chunked run
/// (0 otherwise). Where the C entry point declines (a shape or alignment it has no kernel for, exact numerics or
/// row-exact execution), the refusal is [`PlanError::RouteUnavailable`] and the caller runs the decode route instead.
#[allow(clippy::too_many_arguments, reason = "the native entry point's parameter list")]
pub fn plan_mx(
    bits: Bits,
    operands: &Q3Operands,
    scratch: DevicePtr,
    scratch_byte_size: u64,
    sm_count: u32,
    split_rows: u32,
    exact_numerics: bool,
    row_exact: bool,
) -> Result<Q3Plan, PlanError> {
    let Q3Operands {
        input,
        weights,
        output,
        rows,
        in_features: width,
        out_features: weight_rows,
        weights_byte_size,
    } = *operands;
    if rows == 0
        || !width.is_multiple_of(128)
        || !weight_rows.is_multiple_of(128)
        || !input.get().is_multiple_of(16)
        || !weights.get().is_multiple_of(4)
        || !output.get().is_multiple_of(4)
    {
        return Err(PlanError::RouteUnavailable);
    }
    // Activations are quantized, so exact numerics and row-exact execution keep the BF16 kernels.
    if exact_numerics || row_exact {
        return Err(PlanError::RouteUnavailable);
    }
    let (high_offset, scale_offset) = mx_offsets(bits, weight_rows, width);
    let groups = u64::from(weight_rows) * u64::from(width / 64);
    if weights_byte_size != scale_offset + groups * 2 {
        return Err(PlanError::FormatMismatch);
    }
    if scratch == DevicePtr::NULL
        || !scratch.get().is_multiple_of(256)
        || scratch_byte_size < mx_scratch_bytes(rows, width, weight_rows, sm_count, split_rows)
    {
        return Err(PlanError::InvalidArgument);
    }
    let splits = mx_splits(rows, width, weight_rows, sm_count, split_rows);
    let grid = u64::from(rows).div_ceil(128) * u64::from(weight_rows / 128);
    let blocks = (u64::from(rows) * u64::from(width / 32)).div_ceil(256);
    if grid > MAX_GRID || blocks > MAX_GRID {
        return Err(PlanError::SizeOverflow);
    }
    let plane = align256(u64::from(rows) * u64::from(width));
    let hi = scratch;
    let lo = advance(scratch, plane);
    let scales = advance(scratch, 2 * plane);
    let quantize = Launch::new(
        MX_QUANTIZE,
        [blocks as u32, 1, 1],
        [256, 1, 1],
        0,
        Arguments::new()
            .pointer(input)
            .pointer(hi)
            .pointer(lo)
            .pointer(scales)
            .u32(rows)
            .u32(width),
    );
    let mut plan = Q3Plan::single(quantize);
    let partials = if splits > 1 {
        advance(scales, align256(u64::from(rows) * u64::from(width / 32)))
    } else {
        DevicePtr::NULL
    };
    plan.push(Launch::new(
        MX_LINEAR[bits.index()],
        [grid as u32, splits, 1],
        [384, 1, 1],
        MX_SHARED_BYTES,
        Arguments::new()
            .pointer(hi)
            .pointer(lo)
            .pointer(scales)
            .pointer(weights)
            .pointer(output)
            .u32(rows)
            .u32(weight_rows)
            .u32(width)
            .u64(high_offset)
            .u64(scale_offset)
            .pointer(partials),
    ));
    if splits > 1 {
        // `count` is a 32-bit value in C, as `rows * weight_rows`: the wrap is kept as C has it.
        let count = rows.wrapping_mul(weight_rows);
        plan.push(Launch::new(
            MX_REDUCE,
            [(count / 4).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            Arguments::new()
                .pointer(partials)
                .pointer(output)
                .u32(count)
                .u32(splits),
        ));
    }
    Ok(plan)
}

#[cfg(test)]
mod tests;
