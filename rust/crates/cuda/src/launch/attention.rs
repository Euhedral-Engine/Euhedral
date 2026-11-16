//! The attention launches: the host dispatch of the native library's attention wrappers
//! (`native/src/host/qwen_layer_ops.c`), as plain functions from a shape to the launches the wrapper would issue, in
//! its order, with its grids, blocks, dynamic shared bytes and arguments.
//!
//! Each function pushes its [`Launch`]es into the caller's sink as it plans them and allocates nothing. Where the C
//! wrapper returns an error status the function returns the matching [`PlanError`]; launches already pushed stay
//! pushed, as the C wrapper's earlier launches stay issued. The thread-local modes of the C library (row-exact
//! execution, exact numerics) are explicit values in [`Modes`].
//!
//! The thresholds and tables are the C constants, ported verbatim; each names its C line.

use crate::DevicePtr;
use crate::kernel;
use crate::launch::catalog::{Arguments, KernelId, Launch};
use crate::launch::{Modes, PlanError, advance};

/// The format of the KV cache pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheFormat {
    Nvfp4,
    Bf16,
}

/// The attention kernels over one cache format (`struct attention_kernels`, `qwen_layer_ops.c:53-58`).
struct Kernels {
    append: KernelId,
    prefill: KernelId,
    prefill_exact: KernelId,
    prefill_fa2: KernelId,
    decode: KernelId,
    decode_exact: KernelId,
    decode_gqa: KernelId,
    merge: KernelId,
    decode_rows: KernelId,
    decode_gqa_rows: KernelId,
    merge_rows: KernelId,
    /// `fused[t - 2]` serves `t` mma tiles; NVFP4 only.
    fused: Option<[KernelId; 5]>,
}

const NVFP4: Kernels = Kernels {
    append: kernel!("euhedral_attention_kv_append_nvfp4"),
    prefill: kernel!("euhedral_attention_prefill32_nvfp4"),
    prefill_exact: kernel!("euhedral_attention_prefill32_nvfp4_exact"),
    prefill_fa2: kernel!("euhedral_attention_prefill_fa2_nvfp4"),
    decode: kernel!("euhedral_attention_decode_nvfp4"),
    decode_exact: kernel!("euhedral_attention_decode_nvfp4_exact"),
    decode_gqa: kernel!("euhedral_attention_decode_gqa_nvfp4"),
    merge: kernel!("euhedral_attention_merge_nvfp4"),
    decode_rows: kernel!("euhedral_attention_decode_nvfp4_rows"),
    decode_gqa_rows: kernel!("euhedral_attention_decode_gqa_nvfp4_rows"),
    merge_rows: kernel!("euhedral_attention_merge_nvfp4_rows"),
    fused: Some([
        kernel!("euhedral_attention_decode_gqa_nvfp4_fused2"),
        kernel!("euhedral_attention_decode_gqa_nvfp4_fused3"),
        kernel!("euhedral_attention_decode_gqa_nvfp4_fused4"),
        kernel!("euhedral_attention_decode_gqa_nvfp4_fused5"),
        kernel!("euhedral_attention_decode_gqa_nvfp4_fused6"),
    ]),
};

const BF16: Kernels = Kernels {
    append: kernel!("euhedral_attention_kv_append_bf16"),
    prefill: kernel!("euhedral_attention_prefill32_bf16"),
    prefill_exact: kernel!("euhedral_attention_prefill32_bf16_exact"),
    prefill_fa2: kernel!("euhedral_attention_prefill_fa2_bf16"),
    decode: kernel!("euhedral_attention_decode_bf16"),
    decode_exact: kernel!("euhedral_attention_decode_bf16_exact"),
    decode_gqa: kernel!("euhedral_attention_decode_gqa_bf16"),
    merge: kernel!("euhedral_attention_merge_bf16"),
    decode_rows: kernel!("euhedral_attention_decode_bf16_rows"),
    decode_gqa_rows: kernel!("euhedral_attention_decode_gqa_bf16_rows"),
    merge_rows: kernel!("euhedral_attention_merge_bf16_rows"),
    fused: None,
};

impl CacheFormat {
    const fn kernels(self) -> &'static Kernels {
        match self {
            CacheFormat::Nvfp4 => &NVFP4,
            CacheFormat::Bf16 => &BF16,
        }
    }
}

const QK_NORM_ROPE: KernelId = kernel!("euhedral_attention_qk_norm_rope_bf16");
const QK_NORM_ROPE_ROWS: KernelId = kernel!("euhedral_attention_qk_norm_rope_rows_bf16");

/// `FA2_SHARED_BYTES`, `qwen_layer_ops.c:50`.
const FA2_SHARED_BYTES: u32 = 67584;
/// The only head dimension the kernels serve (`head_dim == 256` checks, `qwen_layer_ops.c:630,689,748`).
const HEAD_DIM: u32 = 256;

/// `fused_shared_bytes`, `qwen_layer_ops.c:61-63`: two FP16 V tiles, the scores and probabilities of each query tile,
/// four raw K and V tiles, the pair table and the mbarriers.
const fn fused_shared_bytes(tiles: u32) -> u32 {
    2 * 16 * 264 * 2 + tiles * 1024 + tiles * 768 + 2 * 4 * 16 * 144 + 1024 + 8 * (12 + 4 * tiles)
}

/// `fused_from`, `qwen_layer_ops.c:66-69`: the fewest keys at which the fused verifier kernel beats the row twins, by
/// verified rows (machine-measured, 6 query heads per KV head).
const fn fused_from(rows: u32) -> u32 {
    // qwen_layer_ops.c:67, verbatim.
    const FROM: [u32; 9] = [u32::MAX, u32::MAX, 49152, 16384, 4096, 2048, 16384, 16384, 12288];
    if rows < 9 { FROM[rows as usize] } else { u32::MAX }
}

fn launch(kernel: KernelId, grid: [u32; 3], block: u32, shared_bytes: u32, args: Arguments) -> Launch {
    Launch::new(kernel, grid, [block, 1, 1], shared_bytes, args)
}

/// The shape of `euhedral_cuda_attention_kv_append_{nvfp4,bf16}` (`qwen_layer_ops.c:523,537`).
#[derive(Clone, Copy, Debug)]
pub struct AppendShape {
    pub query_key: DevicePtr,
    pub gate_value: DevicePtr,
    pub key_cache: DevicePtr,
    pub value_cache: DevicePtr,
    pub position: DevicePtr,
    pub rows: u32,
    pub query_width: u32,
    pub key_value_width: u32,
    /// The host's copy of the position (only validated; the kernel reads `position`).
    pub start_position: u64,
}

/// `kv_append`, `qwen_layer_ops.c:552-582`: one launch.
pub fn plan_kv_append(
    format: CacheFormat,
    shape: &AppendShape,
    sink: &mut impl FnMut(Launch),
) -> Result<(), PlanError> {
    let s = shape;
    if s.query_key.is_null()
        || s.gate_value.is_null()
        || s.key_cache.is_null()
        || s.value_cache.is_null()
        || s.position.is_null()
        || s.rows == 0
        || s.query_width == 0
        || s.key_value_width == 0
        || !s.query_width.is_multiple_of(s.key_value_width)
        || !s.key_value_width.is_multiple_of(256)
        || !s.query_width.is_multiple_of(256)
        || s.start_position > u64::from(u32::MAX) - u64::from(s.rows)
    {
        return Err(PlanError::InvalidArgument);
    }
    let count = u64::from(s.rows) * u64::from(s.key_value_width);
    if count > u64::from(u32::MAX) {
        return Err(PlanError::SizeOverflow);
    }
    let args = Arguments::new()
        .pointer(s.query_key)
        .pointer(s.gate_value)
        .pointer(s.key_cache)
        .pointer(s.value_cache)
        .u32(s.rows)
        .u32(s.query_width)
        .u32(s.key_value_width)
        .pointer(s.position);
    // qwen_layer_ops.c:581.
    let grid = (count / 256).div_ceil(4) as u32;
    sink(launch(format.kernels().append, [grid, 1, 1], 128, 0, args));
    Ok(())
}

/// The shape of `euhedral_cuda_attention_qk_norm_rope_bf16` (`qwen_layer_ops.c:589`).
#[derive(Clone, Copy, Debug)]
pub struct RopeShape {
    pub query_key: DevicePtr,
    pub query_norm: DevicePtr,
    pub key_norm: DevicePtr,
    pub output: DevicePtr,
    pub position: DevicePtr,
    pub rows: u32,
    pub query_heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub start_position: u64,
    pub epsilon: f32,
    pub rope_theta: f64,
}

/// `qk_norm_rope`, `qwen_layer_ops.c:609-654`. The thread's modes are `modes`.
pub fn plan_qk_norm_rope(shape: &RopeShape, modes: Modes, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    plan_rope(shape, 0, modes, sink)
}

fn plan_rope(
    shape: &RopeShape,
    position_offset: u64,
    modes: Modes,
    sink: &mut impl FnMut(Launch),
) -> Result<(), PlanError> {
    let s = shape;
    let row_exact = modes.row_exact;
    let row_exact_heads = s.rows > 1 && row_exact;
    // qwen_layer_ops.c:617-627: past u32 blocks, one row at a time.
    if row_exact_heads
        && !s.query_key.is_null()
        && !s.output.is_null()
        && u64::from(s.rows) * (u64::from(s.query_heads) + u64::from(s.key_value_heads)) > u64::from(u32::MAX)
    {
        let width = (u64::from(s.query_heads) + u64::from(s.key_value_heads)) * u64::from(s.head_dim) * 2;
        for row in 0..s.rows {
            let step = u64::from(row).wrapping_mul(width);
            let one = RopeShape {
                query_key: advance(s.query_key, step),
                output: advance(s.output, step),
                rows: 1,
                start_position: s.start_position.wrapping_add(u64::from(row)),
                ..*s
            };
            plan_rope(&one, position_offset.wrapping_add(u64::from(row)), modes, sink)?;
        }
        return Ok(());
    }
    if s.query_key.is_null()
        || s.query_norm.is_null()
        || s.key_norm.is_null()
        || s.output.is_null()
        || s.position.is_null()
        || s.rows == 0
        || s.query_heads == 0
        || s.key_value_heads == 0
        || !s.query_heads.is_multiple_of(s.key_value_heads)
        || s.head_dim != HEAD_DIM
        || s.rotary_dim == 0
        || s.rotary_dim > s.head_dim
        || (s.rotary_dim & 1) != 0
        || s.start_position > u64::MAX - u64::from(s.rows)
        || !s.epsilon.is_finite()
        || s.epsilon <= 0.0
        || !s.rope_theta.is_finite()
        || s.rope_theta <= 0.0
    {
        return Err(PlanError::InvalidArgument);
    }
    let heads = u64::from(s.query_heads) + u64::from(s.key_value_heads);
    let blocks = u64::from(s.rows) * heads;
    if heads > u64::from(u32::MAX) || blocks > u64::from(u32::MAX) {
        return Err(PlanError::SizeOverflow);
    }
    let args = Arguments::new()
        .pointer(s.query_key)
        .pointer(s.query_norm)
        .pointer(s.key_norm)
        .pointer(s.output)
        .u32(s.rows)
        .u32(s.query_heads)
        .u32(s.key_value_heads)
        .u32(s.head_dim)
        .u32(s.rotary_dim)
        .pointer(s.position)
        .u64(position_offset)
        .f32(s.epsilon)
        .f64(s.rope_theta);
    // qwen_layer_ops.c:650-653: a single row keeps one CTA per head; from two rows one CTA per row.
    if s.rows > 1 && !row_exact_heads {
        sink(launch(QK_NORM_ROPE_ROWS, [s.rows, 1, 1], 256, 0, args));
    } else {
        sink(launch(QK_NORM_ROPE, [blocks as u32, 1, 1], s.head_dim, 0, args));
    }
    Ok(())
}

/// The shape of `euhedral_cuda_attention_causal_{nvfp4,bf16}` (`qwen_layer_ops.c:661,670`).
#[derive(Clone, Copy, Debug)]
pub struct CausalShape {
    pub query_key: DevicePtr,
    pub gate: DevicePtr,
    pub keys: DevicePtr,
    pub values: DevicePtr,
    pub output: DevicePtr,
    /// The decode partials (`query_heads * 64 * 258` floats per row); null when the caller brings none.
    pub scratch: DevicePtr,
    pub position: DevicePtr,
    pub rows: u32,
    pub query_heads: u32,
    pub key_heads: u32,
    pub head_dim: u32,
    pub cache_length: u32,
    /// The host's copy of the quantum's position, which sizes the grids.
    pub start: u64,
}

/// `attention_causal`, `qwen_layer_ops.c:681-813`, with the public wrappers' position check (`:665,674`).
pub fn plan_causal(
    format: CacheFormat,
    shape: &CausalShape,
    modes: Modes,
    sink: &mut impl FnMut(Launch),
) -> Result<(), PlanError> {
    if shape.position.is_null() {
        return Err(PlanError::InvalidArgument);
    }
    plan_causal_at(format.kernels(), shape, 0, modes, sink)
}

fn plan_causal_at(
    kernels: &Kernels,
    shape: &CausalShape,
    position_offset: u64,
    modes: Modes,
    sink: &mut impl FnMut(Launch),
) -> Result<(), PlanError> {
    let s = shape;
    let rows = s.rows;
    let pointers =
        !s.query_key.is_null() && !s.gate.is_null() && !s.keys.is_null() && !s.values.is_null() && !s.output.is_null();
    // qwen_layer_ops.c:688-691: the row twins.
    if rows > 1
        && (modes.row_exact || rows <= 8)
        && pointers
        && !s.scratch.is_null()
        && s.query_heads != 0
        && s.key_heads != 0
        && s.query_heads.is_multiple_of(s.key_heads)
        && s.head_dim == HEAD_DIM
        && rows <= 64
        && s.start <= u64::from(s.cache_length)
        && u64::from(rows) <= u64::from(s.cache_length) - s.start
        && s.start + u64::from(rows) <= u64::from(u32::MAX)
        && position_offset == 0
        && !modes.exact_numerics
    {
        plan_row_twins(kernels, s, sink);
        return Ok(());
    }
    // qwen_layer_ops.c:736-746: row-exact: one-row decode per row, pointers advanced by the row's byte widths.
    if rows > 1 && modes.row_exact && !s.query_key.is_null() && !s.gate.is_null() && !s.output.is_null() {
        let width = (u64::from(s.query_heads) + u64::from(s.key_heads)) * u64::from(s.head_dim) * 2;
        let out = u64::from(s.query_heads) * u64::from(s.head_dim) * 2;
        for row in 0..rows {
            let one = CausalShape {
                query_key: advance(s.query_key, u64::from(row).wrapping_mul(width)),
                gate: advance(s.gate, u64::from(row).wrapping_mul(width)),
                output: advance(s.output, u64::from(row).wrapping_mul(out)),
                rows: 1,
                start: s.start.wrapping_add(u64::from(row)),
                ..*s
            };
            plan_causal_at(kernels, &one, position_offset.wrapping_add(u64::from(row)), modes, sink)?;
        }
        return Ok(());
    }
    if !pointers
        || rows == 0
        || s.query_heads == 0
        || s.key_heads == 0
        || !s.query_heads.is_multiple_of(s.key_heads)
        || s.head_dim != HEAD_DIM
        || s.cache_length == 0
        || s.start > u64::from(s.cache_length)
        || u64::from(rows) > u64::from(s.cache_length) - s.start
        || (rows == 1 && s.scratch.is_null())
    {
        return Err(PlanError::InvalidArgument);
    }
    let grid = u64::from(rows).div_ceil(32) * u64::from(s.query_heads);
    if grid > u64::from(u32::MAX) || u64::from(s.query_heads) * 64 > u64::from(u32::MAX) {
        return Err(PlanError::SizeOverflow);
    }
    if rows > 1 {
        plan_prefill(kernels, s, grid as u32, modes.exact_numerics, sink);
    } else {
        plan_decode(kernels, s, position_offset, modes.exact_numerics, sink);
    }
    Ok(())
}

/// `qwen_layer_ops.c:700-734`: row j is one-row decode at position `start + j`, all rows in one launch each for
/// decode, the fused verifier, GQA decode and merge.
fn plan_row_twins(kernels: &Kernels, s: &CausalShape, sink: &mut impl FnMut(Launch)) {
    let rows = s.rows;
    let (query_heads, key_heads) = (s.query_heads, s.key_heads);
    let group = query_heads / key_heads;
    // :701
    let from = if group <= 8 { 2048 } else { u32::MAX };
    // :703-704
    let first = (s.start as u32).wrapping_add(1);
    let last = (s.start as u32).wrapping_add(rows);
    let row_stride = u64::from(query_heads) * 64 * 258;
    if first < from {
        // :708-709
        let below = if last < from { last } else { from - 1 };
        let splits = (below.wrapping_add(47) / 48).min(64);
        let args = Arguments::new()
            .pointer(s.query_key)
            .pointer(s.keys)
            .pointer(s.values)
            .u32(query_heads)
            .u32(key_heads)
            .pointer(s.position)
            .pointer(s.scratch)
            .u64(row_stride)
            .u32(from);
        sink(launch(
            kernels.decode_rows,
            [query_heads.wrapping_mul(splits), rows, 1],
            128,
            0,
            args,
        ));
    }
    if last >= from {
        // :715-716
        let splits = (last.wrapping_add(31) / 32).min(64);
        // :719-720: the fused kernel serves the rows that share the last row's split partition.
        let tiles = (rows.wrapping_mul(group).wrapping_add(7) / 8).max(2);
        let fused_kernel = match kernels.fused {
            Some(fused) if tiles <= 6 && last >= fused_from(rows) => Some(fused[(tiles - 2) as usize]),
            _ => None,
        };
        if let Some(kernel) = fused_kernel {
            let args = Arguments::new()
                .pointer(s.query_key)
                .pointer(s.keys)
                .pointer(s.values)
                .u32(query_heads)
                .u32(key_heads)
                .pointer(s.position)
                .pointer(s.scratch)
                .u64(row_stride)
                .u32(rows)
                .u32(from);
            let grid = key_heads.wrapping_mul(splits);
            sink(launch(
                kernel,
                [grid, 1, 1],
                32 * (2 * tiles + 1),
                fused_shared_bytes(tiles),
                args,
            ));
        }
        let args = Arguments::new()
            .pointer(s.query_key)
            .pointer(s.keys)
            .pointer(s.values)
            .u32(query_heads)
            .u32(key_heads)
            .pointer(s.position)
            .pointer(s.scratch)
            .u64(row_stride)
            .u32(from)
            .u32(u32::from(fused_kernel.is_some()));
        sink(launch(
            kernels.decode_gqa_rows,
            [key_heads.wrapping_mul(splits), rows, 1],
            96,
            0,
            args,
        ));
    }
    // :732-733
    let args = Arguments::new()
        .pointer(s.gate)
        .pointer(s.output)
        .pointer(s.scratch)
        .u32(query_heads)
        .u32(key_heads)
        .pointer(s.position)
        .u64(row_stride)
        .u32(from);
    sink(launch(kernels.merge_rows, [query_heads, rows, 1], 128, 0, args));
}

/// `qwen_layer_ops.c:758-775`: more than one row, not served by the twins.
fn plan_prefill(kernels: &Kernels, s: &CausalShape, grid: u32, exact: bool, sink: &mut impl FnMut(Launch)) {
    let args = Arguments::new()
        .pointer(s.query_key)
        .pointer(s.gate)
        .pointer(s.keys)
        .pointer(s.values)
        .pointer(s.output)
        .u32(s.rows)
        .u32(s.query_heads)
        .u32(s.key_heads)
        .u32(s.head_dim)
        .u32(s.cache_length)
        .u64(s.start);
    let group = s.query_heads / s.key_heads;
    // :767-768: the FlashAttention-2 leaf where its grid carries enough work (machine-measured, see the C comment).
    if !exact && (1..=6).contains(&group) && (s.rows >= 512 || (s.rows >= 128 && u64::from(s.cache_length) >= 2048)) {
        // :769-771
        let fa2_grid = u64::from(s.rows).div_ceil(16) * u64::from(s.key_heads);
        if fa2_grid <= u64::from(u32::MAX) {
            sink(launch(
                kernels.prefill_fa2,
                [fa2_grid as u32, 1, 1],
                32 * (group + 2),
                FA2_SHARED_BYTES,
                args,
            ));
            return;
        }
    }
    // :773-774
    let kernel = if exact { kernels.prefill_exact } else { kernels.prefill };
    sink(launch(kernel, [grid, 1, 1], 128, 0, args));
}

/// `qwen_layer_ops.c:776-811`: one row, the decode kernel then the merge.
fn plan_decode(kernels: &Kernels, s: &CausalShape, position_offset: u64, exact: bool, sink: &mut impl FnMut(Launch)) {
    // :780-782: each CTA scans about 48 keys.
    let length = (s.start as u32).wrapping_add(1);
    // The sums are u64 in C (:781, :798), so the count does not wrap near u32::MAX keys.
    let mut splits = (u64::from(length) + 47).div_euclid(48).min(64) as u32;
    // :791-792
    let mut decode = if exact { kernels.decode_exact } else { kernels.decode };
    let mut grid = s.query_heads.wrapping_mul(splits);
    let mut block = 128;
    // :796-803: relaxed numerics from 2048 keys: one three-warp tensor-core CTA per (KV head, 32-key split).
    let group = s.query_heads / s.key_heads;
    if !exact && length >= 2048 && group <= 8 {
        splits = (u64::from(length) + 31).div_euclid(32).min(64) as u32;
        decode = kernels.decode_gqa;
        grid = s.key_heads.wrapping_mul(splits);
        block = 96;
    }
    let args = Arguments::new()
        .pointer(s.query_key)
        .pointer(s.gate)
        .pointer(s.keys)
        .pointer(s.values)
        .pointer(s.output)
        .u32(s.rows)
        .u32(s.query_heads)
        .u32(s.key_heads)
        .u32(s.head_dim)
        .pointer(s.position)
        .u64(position_offset)
        .pointer(s.scratch)
        .u32(splits);
    sink(launch(decode, [grid, 1, 1], block, 0, args));
    // :788, 806
    let merge = Arguments::new()
        .pointer(s.gate)
        .pointer(s.output)
        .pointer(s.scratch)
        .u32(s.query_heads)
        .u32(s.key_heads)
        .u32(splits);
    sink(launch(kernels.merge, [s.query_heads, 1, 1], 128, 0, merge));
}

#[cfg(test)]
mod tests {
    use super::*;

    const QK: u64 = 0x1000;
    const GATE: u64 = 0x2000;
    const KEYS: u64 = 0x3000;
    const VALUES: u64 = 0x4000;
    const OUT: u64 = 0x5000;
    const SCRATCH: u64 = 0x6000;
    const POSITION: u64 = 0x7000;

    fn ptr(address: u64) -> DevicePtr {
        // SAFETY: never dereferenced; the plans only carry the address.
        unsafe { DevicePtr::from_raw(address) }
    }

    /// Qwen's shape: 24 query heads over 4 KV heads (group 6).
    fn shape(rows: u32, start: u64) -> CausalShape {
        CausalShape {
            query_key: ptr(QK),
            gate: ptr(GATE),
            keys: ptr(KEYS),
            values: ptr(VALUES),
            output: ptr(OUT),
            scratch: ptr(SCRATCH),
            position: ptr(POSITION),
            rows,
            query_heads: 24,
            key_heads: 4,
            head_dim: 256,
            cache_length: 100_000,
            start,
        }
    }

    fn with_group(mut s: CausalShape, query_heads: u32, key_heads: u32) -> CausalShape {
        s.query_heads = query_heads;
        s.key_heads = key_heads;
        s
    }

    const RELAXED: Modes = Modes {
        row_exact: false,
        exact_numerics: false,
    };
    const ROW_EXACT: Modes = Modes {
        row_exact: true,
        exact_numerics: false,
    };
    const EXACT: Modes = Modes {
        row_exact: false,
        exact_numerics: true,
    };

    fn causal(format: CacheFormat, s: &CausalShape, modes: Modes) -> Result<Vec<Launch>, PlanError> {
        let mut out = Vec::new();
        plan_causal(format, s, modes, &mut |launch| out.push(launch))?;
        Ok(out)
    }

    fn nvfp4(s: &CausalShape, modes: Modes) -> Vec<Launch> {
        causal(CacheFormat::Nvfp4, s, modes).unwrap()
    }

    fn names(launches: &[Launch]) -> Vec<&'static str> {
        launches.iter().map(|l| l.kernel.name()).collect()
    }

    fn words(launch: &Launch) -> Vec<u64> {
        launch.args.words().to_vec()
    }

    fn sizes(launch: &Launch) -> Vec<u8> {
        launch.args.sizes().to_vec()
    }

    // ---- one-row decode (qwen_layer_ops.c:776-811) ----

    #[test]
    fn decode_split_count_follows_48_keys_per_split() {
        // :780-782: length = start + 1, splits = ceil(length / 48) <= 64; decode grid = query_heads * splits.
        for (start, splits) in [(0u64, 1u32), (47, 1), (48, 2), (95, 2), (96, 3), (2045, 43)] {
            let plan = nvfp4(&shape(1, start), RELAXED);
            assert_eq!(
                names(&plan),
                ["euhedral_attention_decode_nvfp4", "euhedral_attention_merge_nvfp4"]
            );
            assert_eq!(plan[0].grid, [24 * splits, 1, 1], "start {start}");
            assert_eq!(plan[0].block, [128, 1, 1]);
            assert_eq!(plan[0].shared_bytes, 0);
            assert_eq!(*plan[0].args.words().last().unwrap(), u64::from(splits));
            assert_eq!(*plan[1].args.words().last().unwrap(), u64::from(splits));
            assert_eq!(plan[1].grid, [24, 1, 1]);
            assert_eq!(plan[1].block, [128, 1, 1]);
        }
    }

    #[test]
    fn decode_splits_cap_at_64() {
        // 64 * 48 = 3072 keys fill the cap; one more key stays at 64. Group 9 never takes the GQA kernel (:797).
        let wide = |start| with_group(shape(1, start), 36, 4);
        let at = nvfp4(&wide(3071), RELAXED);
        assert_eq!(at[0].grid, [36 * 64, 1, 1]);
        let above = nvfp4(&wide(3072), RELAXED);
        assert_eq!(above[0].grid, [36 * 64, 1, 1]);
        assert_eq!(*above[0].args.words().last().unwrap(), 64);
        let below = nvfp4(&wide(3070), RELAXED);
        assert_eq!(*below[0].args.words().last().unwrap(), 64);
        let lower = nvfp4(&wide(3023), RELAXED); // length 3024 = 63 * 48
        assert_eq!(*lower[0].args.words().last().unwrap(), 63);
    }

    #[test]
    fn decode_gqa_starts_at_2048_keys() {
        // :797: relaxed, length >= 2048, group <= 8: the GQA kernel, kv_heads * ceil(length / 32) (<= 64), 96 threads.
        let below = nvfp4(&shape(1, 2046), RELAXED); // length 2047
        assert_eq!(names(&below)[0], "euhedral_attention_decode_nvfp4");
        assert_eq!(below[0].grid, [24 * 43, 1, 1]); // ceil(2047 / 48) = 43
        for start in [2047u64, 2048] {
            let plan = nvfp4(&shape(1, start), RELAXED);
            assert_eq!(
                names(&plan),
                ["euhedral_attention_decode_gqa_nvfp4", "euhedral_attention_merge_nvfp4"]
            );
            assert_eq!(plan[0].grid, [4 * 64, 1, 1]);
            assert_eq!(plan[0].block, [96, 1, 1]);
            // The merge takes the GQA split count.
            assert_eq!(*plan[1].args.words().last().unwrap(), 64);
            assert_eq!(*plan[0].args.words().last().unwrap(), 64);
            assert_eq!(plan[1].grid, [24, 1, 1]);
        }
    }

    #[test]
    fn decode_gqa_group_limit_and_exact_numerics() {
        // Group 8 is served, group 9 is not (:797); exact numerics never take it and use the exact kernel (:791).
        let eight = nvfp4(&with_group(shape(1, 5000), 32, 4), RELAXED);
        assert_eq!(names(&eight)[0], "euhedral_attention_decode_gqa_nvfp4");
        let nine = nvfp4(&with_group(shape(1, 5000), 36, 4), RELAXED);
        assert_eq!(names(&nine)[0], "euhedral_attention_decode_nvfp4");
        let exact = nvfp4(&shape(1, 5000), EXACT);
        assert_eq!(
            names(&exact),
            [
                "euhedral_attention_decode_nvfp4_exact",
                "euhedral_attention_merge_nvfp4"
            ]
        );
        assert_eq!(exact[0].grid, [24 * 64, 1, 1]);
        let bf16 = causal(CacheFormat::Bf16, &shape(1, 5000), RELAXED).unwrap();
        assert_eq!(
            names(&bf16),
            ["euhedral_attention_decode_gqa_bf16", "euhedral_attention_merge_bf16"]
        );
    }

    #[test]
    fn decode_arguments_are_the_native_parameter_list() {
        // :786-788: 5 pointers, rows, query_heads, key_heads, head_dim, position pointer, offset u64, scratch, splits.
        let plan = nvfp4(&shape(1, 47), RELAXED);
        assert_eq!(
            words(&plan[0]),
            [QK, GATE, KEYS, VALUES, OUT, 1, 24, 4, 256, POSITION, 0, SCRATCH, 1]
        );
        assert_eq!(sizes(&plan[0]), [8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 4]);
        // :788: gate, output, scratch, query_heads, key_heads, splits.
        assert_eq!(words(&plan[1]), [GATE, OUT, SCRATCH, 24, 4, 1]);
        assert_eq!(sizes(&plan[1]), [8, 8, 8, 4, 4, 4]);
    }

    #[test]
    fn decode_needs_scratch_and_a_position() {
        let mut s = shape(1, 5);
        s.scratch = DevicePtr::NULL;
        assert_eq!(causal(CacheFormat::Nvfp4, &s, RELAXED), Err(PlanError::InvalidArgument));
        let mut s = shape(1, 5);
        s.position = DevicePtr::NULL;
        assert_eq!(causal(CacheFormat::Nvfp4, &s, RELAXED), Err(PlanError::InvalidArgument));
        // start must be below the cache length (:749).
        let mut s = shape(1, 10);
        s.cache_length = 10;
        assert_eq!(causal(CacheFormat::Nvfp4, &s, RELAXED), Err(PlanError::InvalidArgument));
        s.cache_length = 11;
        assert!(causal(CacheFormat::Nvfp4, &s, RELAXED).is_ok());
    }

    // ---- prefill (qwen_layer_ops.c:758-775) ----

    fn prefill(rows: u32, cache: u32, query_heads: u32, key_heads: u32, modes: Modes) -> Launch {
        let mut s = with_group(shape(rows, 0), query_heads, key_heads);
        s.cache_length = cache;
        s.scratch = DevicePtr::NULL; // no scratch: the twins are out
        let plan = nvfp4(&s, modes);
        assert_eq!(plan.len(), 1);
        plan[0]
    }

    #[test]
    fn prefill_tile_grid_is_32_rows_per_query_head() {
        // :751: ceil(rows / 32) * query_heads, 128 threads, no shared memory.
        for (rows, tiles) in [(2u32, 1u32), (32, 1), (33, 2), (64, 2), (65, 3), (127, 4)] {
            let launch = prefill(rows, 4096, 24, 4, RELAXED);
            assert_eq!(
                launch.kernel.name(),
                "euhedral_attention_prefill32_nvfp4",
                "rows {rows}"
            );
            assert_eq!(launch.grid, [24 * tiles, 1, 1]);
            assert_eq!(launch.block, [128, 1, 1]);
            assert_eq!(launch.shared_bytes, 0);
            assert!(!launch.programmatic);
        }
        let exact = prefill(40, 4096, 24, 4, EXACT);
        assert_eq!(exact.kernel.name(), "euhedral_attention_prefill32_nvfp4_exact");
        assert_eq!(exact.grid, [48, 1, 1]);
        // :759-760: 5 pointers, rows, query_heads, key_heads, head_dim, cache_length, then start as u64.
        let launch = prefill(40, 4096, 24, 4, RELAXED);
        assert_eq!(words(&launch), [QK, GATE, KEYS, VALUES, OUT, 40, 24, 4, 256, 4096, 0]);
        assert_eq!(sizes(&launch), [8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 8]);
    }

    #[test]
    fn fa2_takes_512_rows_or_128_rows_over_2048_keys() {
        // :767-768. 512 rows: any cache; 128..511 rows: cache_length >= 2048.
        let at = prefill(512, 512, 24, 4, RELAXED);
        assert_eq!(at.kernel.name(), "euhedral_attention_prefill_fa2_nvfp4");
        // :769-771: ceil(512 / 16) * kv_heads CTAs of 32 * (group + 2) threads, 67584 shared bytes.
        assert_eq!(at.grid, [32 * 4, 1, 1]);
        assert_eq!(at.block, [32 * 8, 1, 1]);
        assert_eq!(at.shared_bytes, 67584);
        assert!(!at.programmatic);
        assert_eq!(
            prefill(511, 2047, 24, 4, RELAXED).kernel.name(),
            "euhedral_attention_prefill32_nvfp4"
        );
        assert_eq!(prefill(513, 513, 24, 4, RELAXED).grid, [33 * 4, 1, 1]);
        assert_eq!(
            prefill(128, 2047, 24, 4, RELAXED).kernel.name(),
            "euhedral_attention_prefill32_nvfp4"
        );
        assert_eq!(
            prefill(128, 2048, 24, 4, RELAXED).kernel.name(),
            "euhedral_attention_prefill_fa2_nvfp4"
        );
        assert_eq!(
            prefill(127, 2048, 24, 4, RELAXED).kernel.name(),
            "euhedral_attention_prefill32_nvfp4"
        );
        assert_eq!(
            prefill(128, 2049, 24, 4, RELAXED).kernel.name(),
            "euhedral_attention_prefill_fa2_nvfp4"
        );
        // The tile is 16 rows: 129 rows is 9 tiles.
        assert_eq!(prefill(129, 2048, 24, 4, RELAXED).grid, [9 * 4, 1, 1]);
    }

    #[test]
    fn fa2_group_limits_and_exact_numerics() {
        // Group 1..=6 only (:767); the block is 32 * (group + 2).
        let one = prefill(512, 512, 4, 4, RELAXED);
        assert_eq!(one.kernel.name(), "euhedral_attention_prefill_fa2_nvfp4");
        assert_eq!(one.block, [96, 1, 1]);
        let six = prefill(512, 512, 24, 4, RELAXED);
        assert_eq!(six.block, [256, 1, 1]);
        let seven = prefill(512, 512, 28, 4, RELAXED);
        assert_eq!(seven.kernel.name(), "euhedral_attention_prefill32_nvfp4");
        assert_eq!(seven.grid, [16 * 28, 1, 1]);
        assert_eq!(
            prefill(512, 512, 24, 4, EXACT).kernel.name(),
            "euhedral_attention_prefill32_nvfp4_exact"
        );
        let bf16 = {
            let mut s = shape(512, 0);
            s.cache_length = 512;
            s.scratch = DevicePtr::NULL;
            causal(CacheFormat::Bf16, &s, RELAXED).unwrap()
        };
        assert_eq!(names(&bf16), ["euhedral_attention_prefill_fa2_bf16"]);
    }

    #[test]
    fn prefill_validation() {
        let mut s = shape(512, 1000);
        s.scratch = DevicePtr::NULL;
        s.cache_length = 1511; // rows > cache_length - start (:749)
        assert_eq!(causal(CacheFormat::Nvfp4, &s, RELAXED), Err(PlanError::InvalidArgument));
        s.cache_length = 1512;
        assert!(causal(CacheFormat::Nvfp4, &s, RELAXED).is_ok());
        // :752: query_heads * 64 past u32.
        let mut huge = shape(2, 0);
        huge.scratch = DevicePtr::NULL;
        huge.query_heads = 1 << 26;
        huge.key_heads = 1;
        assert_eq!(causal(CacheFormat::Nvfp4, &huge, RELAXED), Err(PlanError::SizeOverflow));
    }

    // ---- the row twins (qwen_layer_ops.c:688-734) ----

    #[test]
    fn twins_serve_up_to_eight_rows_with_scratch() {
        // :688: rows <= 8 with scratch, not exact; rows 9 falls to prefill, no scratch falls to prefill.
        for rows in 2..=8u32 {
            let plan = nvfp4(&shape(rows, 100), RELAXED);
            assert_eq!(
                names(&plan),
                [
                    "euhedral_attention_decode_nvfp4_rows",
                    "euhedral_attention_merge_nvfp4_rows"
                ],
                "rows {rows}"
            );
            assert_eq!(plan[0].grid[1], rows);
            assert_eq!(plan[1].grid, [24, rows, 1]);
        }
        assert_eq!(
            names(&nvfp4(&shape(9, 100), RELAXED)),
            ["euhedral_attention_prefill32_nvfp4"]
        );
        let mut bare = shape(4, 100);
        bare.scratch = DevicePtr::NULL;
        assert_eq!(names(&nvfp4(&bare, RELAXED)), ["euhedral_attention_prefill32_nvfp4"]);
        // Exact numerics leave the twins (:691).
        assert_eq!(
            names(&nvfp4(&shape(4, 100), EXACT)),
            ["euhedral_attention_prefill32_nvfp4_exact"]
        );
    }

    #[test]
    fn twin_below_2048_keys() {
        // rows 2, start 100: first 101, last 102 < from 2048 (:701-713). below = last, splits = ceil(102 / 48) = 3.
        let plan = nvfp4(&shape(2, 100), RELAXED);
        assert_eq!(plan.len(), 2);
        let stride = 24u64 * 64 * 258;
        assert_eq!(plan[0].grid, [24 * 3, 2, 1]);
        assert_eq!(plan[0].block, [128, 1, 1]);
        assert_eq!(plan[0].shared_bytes, 0);
        assert!(plan[0].programmatic);
        assert_eq!(
            words(&plan[0]),
            [QK, KEYS, VALUES, 24, 4, POSITION, SCRATCH, stride, 2048]
        );
        assert_eq!(sizes(&plan[0]), [8, 8, 8, 4, 4, 8, 8, 8, 4]);
        // :732: gate, output, scratch, query_heads, key_heads, position, row_stride, from.
        assert_eq!(words(&plan[1]), [GATE, OUT, SCRATCH, 24, 4, POSITION, stride, 2048]);
        assert_eq!(sizes(&plan[1]), [8, 8, 8, 4, 4, 8, 8, 4]);
        assert_eq!(plan[1].block, [128, 1, 1]);
    }

    #[test]
    fn twin_decode_rows_split_boundaries() {
        // :708-709: below = min(last, 2047); splits = ceil(below / 48) <= 64. 48 splits hold 2304 keys, so the 2047
        // cap gives 43. Rows 3, start 45: last 48 -> 1 split; start 46: last 49 -> 2.
        assert_eq!(nvfp4(&shape(3, 45), RELAXED)[0].grid, [24, 3, 1]);
        assert_eq!(nvfp4(&shape(3, 46), RELAXED)[0].grid, [48, 3, 1]);
        // Last row exactly at 2047: no GQA part, 43 splits.
        assert_eq!(nvfp4(&shape(3, 2044), RELAXED).len(), 2);
        assert_eq!(nvfp4(&shape(3, 2044), RELAXED)[0].grid, [24 * 43, 3, 1]);
        // Group 9 keeps the per-head kernel for every key (from = u32::MAX, :701): 64 splits well past 3072 keys.
        let wide = nvfp4(&with_group(shape(2, 5000), 36, 4), RELAXED);
        assert_eq!(
            names(&wide),
            [
                "euhedral_attention_decode_nvfp4_rows",
                "euhedral_attention_merge_nvfp4_rows"
            ]
        );
        assert_eq!(wide[0].grid, [36 * 64, 2, 1]);
        assert_eq!(*wide[0].args.words().last().unwrap(), u64::from(u32::MAX));
        // Group 8 still has the GQA kernel.
        let eight = nvfp4(&with_group(shape(2, 5000), 32, 4), RELAXED);
        assert_eq!(names(&eight)[0], "euhedral_attention_decode_gqa_nvfp4_rows");
    }

    #[test]
    fn twin_crossing_2048_runs_decode_then_gqa_then_merge() {
        // rows 4, start 2045: first 2046 < 2048 <= last 2049. decode_rows below = 2047 -> 43 splits (:708);
        // GQA splits = ceil(2049 / 32) = 65 -> 64 (:715-716). tiles = (4 * 6 + 7) / 8 = 3; fused_from(4) = 4096 > 2049.
        let plan = nvfp4(&shape(4, 2045), RELAXED);
        assert_eq!(
            names(&plan),
            [
                "euhedral_attention_decode_nvfp4_rows",
                "euhedral_attention_decode_gqa_nvfp4_rows",
                "euhedral_attention_merge_nvfp4_rows"
            ]
        );
        assert_eq!(plan[0].grid, [24 * 43, 4, 1]);
        assert_eq!(plan[1].grid, [4 * 64, 4, 1]);
        assert_eq!(plan[1].block, [96, 1, 1]);
        let stride = 24u64 * 64 * 258;
        // :728: ..., row_stride, from, fused (0).
        assert_eq!(
            words(&plan[1]),
            [QK, KEYS, VALUES, 24, 4, POSITION, SCRATCH, stride, 2048, 0]
        );
        assert_eq!(sizes(&plan[1]), [8, 8, 8, 4, 4, 8, 8, 8, 4, 4]);
        assert_eq!(plan[2].grid, [24, 4, 1]);
        // Starting at 2047: first 2048 is not below from, so no per-head part.
        let at = nvfp4(&shape(2, 2047), RELAXED);
        assert_eq!(names(&at)[0], "euhedral_attention_decode_gqa_nvfp4_rows");
        // first = 2047 (start 2046, rows 1 -> not twin); rows 2 start 2045: first 2046, last 2047: per-head only.
        assert_eq!(
            names(&nvfp4(&shape(2, 2045), RELAXED))[0],
            "euhedral_attention_decode_nvfp4_rows"
        );
        assert_eq!(nvfp4(&shape(2, 2045), RELAXED).len(), 2);
        // first = 2047 exactly (start 2046), rows 2: last 2048: both parts.
        assert_eq!(nvfp4(&shape(2, 2046), RELAXED).len(), 3);
    }

    #[test]
    fn fused_verifier_from_table_by_rows() {
        // fused_from (:66-67): rows 2..=8 -> 49152, 16384, 4096, 2048, 16384, 16384, 12288 keys. The fused kernel
        // runs before the GQA twin when last >= fused_from(rows), tiles = max(2, ceil(rows * group / 8)) <= 6.
        let fused = |rows: u32, group_heads: (u32, u32), last: u64| {
            let s = with_group(shape(rows, last - u64::from(rows)), group_heads.0, group_heads.1);
            let plan = nvfp4(&s, RELAXED);
            plan.iter().any(|l| l.kernel.name().contains("fused"))
        };
        let g6 = (24, 4);
        for (rows, from) in [
            (2u32, 49152u64),
            (3, 16384),
            (4, 4096),
            (5, 2048),
            (6, 16384),
            (7, 16384),
            (8, 12288),
        ] {
            assert!(!fused(rows, g6, from - 1), "rows {rows} below {from}");
            assert!(fused(rows, g6, from), "rows {rows} at {from}");
            assert!(fused(rows, g6, from + 1), "rows {rows} above {from}");
        }
    }

    #[test]
    fn fused_launch_geometry_and_order() {
        // rows 5, group 6: tiles = (30 + 7) / 8 = 4 -> fused4. start 2043: last 2048 = fused_from(5).
        let plan = nvfp4(&shape(5, 2043), RELAXED);
        assert_eq!(
            names(&plan),
            [
                "euhedral_attention_decode_nvfp4_rows",
                "euhedral_attention_decode_gqa_nvfp4_fused4",
                "euhedral_attention_decode_gqa_nvfp4_rows",
                "euhedral_attention_merge_nvfp4_rows"
            ]
        );
        // decode_rows serves positions below 2048: below = 2047, 43 splits.
        assert_eq!(plan[0].grid, [24 * 43, 5, 1]);
        // splits = ceil(2048 / 32) = 64. Fused: kv_heads * splits CTAs of 32 * (2 * 4 + 1) threads.
        assert_eq!(plan[1].grid, [4 * 64, 1, 1]);
        assert_eq!(plan[1].block, [288, 1, 1]);
        // 2 * 16 * 264 * 2 + 4 * 1024 + 4 * 768 + 2 * 4 * 16 * 144 + 1024 + 8 * (12 + 16) = 43744 (:61-63).
        assert_eq!(plan[1].shared_bytes, 43744);
        assert!(plan[1].programmatic);
        let stride = 24u64 * 64 * 258;
        // :723: ..., row_stride, rows, from.
        assert_eq!(
            words(&plan[1]),
            [QK, KEYS, VALUES, 24, 4, POSITION, SCRATCH, stride, 5, 2048]
        );
        assert_eq!(sizes(&plan[1]), [8, 8, 8, 4, 4, 8, 8, 8, 4, 4]);
        // The GQA twin follows with fused = 1.
        assert_eq!(plan[2].grid, [4 * 64, 5, 1]);
        assert_eq!(*plan[2].args.words().last().unwrap(), 1);
    }

    #[test]
    fn fused_kernel_follows_the_tile_count() {
        // tiles = max(2, ceil(rows * group / 8)), kernel fused[tiles - 2], shared bytes of that tile count.
        let last = 60_000u64;
        let one = |rows: u32, qh: u32, kh: u32| {
            let plan = nvfp4(&with_group(shape(rows, last - u64::from(rows)), qh, kh), RELAXED);
            plan.iter().find(|l| l.kernel.name().contains("fused")).copied()
        };
        // rows 2, group 6: (12 + 7) / 8 = 2 -> fused2 (40096 bytes, 160 threads).
        let two = one(2, 24, 4).unwrap();
        assert_eq!(two.kernel.name(), "euhedral_attention_decode_gqa_nvfp4_fused2");
        assert_eq!(two.shared_bytes, 40096);
        assert_eq!(two.block, [160, 1, 1]);
        // rows 8, group 1: 8 / 8 = 1 -> raised to 2.
        assert_eq!(
            one(8, 4, 4).unwrap().kernel.name(),
            "euhedral_attention_decode_gqa_nvfp4_fused2"
        );
        // rows 3 group 6: (18 + 7) / 8 = 3. rows 6 group 6: (36 + 7) / 8 = 5. rows 8 group 6: 55 / 8 = 6.
        assert_eq!(
            one(3, 24, 4).unwrap().kernel.name(),
            "euhedral_attention_decode_gqa_nvfp4_fused3"
        );
        assert_eq!(one(3, 24, 4).unwrap().shared_bytes, 41920);
        assert_eq!(
            one(6, 24, 4).unwrap().kernel.name(),
            "euhedral_attention_decode_gqa_nvfp4_fused5"
        );
        assert_eq!(one(6, 24, 4).unwrap().shared_bytes, 45568);
        let six = one(8, 24, 4).unwrap();
        assert_eq!(six.kernel.name(), "euhedral_attention_decode_gqa_nvfp4_fused6");
        assert_eq!(six.shared_bytes, 47392);
        assert_eq!(six.block, [416, 1, 1]);
        // rows 8 group 8: 71 / 8 = 8 tiles > 6: no fused kernel.
        assert!(one(8, 32, 4).is_none());
        // rows 5 group 8: (40 + 7) / 8 = 5 tiles: fused5.
        assert_eq!(
            one(5, 32, 4).unwrap().kernel.name(),
            "euhedral_attention_decode_gqa_nvfp4_fused5"
        );
    }

    #[test]
    fn fused_never_for_bf16_and_never_past_eight_rows() {
        // BF16 has no fused kernels (:720: kernels->fused[0] != NULL).
        let bf16 = causal(CacheFormat::Bf16, &shape(5, 59_000), RELAXED).unwrap();
        assert_eq!(
            names(&bf16),
            [
                "euhedral_attention_decode_gqa_bf16_rows",
                "euhedral_attention_merge_bf16_rows"
            ]
        );
        assert_eq!(*bf16[0].args.words().last().unwrap(), 0);
        // Row-exact with 9 rows: fused_from(9) = u32::MAX (:68).
        let mut s = with_group(shape(9, 100_000), 4, 4);
        s.cache_length = 200_000;
        let plan = nvfp4(&s, ROW_EXACT);
        assert!(plan.iter().all(|l| !l.kernel.name().contains("fused")));
        assert_eq!(names(&plan)[0], "euhedral_attention_decode_gqa_nvfp4_rows");
        // Row 8 on the same shape does fuse at 12288.
        let mut s = with_group(shape(8, 12_280), 4, 4);
        s.cache_length = 200_000;
        assert!(nvfp4(&s, RELAXED).iter().any(|l| l.kernel.name().contains("fused")));
    }

    #[test]
    fn twin_validation_falls_through_to_the_other_branches() {
        // rows 5 with start + rows past the cache: not twin, not row-exact: validation rejects (:749).
        let mut s = shape(5, 96);
        s.cache_length = 100;
        assert_eq!(causal(CacheFormat::Nvfp4, &s, RELAXED), Err(PlanError::InvalidArgument));
        s.cache_length = 101;
        assert_eq!(names(&nvfp4(&s, RELAXED))[0], "euhedral_attention_decode_nvfp4_rows");
        // Row-exact with a short cache for 65 rows: twin out (rows > 64), recursion rows hit validation.
        let mut s = shape(65, 10);
        s.cache_length = 74;
        assert_eq!(
            causal(CacheFormat::Nvfp4, &s, ROW_EXACT),
            Err(PlanError::InvalidArgument)
        );
    }

    // ---- row-exact recursion (qwen_layer_ops.c:736-746) ----

    #[test]
    fn row_exact_over_64_rows_recurses_per_row_with_offsets() {
        let width = (24u64 + 4) * 256 * 2;
        let out = 24u64 * 256 * 2;
        let s = shape(65, 100);
        let plan = nvfp4(&s, ROW_EXACT);
        assert_eq!(plan.len(), 130);
        for row in [0u64, 1, 2, 3, 64] {
            let decode = &plan[2 * row as usize];
            let merge = &plan[2 * row as usize + 1];
            assert_eq!(decode.kernel.name(), "euhedral_attention_decode_nvfp4");
            assert_eq!(merge.kernel.name(), "euhedral_attention_merge_nvfp4");
            // :740-742: query_key and gate advance by width, output by out; start by row; offset arg by row.
            assert_eq!(
                words(decode),
                [
                    QK + row * width,
                    GATE + row * width,
                    KEYS,
                    VALUES,
                    OUT + row * out,
                    1,
                    24,
                    4,
                    256,
                    POSITION,
                    row,
                    SCRATCH,
                    { (100 + row + 1).div_ceil(48) }
                ]
            );
            assert_eq!(merge.args.words()[0], GATE + row * width);
            assert_eq!(merge.args.words()[1], OUT + row * out);
        }
    }

    #[test]
    fn row_exact_recursion_row_counts() {
        // Exact numerics turn the twins off even for few rows, so rows 1, 2, an odd count and a large count recurse.
        let both = Modes {
            row_exact: true,
            exact_numerics: true,
        };
        let count = |rows: u32| nvfp4(&shape(rows, 10), both).len();
        assert_eq!(count(1), 2); // one row is plain decode, the recursion is not entered (:736 needs rows > 1)
        assert_eq!(count(2), 4);
        assert_eq!(count(7), 14);
        assert_eq!(count(1000), 2000);
        let plan = nvfp4(&shape(3, 10), both);
        assert_eq!(names(&plan)[0], "euhedral_attention_decode_nvfp4_exact");
        assert_eq!(plan[2].args.words()[10], 1);
        assert_eq!(plan[4].args.words()[10], 2);
        // The last launch of the large count is the last row's merge.
        let mut last = None;
        plan_causal(CacheFormat::Nvfp4, &shape(1000, 10), both, &mut |l| last = Some(l)).unwrap();
        assert_eq!(last.unwrap().kernel.name(), "euhedral_attention_merge_nvfp4");
    }

    #[test]
    fn row_exact_recursion_uses_gqa_from_each_rows_own_length() {
        // Exact numerics off, 65 rows: row j has length start + j + 1; rows from 2048 keys take GQA (:797).
        let mut s = shape(65, 2000);
        s.cache_length = 3000;
        let plan = nvfp4(&s, ROW_EXACT);
        let decodes: Vec<_> = plan.iter().step_by(2).collect();
        // length 2047 is row 46 (2000 + 46 + 1).
        assert_eq!(decodes[46].kernel.name(), "euhedral_attention_decode_nvfp4");
        assert_eq!(decodes[47].kernel.name(), "euhedral_attention_decode_gqa_nvfp4");
    }

    #[test]
    fn row_exact_without_scratch_is_rejected_on_the_first_row() {
        let mut s = shape(2, 10);
        s.scratch = DevicePtr::NULL;
        let mut count = 0;
        let result = plan_causal(CacheFormat::Nvfp4, &s, ROW_EXACT, &mut |_| count += 1);
        assert_eq!(result, Err(PlanError::InvalidArgument));
        assert_eq!(count, 0);
    }

    #[test]
    fn row_exact_with_scratch_prefers_the_twins_up_to_64_rows() {
        let plan = nvfp4(&shape(64, 10), ROW_EXACT);
        assert_eq!(
            names(&plan),
            [
                "euhedral_attention_decode_nvfp4_rows",
                "euhedral_attention_merge_nvfp4_rows"
            ]
        );
        assert_eq!(plan[0].grid, [24 * 2, 64, 1]); // last 74 -> ceil(74 / 48) = 2
    }

    #[test]
    fn decode_split_counts_do_not_wrap_near_the_key_limit() {
        // C sums in u64 (:781, :798): length = 4294967294 (start 4294967293) gives (length + 47) / 48 and
        // (length + 31) / 32, both far above 64, so 64 splits. Group 6 and relaxed: GQA, grid 4 * 64.
        let mut s = shape(1, 4_294_967_293);
        s.cache_length = 4_294_967_294;
        let gqa = nvfp4(&s, RELAXED);
        assert_eq!(gqa[0].grid, [256, 1, 1]);
        assert_eq!(*gqa[0].args.words().last().unwrap(), 64);
        assert_eq!(*gqa[1].args.words().last().unwrap(), 64);
        // 48-key path: exact numerics, grid query_heads * 64.
        let per_head = nvfp4(&s, EXACT);
        assert_eq!(per_head[0].grid, [24 * 64, 1, 1]);
        assert_eq!(*per_head[0].args.words().last().unwrap(), 64);
        assert_eq!(*per_head[1].args.words().last().unwrap(), 64);
    }

    #[test]
    fn plans_never_panic_on_pointer_wrap() {
        // C's pointer arithmetic wraps; so does the plan.
        let mut s = shape(65, 10);
        s.query_key = ptr(u64::MAX - 100);
        s.cache_length = 200;
        assert!(causal(CacheFormat::Nvfp4, &s, ROW_EXACT).is_ok());
    }

    // ---- KV append (qwen_layer_ops.c:552-582) ----

    fn append_shape(rows: u32) -> AppendShape {
        AppendShape {
            query_key: ptr(QK),
            gate_value: ptr(GATE),
            key_cache: ptr(KEYS),
            value_cache: ptr(VALUES),
            position: ptr(POSITION),
            rows,
            query_width: 6144,
            key_value_width: 1024,
            start_position: 0,
        }
    }

    fn append(format: CacheFormat, s: &AppendShape) -> Result<Vec<Launch>, PlanError> {
        let mut out = Vec::new();
        plan_kv_append(format, s, &mut |l| out.push(l))?;
        Ok(out)
    }

    #[test]
    fn kv_append_grid_is_rows_times_width_over_1024() {
        // :581: (rows * 1024 / 256 + 3) / 4 = (4 * rows + 3) / 4 = rows for width 1024; 128 threads.
        for rows in [1u32, 2, 3, 8, 33, 512] {
            let plan = append(CacheFormat::Nvfp4, &append_shape(rows)).unwrap();
            assert_eq!(plan.len(), 1);
            assert_eq!(plan[0].kernel.name(), "euhedral_attention_kv_append_nvfp4");
            assert_eq!(plan[0].grid, [rows, 1, 1]);
            assert_eq!(plan[0].block, [128, 1, 1]);
            assert!(plan[0].programmatic);
        }
        // Width 256 and rows 5: count / 256 = 5, (5 + 3) / 4 = 2; rows 4 -> 7 / 4 = 1; rows 6 -> 9 / 4 = 2.
        let narrow = |rows| AppendShape {
            key_value_width: 256,
            query_width: 256,
            ..append_shape(rows)
        };
        for (rows, grid) in [(4u32, 1u32), (5, 2), (6, 2), (9, 3)] {
            assert_eq!(
                append(CacheFormat::Bf16, &narrow(rows)).unwrap()[0].grid,
                [grid, 1, 1],
                "rows {rows}"
            );
        }
        let plan = append(CacheFormat::Bf16, &append_shape(2)).unwrap();
        assert_eq!(plan[0].kernel.name(), "euhedral_attention_kv_append_bf16");
        // :579-580.
        assert_eq!(words(&plan[0]), [QK, GATE, KEYS, VALUES, 2, 6144, 1024, POSITION]);
        assert_eq!(sizes(&plan[0]), [8, 8, 8, 8, 4, 4, 4, 8]);
    }

    #[test]
    fn kv_append_validation() {
        let bad = |s: AppendShape| append(CacheFormat::Nvfp4, &s);
        assert_eq!(
            bad(AppendShape {
                rows: 0,
                ..append_shape(1)
            }),
            Err(PlanError::InvalidArgument)
        );
        assert_eq!(
            bad(AppendShape {
                key_value_width: 1000,
                ..append_shape(1)
            }),
            Err(PlanError::InvalidArgument)
        );
        assert_eq!(
            bad(AppendShape {
                query_width: 6000,
                ..append_shape(1)
            }),
            Err(PlanError::InvalidArgument)
        );
        assert_eq!(
            bad(AppendShape {
                position: DevicePtr::NULL,
                ..append_shape(1)
            }),
            Err(PlanError::InvalidArgument)
        );
        // start_position > u32::MAX - rows
        let edge = u64::from(u32::MAX) - 4;
        assert!(
            bad(AppendShape {
                start_position: edge,
                ..append_shape(4)
            })
            .is_ok()
        );
        assert_eq!(
            bad(AppendShape {
                start_position: edge + 1,
                ..append_shape(4)
            }),
            Err(PlanError::InvalidArgument)
        );
        // count = rows * width past u32 (:568): 4194304 * 1024 = 2^32.
        assert_eq!(bad(append_shape(1 << 22)), Err(PlanError::SizeOverflow));
        assert!(bad(append_shape((1 << 22) - 1)).is_ok());
    }

    // ---- QK norm and RoPE (qwen_layer_ops.c:609-654) ----

    fn rope_shape(rows: u32) -> RopeShape {
        RopeShape {
            query_key: ptr(QK),
            query_norm: ptr(0x8000),
            key_norm: ptr(0x9000),
            output: ptr(OUT),
            position: ptr(POSITION),
            rows,
            query_heads: 24,
            key_value_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
            start_position: 0,
            epsilon: 1e-6,
            rope_theta: 10_000_000.0,
        }
    }

    fn rope(s: &RopeShape, row_exact: bool) -> Result<Vec<Launch>, PlanError> {
        let mut out = Vec::new();
        let modes = Modes {
            row_exact,
            exact_numerics: false,
        };
        plan_qk_norm_rope(s, modes, &mut |l| out.push(l))?;
        Ok(out)
    }

    #[test]
    fn rope_one_row_uses_one_cta_per_head() {
        // :653: blocks = rows * (query_heads + kv_heads), head_dim threads.
        let plan = rope(&rope_shape(1), false).unwrap();
        assert_eq!(names(&plan), ["euhedral_attention_qk_norm_rope_bf16"]);
        assert_eq!(plan[0].grid, [28, 1, 1]);
        assert_eq!(plan[0].block, [256, 1, 1]);
        assert!(plan[0].programmatic);
        // :648-649: f32 epsilon and f64 theta as their bits; the offset is 0.
        let args = &plan[0].args;
        assert_eq!(sizes(&plan[0]), [8, 8, 8, 8, 4, 4, 4, 4, 4, 8, 8, 4, 8]);
        assert_eq!(
            &args.words()[..11],
            [QK, 0x8000, 0x9000, OUT, 1, 24, 4, 256, 64, POSITION, 0]
        );
        assert_eq!(args.f32_at(11), Some(1e-6));
        assert_eq!(args.f64_at(12), Some(10_000_000.0));
    }

    #[test]
    fn rope_rows_kernel_from_two_rows_unless_row_exact() {
        // :651-652: one CTA per row, 256 threads.
        for rows in [2u32, 3, 33, 512] {
            let plan = rope(&rope_shape(rows), false).unwrap();
            assert_eq!(names(&plan), ["euhedral_attention_qk_norm_rope_rows_bf16"]);
            assert_eq!(plan[0].grid, [rows, 1, 1]);
            assert_eq!(plan[0].block, [256, 1, 1]);
            // Row-exact: the per-head kernel over rows * heads blocks (:616, :653).
            let exact = rope(&rope_shape(rows), true).unwrap();
            assert_eq!(names(&exact), ["euhedral_attention_qk_norm_rope_bf16"]);
            assert_eq!(exact[0].grid, [rows * 28, 1, 1]);
        }
        // One row row-exact is the plain per-head launch.
        assert_eq!(rope(&rope_shape(1), true).unwrap()[0].grid, [28, 1, 1]);
    }

    #[test]
    fn rope_row_exact_recursion_past_u32_blocks() {
        // :617-626: 3 rows * (2^31 + 1) heads > u32::MAX: one launch per row, pointers advanced by width,
        // offset and the host position advanced by row.
        let mut s = rope_shape(3);
        s.query_heads = 1 << 31;
        s.key_value_heads = 1;
        let heads = (1u64 << 31) + 1;
        let width = heads * 256 * 2;
        let plan = rope(&s, true).unwrap();
        assert_eq!(plan.len(), 3);
        for (row, launch) in plan.iter().enumerate() {
            let row = row as u64;
            assert_eq!(launch.kernel.name(), "euhedral_attention_qk_norm_rope_bf16");
            assert_eq!(launch.grid, [heads as u32, 1, 1]);
            assert_eq!(launch.args.words()[0], QK + row * width);
            assert_eq!(launch.args.words()[3], OUT + row * width);
            assert_eq!(launch.args.words()[4], 1);
            assert_eq!(launch.args.words()[10], row);
        }
        // Not row-exact: the rows kernel would need rows CTAs, which fit; the host checks blocks first (:636).
        assert_eq!(rope(&s, false), Err(PlanError::SizeOverflow));
    }

    #[test]
    fn rope_validation() {
        let bad = |f: fn(&mut RopeShape)| {
            let mut s = rope_shape(2);
            f(&mut s);
            rope(&s, false)
        };
        assert_eq!(bad(|s| s.head_dim = 128), Err(PlanError::InvalidArgument));
        assert_eq!(bad(|s| s.rotary_dim = 63), Err(PlanError::InvalidArgument));
        assert_eq!(bad(|s| s.rotary_dim = 257), Err(PlanError::InvalidArgument));
        assert_eq!(bad(|s| s.epsilon = f32::NAN), Err(PlanError::InvalidArgument));
        assert_eq!(bad(|s| s.rope_theta = 0.0), Err(PlanError::InvalidArgument));
        assert_eq!(bad(|s| s.query_heads = 25), Err(PlanError::InvalidArgument));
        assert_eq!(bad(|s| s.start_position = u64::MAX), Err(PlanError::InvalidArgument));
        assert!(bad(|s| s.start_position = u64::MAX - 2).is_ok());
    }
}
