//! The DFlash2 drafter's launches: plain functions from a shape to the launches that run it.
//!
//! Each function pushes its [`Launch`]es into the caller's sink and allocates nothing. A shape the kernels refuse
//! returns the matching [`PlanError`].

use crate::DevicePtr;
use crate::kernel;
use crate::launch::PlanError;
use crate::launch::catalog::{Arguments, KernelId, Launch};

const LINEAR: KernelId = kernel!("euhedral_dflash_linear_bf16");
const LINEAR_ROWS: KernelId = kernel!("euhedral_dflash_linear_rows_bf16");
const LINEAR_SPLIT: KernelId = kernel!("euhedral_dflash_linear_split_bf16");
const LINEAR_ROWS_SPLIT: KernelId = kernel!("euhedral_dflash_linear_rows_split_bf16");
const RMS_NORM: KernelId = kernel!("euhedral_dflash_rms_norm_bf16");
const CONV: KernelId = kernel!("euhedral_dflash_conv_bf16");
const CONTEXT_KV: KernelId = kernel!("euhedral_dflash_context_kv_bf16");
const BLOCK_QK: KernelId = kernel!("euhedral_dflash_block_qk_bf16");
const ATTENTION: KernelId = kernel!("euhedral_dflash_attention_tc_bf16");
const ATTENTION_MERGE: KernelId = kernel!("euhedral_dflash_attention_merge_bf16");
const SWIGLU: KernelId = kernel!("euhedral_dflash_swiglu_bf16");
const TOPK_PARTIAL: KernelId = kernel!("euhedral_dflash_topk_partial_bf16");
const TOPK_MERGE: KernelId = kernel!("euhedral_dflash_topk_merge_bf16");
const SELECT: KernelId = kernel!("euhedral_dflash_select_bf16");

/// The key splits of the attention kernel (`dflash::kAttentionSplits`).
const ATTENTION_SPLITS: u32 = 8;
/// Its dynamic shared bytes (`dflash::kAttentionSharedBytes`).
const ATTENTION_SHARED_BYTES: u32 = 32 * 276 * 4 + 2 * 32 * 280 * 2 + 16 * 136 * 2 + 64 * 4;
/// The slices of the vocabulary the top-k's first pass splits each row into.
const TOPK_SPLITS: u32 = 64;
/// The candidates each slice keeps, and so the width of the top-k.
pub const TOPK_WIDTH: u32 = 16;
/// Outputs up to this wide split K over a CTA's four warps.
const LINEAR_SPLIT_MAX_OUTPUTS: u32 = 1536;
/// The only head dimension the kernels serve.
const HEAD_DIM: u32 = 128;

/// The shape of `dflash_linear_bf16`.
#[derive(Clone, Copy, Debug)]
pub struct LinearShape {
    pub input: DevicePtr,
    pub weights: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub in_features: u32,
    pub out_features: u32,
}

/// `dflash_linear_bf16`: one launch. Up to 16 rows stream the weights once; more rows reuse each weight
/// load over 64 rows. Outputs up to 1536 wide split K in four, in both; each row gets the same bits either way.
pub fn plan_linear(shape: &LinearShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    let split = s.out_features <= LINEAR_SPLIT_MAX_OUTPUTS;
    let tile = if split { 8 } else { 32 };
    if s.input.is_null()
        || s.weights.is_null()
        || s.output.is_null()
        || s.rows == 0
        || s.in_features == 0
        || !s.in_features.is_multiple_of(32)
        || s.out_features == 0
        || !s.out_features.is_multiple_of(tile)
    {
        return Err(PlanError::InvalidArgument);
    }
    let args = Arguments::new()
        .pointer(s.input)
        .pointer(s.weights)
        .pointer(s.output)
        .u32(s.rows)
        .u32(s.in_features)
        .u32(s.out_features);
    let (kernel, grid) = match (split, s.rows <= 16) {
        (true, true) => (LINEAR_SPLIT, [s.out_features / 8, 1, 1]),
        (true, false) => (LINEAR_ROWS_SPLIT, [s.out_features / 8, s.rows.div_ceil(64), 1]),
        (false, true) => (LINEAR, [s.out_features / 32, 1, 1]),
        (false, false) => (LINEAR_ROWS, [s.out_features / 32, s.rows.div_ceil(64), 1]),
    };
    sink(Launch::new(kernel, grid, [128, 1, 1], 0, args));
    Ok(())
}

/// The shape of `dflash_rms_norm_bf16`.
#[derive(Clone, Copy, Debug)]
pub struct RmsNormShape {
    pub input: DevicePtr,
    pub weight: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub width: u32,
    pub epsilon: f32,
}

/// `dflash_rms_norm_bf16`: one launch, a block per row.
pub fn plan_rms_norm(shape: &RmsNormShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.input.is_null()
        || s.weight.is_null()
        || s.output.is_null()
        || s.rows == 0
        || s.width == 0
        || !s.epsilon.is_finite()
    {
        return Err(PlanError::InvalidArgument);
    }
    let args = Arguments::new()
        .pointer(s.input)
        .pointer(s.weight)
        .pointer(s.output)
        .u32(s.width)
        .f32(s.epsilon);
    sink(Launch::new(RMS_NORM, [s.rows, 1, 1], [256, 1, 1], 0, args));
    Ok(())
}

/// The shape of `dflash_conv_bf16`.
#[derive(Clone, Copy, Debug)]
pub struct ConvShape {
    pub input: DevicePtr,
    pub dynamic: DevicePtr,
    pub base: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub width: u32,
    pub group: u32,
    pub taps: u32,
    /// 0 prepares the dynamic taps, 1 finishes the convolution.
    pub part: u32,
}

/// `dflash_conv_bf16`: one launch over every element.
pub fn plan_conv(shape: &ConvShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.input.is_null()
        || s.dynamic.is_null()
        || s.base.is_null()
        || s.output.is_null()
        || s.rows == 0
        || s.width == 0
        || s.group == 0
        || !s.width.is_multiple_of(s.group)
        || s.taps == 0
        || s.part > 1
    {
        return Err(PlanError::InvalidArgument);
    }
    let count = u64::from(s.rows) * u64::from(s.width);
    if count > u64::from(u32::MAX) {
        return Err(PlanError::SizeOverflow);
    }
    let args = Arguments::new()
        .pointer(s.input)
        .pointer(s.dynamic)
        .pointer(s.base)
        .pointer(s.output)
        .u32(s.rows)
        .u32(s.width)
        .u32(s.group)
        .u32(s.taps)
        .u32(s.part);
    sink(Launch::new(
        CONV,
        [count.div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        0,
        args,
    ));
    Ok(())
}

/// The shape of `dflash_context_kv_bf16`.
#[derive(Clone, Copy, Debug)]
pub struct ContextKvShape {
    pub kv: DevicePtr,
    pub key_norm: DevicePtr,
    pub ring_keys: DevicePtr,
    pub ring_values: DevicePtr,
    pub rows: u32,
    /// The device address of the u64 position of the first row.
    pub position: DevicePtr,
    pub window: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
    pub epsilon: f32,
    pub theta: f32,
}

/// `dflash_context_kv_bf16`: one launch, a block per row and head. More rows than the window would write
/// a ring slot twice in one launch, so they are refused.
pub fn plan_context_kv(shape: &ContextKvShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.kv.is_null()
        || s.key_norm.is_null()
        || s.ring_keys.is_null()
        || s.ring_values.is_null()
        || s.position.is_null()
        || s.rows == 0
        || s.window == 0
        || s.rows > s.window
        || s.key_value_heads == 0
        || s.head_dim != HEAD_DIM
    {
        return Err(PlanError::InvalidArgument);
    }
    let args = Arguments::new()
        .pointer(s.kv)
        .pointer(s.key_norm)
        .pointer(s.ring_keys)
        .pointer(s.ring_values)
        .pointer(s.position)
        .u32(s.window)
        .u32(s.key_value_heads)
        .f32(s.epsilon)
        .f32(s.theta);
    sink(Launch::new(
        CONTEXT_KV,
        [s.rows, s.key_value_heads, 1],
        [128, 1, 1],
        0,
        args,
    ));
    Ok(())
}

/// The shape of `dflash_block_qk_bf16`.
#[derive(Clone, Copy, Debug)]
pub struct BlockQkShape {
    pub query: DevicePtr,
    pub kv: DevicePtr,
    pub query_norm: DevicePtr,
    pub key_norm: DevicePtr,
    pub query_out: DevicePtr,
    pub key_out: DevicePtr,
    pub rows: u32,
    pub position: DevicePtr,
    pub heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
    pub epsilon: f32,
    pub theta: f32,
}

/// `dflash_block_qk_bf16`: one launch, a block per row and (query or key) head.
pub fn plan_block_qk(shape: &BlockQkShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.query.is_null()
        || s.kv.is_null()
        || s.query_norm.is_null()
        || s.key_norm.is_null()
        || s.query_out.is_null()
        || s.key_out.is_null()
        || s.position.is_null()
        || s.rows == 0
        || s.heads == 0
        || s.key_value_heads == 0
        || s.head_dim != HEAD_DIM
    {
        return Err(PlanError::InvalidArgument);
    }
    let args = Arguments::new()
        .pointer(s.query)
        .pointer(s.kv)
        .pointer(s.query_norm)
        .pointer(s.key_norm)
        .pointer(s.query_out)
        .pointer(s.key_out)
        .pointer(s.position)
        .u32(s.heads)
        .u32(s.key_value_heads)
        .f32(s.epsilon)
        .f32(s.theta);
    sink(Launch::new(
        BLOCK_QK,
        [s.rows, s.heads + s.key_value_heads, 1],
        [128, 1, 1],
        0,
        args,
    ));
    Ok(())
}

/// The shape of `dflash_attention_bf16`.
#[derive(Clone, Copy, Debug)]
pub struct AttentionShape {
    pub query: DevicePtr,
    pub block_keys: DevicePtr,
    pub kv: DevicePtr,
    pub ring_keys: DevicePtr,
    pub ring_values: DevicePtr,
    pub output: DevicePtr,
    /// The partial results of the key splits.
    pub scratch: DevicePtr,
    pub rows: u32,
    pub position: DevicePtr,
    pub window: u32,
    pub heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
}

/// `dflash_attention_bf16`: the attention over the ring and the block, then the merge of the key splits.
/// A key-value head serves at most four query heads, and all the queries of a block (rows times that) at most 32.
pub fn plan_attention(shape: &AttentionShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.query.is_null()
        || s.block_keys.is_null()
        || s.kv.is_null()
        || s.ring_keys.is_null()
        || s.ring_values.is_null()
        || s.output.is_null()
        || s.scratch.is_null()
        || s.position.is_null()
        || s.rows == 0
        || s.window == 0
        || s.window > 2048
        || s.key_value_heads == 0
        || !s.heads.is_multiple_of(s.key_value_heads)
        || s.heads / s.key_value_heads > 4
        || u64::from(s.rows) * u64::from(s.heads / s.key_value_heads) > 32
        || s.head_dim != HEAD_DIM
    {
        return Err(PlanError::InvalidArgument);
    }
    let attend = Arguments::new()
        .pointer(s.query)
        .pointer(s.block_keys)
        .pointer(s.kv)
        .pointer(s.ring_keys)
        .pointer(s.ring_values)
        .pointer(s.scratch)
        .pointer(s.position)
        .u32(s.rows)
        .u32(s.window)
        .u32(s.heads)
        .u32(s.key_value_heads);
    sink(Launch::new(
        ATTENTION,
        [s.key_value_heads, ATTENTION_SPLITS, 1],
        [128, 1, 1],
        ATTENTION_SHARED_BYTES,
        attend,
    ));
    let merge = Arguments::new().pointer(s.scratch).pointer(s.output).u32(s.heads);
    sink(Launch::new(
        ATTENTION_MERGE,
        [s.rows, s.heads, 1],
        [128, 1, 1],
        0,
        merge,
    ));
    Ok(())
}

/// The shape of `dflash_swiglu_bf16`.
#[derive(Clone, Copy, Debug)]
pub struct SwigluShape {
    pub gate_up: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub intermediate: u32,
}

/// `dflash_swiglu_bf16`: one launch over every element.
pub fn plan_swiglu(shape: &SwigluShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.gate_up.is_null() || s.output.is_null() || s.rows == 0 || s.intermediate == 0 {
        return Err(PlanError::InvalidArgument);
    }
    let count = u64::from(s.rows) * u64::from(s.intermediate);
    if count > u64::from(u32::MAX) {
        return Err(PlanError::SizeOverflow);
    }
    let args = Arguments::new()
        .pointer(s.gate_up)
        .pointer(s.output)
        .u32(s.rows)
        .u32(s.intermediate);
    sink(Launch::new(
        SWIGLU,
        [count.div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        0,
        args,
    ));
    Ok(())
}

/// The bytes of scratch the top-k of `rows` rows needs (`dflash_topk_scratch_bytes`): each slice's
/// candidates, as 4-byte values and 4-byte indices.
pub const fn topk_scratch_bytes(rows: u32) -> u64 {
    rows as u64 * TOPK_SPLITS as u64 * TOPK_WIDTH as u64 * 8
}

/// The shape of `dflash_topk_bf16`.
#[derive(Clone, Copy, Debug)]
pub struct TopkShape {
    pub logits: DevicePtr,
    pub rows: u32,
    pub vocabulary: u32,
    /// `rows` x 16 floats.
    pub values: DevicePtr,
    /// `rows` x 16 int32.
    pub indices: DevicePtr,
    /// [`topk_scratch_bytes`] bytes.
    pub scratch: DevicePtr,
}

/// `dflash_topk_bf16`: each slice of every row keeps its top 16, then each row's candidates merge.
pub fn plan_topk(shape: &TopkShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.logits.is_null()
        || s.values.is_null()
        || s.indices.is_null()
        || s.scratch.is_null()
        || s.rows == 0
        || s.vocabulary < TOPK_WIDTH * TOPK_SPLITS
    {
        return Err(PlanError::InvalidArgument);
    }
    let partial_values = s.scratch;
    let partial_indices = super::advance(
        s.scratch,
        u64::from(s.rows) * u64::from(TOPK_SPLITS) * u64::from(TOPK_WIDTH) * 4,
    );
    let first = Arguments::new()
        .pointer(s.logits)
        .u32(s.vocabulary)
        .pointer(partial_values)
        .pointer(partial_indices);
    sink(Launch::new(
        TOPK_PARTIAL,
        [s.rows, TOPK_SPLITS, 1],
        [256, 1, 1],
        0,
        first,
    ));
    let second = Arguments::new()
        .pointer(partial_values)
        .pointer(partial_indices)
        .u32(TOPK_SPLITS)
        .pointer(s.values)
        .pointer(s.indices);
    sink(Launch::new(TOPK_MERGE, [s.rows, 1, 1], [256, 1, 1], 0, second));
    Ok(())
}

/// The shape of `dflash_select_bf16`.
#[derive(Clone, Copy, Debug)]
pub struct SelectShape {
    pub hidden: DevicePtr,
    pub values: DevicePtr,
    pub indices: DevicePtr,
    pub predecessor: DevicePtr,
    pub successor: DevicePtr,
    /// The device address of the anchor token (int32).
    pub anchor: DevicePtr,
    pub positions: u32,
    pub rank: u32,
    /// `positions` int32 tokens.
    pub tokens: DevicePtr,
    /// `positions` floats.
    pub scores: DevicePtr,
}

/// `dflash_select_bf16`: one block chooses every position's token in order.
pub fn plan_select(shape: &SelectShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.hidden.is_null()
        || s.values.is_null()
        || s.indices.is_null()
        || s.predecessor.is_null()
        || s.successor.is_null()
        || s.anchor.is_null()
        || s.tokens.is_null()
        || s.scores.is_null()
        || s.positions == 0
        || s.rank == 0
        || s.rank > 1024
    {
        return Err(PlanError::InvalidArgument);
    }
    let args = Arguments::new()
        .pointer(s.hidden)
        .pointer(s.values)
        .pointer(s.indices)
        .pointer(s.predecessor)
        .pointer(s.successor)
        .pointer(s.anchor)
        .u32(s.positions)
        .u32(s.rank)
        .pointer(s.tokens)
        .pointer(s.scores);
    sink(Launch::new(SELECT, [1, 1, 1], [512, 1, 1], 0, args));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptr(address: u64) -> DevicePtr {
        // SAFETY: the planner never dereferences a pointer.
        unsafe { DevicePtr::from_raw(address) }
    }

    fn collect(
        plan: impl FnOnce(&mut dyn FnMut(Launch)) -> Result<(), PlanError>,
    ) -> (Vec<Launch>, Result<(), PlanError>) {
        let mut launches = Vec::new();
        let result = plan(&mut |launch| launches.push(launch));
        (launches, result)
    }

    fn linear(rows: u32, out_features: u32) -> (Vec<Launch>, Result<(), PlanError>) {
        let shape = LinearShape {
            input: ptr(0x1000),
            weights: ptr(0x2000),
            output: ptr(0x3000),
            rows,
            in_features: 5120,
            out_features,
        };
        collect(|sink| plan_linear(&shape, &mut |l| sink(l)))
    }

    #[test]
    fn linear_picks_the_kernel_by_output_width_and_row_count() {
        // up to 1536 outputs split K (tile of 8 outputs), wider ones do not (tile of 32); up to 16 rows take
        // the streaming kernel, more the 64-row kernel.
        for (rows, outputs, kernel, grid) in [
            (7, 1024, kernel!("euhedral_dflash_linear_split_bf16"), [128, 1, 1]),
            (17, 1536, kernel!("euhedral_dflash_linear_rows_split_bf16"), [192, 1, 1]),
            (16, 2048, kernel!("euhedral_dflash_linear_bf16"), [64, 1, 1]),
            (130, 34816, kernel!("euhedral_dflash_linear_rows_bf16"), [1088, 3, 1]),
        ] {
            let (launches, result) = linear(rows, outputs);
            assert_eq!(result, Ok(()));
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].kernel, kernel);
            let mut expected = grid;
            if rows > 16 {
                expected[1] = rows.div_ceil(64);
            }
            assert_eq!(launches[0].grid, expected, "rows {rows} outputs {outputs}");
            assert_eq!(launches[0].block, [128, 1, 1]);
            assert_eq!(
                launches[0].args.words(),
                &[0x1000, 0x2000, 0x3000, u64::from(rows), 5120, u64::from(outputs)]
            );
        }
    }

    #[test]
    fn linear_refuses_what_the_native_wrapper_refuses() {
        // Outputs must be a multiple of the tile (8 up to 1536, else 32), inputs of 32.
        for (rows, outputs) in [(0, 1024), (1, 0), (1, 1028), (1, 1540)] {
            assert_eq!(
                linear(rows, outputs).1,
                Err(PlanError::InvalidArgument),
                "{rows} {outputs}"
            );
        }
        let shape = LinearShape {
            input: ptr(1),
            weights: ptr(2),
            output: ptr(3),
            rows: 1,
            in_features: 33,
            out_features: 8,
        };
        assert_eq!(
            collect(|sink| plan_linear(&shape, &mut |l| sink(l))).1,
            Err(PlanError::InvalidArgument)
        );
        let shape = LinearShape {
            input: DevicePtr::NULL,
            in_features: 32,
            ..shape
        };
        assert_eq!(
            collect(|sink| plan_linear(&shape, &mut |l| sink(l))).1,
            Err(PlanError::InvalidArgument)
        );
    }

    #[test]
    fn attention_is_the_split_pass_then_the_merge() {
        let shape = AttentionShape {
            query: ptr(1),
            block_keys: ptr(2),
            kv: ptr(3),
            ring_keys: ptr(4),
            ring_values: ptr(5),
            output: ptr(6),
            scratch: ptr(7),
            rows: 8,
            position: ptr(9),
            window: 2048,
            heads: 32,
            key_value_heads: 8,
            head_dim: 128,
        };
        let (launches, result) = collect(|sink| plan_attention(&shape, &mut |l| sink(l)));
        assert_eq!(result, Ok(()));
        assert_eq!(launches.len(), 2);
        assert_eq!(launches[0].kernel, kernel!("euhedral_dflash_attention_tc_bf16"));
        assert_eq!(launches[0].grid, [8, 8, 1]);
        assert_eq!(launches[0].shared_bytes, 75776);
        assert_eq!(launches[0].args.words(), &[1, 2, 3, 4, 5, 7, 9, 8, 2048, 32, 8]);
        assert_eq!(launches[1].kernel, kernel!("euhedral_dflash_attention_merge_bf16"));
        assert_eq!(launches[1].grid, [8, 32, 1]);
        assert_eq!(launches[1].args.words(), &[7, 6, 32]);
        // Nine rows of four queries per key head are 36 queries: more than the kernel's 32.
        let (launches, result) = collect(|sink| plan_attention(&AttentionShape { rows: 9, ..shape }, &mut |l| sink(l)));
        assert_eq!((launches.len(), result), (0, Err(PlanError::InvalidArgument)));
        let window = AttentionShape { window: 2049, ..shape };
        assert_eq!(
            collect(|sink| plan_attention(&window, &mut |l| sink(l))).1,
            Err(PlanError::InvalidArgument)
        );
    }

    #[test]
    fn context_kv_refuses_more_rows_than_the_window() {
        let shape = ContextKvShape {
            kv: ptr(1),
            key_norm: ptr(2),
            ring_keys: ptr(3),
            ring_values: ptr(4),
            rows: 2048,
            position: ptr(5),
            window: 2048,
            key_value_heads: 8,
            head_dim: 128,
            epsilon: 1e-6,
            theta: 1e7,
        };
        let (launches, result) = collect(|sink| plan_context_kv(&shape, &mut |l| sink(l)));
        assert_eq!(result, Ok(()));
        assert_eq!(launches[0].grid, [2048, 8, 1]);
        assert_eq!(launches[0].args.f32_at(7), Some(1e-6));
        let over = ContextKvShape { rows: 2049, ..shape };
        assert_eq!(
            collect(|sink| plan_context_kv(&over, &mut |l| sink(l))).1,
            Err(PlanError::InvalidArgument)
        );
        let head = ContextKvShape { head_dim: 64, ..shape };
        assert_eq!(
            collect(|sink| plan_context_kv(&head, &mut |l| sink(l))).1,
            Err(PlanError::InvalidArgument)
        );
    }

    #[test]
    fn topk_puts_the_merge_candidates_after_the_slice_values() {
        let shape = TopkShape {
            logits: ptr(0x100),
            rows: 7,
            vocabulary: 248_320,
            values: ptr(0x200),
            indices: ptr(0x300),
            scratch: ptr(0x1000),
        };
        let (launches, result) = collect(|sink| plan_topk(&shape, &mut |l| sink(l)));
        assert_eq!(result, Ok(()));
        assert_eq!(launches.len(), 2);
        let indices_at = 0x1000 + 7 * 64 * 16 * 4;
        assert_eq!(launches[0].grid, [7, 64, 1]);
        assert_eq!(launches[0].args.words(), &[0x100, 248_320, 0x1000, indices_at]);
        assert_eq!(launches[1].grid, [7, 1, 1]);
        assert_eq!(launches[1].args.words(), &[0x1000, indices_at, 64, 0x200, 0x300]);
        assert_eq!(topk_scratch_bytes(7), 7 * 64 * 16 * 8);
        // The slices keep 16 each: a vocabulary under 1024 cannot fill them.
        let short = TopkShape {
            vocabulary: 1023,
            ..shape
        };
        assert_eq!(
            collect(|sink| plan_topk(&short, &mut |l| sink(l))).1,
            Err(PlanError::InvalidArgument)
        );
    }

    #[test]
    fn select_is_one_block_and_refuses_a_rank_past_1024() {
        let shape = SelectShape {
            hidden: ptr(1),
            values: ptr(2),
            indices: ptr(3),
            predecessor: ptr(4),
            successor: ptr(5),
            anchor: ptr(6),
            positions: 7,
            rank: 256,
            tokens: ptr(7),
            scores: ptr(8),
        };
        let (launches, result) = collect(|sink| plan_select(&shape, &mut |l| sink(l)));
        assert_eq!(result, Ok(()));
        assert_eq!((launches[0].grid, launches[0].block), ([1, 1, 1], [512, 1, 1]));
        assert_eq!(launches[0].args.words(), &[1, 2, 3, 4, 5, 6, 7, 256, 7, 8]);
        let wide = SelectShape { rank: 1025, ..shape };
        assert_eq!(
            collect(|sink| plan_select(&wide, &mut |l| sink(l))).1,
            Err(PlanError::InvalidArgument)
        );
    }

    #[test]
    fn conv_and_swiglu_and_norm_grids_follow_the_element_count() {
        let conv = ConvShape {
            input: ptr(1),
            dynamic: ptr(2),
            base: ptr(3),
            output: ptr(4),
            rows: 3,
            width: 5120,
            group: 16,
            taps: 2,
            part: 1,
        };
        let (launches, _) = collect(|sink| plan_conv(&conv, &mut |l| sink(l)));
        assert_eq!(launches[0].grid, [60, 1, 1]);
        let bad = ConvShape { part: 2, ..conv };
        assert_eq!(
            collect(|sink| plan_conv(&bad, &mut |l| sink(l))).1,
            Err(PlanError::InvalidArgument)
        );
        let bad = ConvShape { width: 5121, ..conv };
        assert_eq!(
            collect(|sink| plan_conv(&bad, &mut |l| sink(l))).1,
            Err(PlanError::InvalidArgument)
        );
        let swiglu = SwigluShape {
            gate_up: ptr(1),
            output: ptr(2),
            rows: 7,
            intermediate: 17408,
        };
        let (launches, _) = collect(|sink| plan_swiglu(&swiglu, &mut |l| sink(l)));
        assert_eq!(launches[0].grid, [476, 1, 1]);
        let norm = RmsNormShape {
            input: ptr(1),
            weight: ptr(2),
            output: ptr(3),
            rows: 7,
            width: 5120,
            epsilon: f32::NAN,
        };
        assert_eq!(
            collect(|sink| plan_rms_norm(&norm, &mut |l| sink(l))).1,
            Err(PlanError::InvalidArgument)
        );
        let norm = RmsNormShape { epsilon: 1e-6, ..norm };
        let (launches, _) = collect(|sink| plan_rms_norm(&norm, &mut |l| sink(l)));
        assert_eq!((launches[0].grid, launches[0].block), ([7, 1, 1], [256, 1, 1]));
    }

    #[test]
    fn block_qk_runs_a_block_per_row_and_head() {
        let shape = BlockQkShape {
            query: ptr(1),
            kv: ptr(2),
            query_norm: ptr(3),
            key_norm: ptr(4),
            query_out: ptr(5),
            key_out: ptr(6),
            rows: 8,
            position: ptr(7),
            heads: 32,
            key_value_heads: 8,
            head_dim: 128,
            epsilon: 1e-6,
            theta: 1e7,
        };
        let (launches, result) = collect(|sink| plan_block_qk(&shape, &mut |l| sink(l)));
        assert_eq!(result, Ok(()));
        assert_eq!(launches[0].grid, [8, 40, 1]);
        assert_eq!(launches[0].args.words()[..9], [1, 2, 3, 4, 5, 6, 7, 32, 8]);
    }
}
