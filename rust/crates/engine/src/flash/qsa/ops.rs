//! The launches of the sparse attention's kernels (`QsaOps`): the same kernel, grid, block, shared bytes and argument
//! order as Java, with the geometry each kernel is written for checked where Java checks it. Each returns the
//! [`Launch`] and queues nothing, as the dense part's [`ops`](crate::flash::ops) do. A launch that has nothing to do
//! (no block completed) returns `None`.

use euhedral_inference_cuda::DevicePtr;

use crate::flash::FlashError;
use crate::flash::kernels::Qwen4Kernel;
use crate::flash::ops::{aligned, blocks, positive};
use crate::formats::shapes::{KernelArguments, Launch};

/// Tokens per indexer block.
pub const BLOCK_TOKENS: u32 = 4;
/// Values of an indexer key or query head.
pub const INDEX_WIDTH: u32 = 128;
/// Indexer query heads.
pub const INDEX_HEADS: u32 = 4;
/// Rows a scores launch covers per CTA row group.
const SCORE_ROWS_PER_CTA: u32 = 32;
/// Blocks a scores launch covers per CTA.
const SCORE_BLOCKS_PER_CTA: u32 = 64;
/// Floats of one (row, head, split) partial: 256 values, the running maximum and the running sum.
pub const PARTIAL_FLOATS: u32 = 258;
/// The most a position may be: the kernels carry positions in 24 bits.
pub const POSITION_LIMIT: u64 = 1 << 24;

/// Blocks completed by `rows` tokens appended at `start`.
pub fn completed_blocks(start: u32, rows: u32) -> u32 {
    ((u64::from(start) + u64::from(rows)) / u64::from(BLOCK_TOKENS) - u64::from(start / BLOCK_TOKENS)) as u32
}

/// Per-head RMS norm with the one weight of `width` values shared by every head, then RoPE on the first 64 values of
/// each head at position `position + row * position_step` (`QsaOps.headNormRope`). Strides are in elements; the input
/// and the output may be the same memory.
#[derive(Clone, Copy, Debug)]
pub struct HeadNormRope {
    pub input: DevicePtr,
    pub weight: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub heads: u32,
    pub width: u32,
    pub in_row_stride: u32,
    pub in_head_stride: u32,
    pub out_row_stride: u32,
    pub out_head_stride: u32,
    /// 64 or 0.
    pub rotary: u32,
    pub position: u32,
    pub position_step: u32,
    pub epsilon: f32,
    pub theta: f32,
}

impl HeadNormRope {
    pub fn launch(&self) -> Result<Launch, FlashError> {
        const OP: &str = "head_norm_rope";
        positive(OP, &[("rows", self.rows), ("heads", self.heads), ("width", self.width)])?;
        if !self.width.is_multiple_of(32) || self.width > 256 {
            return Err(FlashError::refused(
                OP,
                format!("a head is 32 to 256 values in steps of 32, got {}", self.width),
            ));
        }
        if self.rotary != 0 && self.rotary != 64 {
            return Err(FlashError::refused(
                OP,
                format!("the kernel rotates 64 values or none, got {}", self.rotary),
            ));
        }
        if self.rotary != 0 && self.width < 64 {
            return Err(FlashError::refused(
                OP,
                format!("rotary 64 needs a head of 64 values or more, got {}", self.width),
            ));
        }
        let last = u64::from(self.position) + u64::from(self.rows) * u64::from(self.position_step);
        if last >= POSITION_LIMIT {
            return Err(FlashError::refused(
                OP,
                format!("positions must stay below 2^24, these reach {last}"),
            ));
        }
        aligned(
            OP,
            4,
            &[("input", self.input), ("output", self.output), ("weight", self.weight)],
        )?;
        let units = u64::from(self.rows) * u64::from(self.heads);
        Ok(Qwen4Kernel::HeadNormRope.launch(
            [blocks(OP, units, 4)?, 1, 1],
            [128, 1, 1],
            KernelArguments::new()
                .pointer(self.input)
                .pointer(self.weight)
                .pointer(self.output)
                .int32(self.rows)
                .int32(self.heads)
                .int32(self.width)
                .int32(self.in_row_stride)
                .int32(self.in_head_stride)
                .int32(self.out_row_stride)
                .int32(self.out_head_stride)
                .int32(self.rotary)
                .int32(self.position)
                .int32(self.position_step)
                .float32(self.epsilon)
                .float32(self.theta),
        ))
    }
}

/// Pools the blocks a chunk of `rows` tokens at `start` completes: each the BF16 mean of its four raw keys (the chunk's
/// rows at `raw`, row stride `raw_stride`, or the `tail` of the previous chunks), written to `blocks[start / 4 ..]`
/// still to be normalized and rotated (`QsaOps.poolKeys`). `None` when the chunk completes none.
pub fn pool_keys(
    raw: DevicePtr,
    tail: DevicePtr,
    blocks_out: DevicePtr,
    rows: u32,
    start: u32,
    raw_stride: u32,
) -> Result<Option<Launch>, FlashError> {
    const OP: &str = "qsa_pool_keys";
    let count = completed_blocks(start, rows);
    if count == 0 {
        return Ok(None);
    }
    aligned(OP, 4, &[("raw", raw), ("tail", tail), ("blocks", blocks_out)])?;
    Ok(Some(
        Qwen4Kernel::QsaPoolKeys.launch(
            [blocks(OP, u64::from(count), 4)?, 1, 1],
            [128, 1, 1],
            KernelArguments::new()
                .pointer(raw)
                .pointer(tail)
                .pointer(blocks_out)
                .int32(count)
                .int32(start)
                .int32(raw_stride),
        ),
    ))
}

/// The raw keys of the incomplete trailing block after the chunk, written to `tail_out`, which is not `tail_in`
/// (`QsaOps.tail`).
pub fn tail(
    raw: DevicePtr,
    tail_in: DevicePtr,
    tail_out: DevicePtr,
    rows: u32,
    start: u32,
    raw_stride: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "qsa_tail";
    positive(OP, &[("rows", rows)])?;
    aligned(OP, 4, &[("raw", raw), ("tail in", tail_in), ("tail out", tail_out)])?;
    if tail_in == tail_out {
        return Err(FlashError::refused(
            OP,
            format!(
                "the tail a chunk leaves is written beside the one it reads, both are {:#x}",
                tail_in.get()
            ),
        ));
    }
    Ok(Qwen4Kernel::QsaTail.launch(
        [1, 1, 1],
        [128, 1, 1],
        KernelArguments::new()
            .pointer(raw)
            .pointer(tail_in)
            .pointer(tail_out)
            .int32(rows)
            .int32(start)
            .int32(raw_stride),
    ))
}

/// The scores of chunk rows `row_begin ..< row_begin + tile_rows` against the first `blocks_total` block keys, into
/// `scores[tile_rows][score_stride]` (`QsaOps.scores`).
#[derive(Clone, Copy, Debug)]
pub struct Scores {
    pub queries: DevicePtr,
    pub keys: DevicePtr,
    pub scores: DevicePtr,
    pub row_begin: u32,
    pub tile_rows: u32,
    pub start: u32,
    pub query_row_stride: u32,
    pub score_stride: u32,
    pub blocks_total: u32,
}

impl Scores {
    pub fn launch(&self) -> Result<Launch, FlashError> {
        const OP: &str = "qsa_scores";
        positive(OP, &[("tile rows", self.tile_rows), ("blocks", self.blocks_total)])?;
        if !self.query_row_stride.is_multiple_of(2) {
            return Err(FlashError::refused(
                OP,
                format!(
                    "queries are read in pairs, got a row stride of {}",
                    self.query_row_stride
                ),
            ));
        }
        aligned(OP, 4, &[("queries", self.queries), ("keys", self.keys)])?;
        Ok(Qwen4Kernel::QsaScores.launch(
            [
                blocks(OP, u64::from(self.blocks_total), u64::from(SCORE_BLOCKS_PER_CTA))?,
                blocks(OP, u64::from(self.tile_rows), u64::from(SCORE_ROWS_PER_CTA))?,
                1,
            ],
            [256, 1, 1],
            KernelArguments::new()
                .pointer(self.queries)
                .pointer(self.keys)
                .pointer(self.scores)
                .int32(self.row_begin)
                .int32(self.tile_rows)
                .int32(self.start)
                .int32(self.query_row_stride)
                .int32(self.score_stride)
                .int32(self.blocks_total),
        ))
    }
}

/// The selection of the blocks of chunk rows `row_begin ..< row_begin + tile_rows`: `ids[row][0 ..< counts[row]]`
/// ascending (`QsaOps.select`).
#[allow(
    clippy::too_many_arguments,
    reason = "the operands of one launch, as the kernel takes them"
)]
pub fn select(
    scores: DevicePtr,
    ids: DevicePtr,
    counts: DevicePtr,
    row_begin: u32,
    tile_rows: u32,
    start: u32,
    score_stride: u32,
    budget: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "qsa_select";
    positive(OP, &[("tile rows", tile_rows), ("budget", budget)])?;
    Ok(Qwen4Kernel::QsaSelect.launch(
        [tile_rows, 1, 1],
        [1024, 1, 1],
        KernelArguments::new()
            .pointer(scores)
            .pointer(ids)
            .pointer(counts)
            .int32(row_begin)
            .int32(start)
            .int32(score_stride)
            .int32(budget),
    ))
}

/// The sparse attention of `rows` query rows at `start ..` over their selected blocks (`ids` null: every block, the
/// dense causal prefix) and tail tokens (`QsaOps.attention`). With one split the finished rows go to `gated` (and
/// `core` unless null); with more, partials go to `partial` and [`merge`] finishes them.
#[derive(Clone, Copy, Debug)]
pub struct Attention {
    pub queries: DevicePtr,
    pub gate: DevicePtr,
    pub key_pages: DevicePtr,
    pub value_pages: DevicePtr,
    pub ids: DevicePtr,
    pub counts: DevicePtr,
    pub partial: DevicePtr,
    pub core: DevicePtr,
    pub gated: DevicePtr,
    pub rows: u32,
    pub query_heads: u32,
    pub key_heads: u32,
    pub start: u32,
    pub splits: u32,
    pub budget: u32,
    pub query_row_stride: u32,
    pub query_head_stride: u32,
    pub gate_row_stride: u32,
    pub gate_head_stride: u32,
}

impl Attention {
    pub fn launch(&self) -> Result<Launch, FlashError> {
        const OP: &str = "qsa_attention";
        positive(
            OP,
            &[
                ("rows", self.rows),
                ("query heads", self.query_heads),
                ("key heads", self.key_heads),
                ("splits", self.splits),
            ],
        )?;
        if !self.query_heads.is_multiple_of(self.key_heads) || self.query_heads / self.key_heads > 16 {
            return Err(FlashError::refused(
                OP,
                format!(
                    "a KV head's group of query heads has at most 16 heads, got {} over {}",
                    self.query_heads, self.key_heads
                ),
            ));
        }
        if self.splits > 1 && self.partial.is_null() {
            return Err(FlashError::refused(
                OP,
                format!("{} splits need a partial buffer", self.splits),
            ));
        }
        if self.ids.is_null() != self.counts.is_null() {
            return Err(FlashError::refused(OP, "ids and counts go together".to_owned()));
        }
        let units = u64::from(self.rows) * u64::from(self.key_heads) * u64::from(self.splits);
        let grid = u32::try_from(units).map_err(|_| FlashError::refused(OP, format!("{units} units exceed a grid")))?;
        Ok(Qwen4Kernel::QsaAttention.launch_shared(
            [grid, 1, 1],
            [96, 1, 1],
            Qwen4Kernel::QsaAttention.shared_bytes(),
            KernelArguments::new()
                .pointer(self.queries)
                .pointer(self.gate)
                .pointer(self.key_pages)
                .pointer(self.value_pages)
                .pointer(self.ids)
                .pointer(self.counts)
                .pointer(self.partial)
                .pointer(self.core)
                .pointer(self.gated)
                .int32(self.rows)
                .int32(self.query_heads)
                .int32(self.key_heads)
                .int32(self.start)
                .int32(self.splits)
                .int32(self.budget)
                .int32(self.query_row_stride)
                .int32(self.query_head_stride)
                .int32(self.gate_row_stride)
                .int32(self.gate_head_stride),
        ))
    }
}

/// The combination of the key splits of [`Attention`], writing `core` (unless null) and `gated` (`QsaOps.merge`).
#[allow(
    clippy::too_many_arguments,
    reason = "the operands of one launch, as the kernel takes them"
)]
pub fn merge(
    partial: DevicePtr,
    gate: DevicePtr,
    core: DevicePtr,
    gated: DevicePtr,
    rows: u32,
    query_heads: u32,
    splits: u32,
    gate_row_stride: u32,
    gate_head_stride: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "qsa_merge";
    positive(OP, &[("rows", rows), ("query heads", query_heads), ("splits", splits)])?;
    let units = u64::from(rows) * u64::from(query_heads);
    Ok(Qwen4Kernel::QsaMerge.launch(
        [blocks(OP, units, 4)?, 1, 1],
        [128, 1, 1],
        KernelArguments::new()
            .pointer(partial)
            .pointer(gate)
            .pointer(core)
            .pointer(gated)
            .int32(rows)
            .int32(query_heads)
            .int32(splits)
            .int32(gate_row_stride)
            .int32(gate_head_stride),
    ))
}

/// The quantization of `rows` rows of BF16 keys and values (`key_heads` heads of 256) into the cache pages at
/// positions `start ..`; `key_pages` and `value_pages` are the device page tables of the sequence's cache
/// (`QsaOps.kvAppend`).
#[allow(
    clippy::too_many_arguments,
    reason = "the operands of one launch, as the kernel takes them"
)]
pub fn kv_append(
    keys: DevicePtr,
    values: DevicePtr,
    key_pages: DevicePtr,
    value_pages: DevicePtr,
    rows: u32,
    key_heads: u32,
    key_row_stride: u32,
    value_row_stride: u32,
    start: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "qsa_kv_append";
    positive(OP, &[("rows", rows), ("key heads", key_heads)])?;
    aligned(OP, 16, &[("keys", keys), ("values", values)])?;
    let units = u64::from(rows) * u64::from(key_heads);
    Ok(Qwen4Kernel::QsaKvAppend.launch(
        [blocks(OP, units, 4)?, 1, 1],
        [128, 1, 1],
        KernelArguments::new()
            .pointer(keys)
            .pointer(values)
            .pointer(key_pages)
            .pointer(value_pages)
            .int32(rows)
            .int32(key_heads)
            .int32(key_row_stride)
            .int32(value_row_stride)
            .int32(start),
    ))
}

#[cfg(test)]
mod tests;
