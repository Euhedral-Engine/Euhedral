//! The sparse attention block of a Flash-Next layer (`QsaLayer.run`): the query/gate, key and value projections and the
//! indexer projection, the norms and RoPE, the keys and values into the cache pages, the pooled indexer keys, the
//! selection of each row's blocks, the attention over them and the incomplete block's tail with the sigmoid gate, and
//! the output projection.
//!
//! The indexer scores blocks of four tokens; a row keeps its top `min(budget, blocks it sees)` blocks, the lowest ids
//! among equal scores, and attends over their tokens plus the tail (`docs/flash-next/FLASH_NEXT_QSA.md`). Any chunking
//! of a sequence gives the same result: a block key is computed once, when its block completes, and the raw keys of the
//! incomplete block are carried in the [`QsaState`](state::QsaState).
//!
//! [`QsaBody`] holds the shape, the weights and the addresses of the layer's indexer buffers; where a call's rows lie,
//! the cache's page tables and the tail slot come with the call ([`ChunkAt`]), because the sequence's owner moves them
//! between chunks.

pub mod ops;
pub mod state;

use euhedral_inference_cuda::DevicePtr;
use euhedral_inference_edrl::{AttentionConfig, QsaConfig};

use self::ops::{
    Attention, BLOCK_TOKENS, HeadNormRope, INDEX_HEADS, INDEX_WIDTH, PARTIAL_FLOATS, POSITION_LIMIT, Scores,
    completed_blocks,
};
use self::state::{QsaBuffers, QsaState, max_blocks};
use super::block::{BlockBody, BlockCall, ChunkAt};
use super::buffers::{ALIGNMENT, Carver, bf16_bytes};
use super::linear::{Nvfp4Device, Nvfp4Weight};
use super::sequence::{Op, OpList};
use super::{FlashError, Weight};

const OP: &str = "qsa layer";
/// The key and value head width, and the values RoPE rotates.
const HEAD_DIM: u32 = 256;
const ROTARY_DIM: u32 = 64;
/// Units (row, KV head, key split) a decode-sized launch aims for.
const TARGET_UNITS: u32 = 512;
/// Keys a split holds at least.
const MIN_SPLIT_KEYS: u32 = 32;
const MAX_SPLITS: u32 = 64;
/// The default size of the scores scratch: rows are scored in tiles that fit it.
pub const DEFAULT_SCORE_BYTES: u64 = 16 << 20;

/// The sizes of one sparse-attention layer, checked against what the kernels serve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QsaShape {
    hidden: u32,
    query_heads: u32,
    key_heads: u32,
    rope_theta: f32,
    epsilon: f32,
    budget_tokens: u32,
    max_tokens: u32,
    /// The FP32 scores scratch in bytes, 256-aligned, at least one row of a sequence of `max_tokens`.
    score_bytes: u64,
}

impl QsaShape {
    /// The layer of `attention` and `qsa` over `hidden` values for sequences of up to `max_tokens` positions, with
    /// the norm's `epsilon`. `score_bytes` bounds the scores scratch (a tile of rows times the blocks they see); it is
    /// raised to hold one row of a sequence of `max_tokens`, as Java does. Refuses what the kernels are not written
    /// for.
    pub fn new(
        hidden: u32,
        attention: &AttentionConfig,
        qsa: &QsaConfig,
        epsilon: f64,
        max_tokens: u32,
        score_bytes: u64,
    ) -> Result<Self, FlashError> {
        if qsa.compress_ratio != BLOCK_TOKENS || qsa.num_heads != INDEX_HEADS || qsa.head_dim != INDEX_WIDTH {
            return Err(FlashError::refused(
                OP,
                format!(
                    "the kernels are written for {INDEX_HEADS} indexer heads of {INDEX_WIDTH} over blocks of {BLOCK_TOKENS}, got {} heads of {} over blocks of {}",
                    qsa.num_heads, qsa.head_dim, qsa.compress_ratio
                ),
            ));
        }
        if qsa.kv_heads != 1 {
            return Err(FlashError::refused(
                OP,
                format!(
                    "the indexer pools one raw key per token, got {} key heads",
                    qsa.kv_heads
                ),
            ));
        }
        if qsa.indexer_budget == 0 || !qsa.indexer_budget.is_multiple_of(BLOCK_TOKENS) {
            return Err(FlashError::refused(
                OP,
                format!(
                    "the budget is a positive number of whole blocks of {BLOCK_TOKENS} tokens, got {}",
                    qsa.indexer_budget
                ),
            ));
        }
        let rotary = (f64::from(attention.head_dim) * attention.partial_rotary_factor) as u32;
        if attention.head_dim != HEAD_DIM || rotary != ROTARY_DIM {
            return Err(FlashError::refused(
                OP,
                format!(
                    "the kernels are written for heads of {HEAD_DIM} with {ROTARY_DIM} rotary values, got {} with {rotary}",
                    attention.head_dim
                ),
            ));
        }
        if attention.num_kv_heads == 0
            || !attention.num_heads.is_multiple_of(attention.num_kv_heads)
            || attention.num_heads / attention.num_kv_heads > 16
        {
            return Err(FlashError::refused(
                OP,
                format!(
                    "a group of query heads has at most 16 heads, got {} query heads over {} key heads",
                    attention.num_heads, attention.num_kv_heads
                ),
            ));
        }
        if max_tokens == 0 || u64::from(max_tokens) >= POSITION_LIMIT {
            return Err(FlashError::refused(
                OP,
                format!("max tokens is 1 to 2^24 - 1, got {max_tokens}"),
            ));
        }
        let widest = u64::from(attention.num_heads) * u64::from(HEAD_DIM) * 2;
        if hidden == 0 || !hidden.is_multiple_of(8) || widest > u64::from(u32::MAX) / 2 || !epsilon.is_finite() {
            return Err(FlashError::refused(
                OP,
                format!("hidden {hidden} (a positive multiple of 8), query width {widest}, epsilon {epsilon}"),
            ));
        }
        let one_row = u64::from(score_stride(max_blocks(max_tokens))) * 4;
        Ok(Self {
            hidden,
            query_heads: attention.num_heads,
            key_heads: attention.num_kv_heads,
            rope_theta: attention.rope_theta as f32,
            epsilon: epsilon as f32,
            budget_tokens: qsa.indexer_budget,
            max_tokens,
            score_bytes: score_bytes.max(one_row).next_multiple_of(ALIGNMENT),
        })
    }

    /// The blocks a row keeps at most.
    pub fn budget_blocks(&self) -> u32 {
        self.budget_tokens / BLOCK_TOKENS
    }

    /// The positions a sequence may hold.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// The bytes of the scores scratch.
    pub fn score_bytes(&self) -> u64 {
        self.score_bytes
    }

    fn query_width(&self) -> u32 {
        self.query_heads * HEAD_DIM
    }

    fn key_value_width(&self) -> u32 {
        self.key_heads * HEAD_DIM
    }

    fn index_projection_width(&self) -> u32 {
        (INDEX_HEADS + 1) * INDEX_WIDTH
    }

    /// Key splits of a launch of `rows` rows whose longest key list has `max_keys` keys.
    fn splits_for(&self, rows: u32, max_keys: u32) -> u32 {
        (max_keys / MIN_SPLIT_KEYS).clamp(1, self.splits_cap(rows))
    }

    fn splits_cap(&self, rows: u32) -> u32 {
        let units = u64::from(rows) * u64::from(self.key_heads);
        (u64::from(TARGET_UNITS) / units).clamp(1, u64::from(MAX_SPLITS)) as u32
    }

    /// The bytes of split partials a call of up to `rows` rows can need: the largest over the sizes that split.
    fn partial_bytes(&self, rows: u32) -> u64 {
        (1..=rows)
            .take_while(|rows| self.splits_cap(*rows) > 1)
            .map(|rows| {
                u64::from(rows)
                    * u64::from(self.query_heads)
                    * u64::from(self.splits_cap(rows))
                    * u64::from(PARTIAL_FLOATS)
                    * 4
            })
            .max()
            .unwrap_or(0)
    }

    fn carve(&self, carver: &mut Carver, rows: u32) -> QsaScratch {
        QsaScratch {
            query: carver.take(bf16_bytes(rows, 2 * self.query_width())),
            key: carver.take(bf16_bytes(rows, self.key_value_width())),
            key_normed: carver.take(bf16_bytes(rows, self.key_value_width())),
            value: carver.take(bf16_bytes(rows, self.key_value_width())),
            index: carver.take(bf16_bytes(rows, self.index_projection_width())),
            scores: carver.take(self.score_bytes),
            ids: carver.take(u64::from(rows) * u64::from(self.budget_blocks()) * 4),
            counts: carver.take(u64::from(rows) * 4),
            partial: carver.take(self.partial_bytes(rows)),
            gated: carver.take(bf16_bytes(rows, self.query_width())),
        }
    }

    /// The bytes of the scratch of `rows` rows.
    pub fn scratch_bytes(&self, rows: u32) -> u64 {
        let mut carver = Carver::new(DevicePtr::NULL);
        self.carve(&mut carver, rows);
        carver.bytes()
    }

    /// The linears of the layer as `(in_features, out_features)`: the query and gate, the key and the value (alike)
    /// and the output.
    fn linear_shapes(&self) -> [(u32, u32); 3] {
        [
            (self.hidden, 2 * self.query_width()),
            (self.hidden, self.key_value_width()),
            (self.query_width(), self.hidden),
        ]
    }

    /// Whether any row of a chunk of `rows` rows at `start` chooses among more blocks than the budget: only then do
    /// the scores and the selection exist.
    fn selects(&self, rows: u32, start: u32) -> bool {
        u64::from(start) + u64::from(rows) > u64::from(self.budget_tokens + BLOCK_TOKENS - 1)
    }

    /// The rows scored at a time: as many as the scores scratch holds for `selecting_rows` rows seeing the blocks of
    /// a sequence that ends at position `end`.
    fn score_tile_rows(&self, selecting_rows: u32, end: u32) -> u32 {
        let stride = u64::from(score_stride(end / BLOCK_TOKENS));
        (self.score_bytes / 4 / stride).clamp(1, u64::from(selecting_rows)) as u32
    }
}

/// The scores row stride in floats: the blocks rounded up to a multiple of 32.
fn score_stride(blocks: u32) -> u32 {
    blocks.next_multiple_of(32)
}

#[derive(Clone, Copy, Debug)]
struct QsaScratch {
    /// `[rows][queryHeads][query | gate]`.
    query: DevicePtr,
    key: DevicePtr,
    key_normed: DevicePtr,
    value: DevicePtr,
    /// `[rows][4 indexer heads | raw key]`.
    index: DevicePtr,
    scores: DevicePtr,
    ids: DevicePtr,
    counts: DevicePtr,
    partial: DevicePtr,
    gated: DevicePtr,
}

/// The weights of one sparse-attention layer: the NVFP4 projections, and the BF16 indexer projection and norms (the
/// norm weights are shared by all heads; `index` is `[(heads + 1) * 128][hidden]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QsaWeights {
    pub query: Nvfp4Weight,
    pub key: Nvfp4Weight,
    pub value: Nvfp4Weight,
    pub output: Nvfp4Weight,
    pub query_norm: Weight,
    pub key_norm: Weight,
    pub index: Weight,
    pub index_query_norm: Weight,
    pub index_key_norm: Weight,
}

/// The sparse attention block as the body of a layer.
#[derive(Clone, Debug)]
pub struct QsaBody {
    shape: QsaShape,
    weights: QsaWeights,
    buffers: QsaBuffers,
    device: Nvfp4Device,
}

impl QsaBody {
    /// The body of a layer over `state`, which must hold at least the tokens the shape names: the block keys and the
    /// norm of a pooled block are written within the state's buffers.
    pub fn new(
        shape: QsaShape,
        weights: QsaWeights,
        state: &QsaState,
        device: Nvfp4Device,
    ) -> Result<Self, FlashError> {
        if state.max_tokens() < shape.max_tokens {
            return Err(FlashError::refused(
                OP,
                format!(
                    "the state holds {} tokens, the layer's shape {}",
                    state.max_tokens(),
                    shape.max_tokens
                ),
            ));
        }
        Ok(Self {
            shape,
            weights,
            buffers: state.buffers(),
            device,
        })
    }

    /// Refuses a chunk the layer's state cannot hold or has no cache for.
    fn check_chunk(&self, rows: u32, chunk: &ChunkAt) -> Result<(), FlashError> {
        let end = u64::from(chunk.start) + u64::from(rows);
        if end > u64::from(self.shape.max_tokens) {
            return Err(FlashError::refused(
                OP,
                format!(
                    "{rows} rows at {} would take the sequence to {end} tokens, past the {} the state holds",
                    chunk.start, self.shape.max_tokens
                ),
            ));
        }
        if chunk.key_table.is_null() || chunk.value_table.is_null() {
            return Err(FlashError::refused(
                OP,
                "the chunk names no K and V page tables of the layer's cache".to_owned(),
            ));
        }
        if chunk.tail_slot > 1 {
            return Err(FlashError::refused(
                OP,
                format!("the raw tail has two slots, the chunk reads slot {}", chunk.tail_slot),
            ));
        }
        Ok(())
    }
}

impl BlockBody for QsaBody {
    fn scratch_bytes(&self, rows: u32) -> u64 {
        self.shape.scratch_bytes(rows)
    }

    fn expansion_bytes(&self, rows: u32) -> u64 {
        self.shape
            .linear_shapes()
            .iter()
            .map(|(k, n)| self.device.expansion_bytes(rows, *k, *n))
            .max()
            .unwrap_or(0)
    }

    fn ops(&self, call: &BlockCall) -> Result<OpList<'static>, FlashError> {
        let shape = &self.shape;
        let (rows, start, chunk) = (call.rows, call.chunk.start, call.chunk);
        self.check_chunk(rows, &chunk)?;
        let scratch = shape.carve(&mut Carver::new(call.scratch), rows);
        let mut list = OpList::new();
        self.project(call, &scratch, &mut list)?;
        self.norm_and_rope(rows, start, &scratch, &mut list)?;
        list.push(Op::Launch(ops::kv_append(
            scratch.key_normed,
            scratch.value,
            chunk.key_table,
            chunk.value_table,
            rows,
            shape.key_heads,
            shape.key_value_width(),
            shape.key_value_width(),
            start,
        )?));
        self.index_keys(rows, &chunk, &scratch, &mut list)?;
        let selecting = self.select(rows, start, &scratch, &mut list)?;
        self.attend(rows, &chunk, &scratch, selecting, &mut list)?;
        list.push(self.device.linear(
            scratch.gated,
            self.weights.output,
            call.output,
            rows,
            shape.query_width(),
            shape.hidden,
            call.expansion,
        )?);
        Ok(list)
    }
}

impl QsaBody {
    /// The query/gate, key and value projections and the indexer projection of `rows` rows.
    fn project(&self, call: &BlockCall, scratch: &QsaScratch, list: &mut OpList<'static>) -> Result<(), FlashError> {
        let (shape, weights, rows) = (&self.shape, &self.weights, call.rows);
        let mut linear = |weight: Nvfp4Weight, output, n| -> Result<(), FlashError> {
            list.push(
                self.device
                    .linear(call.input, weight, output, rows, shape.hidden, n, call.expansion)?,
            );
            Ok(())
        };
        linear(weights.query, scratch.query, 2 * shape.query_width())?;
        linear(weights.key, scratch.key, shape.key_value_width())?;
        linear(weights.value, scratch.value, shape.key_value_width())?;
        list.push(Op::Launch(super::ops::linear_bf16(
            call.input,
            weights.index.address,
            scratch.index,
            rows,
            shape.hidden,
            shape.index_projection_width(),
            call.exact,
        )?));
        Ok(())
    }

    /// q_norm and RoPE in place on the query half of every head's `[q | gate]` (the gate stays where it is), k_norm and
    /// RoPE into `key_normed`, the indexer's q_layernorm and RoPE in place on its queries.
    fn norm_and_rope(
        &self,
        rows: u32,
        start: u32,
        scratch: &QsaScratch,
        list: &mut OpList<'static>,
    ) -> Result<(), FlashError> {
        let (shape, weights) = (&self.shape, &self.weights);
        let query_row = 2 * shape.query_width();
        let query_head = 2 * HEAD_DIM;
        let kv_width = shape.key_value_width();
        let index_width = shape.index_projection_width();
        let norm = |input, weight: Weight, output, heads, width, strides: [u32; 4]| HeadNormRope {
            input,
            weight: weight.address,
            output,
            rows,
            heads,
            width,
            in_row_stride: strides[0],
            in_head_stride: strides[1],
            out_row_stride: strides[2],
            out_head_stride: strides[3],
            rotary: ROTARY_DIM,
            position: start,
            position_step: 1,
            epsilon: shape.epsilon,
            theta: shape.rope_theta,
        };
        for launch in [
            norm(
                scratch.query,
                weights.query_norm,
                scratch.query,
                shape.query_heads,
                HEAD_DIM,
                [query_row, query_head, query_row, query_head],
            ),
            norm(
                scratch.key,
                weights.key_norm,
                scratch.key_normed,
                shape.key_heads,
                HEAD_DIM,
                [kv_width, HEAD_DIM, kv_width, HEAD_DIM],
            ),
            norm(
                scratch.index,
                weights.index_query_norm,
                scratch.index,
                INDEX_HEADS,
                INDEX_WIDTH,
                [index_width, INDEX_WIDTH, index_width, INDEX_WIDTH],
            ),
        ] {
            list.push(Op::Launch(launch.launch()?));
        }
        Ok(())
    }

    /// Pools the blocks the chunk completes (the raw key is the last 128 values of an indexer row), normalizes and
    /// rotates them into the block keys, and leaves the raw keys of the incomplete block as the new tail.
    fn index_keys(
        &self,
        rows: u32,
        chunk: &ChunkAt,
        scratch: &QsaScratch,
        list: &mut OpList<'static>,
    ) -> Result<(), FlashError> {
        let (shape, buffers) = (&self.shape, &self.buffers);
        let start = chunk.start;
        let index_width = shape.index_projection_width();
        let raw = scratch
            .index
            .offset(u64::from(INDEX_HEADS) * u64::from(INDEX_WIDTH) * 2);
        let tail_in = buffers
            .tails
            .offset(u64::from(chunk.tail_slot) * QsaBuffers::TAIL_BYTES);
        let tail_out = buffers
            .tails
            .offset(u64::from(chunk.tail_slot ^ 1) * QsaBuffers::TAIL_BYTES);
        if let Some(pool) = ops::pool_keys(raw, tail_in, buffers.block_keys, rows, start, index_width)? {
            list.push(Op::Launch(pool));
            let completed = completed_blocks(start, rows);
            let first = start / BLOCK_TOKENS;
            let fresh = buffers.block_keys.offset(u64::from(first) * u64::from(INDEX_WIDTH) * 2);
            list.push(Op::Launch(
                HeadNormRope {
                    input: fresh,
                    weight: self.weights.index_key_norm.address,
                    output: fresh,
                    rows: completed,
                    heads: 1,
                    width: INDEX_WIDTH,
                    in_row_stride: INDEX_WIDTH,
                    in_head_stride: INDEX_WIDTH,
                    out_row_stride: INDEX_WIDTH,
                    out_head_stride: INDEX_WIDTH,
                    rotary: ROTARY_DIM,
                    position: first * BLOCK_TOKENS,
                    position_step: BLOCK_TOKENS,
                    epsilon: shape.epsilon,
                    theta: shape.rope_theta,
                }
                .launch()?,
            ));
        }
        list.push(Op::Launch(ops::tail(raw, tail_in, tail_out, rows, start, index_width)?));
        Ok(())
    }

    /// Scores and selects the blocks of every row. A row keeps every block while it sees at most the budget of them, so
    /// the rows before the first one that chooses just list their blocks. Returns whether `ids` and `counts` were
    /// written.
    fn select(
        &self,
        rows: u32,
        start: u32,
        scratch: &QsaScratch,
        list: &mut OpList<'static>,
    ) -> Result<bool, FlashError> {
        let shape = &self.shape;
        if !shape.selects(rows, start) {
            return Ok(false);
        }
        let budget = shape.budget_blocks();
        let first_selecting = (shape.budget_tokens + BLOCK_TOKENS - 1).saturating_sub(start);
        if first_selecting > 0 {
            list.push(Op::Launch(ops::select(
                scratch.scores,
                scratch.ids,
                scratch.counts,
                0,
                first_selecting,
                start,
                0,
                budget,
            )?));
        }
        let end = start + rows;
        let stride = score_stride(end / BLOCK_TOKENS);
        let tile = shape.score_tile_rows(rows - first_selecting, end);
        for begin in (first_selecting..rows).step_by(tile as usize) {
            let count = tile.min(rows - begin);
            list.push(Op::Launch(
                Scores {
                    queries: scratch.index,
                    keys: self.buffers.block_keys,
                    scores: scratch.scores,
                    row_begin: begin,
                    tile_rows: count,
                    start,
                    query_row_stride: shape.index_projection_width(),
                    score_stride: stride,
                    blocks_total: (start + begin + count) / BLOCK_TOKENS,
                }
                .launch()?,
            ));
            list.push(Op::Launch(ops::select(
                scratch.scores,
                scratch.ids,
                scratch.counts,
                begin,
                count,
                start,
                stride,
                budget,
            )?));
        }
        Ok(true)
    }

    /// The attention of every row over its selected blocks and tail, with the sigmoid gate, into `gated`.
    fn attend(
        &self,
        rows: u32,
        chunk: &ChunkAt,
        scratch: &QsaScratch,
        selecting: bool,
        list: &mut OpList<'static>,
    ) -> Result<(), FlashError> {
        let shape = &self.shape;
        let start = chunk.start;
        let query_row = 2 * shape.query_width();
        let query_head = 2 * HEAD_DIM;
        let max_keys = (shape.budget_tokens + BLOCK_TOKENS - 1).min(start + rows);
        let splits = shape.splits_for(rows, max_keys);
        let gate = scratch.query.offset(u64::from(HEAD_DIM) * 2);
        let partial = if splits > 1 { scratch.partial } else { DevicePtr::NULL };
        let (ids, counts) = if selecting {
            (scratch.ids, scratch.counts)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        list.push(Op::Launch(
            Attention {
                queries: scratch.query,
                gate,
                key_pages: chunk.key_table,
                value_pages: chunk.value_table,
                ids,
                counts,
                partial,
                core: DevicePtr::NULL,
                gated: scratch.gated,
                rows,
                query_heads: shape.query_heads,
                key_heads: shape.key_heads,
                start,
                splits,
                budget: shape.budget_blocks(),
                query_row_stride: query_row,
                query_head_stride: query_head,
                gate_row_stride: query_row,
                gate_head_stride: query_head,
            }
            .launch()?,
        ));
        if splits > 1 {
            list.push(Op::Launch(ops::merge(
                scratch.partial,
                gate,
                DevicePtr::NULL,
                scratch.gated,
                rows,
                shape.query_heads,
                splits,
                query_row,
                query_head,
            )?));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
