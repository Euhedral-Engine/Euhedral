//! The per-layer n-gram embedding (`Ple`, with the `PLEIDS`, `PLEGATHER` and `PLE` stages of `Stages`).
//!
//! Three kinds of stage make one quantum's embedding:
//!
//! - [`PleIdsStage`] (host) hashes the quantum's tokens into row ids, over the context the sequence carried;
//! - [`PleGatherStage`] (host, one per part, up to [`PARTS`]) reads its part's records from the mapped table into a
//!   pinned loan of its own and publishes it;
//! - [`PleApply`] (device) uploads the parts once all have arrived, expands the records, projects, gates and convolves
//!   them, and adds the result into the residual streams.
//!
//! What the host stages share with the device stage travels in the quantum ([`PleRows`]), which they reach through
//! [`Quantum::ple`]; no stage holds a lock or waits for another: the graph's edges order them.

mod ngram;

use std::ops::Range;
use std::sync::{Arc, OnceLock};

use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{DevicePtr, KernelOps, PinnedPool, PooledPinned};

pub use self::ngram::{NgramIds, NgramRows, PARTS, PartSlots, PublishError, ShardBytes, part_count, part_range};
use super::buffers::{Carver, FlashRegions, ScratchSizes, StreamShape, bf16_bytes};
use super::linear::{Nvfp4Device, Nvfp4Weight};
use super::sequence::{Op, OpList, queue_all};
use super::{FlashError, Weight, ops};
use crate::stages::{Launch, Quantum, fatal};

/// The ids of a quantum's rows and the context after them.
#[derive(Debug)]
struct Hashed {
    ids: Vec<u64>,
    next_context: Box<[i32]>,
}

/// One part of the gather: the pinned records of its rows. The rows are those of [`part_range`].
#[derive(Debug)]
pub struct GatheredPart {
    loan: PooledPinned,
}

/// What the stages of one quantum's n-gram embedding hand to one another. Built with the context the sequence carried
/// into the quantum; the sequence takes [`PleRows::next_context`] once the quantum has committed, so an abandoned
/// quantum leaves the context as it was.
#[derive(Debug)]
pub struct PleRows {
    context: Box<[i32]>,
    hashed: OnceLock<Hashed>,
    parts: PartSlots<GatheredPart>,
}

impl PleRows {
    /// The hand-over for a quantum whose sequence carried `context`.
    pub fn new(context: Box<[i32]>) -> Self {
        Self {
            context,
            hashed: OnceLock::new(),
            parts: PartSlots::new(PARTS),
        }
    }

    /// The context after the quantum's tokens, once the ids stage has run.
    pub fn next_context(&self) -> Option<&[i32]> {
        self.hashed.get().map(|hashed| &*hashed.next_context)
    }

    fn ids(&self) -> Option<&[u64]> {
        self.hashed.get().map(|hashed| hashed.ids.as_slice())
    }
}

/// The ids stage: the row ids of the quantum's tokens.
#[derive(Clone, Debug)]
pub struct PleIdsStage {
    ids: Arc<NgramIds>,
}

impl PleIdsStage {
    pub fn new(ids: Arc<NgramIds>) -> Self {
        Self { ids }
    }

    /// Hashes `tokens`, the tokens of a quantum of `quantum_rows` rows, into `rows`. Refuses a token count that is not
    /// the rows and ids that were computed already.
    pub fn hash(&self, rows: &PleRows, tokens: &[u32], quantum_rows: usize) -> Result<(), StageError> {
        if tokens.len() != quantum_rows {
            return Err(fatal(format!(
                "a quantum of {quantum_rows} rows holds {} tokens",
                tokens.len()
            )));
        }
        let (ids, next_context) = self.ids.compute(&rows.context, tokens).map_err(fatal)?;
        rows.hashed
            .set(Hashed { ids, next_context })
            .map_err(|_| fatal("the n-gram ids of the quantum were computed twice"))
    }
}

/// The hand-over of `quantum`, or the failure of a quantum built without one.
fn rows_of(quantum: &Quantum) -> Result<&Arc<PleRows>, StageError> {
    quantum
        .ple
        .as_ref()
        .ok_or_else(|| fatal("the quantum carries no n-gram embedding rows"))
}

impl Stage<Quantum> for PleIdsStage {
    fn run(&self, quantum: &Quantum, _cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        self.hash(rows_of(quantum)?, &quantum.tokens, quantum.rows)?;
        Ok(Submitted::done())
    }
}

/// One part of the gather.
#[derive(Clone, Debug)]
pub struct PleGatherStage {
    part: usize,
    table: Arc<NgramRows>,
    pool: Arc<PinnedPool>,
}

impl PleGatherStage {
    /// Part `part` of the gather from `table`, into loans of `pool`. Refuses a part past [`PARTS`].
    pub fn new(part: usize, table: Arc<NgramRows>, pool: Arc<PinnedPool>) -> Result<Self, FlashError> {
        if part >= PARTS {
            return Err(FlashError::refused("ple gather", format!("part {part} of {PARTS}")));
        }
        Ok(Self { part, table, pool })
    }
}

impl Stage<Quantum> for PleGatherStage {
    fn run(&self, quantum: &Quantum, _cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        self.gather(rows_of(quantum)?)?;
        Ok(Submitted::done())
    }
}

impl PleGatherStage {
    /// Reads this part's records of `rows`' ids and publishes them. Refuses a run before the ids and a part published
    /// twice.
    pub fn gather(&self, rows: &PleRows) -> Result<(), StageError> {
        let ids = rows
            .ids()
            .ok_or_else(|| fatal(format!("part {} ran before the n-gram ids", self.part)))?;
        let range = part_range(ids.len(), self.part);
        if range.is_empty() {
            return Ok(());
        }
        let bytes = range.len() * self.table.record_bytes() as usize;
        let mut loan = self.pool.acquire(bytes).map_err(fatal)?;
        self.table
            .gather(ids, range.clone(), &mut loan.buffer_mut().as_mut_slice()[..bytes])
            .map_err(fatal)?;
        rows.parts
            .publish(self.part, GatheredPart { loan })
            .map_err(|(why, _)| fatal(format!("part {} of the gather: {why:?}", self.part)))
    }
}

/// The sizes of the embedding, checked against the table and the streams.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PleShape {
    hidden: u32,
    streams: u32,
    heads: u32,
    row_width: u32,
    record_bytes: u32,
    taps: u32,
    dilation: u32,
    epsilon: f32,
}

impl PleShape {
    /// The embedding of the rows `ids` addresses in `table`, convolved over `taps` rows at `dilation`. The heads, the
    /// row width and the record size are those of the id function and the store, so the two cannot disagree (Java's
    /// constructor refuses "the id function and the store disagree"). Refuses a width that is not the heads' rows
    /// concatenated.
    #[allow(clippy::too_many_arguments, reason = "the sizes Java's constructor takes")]
    pub fn new(
        hidden: u32,
        streams: u32,
        embed_dim: u32,
        ids: &NgramIds,
        table: &NgramRows,
        taps: u32,
        dilation: u32,
        epsilon: f32,
    ) -> Result<Self, FlashError> {
        let heads =
            u32::try_from(ids.heads()).map_err(|_| FlashError::refused("ple", format!("{} heads", ids.heads())))?;
        let (row_width, record_bytes) = (table.row_width(), table.record_bytes());
        const OP: &str = "ple";
        if u64::from(heads) * u64::from(row_width) != u64::from(embed_dim) || heads == 0 {
            return Err(FlashError::refused(
                OP,
                format!("{heads} heads of {row_width} values do not concatenate to the embedding width {embed_dim}"),
            ));
        }
        let state_ok = hidden > 0 && streams > 0 && u64::from(hidden) * u64::from(streams) <= u64::from(u32::MAX);
        let history = u64::from(taps.saturating_sub(1)) * u64::from(dilation);
        if !state_ok
            || taps < 2
            || dilation == 0
            || history > 32
            || !row_width.is_multiple_of(16)
            || !epsilon.is_finite()
        {
            return Err(FlashError::refused(
                OP,
                format!(
                    "hidden {hidden}, streams {streams}, taps {taps}, dilation {dilation} (history {history} of at most 32), rows of {row_width} (a multiple of 16), epsilon {epsilon}"
                ),
            ));
        }
        Ok(Self {
            hidden,
            streams,
            heads,
            row_width,
            record_bytes,
            taps,
            dilation,
            epsilon,
        })
    }

    fn state_width(&self) -> u32 {
        self.hidden * self.streams
    }

    fn embed_dim(&self) -> u32 {
        self.heads * self.row_width
    }

    /// The rows of convolution history a sequence keeps.
    pub fn history_rows(&self) -> u32 {
        (self.taps - 1) * self.dilation
    }

    fn carve(&self, carver: &mut Carver, rows: u32) -> PleScratch {
        let width = self.state_width();
        PleScratch {
            records: carver.take(u64::from(rows) * u64::from(self.heads) * u64::from(self.record_bytes)),
            embedding: carver.take(bf16_bytes(rows, self.embed_dim())),
            key: carver.take(bf16_bytes(rows, width)),
            key_normed: carver.take(bf16_bytes(rows, width)),
            value: carver.take(bf16_bytes(rows, self.hidden)),
            query: carver.take(bf16_bytes(rows, width)),
            gated: carver.take(bf16_bytes(rows, width)),
            gated_normed: carver.take(bf16_bytes(rows, width)),
        }
    }

    /// The expansion scratch the two projections take for `rows` rows on `device`: what [`PleApply::new`] asks of the
    /// regions.
    pub fn expansion_bytes(&self, device: &Nvfp4Device, rows: u32) -> u64 {
        let embed_dim = self.embed_dim();
        device
            .expansion_bytes(rows, embed_dim, self.state_width())
            .max(device.expansion_bytes(rows, embed_dim, self.hidden))
    }

    /// The bytes of the scratch of `rows` rows.
    pub fn scratch_bytes(&self, rows: u32) -> u64 {
        let mut carver = Carver::new(DevicePtr::NULL);
        self.carve(&mut carver, rows);
        carver.bytes()
    }
}

#[derive(Clone, Copy, Debug)]
struct PleScratch {
    records: DevicePtr,
    embedding: DevicePtr,
    key: DevicePtr,
    key_normed: DevicePtr,
    value: DevicePtr,
    query: DevicePtr,
    gated: DevicePtr,
    gated_normed: DevicePtr,
}

/// The weights of the embedding: the NVFP4 key and value projections, and the BF16 norms and convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PleWeights {
    pub key_proj: Nvfp4Weight,
    pub value_proj: Nvfp4Weight,
    pub norm_key: Weight,
    pub norm_query: Weight,
    pub norm_conv: Weight,
    pub convolution: Weight,
}

/// The device stage: the upload of the gathered records and the embedding's arithmetic.
#[derive(Clone, Debug)]
pub struct PleApply {
    shape: PleShape,
    weights: PleWeights,
    /// The sequence's convolution history (`(taps - 1) * dilation` rows of the state width).
    history: DevicePtr,
    regions: FlashRegions,
    device: Nvfp4Device,
}

impl PleApply {
    /// Refuses regions of another stream shape or with less scratch than the embedding takes for their rows.
    pub fn new(
        shape: PleShape,
        weights: PleWeights,
        history: DevicePtr,
        regions: FlashRegions,
        device: Nvfp4Device,
    ) -> Result<Self, FlashError> {
        let rows = regions.capacity_rows();
        regions.require(
            "ple",
            StreamShape::new(shape.hidden, shape.streams)?,
            ScratchSizes {
                layer: shape.scratch_bytes(rows),
                expansion: shape.expansion_bytes(&device, rows),
                ..ScratchSizes::default()
            },
        )?;
        Ok(Self {
            shape,
            weights,
            history,
            regions,
            device,
        })
    }

    /// The operations after the records are on the device, in Java's order, ending with the add into the streams.
    pub fn ops(&self, rows: u32) -> Result<OpList<'static>, FlashError> {
        let shape = &self.shape;
        let weights = &self.weights;
        let scratch = shape.carve(&mut Carver::new(self.regions.layer_scratch), rows);
        let width = shape.state_width();
        let streams = self.regions.state;
        let expansion = self.regions.expansion_scratch();
        let linear = |weight: Nvfp4Weight, output, n| {
            self.device
                .linear(scratch.embedding, weight, output, rows, shape.embed_dim(), n, expansion)
        };
        let mut list = OpList::new();
        list.push(Op::Launch(ops::ngram_expand(
            scratch.records,
            scratch.embedding,
            rows * shape.heads,
            shape.row_width,
            shape.record_bytes,
        )?));
        list.push(linear(weights.key_proj, scratch.key, width)?);
        list.push(Op::Launch(ops::grouped_rms_norm(
            scratch.key,
            weights.norm_key.address,
            scratch.key_normed,
            rows,
            shape.streams,
            shape.hidden,
            shape.epsilon,
        )?));
        list.push(linear(weights.value_proj, scratch.value, shape.hidden)?);
        list.push(Op::Launch(ops::grouped_rms_norm(
            streams,
            weights.norm_query.address,
            scratch.query,
            rows,
            shape.streams,
            shape.hidden,
            shape.epsilon,
        )?));
        list.push(Op::Launch(ops::ple_gate(
            scratch.key_normed,
            scratch.query,
            scratch.value,
            scratch.gated,
            rows,
            shape.streams,
            shape.hidden,
        )?));
        list.push(Op::Launch(ops::grouped_rms_norm(
            scratch.gated,
            weights.norm_conv.address,
            scratch.gated_normed,
            rows,
            shape.streams,
            shape.hidden,
            shape.epsilon,
        )?));
        list.push(Op::Launch(ops::ple_conv(
            scratch.gated_normed,
            scratch.gated,
            self.history,
            weights.convolution.address,
            self.regions.ple_output,
            rows,
            width,
            shape.taps,
            shape.dilation,
        )?));
        list.push(Op::Launch(ops::conv_history(
            scratch.gated_normed,
            self.history,
            rows,
            width,
            shape.history_rows(),
        )?));
        list.push(Op::ResidualAdd {
            residual: streams,
            delta: self.regions.ple_output,
            output: streams,
            rows,
            width,
        });
        Ok(list)
    }

    /// The destination and the byte count of the upload of the records of `range` into the staged records of `rows`
    /// rows.
    fn upload_extent(&self, rows: u32, range: &Range<usize>) -> (DevicePtr, usize) {
        let records = shape_records(&self.shape, &self.regions, rows);
        let record = u64::from(self.shape.record_bytes);
        (
            records.offset(range.start as u64 * record),
            range.len() * self.shape.record_bytes as usize,
        )
    }

    /// The uploads of `parts` to their places in the records, in part order.
    fn uploads<'a>(&self, rows: u32, parts: &'a [(Range<usize>, &'a GatheredPart)]) -> OpList<'a> {
        parts
            .iter()
            .map(|(range, part)| {
                let (destination, bytes) = self.upload_extent(rows, range);
                Op::Upload {
                    destination,
                    source: part.loan.buffer(),
                    offset: 0,
                    bytes,
                }
            })
            .collect()
    }

    /// The gathered parts of a quantum of `rows` rows, in row order. Refuses rows past the regions' capacity, ids that
    /// are not there or not `rows` of the heads, and a part that never arrived.
    fn parts<'a>(
        &self,
        rows: u32,
        hand_over: &'a PleRows,
    ) -> Result<Vec<(Range<usize>, &'a GatheredPart)>, StageError> {
        if rows > self.regions.capacity_rows() {
            return Err(fatal(format!(
                "a quantum of {rows} rows exceeds the {} rows the flash regions hold",
                self.regions.capacity_rows()
            )));
        }
        let count = hand_over
            .ids()
            .ok_or_else(|| fatal("the n-gram embedding ran before the ids"))?
            .len();
        let expected = u64::from(rows) * u64::from(self.shape.heads);
        if count as u64 != expected {
            return Err(fatal(format!(
                "{count} n-gram row ids for {rows} rows of {} heads, expected {expected}",
                self.shape.heads
            )));
        }
        hand_over
            .parts
            .join(count)
            .map_err(|part| fatal(format!("part {part} of the n-gram gather never arrived")))
    }
}

/// Where the records of `rows` rows are staged on the device.
fn shape_records(shape: &PleShape, regions: &FlashRegions, rows: u32) -> DevicePtr {
    shape.carve(&mut Carver::new(regions.layer_scratch), rows).records
}

impl Launch for PleApply {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.rows == 0
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let rows = quantum.rows_u32()?;
        let parts = self.parts(rows, rows_of(quantum)?)?;
        queue_all(&self.uploads(rows, &parts), kernels).map_err(fatal)?;
        let ops = self.ops(rows).map_err(fatal)?;
        queue_all(&ops, kernels).map_err(fatal)
    }
}

eager!(PleApply);

#[cfg(test)]
mod tests;
