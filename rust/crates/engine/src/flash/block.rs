//! The layer's nodes around its block.
//!
//! A [`BlockStage`] is the mix of the streams for the block, the block, the injection of its output into
//! the streams, and the mix that feeds the MoE block, as one stage. The block is a [`BlockBody`]: GDN is
//! [`super::gdn::GdnBody`]; sparse attention supplies its own. [`EndInject`] injects what the MoE block wrote, and
//! [`FinalMix`] mixes the last row for the output head.

use std::fmt::Debug;
use std::sync::Arc;

use euhedral_graph::StageError;
use euhedral_inference_cuda::{DevicePtr, KernelOps};

use super::FlashError;
use super::buffers::{FlashRegions, ScratchSizes, StreamShape};
use super::hyper::{HyperScratch, HyperShape, HyperWeights};
use super::linear::Expansion;
use super::sequence::{Op, OpList, queue_all};
use crate::stages::attention::KvSource;
use crate::stages::{Launch, Quantum, fatal};

/// Where a call's rows lie in their sequence, and the per-chunk device state of that sequence a token mixer reads. The
/// sequence's owner makes it (for sparse attention from [`QsaState::begin_chunk`](super::qsa::state::QsaState) and the
/// cache's page tables); a mixer with no state per position (GDN) ignores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkAt {
    /// The position of the call's first row in its sequence.
    pub start: u32,
    /// The device tables of the K and V page addresses of the layer's cache.
    pub key_table: DevicePtr,
    pub value_table: DevicePtr,
    /// The slot (0 or 1) of the raw indexer tail the chunk reads; it writes the other.
    pub tail_slot: u32,
}

impl Default for ChunkAt {
    /// The first chunk of a sequence that has no cache: no page tables, which the mixers that need them refuse.
    fn default() -> Self {
        Self {
            start: 0,
            key_table: DevicePtr::NULL,
            value_table: DevicePtr::NULL,
            tail_slot: 0,
        }
    }
}

/// Where one chunk's rows lie in their sequence, for every layer that reads the cache: what the sequence's owner makes
/// when it builds the chunk's quantum, and what [`BlockStage`] takes the layer's [`ChunkAt`] from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkPlacement {
    start: u32,
    tail_slot: u32,
    /// The K and V page tables of each cache layer, by cache layer index.
    tables: Vec<(DevicePtr, DevicePtr)>,
}

impl ChunkPlacement {
    /// The placement of a chunk starting at `start`, reading raw tail slot `tail_slot` (0 or 1), over `tables`.
    /// Refuses a slot that is neither.
    pub fn new(start: u32, tail_slot: u32, tables: Vec<(DevicePtr, DevicePtr)>) -> Result<Self, FlashError> {
        if tail_slot > 1 {
            return Err(FlashError::refused(
                "chunk placement",
                format!("the raw tail has slots 0 and 1, not {tail_slot}"),
            ));
        }
        Ok(Self {
            start,
            tail_slot,
            tables,
        })
    }

    /// The position of the chunk's first row.
    pub fn start(&self) -> u32 {
        self.start
    }

    /// The chunk of the cache layer `layer`: refuses a layer the placement has no tables for.
    pub fn at(&self, layer: usize) -> Result<ChunkAt, String> {
        let (key_table, value_table) = self.tables.get(layer).ok_or_else(|| {
            format!(
                "the chunk's placement has tables for {} cache layers, not layer {layer}",
                self.tables.len()
            )
        })?;
        Ok(ChunkAt {
            start: self.start,
            key_table: *key_table,
            value_table: *value_table,
            tail_slot: self.tail_slot,
        })
    }
}

/// What a block is called with.
#[derive(Clone, Copy, Debug)]
pub struct BlockCall {
    /// The mixed rows the block reads and the rows it writes: `[rows, hidden]` each.
    pub input: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    /// Whether the lane runs the exact-numerics kernels.
    pub exact: bool,
    /// The scratch of the block in flight ([`FlashRegions::layer_scratch`]).
    pub scratch: DevicePtr,
    /// The expansion scratch of the NVFP4 linears' native route.
    pub expansion: Expansion,
    /// Where the rows lie in their sequence.
    pub chunk: ChunkAt,
}

/// The token mixer of a layer between its two hyper-connection sites.
pub trait BlockBody: Send + Sync + Debug {
    /// The operations of the block for `call`, in the order they are queued.
    fn ops(&self, call: &BlockCall) -> Result<OpList<'static>, FlashError>;

    /// The bytes of layer scratch the block takes for `rows` rows.
    fn scratch_bytes(&self, rows: u32) -> u64;

    /// The bytes of expansion scratch its NVFP4 linears take for `rows` rows.
    fn expansion_bytes(&self, rows: u32) -> u64;
}

/// The stream shape of `hyper`.
fn stream_shape(hyper: &HyperShape) -> Result<StreamShape, FlashError> {
    StreamShape::new(hyper.hidden(), hyper.streams())
}

/// Refuses `regions` that do not fit `hyper`'s streams and mix scratch.
fn require_hyper(stage: &'static str, hyper: &HyperShape, regions: &FlashRegions, rows: u32) -> Result<(), FlashError> {
    regions.require(
        stage,
        stream_shape(hyper)?,
        ScratchSizes {
            hyper: hyper.scratch_bytes(rows),
            ..ScratchSizes::default()
        },
    )
}

/// The rows of `quantum`, within the regions' capacity.
fn rows_of(quantum: &Quantum, regions: &FlashRegions) -> Result<u32, StageError> {
    let rows = quantum.rows_u32()?;
    if rows > regions.capacity_rows() {
        return Err(fatal(format!(
            "a quantum of {rows} rows exceeds the {} rows the flash regions hold",
            regions.capacity_rows()
        )));
    }
    Ok(rows)
}

/// The mix for the block, the block, its injection, and the mix for the MoE block.
#[derive(Debug)]
pub struct BlockStage {
    hyper: HyperShape,
    attention_site: HyperWeights,
    moe_site: HyperWeights,
    body: Arc<dyn BlockBody>,
    regions: FlashRegions,
    /// The layer's place among the layers of the sequence's cache, when its body reads one.
    kv_layer: Option<usize>,
}

impl BlockStage {
    /// The stage of a layer whose sites are `attention_site` (before the block) and `moe_site` (after it).
    pub fn new(
        hyper: HyperShape,
        attention_site: HyperWeights,
        moe_site: HyperWeights,
        body: Arc<dyn BlockBody>,
        regions: FlashRegions,
    ) -> Result<Self, FlashError> {
        if attention_site.inject.is_none() || moe_site.inject.is_none() {
            return Err(FlashError::refused(
                "block stage",
                "both sites of a layer carry an injection projection".to_owned(),
            ));
        }
        let capacity = regions.capacity_rows();
        regions.require(
            "block stage",
            stream_shape(&hyper)?,
            ScratchSizes {
                hyper: hyper.scratch_bytes(capacity),
                layer: body.scratch_bytes(capacity),
                expansion: body.expansion_bytes(capacity),
            },
        )?;
        Ok(Self {
            hyper,
            attention_site,
            moe_site,
            body,
            regions,
            kv_layer: None,
        })
    }

    /// The stage of a layer whose block reads the page tables of layer `layer` of the quantum's cache.
    pub fn with_kv_layer(mut self, layer: usize) -> Self {
        self.kv_layer = Some(layer);
        self
    }

    /// The operations for `rows` rows of a sequence's first chunk, in order: the mix for the block, the block, the
    /// injection, the mix for MoE.
    pub fn ops(&self, rows: u32, exact: bool) -> Result<OpList<'static>, FlashError> {
        self.ops_at(rows, exact, ChunkAt::default())
    }

    /// The operations for `rows` rows at `chunk`.
    pub fn ops_at(&self, rows: u32, exact: bool, chunk: ChunkAt) -> Result<OpList<'static>, FlashError> {
        let regions = &self.regions;
        let scratch = self.hyper.scratch(regions.hyper_scratch, rows);
        let mut list = self.hyper.mix(
            &self.attention_site,
            regions.state,
            &scratch,
            regions.mixed,
            rows,
            exact,
        )?;
        list.extend(self.body.ops(&BlockCall {
            input: regions.mixed,
            output: regions.block_output,
            rows,
            exact,
            scratch: regions.layer_scratch,
            expansion: regions.expansion_scratch(),
            chunk,
        })?);
        list.push(
            self.hyper
                .inject(regions.state, regions.block_output, &scratch, regions.state, rows)?,
        );
        list.extend(
            self.hyper
                .mix(&self.moe_site, regions.state, &scratch, regions.mixed, rows, exact)?,
        );
        Ok(list)
    }
}

impl Launch for BlockStage {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.rows == 0
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let rows = rows_of(quantum, &self.regions)?;
        let chunk = self.chunk_of(quantum)?;
        let ops = self.ops_at(rows, kernels.exact_numerics(), chunk).map_err(fatal)?;
        queue_all(&ops, kernels).map_err(fatal)
    }
}

impl BlockStage {
    /// Where the quantum's rows lie: for a layer that reads the cache, the segment's start and its page tables from
    /// the quantum's chunk placement, which a sequence's owner supplies with the position, the tail slot and the
    /// tables. Without one, a launch knows no tail slot: the stage runs only the first chunk of a sequence (slot 0) from
    /// the segment's view, and refuses a later one, because a stale tail would write wrong block keys.
    fn chunk_of(&self, quantum: &Quantum) -> Result<ChunkAt, StageError> {
        let Some(layer) = self.kv_layer else {
            return Ok(ChunkAt::default());
        };
        if let Some(placement) = quantum.chunk_placement() {
            return placement.at(layer).map_err(fatal);
        }
        let segments = quantum.segments();
        let tables = match segments {
            [segment] => segment
                .refs
                .kv(KvSource::Base)
                .map(|view| (view.key_table(layer), view.value_table(layer))),
            _ => None,
        };
        let start = segments.first().map_or(0, |segment| segment.start_position);
        self.place(segments.len(), start, tables).map_err(fatal)
    }

    /// The chunk of a launch of a cache layer's stage over `segments` segments, the first at `start_position`, whose
    /// view of the cache gives `tables` (K and V) for the layer. Refuses a batch, a missing view or table, a position
    /// of 32 bits or more, and a position past the first chunk.
    fn place(
        &self,
        segments: usize,
        start_position: u64,
        tables: Option<(Option<DevicePtr>, Option<DevicePtr>)>,
    ) -> Result<ChunkAt, String> {
        let Some(layer) = self.kv_layer else {
            return Ok(ChunkAt::default());
        };
        if segments != 1 {
            return Err(format!(
                "the block of a cache layer runs one sequence's chunk, the quantum has {segments} segments"
            ));
        }
        let start = u32::try_from(start_position).map_err(|_| format!("position {start_position} exceeds 32 bits"))?;
        let (keys, values) = tables.ok_or_else(|| "the quantum has no view of the cache".to_owned())?;
        let table = |table: Option<DevicePtr>, name: &str| {
            table.ok_or_else(|| format!("the cache has no {name} page table for layer {layer}"))
        };
        Ok(ChunkAt {
            start,
            key_table: table(keys, "K")?,
            value_table: table(values, "V")?,
            tail_slot: unsupplied_tail_slot(start)?,
        })
    }
}

/// The tail slot of a launch that is told none: only a sequence's first chunk has a known one.
fn unsupplied_tail_slot(start: u32) -> Result<u32, String> {
    if start == 0 {
        Ok(0)
    } else {
        Err(format!(
            "a launch at position {start} does not know the raw tail's slot: the sequence's owner supplies it"
        ))
    }
}

/// The injection of the MoE block's output into the streams (`Stages.EndInject`), by the raw weights of the MoE site's
/// mix.
#[derive(Debug)]
pub struct EndInject {
    hyper: HyperShape,
    regions: FlashRegions,
}

impl EndInject {
    /// Refuses regions of another stream shape or with too little mix scratch.
    pub fn new(hyper: HyperShape, regions: FlashRegions) -> Result<Self, FlashError> {
        require_hyper("end inject", &hyper, &regions, regions.capacity_rows())?;
        Ok(Self { hyper, regions })
    }

    /// The injection of `rows` rows.
    pub fn op(&self, rows: u32) -> Result<Op<'static>, FlashError> {
        let regions = &self.regions;
        let scratch = self.hyper.scratch(regions.hyper_scratch, rows);
        self.hyper
            .inject(regions.state, regions.block_output, &scratch, regions.state, rows)
    }
}

impl Launch for EndInject {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.rows == 0
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let rows = rows_of(quantum, &self.regions)?;
        let op = self.op(rows).map_err(fatal)?;
        op.queue(kernels).map_err(fatal)
    }
}

/// The final mixer over the last row (`Stages.Head`'s mix): the streams of the quantum's last row mixed into
/// [`FlashRegions::final_mixed`], with no injection.
#[derive(Debug)]
pub struct FinalMix {
    hyper: HyperShape,
    site: HyperWeights,
    regions: FlashRegions,
}

impl FinalMix {
    /// The final mixer, whose site has no injection projection.
    pub fn new(hyper: HyperShape, site: HyperWeights, regions: FlashRegions) -> Result<Self, FlashError> {
        if site.inject.is_some() {
            return Err(FlashError::refused(
                "final mix",
                "the final mixer has no injection projection".to_owned(),
            ));
        }
        require_hyper("final mix", &hyper, &regions, 1)?;
        Ok(Self { hyper, site, regions })
    }

    /// The operations over the last of `rows` rows.
    pub fn ops(&self, rows: u32, exact: bool) -> Result<OpList<'static>, FlashError> {
        if rows == 0 {
            return Err(FlashError::refused(
                "final mix",
                "a final mix takes at least one row, got 0".to_owned(),
            ));
        }
        let width = u64::from(self.hyper.state_width());
        let last = self.regions.state.offset(u64::from(rows - 1) * width * 2);
        let scratch: HyperScratch = self.hyper.scratch(self.regions.hyper_scratch, 1);
        self.hyper
            .mix(&self.site, last, &scratch, self.regions.final_mixed, 1, exact)
    }
}

impl Launch for FinalMix {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.rows == 0
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let rows = rows_of(quantum, &self.regions)?;
        let ops = self.ops(rows, kernels.exact_numerics()).map_err(fatal)?;
        queue_all(&ops, kernels).map_err(fatal)
    }
}

eager!(BlockStage, EndInject, FinalMix);

#[cfg(test)]
mod tests;
