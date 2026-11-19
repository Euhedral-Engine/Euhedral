//! Quanta: their input records, their capture keys, and how they run (stage by stage, or as one graph body).
//!
//! The sequence's owner builds a [`Quantum`](crate::stages::Quantum) for each step, reserves its cache rows, uploads
//! its input record ([`Quantum::prepare`]) and starts it on an [`Executor`], which picks the graph it runs:
//!
//! - **Stage by stage**: the view's stage graph, each stage on the lane the plan placed it on, ordered behind its
//!   predecessors by markers. Prefill, a quantum whose reservation allocated pages, a quantum that cannot be keyed
//!   (see [`Quantum::capture_key`]), and every quantum when `kernels.graphs.captures-per-view` is 0 run so.
//! - **One graph body**: a quantum with a capture key is one stage on its home lane that runs every launch of the view
//!   in order as a body of the sequence's [`SharedGraphCache`], behind the input record's upload and the reservation
//!   that backs its cache rows. The cache follows RustEE's model: the first sight of a key runs the body as direct
//!   launches, the second records it as a graph, and every sight after replays the recording as it is (the key covers
//!   every argument, so a replay needs no checking pass; a build with debug assertions still runs one and names a key
//!   that forgot a field). It runs on the quantum's home lane, which RustEE's retirement covers whether the run
//!   succeeded or not, and what the run keeps alive for the device is held by the quantum until it is dropped, after
//!   it retired.
//!
//! The host side effects of a quantum (the input record's upload, the KV reservation, the frontiers, the logits
//! bookkeeping) are the owner's, before the body runs and after the quantum retires, so a replay does none of them.

mod capture_key;
pub mod home;
mod input_record;
pub mod whole;

use smallvec::SmallVec;
use std::fmt;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use euhedral_gpu::cuda::{CudaGpu, CudaKernel, CudaQueue};
use euhedral_gpu::{Enqueue, GraphHold, GraphStats, Ran, SharedGraphCache};
use euhedral_graph::{
    EdgeKind, ExternalEdge, GraphError, GraphRuntime, LaneId, Marker, QuantumOutcome, QuantumRef, Stage, StageCx,
    StageError, StageGraph, StageGraphSpec, StageId, StageSpec, Submitted,
};
use euhedral_inference_cuda::launch::ops::RusteeOps;
use euhedral_inference_cuda::{AllocationId, CudaError, KernelOps, PinnedPool, SubmitMode};
use thiserror::Error;

pub use capture_key::{CaptureKey, split_geometry};
pub use input_record::{
    encode as encode_input_record, encode_into as encode_input_record_into,
    encode_positions as encode_input_record_positions,
};

use crate::plan::{MAX_CAPTURED_ROWS, Plan, Variant, View};
use crate::prefix::{Copier, CopyJob};
use crate::stages::DraftSeed;
use crate::stages::Quantum;
use crate::stages::attention::{KvSource, takes_decode_scratch};
use crate::stages::sampling::SelectedRows;
use crate::stages::{LaneTable, QuantumError, QuantumKind, StageLaunch, fatal};
use crate::state::SequenceId;
use crate::workspace::BufferSlot;
use whole::{Launches, Site, run_cached};

/// What a quantum that runs as a graph body carries: its key, what the run holds for the device, and how it ran.
#[derive(Debug)]
pub(crate) struct Capture {
    key: u128,
    /// What the run keeps alive for the device; dropped with the quantum, which the owner drops after it retired.
    hold: OnceLock<GraphHold>,
    ran: OnceLock<Ran>,
}

impl Capture {
    fn new(key: u128) -> Self {
        Self {
            key,
            hold: OnceLock::new(),
            ran: OnceLock::new(),
        }
    }
}

/// Why a quantum's input record could not be uploaded.
#[derive(Debug, Error)]
pub enum PrepareError {
    #[error("the quantum has no token ids for its {rows} rows")]
    NoTokens { rows: usize },
    #[error(transparent)]
    Cuda(#[from] CudaError),
}

impl Quantum {
    /// The view the quantum runs.
    pub fn view(&self) -> View {
        View::of(self.kind, self.rows)
    }

    /// How the quantum's graph body ran, once it has run: `None` for a quantum that runs stage by stage (and until its
    /// body has run). Set by the [`Executor`] that starts it.
    pub fn graph_run(&self) -> Option<Ran> {
        self.capture.as_ref()?.ran.get().copied()
    }

    /// Writes the quantum's input record (its token ids as 32-bit values, then its start position as a 64-bit value
    /// at the next 8-byte boundary) into pinned memory from `pool`, and queues its upload to the workspace's input
    /// slot through `kernels`. The pinned copy is held until the quantum is dropped. Record the marker of the submission
    /// that carried it with [`Quantum::prepared_after`]: the quantum's first stage waits for it on the device. Rows that
    /// read their ids from the step's tokens (`reading_draft_tokens`) or from the selection of the quantum before them
    /// ([`Quantum::token_from`]) need no ids; the record then carries zeros.
    ///
    /// The owner reserves the cache's rows (`KvCache::reserve`, which needs the cache) before it takes the views the
    /// quantum reads ([`SequenceState::refs`](crate::state::SequenceState::refs)); that reservation is not part of
    /// the quantum.
    pub fn prepare(&mut self, pool: &Arc<PinnedPool>, kernels: &mut dyn KernelOps) -> Result<(), PrepareError> {
        let zeros;
        let ids: &[u32] = if self.tokens.len() == self.rows {
            &self.tokens
        } else if self.token_from.is_some() || self.segments().iter().all(|segment| segment.draft_token_read.is_some())
        {
            zeros = vec![0u32; self.rows];
            &zeros
        } else {
            return Err(PrepareError::NoTokens { rows: self.rows });
        };
        let positions: SmallVec<[u64; 1]> = self.segments().iter().map(|segment| segment.start_position).collect();
        let record = encode_input_record_positions(ids, &positions);
        let mut loan = pool.acquire(record.len())?;
        loan.buffer_mut().as_mut_slice()[..record.len()].copy_from_slice(&record);
        // SAFETY: the loan is held by the quantum, which is dropped only after its device work retired, and nothing
        // writes it again.
        unsafe { kernels.copy_upload_to_device(self.slot(BufferSlot::TokenIds), loan.buffer(), 0, record.len()) }?;
        self.staging = Some(loan);
        self.uploaded = true;
        self.prepared = None;
        Ok(())
    }

    /// The marker of the work that uploaded the input record ([`Quantum::prepare`]).
    pub fn prepared_after(&mut self, marker: Marker) {
        self.prepared = Some(marker);
    }

    /// Refuses to run a quantum whose input record was queued for upload without the marker that orders its stages
    /// behind it.
    pub(crate) fn check_prepared(&self) -> Result<(), StageError> {
        if self.uploaded && self.prepared.is_none() {
            Err(fatal(
                "the quantum's input record was queued for upload, but the marker of that work was never recorded \
                 with `prepared_after`",
            ))
        } else {
            Ok(())
        }
    }

    /// The first KV source the quantum's attention reads: the MTP cache for a draft quantum, else the base cache of a
    /// model with attention layers.
    fn attention_source(&self) -> Option<KvSource> {
        if self.kind == QuantumKind::Draft {
            Some(KvSource::Mtp)
        } else if self.config.attention_layers() > 0 {
            Some(KvSource::Base)
        } else {
            None
        }
    }

    /// Checks everything a launch of the quantum would refuse, before anything is queued: a launch error in the
    /// middle of a capture spoils the capture. The builders check the slot ranges as they are called; this checks what
    /// the quantum needs from its sequence and workspace: the views of the caches and their reserved rows, the draft
    /// buffers and seed, the GDN checkpoints of a verification, the decode scratch, and the selection.
    pub fn validate(&self) -> Result<(), QuantumError> {
        let invalid = |what: String| Err(QuantumError::Invalid { what });
        if let Some(what) = self.misuse() {
            return invalid(what.to_string());
        }
        for segment in self.segments() {
            let (start, rows) = (segment.start_position, segment.rows as u64);
            if let Some(source) = self.attention_source() {
                let Some(view) = segment.refs.kv(source) else {
                    return invalid(format!("the quantum has no view of the {source:?} cache"));
                };
                if !view.reserves(start, rows) {
                    return invalid(format!(
                        "an append of {rows} rows at {start} is not reserved: the {source:?} cache held {} rows at admission",
                        view.reserved()
                    ));
                }
                let layers = if source == KvSource::Mtp {
                    1
                } else {
                    self.config.attention_layers()
                };
                if (0..layers).any(|layer| view.key_table(layer).is_none()) {
                    return invalid(format!("a layer of the {source:?} cache has no page table"));
                }
                if takes_decode_scratch(self.kind, segment.rows) {
                    let held = segment.refs.device().scratch.decode_rows();
                    if rows > held {
                        return invalid(format!(
                            "{rows} attention rows exceed the {held} the decode scratch holds"
                        ));
                    }
                }
            }
            if self.kind == QuantumKind::Draft {
                if self.draft_seed.is_none() {
                    return invalid("a draft quantum has no rows to seed it".to_string());
                }
                if segment.refs.device().scratch.draft().is_none() {
                    return invalid("a draft quantum needs the sequence's draft buffers".to_string());
                }
            }
            if self.kind == QuantumKind::Verify {
                for layer in 0..self.config.layer_kinds.len() {
                    let Some(state) = segment.refs.device().gdn.layer(layer) else {
                        continue;
                    };
                    let capacity = state.speculative().map_or(0, |buffers| buffers.capacity);
                    if rows > capacity {
                        return invalid(format!(
                            "a verification of {rows} rows exceeds the {capacity} the speculative buffers of layer {layer} hold"
                        ));
                    }
                }
            }
            if self.logits_rows > 0 {
                let wanted = if self.kind != QuantumKind::Verify {
                    1
                } else if self.segments().len() == 1 {
                    self.logits_rows
                } else {
                    segment.rows
                };
                match (&segment.selection, &segment.row_readback) {
                    (Some(_), Some(_)) => {
                        return invalid("a quantum selects on the device or on the host, not both".to_string());
                    }
                    (Some(selection), None) if wanted > selection.capacity() => {
                        return invalid(format!(
                            "{wanted} selected rows exceed the {} the selection holds",
                            selection.capacity()
                        ));
                    }
                    (None, Some(readback)) if wanted > readback.capacity() => {
                        return invalid(format!(
                            "{wanted} copied rows exceed the {} the row readback holds",
                            readback.capacity()
                        ));
                    }
                    (None, None) => {
                        return invalid(
                            "a quantum that scores logits has no selection words or row readback".to_string(),
                        );
                    }
                    _ => {}
                }
            }
        }
        if self.draft_token_write.is_some() && (self.segment().selection.is_none() || self.logits_rows == 0) {
            return invalid("a drafted token is passed on from a selection, and the quantum has none".to_string());
        }
        Ok(())
    }

    /// The allocation serial of every buffer the quantum's launches read or write, in a fixed order.
    pub fn buffer_generations(&self) -> Vec<AllocationId> {
        let mut generations: Vec<AllocationId> = self.workspace.generations().collect();
        for segment in self.segments() {
            segment.refs.device().generations(&mut generations);
            if let Some(selection) = &segment.selection {
                generations.push(selection.generation());
            }
            if let Some(readback) = &segment.row_readback {
                generations.push(readback.generation());
            }
            // Only the cache the quantum's attention reads: the MTP cache growing does not change a base quantum's graph.
            if let Some(view) = self.attention_source().and_then(|source| segment.refs.kv(source)) {
                view.generations(&mut generations);
            }
        }
        generations
    }

    /// The key of the graph the quantum would be captured into, or `None` when it is not captured: prefill, more than
    /// eight rows, exact numerics, rows that seed from the sequence's seed rows (found only when the stage runs), and a
    /// quantum whose reservation had to allocate pages or upload a page table (it ran work a graph cannot repeat; the
    /// next quantum is captured).
    pub fn capture_key(&self) -> Option<CaptureKey> {
        if self.segments().len() != 1 {
            // A batch runs stage by stage: its segments' sizes and sequences would each have to be part of a key.
            return None;
        }
        let view = self.view();
        if !view.capturable() || self.rows > MAX_CAPTURED_ROWS {
            return None;
        }
        if self.lanes.library().exact_numerics() {
            return None;
        }
        let draft_seed = match self.draft_seed {
            Some(DraftSeed::Address(address)) => Some(address.get()),
            Some(DraftSeed::SeedRow(_)) => return None,
            None => {
                if self.kind == QuantumKind::Draft {
                    return None;
                }
                None
            }
        };
        let (start, rows) = (self.segment().start_position, self.rows as u64);
        if let Some(source) = self.attention_source() {
            let view = self.segment().refs.kv(source)?;
            if view.grew_for(start, rows) || !view.reserves(start, rows) {
                return None;
            }
        }
        let group = (self.config.num_attention_heads / self.config.num_key_value_heads.max(1)) as u64;
        let host_pages = |source| {
            self.segment()
                .refs
                .kv(source)
                .map_or(0, |view| view.reserved_host_pages())
        };
        let source = self.attention_source();
        let slot = |slot: usize| u8::try_from(slot).ok();
        Some(CaptureKey {
            view,
            kind: self.kind,
            rows: self.rows,
            logits: self.logits_rows,
            selects: self.segment().selection.is_some(),
            copies_rows: self.segment().row_readback.as_ref().map(|_| {
                if self.kind == QuantumKind::Verify {
                    SelectedRows::Each
                } else {
                    SelectedRows::Last
                }
            }),
            seeds_draft: self.segment().seeds_draft,
            draft_seed,
            draft_read: self.segment().draft_token_read.and_then(slot),
            draft_write: self
                .draft_token_write
                .and_then(|write| Some((slot(write.slot)?, write.ids.get()))),
            writes_current_token: self.writes_current_token,
            input_record: self.record as u8,
            token_from: self.token_from.as_ref().map(|selection| selection.generation()),
            row_exact: self.mode.row_exact,
            programmatic: self.mode.pdl,
            split_geometry: split_geometry(start, rows, group),
            reserved_host_pages: keyed_host_pages(source, host_pages(KvSource::Base), host_pages(KvSource::Mtp)),
            replay_rows: capture_key::trimmed(self.segment().refs.replay_layers()),
            buffer_generations: self.buffer_generations(),
        })
    }
}

/// The host pages a key names, `[base, mtp]`: those of the cache the quantum's attention reads, since a graph copies only
/// that cache's host pages. The other cache growing past a 256-token page must not re-key the quantum.
fn keyed_host_pages(source: Option<KvSource>, base: u64, mtp: u64) -> [u64; 2] {
    match source {
        Some(KvSource::Base) => [base, 0],
        Some(KvSource::Mtp) => [0, mtp],
        None => [0, 0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_names_the_host_pages_of_the_cache_the_quantum_reads_only() {
        assert_eq!(keyed_host_pages(Some(KvSource::Base), 7, 3), [7, 0]);
        assert_eq!(
            keyed_host_pages(Some(KvSource::Mtp), 7, 3),
            [0, 3],
            "a draft is not re-keyed by the base cache"
        );
        assert_eq!(
            keyed_host_pages(Some(KvSource::Mtp), 8, 3),
            keyed_host_pages(Some(KvSource::Mtp), 7, 3)
        );
        assert_eq!(
            keyed_host_pages(Some(KvSource::Base), 7, 4),
            keyed_host_pages(Some(KvSource::Base), 7, 3)
        );
        assert_eq!(keyed_host_pages(None, 7, 3), [0, 0]);
    }
}

/// The latest marker per lane among `markers`, without those of `own`, the lane that waits.
fn latest_per_lane(markers: impl IntoIterator<Item = Marker>, own: Option<LaneId>) -> Vec<Marker> {
    let mut latest: Vec<Marker> = Vec::new();
    for marker in markers {
        if Some(marker.lane) == own {
            continue;
        }
        match latest.iter_mut().find(|held| held.lane == marker.lane) {
            Some(held) => held.value = held.value.max(marker.value),
            None => latest.push(marker),
        }
    }
    latest
}

/// The launches of a quantum's view as a graph body: each stage with work, in the order of the view.
struct QuantumLaunches<'a> {
    quantum: &'a Quantum,
    launches: &'a [Arc<dyn StageLaunch>],
}

impl Launches<CudaGpu> for QuantumLaunches<'_> {
    fn launch<E: Enqueue<Kernel = CudaKernel>>(&mut self, enq: &mut E) -> Result<(), StageError> {
        let quantum = self.quantum;
        let modes = quantum.lanes.modes(quantum.mode.row_exact);
        let mut ops = RusteeOps::new(enq, quantum.lanes.kernels(), modes).with_pdl(quantum.mode.pdl);
        let kernels: &mut dyn KernelOps = &mut ops;
        for launch in self.launches.iter().filter(|launch| !launch.idle(quantum)) {
            launch.launch(quantum, kernels)?;
        }
        Ok(())
    }
}

/// The one stage of a quantum that runs as a graph body: it runs the quantum's launches on its home lane through the
/// sequence's graph cache.
struct Whole {
    launches: Arc<[Arc<dyn StageLaunch>]>,
    cache: Arc<SharedGraphCache<CudaGpu, u128>>,
    gpu: Arc<CudaGpu>,
}

impl Stage<Quantum> for Whole {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let Some(capture) = &quantum.capture else {
            return Err(fatal("a graph body was started with a quantum that has no capture key"));
        };
        quantum.check_prepared()?;
        let id = cx.lane.ok_or_else(|| fatal("the graph body was given no lane"))?;
        let lane = quantum
            .lanes
            .get(id)
            .ok_or_else(|| fatal(format!("lane {} is not in the lane table", id.0)))?;
        // The markers the quantum depends on are made before the body runs: by a replay on its private stream, else on
        // the lane.
        let busy = self.launches.iter().filter(|launch| !launch.idle(quantum));
        let waits = latest_per_lane(
            cx.waits
                .iter()
                .copied()
                .chain(busy.flat_map(|launch| launch.waits(quantum))),
            Some(id),
        );
        let mut hold = GraphHold::new();
        let mut launches = QuantumLaunches {
            quantum,
            launches: &self.launches,
        };
        let site = Site {
            gpu: &*self.gpu,
            lane: &**lane,
            waits: &waits,
        };
        let made = run_cached(&site, &self.cache, capture.key, &mut launches, &mut hold);
        // A run that did not start keeps nothing; one that failed may have queued work that reads what it kept.
        if !matches!(made, Err(StageError::Retry)) {
            let _ = capture.hold.set(hold);
        }
        let (ran, marker) = made?;
        let _ = capture.ran.set(ran);
        Ok(Submitted::marker(marker))
    }
}

/// The stage graph of a quantum that runs as a graph body, homed on lane `home`: one stage.
fn whole_spec(home: usize) -> StageGraphSpec {
    let mut spec = StageGraphSpec::new();
    spec.stage(StageSpec {
        name: "graph",
        ordered: None,
        lanes: vec![LaneId(home as u16)],
    });
    spec
}

/// A started quantum: its reference (which pins it, see [`QuantumRef`]), and the stages of its graph that a later
/// quantum of the chain is ordered by.
#[derive(Clone, Debug)]
pub struct Started {
    /// The quantum's own number ([`Quantum::id`]).
    pub serial: u64,
    pub reference: QuantumRef,
    /// The stage where the quantum enters its graph.
    pub root: StageId,
    /// The stage whose marker covers the quantum's device work.
    pub sink: StageId,
    /// The unit that reads each staging slot last, which a later quantum's first copy into the slot waits for; empty for
    /// a quantum that stages nothing.
    pub slot_readers: Arc<[Option<StageId>]>,
}

/// The slot readers of a quantum that reads no staging slot: one empty list, shared.
pub fn no_slot_readers() -> Arc<[Option<StageId>]> {
    static NONE: OnceLock<Arc<[Option<StageId>]>> = OnceLock::new();
    Arc::clone(NONE.get_or_init(|| Arc::from(Vec::new())))
}

/// What a new quantum is ordered by in a predecessor: where it entered its graph, the stage whose marker covers its
/// device work, and the unit that reads each staging slot last.
#[derive(Clone, Copy, Debug)]
pub struct Predecessor<'a> {
    pub root: StageId,
    pub sink: StageId,
    pub slot_readers: &'a [Option<StageId>],
}

/// An edge from a stage of the predecessor `after[predecessor]` to a stage of the new quantum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dependency {
    pub predecessor: usize,
    pub from_stage: StageId,
    pub to_stage: StageId,
}

/// The edges a new quantum starts with, behind each predecessor in `after`:
/// - its first stage waits for the predecessor's sink: the chain's token passes from one quantum to the next, and the
///   workspace's buffers are reused;
/// - the first transfer of each staging slot (`first_transfers`: the slot and the transfer's unit) waits for the
///   predecessor's last reader of that slot only, so the copies overlap the predecessor's tail. A predecessor that
///   reads no slot gives its root, which already waited for the quantum before it.
///
/// A predecessor the order no longer holds (retired) is not in `after` and gives nothing. A view that stages nothing
/// has no first transfer and gets the first stage's edges alone.
pub fn external_edges(first_transfers: &[(usize, StageId)], after: &[Predecessor<'_>]) -> SmallVec<[Dependency; 16]> {
    let mut edges = SmallVec::new();
    for (predecessor, previous) in after.iter().enumerate() {
        edges.push(Dependency {
            predecessor,
            from_stage: previous.sink,
            to_stage: StageId(0),
        });
    }
    for (predecessor, previous) in after.iter().enumerate() {
        for &(slot, to_stage) in first_transfers {
            edges.push(Dependency {
                predecessor,
                from_stage: previous
                    .slot_readers
                    .get(slot)
                    .copied()
                    .flatten()
                    .unwrap_or(previous.root),
                to_stage,
            });
        }
    }
    edges
}

/// What stops a quantum from starting.
#[derive(Debug, Error)]
pub enum StartError {
    #[error(transparent)]
    Quantum(#[from] QuantumError),
    #[error("the plan has no graph for {0:?}")]
    NoView(View),
    #[error("the workspace's owner is closed")]
    Closed,
    #[error("starting the quantum panicked")]
    Panicked,
    #[error("the engine has no prefix cache to run a copy job")]
    NoCopier,
    #[error("the pass could not run as one quantum: {0}")]
    Batch(String),
    #[error("the quantum's input record could not be uploaded: {0}")]
    Prepare(String),
    #[error("the sequence stopped offering the step before it started")]
    Withdrawn,
}

/// What uploads a quantum's input record when the executor starts it: the owner's lane, and the pinned pool the record is
/// staged in.
#[derive(Clone)]
pub struct Preparer {
    lanes: LaneTable,
    pool: Arc<PinnedPool>,
    lane: Arc<CudaQueue>,
}

impl fmt::Debug for Preparer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Preparer").finish_non_exhaustive()
    }
}

impl Preparer {
    /// A preparer that uploads on `lane` (the owner's), staging in `pool`.
    pub fn new(lanes: LaneTable, pool: Arc<PinnedPool>, lane: Arc<CudaQueue>) -> Self {
        Self { lanes, pool, lane }
    }

    fn prepare(&self, quantum: &mut Quantum) -> Result<(), StartError> {
        let (_, marker) = self
            .lanes
            .submit_ops(&self.lane, false, |kernels| quantum.prepare(&self.pool, kernels))
            .map_err(|error| StartError::Prepare(error.to_string()))?;
        quantum.prepared_after(marker);
        Ok(())
    }
}

/// The stage graphs of a plan on a runtime, the sequence's graph cache, and the choice of the graph a quantum runs.
pub struct Executor {
    plan: Arc<Plan>,
    /// The lanes the stage graphs run on; a quantum's home is one of them.
    stage_lanes: usize,
    /// The reading of the monotonic clock the home hash is seeded with is the time since this instant.
    origin: Instant,
    /// By variant, the stage-by-stage graph for each home lane.
    eager: Vec<Vec<Arc<StageGraph<Quantum>>>>,
    /// By variant, the one-stage graph of a graph body for each home lane; none for a variant that is not captured, and
    /// none at all when captures are off.
    whole: Vec<Vec<Arc<StageGraph<Quantum>>>>,
    /// The graphs the sequence's quanta recorded, by capture key.
    cache: Arc<SharedGraphCache<CudaGpu, u128>>,
    /// Whether quanta that can be keyed run as graph bodies.
    captures: AtomicBool,
    /// By variant, the unit that reads each staging slot last (empty for a variant that stages nothing).
    slot_readers: Vec<Arc<[Option<StageId>]>>,
    /// Runs the sequence's prefix-cache copy jobs, when it has a cache.
    copier: Option<Copier>,
    /// Uploads the input record of a quantum that arrives unprepared (an offered verification, prepared when its pass
    /// starts, as the one quantum or the batch it is a part of).
    preparer: Option<Preparer>,
}

impl fmt::Debug for Executor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Executor").finish_non_exhaustive()
    }
}

impl Executor {
    /// The graphs of every view of `plan`, run on `runtime` (whose lane pool has `stage_lanes`, at least the plan's
    /// [`Plan::lane_width`], stage lanes first, then lanes the owner uses). The sequence keeps `captures_per_view`
    /// graphs for each view that is captured, recorded on `gpu`; with 0, no quantum is captured. `retire` receives
    /// each quantum, with its outcome, once its device work has completed.
    pub fn new(
        plan: Arc<Plan>,
        runtime: &GraphRuntime,
        stage_lanes: usize,
        gpu: Arc<CudaGpu>,
        captures_per_view: usize,
        retire: impl Fn(Quantum, QuantumOutcome) + Send + Sync + 'static,
    ) -> Result<Executor, GraphError> {
        let retire: Arc<dyn Fn(Quantum, QuantumOutcome) + Send + Sync> = Arc::new(retire);
        let stage_lanes = stage_lanes.max(1);
        let captured = |variant: Variant| captures_per_view > 0 && plan.capturable(variant.representative());
        let views = Variant::ALL
            .into_iter()
            .filter(|&variant| captured(variant) && plan.stages(variant.representative()).is_some())
            .count();
        let cache = Arc::new(SharedGraphCache::with_capacity((captures_per_view * views).max(1)));
        cache.set_replay_as_recorded(true);
        let mut eager = Vec::with_capacity(Variant::ALL.len());
        let mut whole = Vec::with_capacity(Variant::ALL.len());
        for variant in Variant::ALL {
            let view = variant.representative();
            let (Some(entries), Some(units)) = (plan.stages(view), plan.unit_stages(view)) else {
                eager.push(Vec::new());
                whole.push(Vec::new());
                continue;
            };
            let stages: Vec<Arc<dyn Stage<Quantum>>> = units.to_vec();
            // One copy of the graph for each home lane: a quantum runs on the copy of its own home.
            let mut homed = Vec::with_capacity(stage_lanes);
            for home in 0..stage_lanes {
                let Some(spec) = plan.spec_homed(view, home, stage_lanes) else {
                    break;
                };
                let again = Arc::clone(&retire);
                homed.push(StageGraph::new(
                    spec?,
                    stages.clone(),
                    runtime.clone(),
                    move |quantum, outcome| again(quantum, outcome),
                )?);
            }
            eager.push(homed);
            let mut bodies = Vec::new();
            if captured(variant) {
                let launches: Arc<[Arc<dyn StageLaunch>]> =
                    entries.iter().map(|entry| Arc::clone(&entry.launch)).collect();
                let body: Arc<dyn Stage<Quantum>> = Arc::new(Whole {
                    launches,
                    cache: Arc::clone(&cache),
                    gpu: Arc::clone(&gpu),
                });
                for home in 0..stage_lanes {
                    let again = Arc::clone(&retire);
                    bodies.push(StageGraph::new(
                        whole_spec(home),
                        vec![Arc::clone(&body)],
                        runtime.clone(),
                        move |quantum, outcome| again(quantum, outcome),
                    )?);
                }
            }
            whole.push(bodies);
        }
        let slot_readers = Variant::ALL
            .into_iter()
            .map(|variant| match plan.slot_readers(variant.representative()) {
                [] => no_slot_readers(),
                readers => Arc::from(readers),
            })
            .collect();
        Ok(Executor {
            plan,
            stage_lanes,
            origin: Instant::now(),
            eager,
            whole,
            cache,
            captures: AtomicBool::new(captures_per_view > 0),
            slot_readers,
            copier: None,
            preparer: None,
        })
    }

    /// The executor that also uploads the input record of the quanta it is given unprepared.
    #[must_use]
    pub fn with_preparer(mut self, preparer: Preparer) -> Executor {
        self.preparer = Some(preparer);
        self
    }

    /// Starts one quantum that is the batch of `members` (each a verification of its sequence, with the flag its
    /// request is cancelled by): the batch's input record is the first member's own, and it is stopped only by `kill`.
    /// A member refused leaves nothing queued: the record is uploaded last.
    pub(crate) fn start_batch(
        &self,
        members: Vec<(SequenceId, Quantum, Arc<AtomicBool>)>,
        kill: Arc<AtomicBool>,
        after: &[&Started],
    ) -> Result<Started, StartError> {
        let Some((_, first, _)) = members.first() else {
            return Err(StartError::Batch("a pass of no member".to_string()));
        };
        let (record, workspace, config, lanes) = (
            first.record,
            Arc::clone(&first.workspace),
            Arc::clone(&first.config),
            first.lanes.clone(),
        );
        let mut parts = Vec::with_capacity(members.len());
        for (sequence, quantum, cancel) in members {
            parts.push(quantum.into_part(sequence, cancel)?);
        }
        let mode = SubmitMode {
            pdl: false,
            row_exact: true,
        };
        let batch = Quantum::batch(parts, workspace, mode, config, lanes)?.with_input_record(record);
        self.start(batch, kill, after)
    }

    fn prepare(&self, quantum: &mut Quantum) -> Result<(), StartError> {
        match (&self.preparer, quantum.is_prepared()) {
            (Some(preparer), false) => preparer.prepare(quantum),
            _ => Ok(()),
        }
    }

    /// The executor that also runs the copy jobs of the prefix cache with `copier`.
    #[must_use]
    pub fn with_copier(mut self, copier: Copier) -> Executor {
        self.copier = Some(copier);
        self
    }

    /// Starts the copy job `job` behind each of `after`; refused when the executor has no copier.
    pub(crate) fn start_copy(
        &self,
        job: CopyJob,
        kill: Arc<AtomicBool>,
        after: &[&Started],
    ) -> Result<Started, StartError> {
        match &self.copier {
            Some(copier) => copier.start(job, kill, after),
            None => Err(StartError::NoCopier),
        }
    }

    /// One monotonic reading, in nanoseconds, taken as a quantum starts.
    fn clock(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// The plan the executor runs.
    pub fn plan(&self) -> &Arc<Plan> {
        &self.plan
    }

    /// What the sequence's graph cache counted: how its quanta ran as graph bodies.
    pub fn graph_stats(&self) -> GraphStats {
        self.cache.stats()
    }

    /// The graphs the sequence's cache holds, and the most it keeps.
    pub fn graphs(&self) -> (usize, usize) {
        (self.cache.len(), self.cache.capacity())
    }

    /// Whether `key` ([`CaptureKey::digest`]) has a recorded graph.
    pub fn has_graph(&self, key: u128) -> bool {
        self.cache.contains(&key)
    }

    /// Drops every graph the cache holds that no quantum runs (the sequence's end, after its quanta retired).
    pub fn clear_graphs(&self) {
        self.cache.retain(|_| false);
    }

    /// Whether quanta that can be keyed run as graph bodies from now on, or stage by stage. Quanta already started keep
    /// their way; an executor made with no captures cannot capture.
    pub fn set_captures(&self, on: bool) {
        self.captures
            .store(on && !self.whole.iter().all(Vec::is_empty), Ordering::Release);
    }

    /// Starts `quantum`: validates it, and runs it as one body of the graph cache when it has a capture key, else stage
    /// by stage. The new quantum's root waits on the device for the sink of each quantum of `after` (the previous one of
    /// its chain: the chain's token passes from one quantum's selection to the next's embedding, so nothing of the next
    /// could overlap it; and the last one on the workspace, whose buffers it reuses). Keep the previous [`Started`]s
    /// until this call returns, then drop them.
    pub fn start(
        &self,
        mut quantum: Quantum,
        kill: Arc<AtomicBool>,
        after: &[&Started],
    ) -> Result<Started, StartError> {
        quantum.validate()?;
        let view = quantum.view();
        let variant = view.variant();
        let home = home::home_of(quantum.id(), self.clock(), self.stage_lanes);
        let eager = self.eager[variant.index()].get(home).ok_or(StartError::NoView(view))?;
        // Nothing after this can refuse the quantum: the upload is queued last, and a quantum dropped after it would free
        // the pinned record under the copy.
        self.prepare(&mut quantum)?;
        let serial = quantum.id();
        let stages = self.plan.unit_stages(view).map_or(0, <[_]>::len);
        let keyed = self
            .captures
            .load(Ordering::Acquire)
            .then(|| quantum.capture_key())
            .flatten()
            .filter(|_| self.plan.capturable(view));
        let body = keyed.and_then(|key| self.whole[variant.index()].get(home).map(|graph| (key, graph)));
        let captured = body.is_some();
        let (graph, sink) = match body {
            Some((key, graph)) => {
                quantum.capture = Some(Capture::new(key.digest()));
                (graph, StageId(0))
            }
            None => (eager, StageId((stages - 1) as u16)),
        };
        // A captured body is one stage; a staged view is never captured, so its first transfers exist only in the
        // stage-by-stage graph.
        let first_transfers = if captured { &[] } else { self.plan.first_transfers(view) };
        let predecessors: SmallVec<[Predecessor<'_>; 4]> = after
            .iter()
            .map(|previous| Predecessor {
                root: previous.root,
                sink: previous.sink,
                slot_readers: &previous.slot_readers,
            })
            .collect();
        let external = external_edges(first_transfers, &predecessors)
            .into_iter()
            .map(|edge| ExternalEdge {
                from: after[edge.predecessor].reference.clone(),
                from_stage: edge.from_stage,
                to_stage: edge.to_stage,
                kind: EdgeKind::Submission,
            });
        let reference = graph.start(quantum, kill, external);
        Ok(Started {
            serial,
            reference,
            root: StageId(0),
            sink,
            slot_readers: Arc::clone(&self.slot_readers[variant.index()]),
        })
    }
}
