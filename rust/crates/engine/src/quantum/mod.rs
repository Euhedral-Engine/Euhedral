//! Quanta: their input records, their capture keys, and how they run (stage by stage, or as a captured graph).
//!
//! The sequence's owner builds a [`Quantum`](crate::stages::Quantum) for each step, reserves its cache rows, uploads
//! its input record ([`Quantum::prepare`]) and starts it on an [`Executor`], which picks the graph it runs:
//!
//! - **Stage by stage**: the view's stage graph, each stage on the lane the plan placed it on, ordered behind its
//!   predecessors by markers. Prefill, a quantum whose reservation allocated pages, the first sight of a key, and
//!   anything a graph cannot repeat run so.
//! - **Recording**, on the second sight of a key: the quantum runs stage by stage, and a record stage beside its
//!   stages captures the same launches on a lane of its own (relaxed capture, nothing is submitted twice), instantiates
//!   the graph and keeps it in the sequence's [`GraphCache`]. The device never waits for a capture or an
//!   instantiation.
//! - **Replay**: from the next sight after the recording has finished, a quantum with the key is one graph launch on
//!   the home lane, behind the input record's upload and the reservation that backs its cache rows.
//!
//! The host side effects of a quantum (the input record's upload, the KV reservation, the frontiers, the logits
//! bookkeeping) are the owner's, before the graph is launched and after it retires, so a replay needs no checking
//! pass. A test-only oracle (`Executor::check_replay_for_tests`) runs the stages in the C library's check mode and
//! compares what they would submit with what the graph was recorded from.

mod capture_key;
mod graph_cache;
mod input_record;

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use euhedral_graph::{
    EdgeKind, ExternalEdge, GraphError, GraphRuntime, LaneId, Marker, QuantumOutcome, QuantumRef, Stage, StageCx,
    StageError, StageGraph, StageGraphSpec, StageId, StageSpec, Submitted,
};
use euhedral_inference_cuda::{AllocationId, CudaError, CudaLane, GraphExec, PinnedPool, Submission, SubmitMode, ops};
use thiserror::Error;

pub use capture_key::{CaptureKey, split_geometry};
pub use graph_cache::{Admission, GraphCache, GraphSlot, Sighting};
pub use input_record::{encode as encode_input_record, encode_into as encode_input_record_into};

use crate::plan::{MAX_CAPTURED_ROWS, Plan, Variant, View};
use crate::stages::DraftSeed;
use crate::stages::Quantum;
use crate::stages::attention::{KvSource, takes_decode_scratch};
use crate::stages::{QuantumError, QuantumKind, StageLaunch, fatal};
use crate::workspace::BufferSlot;

/// How a quantum runs.
#[derive(Clone, Debug, Default)]
pub enum Disposition {
    /// Stage by stage, through the view's stage graph.
    #[default]
    Eager,
    /// Stage by stage, and the record stage captures the same launches into the slot's graph.
    Record(Arc<GraphSlot>),
    /// One launch of the slot's recorded graph.
    Replay(Arc<GraphSlot>),
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

    /// How the quantum runs; set by the [`Executor`] that starts it.
    pub fn disposition(&self) -> &Disposition {
        &self.disposition
    }

    /// Writes the quantum's input record (its token ids as 32-bit values, then its start position as a 64-bit value
    /// at the next 8-byte boundary) into pinned memory from `pool`, and queues its upload to the workspace's input
    /// slot on `submission`. The pinned copy is held until the quantum is dropped. Finish the submission, and record
    /// its marker with [`Quantum::prepared_after`]: the quantum's first stage waits for it on the device. Rows that
    /// read their ids from the step's tokens (`reading_draft_tokens`) or from the selection of the quantum before them
    /// ([`Quantum::token_from`]) need no ids; the record then carries zeros.
    ///
    /// The owner reserves the cache's rows (`KvCache::reserve`, which needs the cache) before it takes the views the
    /// quantum reads ([`SequenceState::refs`](crate::state::SequenceState::refs)); that reservation is not part of
    /// the quantum.
    pub fn prepare(&mut self, pool: &Arc<PinnedPool>, submission: &Submission<'_>) -> Result<(), PrepareError> {
        let zeros;
        let ids: &[u32] = if self.tokens.len() == self.rows {
            &self.tokens
        } else if self.draft_token_read.is_some() || self.token_from.is_some() {
            zeros = vec![0u32; self.rows];
            &zeros
        } else {
            return Err(PrepareError::NoTokens { rows: self.rows });
        };
        let record = encode_input_record(ids, self.input.start_position());
        let mut loan = pool.acquire(record.len())?;
        loan.buffer_mut().as_mut_slice()[..record.len()].copy_from_slice(&record);
        // SAFETY: the loan is held by the quantum, which is dropped only after its device work retired, and nothing
        // writes it again.
        unsafe {
            ops::copy_upload_to_device(
                submission,
                self.slot(BufferSlot::TokenIds),
                loan.buffer(),
                0,
                record.len(),
            )
        }?;
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
        let start = self.input.start_position();
        let rows = self.rows as u64;
        if let Some(source) = self.attention_source() {
            let Some(view) = self.seq.kv(source) else {
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
            if takes_decode_scratch(self.kind, self.rows) {
                let held = self.seq.device().scratch.decode_rows();
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
            if self.seq.device().scratch.draft().is_none() {
                return invalid("a draft quantum needs the sequence's draft buffers".to_string());
            }
        }
        if self.kind == QuantumKind::Verify {
            for layer in 0..self.config.layer_kinds.len() {
                let Some(state) = self.seq.device().gdn.layer(layer) else {
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
            let Some(selection) = &self.selection else {
                return invalid("a quantum that scores logits has no selection words".to_string());
            };
            let wanted = if self.kind == QuantumKind::Verify {
                self.logits_rows
            } else {
                1
            };
            if wanted > selection.capacity() {
                return invalid(format!(
                    "{wanted} selected rows exceed the {} the selection holds",
                    selection.capacity()
                ));
            }
        }
        if self.draft_token_write.is_some() && (self.selection.is_none() || self.logits_rows == 0) {
            return invalid("a drafted token is passed on from a selection, and the quantum has none".to_string());
        }
        Ok(())
    }

    /// The allocation serial of every buffer the quantum's launches read or write, in a fixed order.
    pub fn buffer_generations(&self) -> Vec<AllocationId> {
        let mut generations: Vec<AllocationId> = self.workspace.generations().collect();
        self.seq.device().generations(&mut generations);
        if let Some(selection) = &self.selection {
            generations.push(selection.generation());
        }
        // Only the cache the quantum's attention reads: the MTP cache growing does not change a base quantum's graph.
        if let Some(view) = self.attention_source().and_then(|source| self.seq.kv(source)) {
            view.generations(&mut generations);
        }
        generations
    }

    /// The key of the graph the quantum would be captured into, or `None` when it is not captured: prefill, more than
    /// eight rows, exact numerics, rows that seed from the sequence's seed rows (found only when the stage runs), and a
    /// quantum whose reservation had to allocate pages or upload a page table (it ran work a graph cannot repeat; the
    /// next quantum is captured).
    pub fn capture_key(&self) -> Option<CaptureKey> {
        let view = self.view();
        if !view.capturable() || self.rows > MAX_CAPTURED_ROWS {
            return None;
        }
        if self.lanes.get(LaneId(0))?.library().exact_numerics() {
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
        let (start, rows) = (self.input.start_position(), self.rows as u64);
        if let Some(source) = self.attention_source() {
            let view = self.seq.kv(source)?;
            if view.grew_for(start, rows) || !view.reserves(start, rows) {
                return None;
            }
        }
        let group = (self.config.num_attention_heads / self.config.num_key_value_heads.max(1)) as u64;
        let host_pages = |source| self.seq.kv(source).map_or(0, |view| view.reserved_host_pages());
        let source = self.attention_source();
        let slot = |slot: usize| u8::try_from(slot).ok();
        Some(CaptureKey {
            view,
            kind: self.kind,
            rows: self.rows,
            logits: self.logits_rows,
            selects: self.selection.is_some(),
            seeds_draft: self.seeds_draft,
            draft_seed,
            draft_read: self.draft_token_read.and_then(slot),
            draft_write: self
                .draft_token_write
                .and_then(|write| Some((slot(write.slot)?, write.ids.get()))),
            writes_current_token: self.writes_current_token,
            input_record: self.record as u8,
            token_from: self.token_from.as_ref().map(|selection| selection.generation()),
            row_exact: self.mode.row_exact,
            split_geometry: split_geometry(start, rows, group),
            reserved_host_pages: keyed_host_pages(source, host_pages(KvSource::Base), host_pages(KvSource::Mtp)),
            replay_rows: capture_key::trimmed(self.seq.replay_layers()),
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

/// The record stage of a quantum that records: beside the quantum's stages it captures the same launches on the capture
/// lane, in the order of the view, and keeps the graph in the quantum's slot. It has no edge to a stage: the capture
/// submits nothing, so it neither waits for the quantum's device work nor delays it, and the quantum stays alive until
/// it is done. Other quanta record nothing.
struct RecordStage {
    launches: Arc<[Arc<dyn StageLaunch>]>,
    lane: LaneId,
}

/// Why a recording failed.
enum RecordFailure {
    /// A stage refused the quantum.
    Stage(StageError),
    /// The device could not capture or instantiate.
    Cuda(CudaError),
}

impl From<CudaError> for RecordFailure {
    fn from(error: CudaError) -> Self {
        RecordFailure::Cuda(error)
    }
}

impl RecordStage {
    /// Records the launches of every stage that has work, in the order of the view, on `lane`.
    fn record(&self, quantum: &Quantum, lane: &CudaLane, mode: SubmitMode) -> Result<GraphExec, RecordFailure> {
        graph_cache::record_on(lane, mode, |submission| {
            for launch in self.launches.iter() {
                if launch.idle(quantum) {
                    continue;
                }
                launch.launch(quantum, submission).map_err(RecordFailure::Stage)?;
            }
            Ok(())
        })
    }

    /// The hash the stages of `quantum` submit in check mode.
    #[cfg(feature = "test-oracles")]
    fn hash(&self, quantum: &Quantum, lane: &CudaLane, mode: SubmitMode) -> Result<u64, RecordFailure> {
        let (outcome, hash) = Submission::hashed(lane, mode, |submission| {
            for launch in self.launches.iter() {
                if launch.idle(quantum) {
                    continue;
                }
                launch.launch(quantum, submission)?;
            }
            Ok::<(), StageError>(())
        })?;
        outcome.map_err(RecordFailure::Stage)?;
        Ok(hash)
    }
}

impl Stage<Quantum> for RecordStage {
    fn run(&self, quantum: &Quantum, _cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let Disposition::Record(slot) = &quantum.disposition else {
            return Ok(Submitted::done());
        };
        let Some(lane) = quantum.lanes.get(self.lane) else {
            slot.release();
            return Err(fatal(format!(
                "the capture lane {} is not in the lane table",
                self.lane.0
            )));
        };
        let mode = SubmitMode {
            // The graph's launches overlap at every position (programmatic dependent launch).
            pdl: true,
            row_exact: quantum.mode.row_exact,
        };
        match self.record(quantum, lane, mode) {
            Ok(graph) => {
                #[cfg(feature = "test-oracles")]
                if let Ok(hash) = self.hash(quantum, lane, mode) {
                    slot.set_recorded_hash(hash);
                }
                slot.publish(graph);
            }
            // The quantum runs its stages whatever the recording does; a stage that refuses it fails the quantum there.
            Err(RecordFailure::Stage(error)) => {
                log::warn!("a quantum's graph was not recorded, a stage refused it: {error:?}");
                slot.release();
            }
            // The lane is held by another recording: this key is tried again at its next sight.
            Err(RecordFailure::Cuda(CudaError::LaneBusy(_))) => slot.release(),
            Err(RecordFailure::Cuda(error)) => {
                log::warn!("a quantum's graph could not be recorded, its key runs stage by stage: {error}");
                slot.abandon();
            }
        }
        Ok(Submitted::done())
    }

    fn abandoned(&self, quantum: &Quantum) {
        if let Disposition::Record(slot) = &quantum.disposition {
            slot.release();
        }
    }
}

/// The one stage of a replayed quantum's graph: it launches the quantum's recorded graph on the home lane and ends
/// with a signalled marker.
struct ReplayStage {
    launches: Arc<[Arc<dyn StageLaunch>]>,
}

impl Stage<Quantum> for ReplayStage {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let Disposition::Replay(slot) = &quantum.disposition else {
            return Err(fatal(
                "a replay stage was started with a quantum that has no recorded graph",
            ));
        };
        let Some(graph) = slot.graph() else {
            return Err(fatal("a replayed quantum's graph is gone"));
        };
        quantum.check_prepared()?;
        let id = cx.lane.ok_or_else(|| fatal("the graph launch was given no lane"))?;
        let lane = quantum
            .lanes
            .get(id)
            .ok_or_else(|| fatal(format!("lane {} is not in the lane table", id.0)))?;
        // A graph does not wait: the markers the quantum depends on are waited for before it launches.
        let waits = latest_per_lane(
            cx.waits
                .iter()
                .copied()
                .chain(self.launches.iter().flat_map(|launch| launch.waits(quantum))),
            Some(id),
        );
        let submission = match Submission::begin(lane, quantum.mode) {
            Ok(submission) => submission,
            Err(CudaError::LaneBusy(_)) => return Err(StageError::Retry),
            Err(error) => return Err(fatal(error)),
        };
        for marker in waits {
            submission.wait(marker).map_err(fatal)?;
        }
        graph.launch(&submission).map_err(fatal)?;
        slot.launched();
        let marker = submission.finish_signalled().map_err(fatal)?;
        Ok(Submitted::marker(marker))
    }
}

/// The stage graph of a replayed quantum: one stage on the home lane.
fn replay_spec() -> StageGraphSpec {
    let mut spec = StageGraphSpec::new();
    spec.stage(StageSpec {
        name: "graph",
        ordered: None,
        lanes: vec![LaneId(0)],
    });
    spec
}

/// A started quantum: its reference (which pins it, see [`QuantumRef`]), and the stages of its graph that a later
/// quantum of the chain is ordered by.
#[derive(Clone, Debug)]
pub struct Started {
    pub reference: QuantumRef,
    /// The stage where the quantum enters its graph.
    pub root: StageId,
    /// The stage whose marker covers the quantum's device work.
    pub sink: StageId,
    /// Whether the quantum is one launch of a recorded graph.
    pub replayed: bool,
    /// Whether the quantum's launches are being recorded beside its stages.
    pub recording: bool,
}

/// What stops a quantum from starting.
#[derive(Debug, Error)]
pub enum StartError {
    #[error(transparent)]
    Quantum(#[from] QuantumError),
    #[error("the plan has no graph for {0:?}")]
    NoView(View),
}

/// The stage graphs of a plan on a runtime, and the choice of the graph a quantum runs.
pub struct Executor {
    plan: Arc<Plan>,
    eager: Vec<Option<Arc<StageGraph<Quantum>>>>,
    replay: Vec<Option<Arc<StageGraph<Quantum>>>>,
}

impl fmt::Debug for Executor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Executor").finish_non_exhaustive()
    }
}

impl Executor {
    /// The graphs of every view of `plan`, run on `runtime` (whose lane pool has the plan's
    /// [`Plan::fan_out_width`] lanes, then `capture`, a lane that only recordings use, and lanes the owner uses). The
    /// graphs of the views that are captured have a record stage on `capture`. `retire` receives each quantum, with
    /// its outcome, once its device work has completed.
    pub fn new(
        plan: Arc<Plan>,
        runtime: &GraphRuntime,
        capture: LaneId,
        retire: impl Fn(Quantum, QuantumOutcome) + Send + Sync + 'static,
    ) -> Result<Executor, GraphError> {
        let retire: Arc<dyn Fn(Quantum, QuantumOutcome) + Send + Sync> = Arc::new(retire);
        let mut eager = Vec::with_capacity(Variant::ALL.len());
        let mut replay = Vec::with_capacity(Variant::ALL.len());
        for variant in Variant::ALL {
            let view = variant.representative();
            let (Some(mut spec), Some(entries)) = (plan.spec(view), plan.stages(view)) else {
                eager.push(None);
                replay.push(None);
                continue;
            };
            let mut stages: Vec<Arc<dyn Stage<Quantum>>> =
                entries.iter().map(|entry| Arc::clone(&entry.stage)).collect();
            let launches: Arc<[Arc<dyn StageLaunch>]> = entries.iter().map(|entry| Arc::clone(&entry.launch)).collect();
            if view.capturable() {
                spec.stage(StageSpec {
                    name: "record",
                    ordered: None,
                    lanes: vec![capture],
                });
                stages.push(Arc::new(RecordStage {
                    launches: Arc::clone(&launches),
                    lane: capture,
                }));
            }
            let again = Arc::clone(&retire);
            eager.push(Some(StageGraph::new(
                spec,
                stages,
                runtime.clone(),
                move |quantum, outcome| again(quantum, outcome),
            )?));
            replay.push(if view.capturable() {
                let again = Arc::clone(&retire);
                Some(StageGraph::new(
                    replay_spec(),
                    vec![Arc::new(ReplayStage { launches }) as Arc<dyn Stage<Quantum>>],
                    runtime.clone(),
                    move |quantum, outcome| again(quantum, outcome),
                )?)
            } else {
                None
            });
        }
        Ok(Executor { plan, eager, replay })
    }

    /// The plan the executor runs.
    pub fn plan(&self) -> &Arc<Plan> {
        &self.plan
    }

    /// Starts `quantum`: validates it, asks `cache` whether to record or replay it, and runs it on the graph that
    /// follows. With `after`, the previous quantum of the chain, the new quantum's root waits on the device for the
    /// previous one's sink (the chain's token passes from one quantum's selection to the next's embedding, so nothing
    /// of the next could overlap it). Keep the previous [`Started`] until this call returns, then drop it.
    pub fn start(
        &self,
        cache: &mut GraphCache,
        mut quantum: Quantum,
        kill: Arc<AtomicBool>,
        after: Option<&Started>,
    ) -> Result<Started, StartError> {
        quantum.validate()?;
        let view = quantum.view();
        let variant = view.variant();
        let eager = self.eager[variant.index()].as_ref().ok_or(StartError::NoView(view))?;
        let stages = self.plan.stages(view).map_or(0, <[_]>::len);
        let stage_by_stage = StageId((stages - 1) as u16);
        let (graph, replayed, recording, sink) =
            match (cache.admit(quantum.capture_key()), &self.replay[variant.index()]) {
                (Admission::Replay(slot), Some(graph)) => {
                    quantum.disposition = Disposition::Replay(slot);
                    (graph, true, false, StageId(0))
                }
                (Admission::Record(slot), _) => {
                    quantum.disposition = Disposition::Record(slot);
                    (eager, false, true, stage_by_stage)
                }
                _ => (eager, false, false, stage_by_stage),
            };
        let external = after
            .map(|previous| ExternalEdge {
                from: previous.reference.clone(),
                from_stage: previous.sink,
                to_stage: StageId(0),
                kind: EdgeKind::Submission,
            })
            .into_iter()
            .collect();
        let reference = graph.start(quantum, kill, external);
        Ok(Started {
            reference,
            root: StageId(0),
            sink,
            replayed,
            recording,
        })
    }

    /// The test oracle: runs the stages of `quantum` in the C library's check mode on `lane` (nothing is submitted;
    /// every launch and copy is hashed) and returns the hash with the one the quantum's graph was recorded from.
    /// They are equal when the graph still repeats what the stages would submit now; a key that omitted something a
    /// launch bakes in shows here as a difference. The quantum must have been admitted to a recorded graph.
    #[cfg(feature = "test-oracles")]
    pub fn check_replay_for_tests(&self, quantum: &Quantum, lane: &CudaLane) -> Result<ReplayCheck, StartError> {
        let (Disposition::Record(slot) | Disposition::Replay(slot)) = &quantum.disposition else {
            return Err(StartError::Quantum(QuantumError::Invalid {
                what: "the quantum was not admitted to a graph".to_string(),
            }));
        };
        Ok(ReplayCheck {
            recorded: slot.recorded_hash(),
            now: self.stages_hash_for_tests(quantum, lane)?,
        })
    }

    /// The hash of what the stages of `quantum` would submit, in the C library's check mode on `lane`: nothing is
    /// submitted.
    #[cfg(feature = "test-oracles")]
    pub fn stages_hash_for_tests(&self, quantum: &Quantum, lane: &CudaLane) -> Result<u64, StartError> {
        let view = quantum.view();
        let entries = self.plan.stages(view).ok_or(StartError::NoView(view))?;
        let mode = SubmitMode {
            pdl: true,
            row_exact: quantum.mode.row_exact,
        };
        let (outcome, hash) = Submission::hashed(lane, mode, |submission| {
            for entry in entries {
                if !entry.launch.idle(quantum) {
                    entry.launch.launch(quantum, submission)?;
                }
            }
            Ok::<(), StageError>(())
        })
        .map_err(|error| {
            StartError::Quantum(QuantumError::Invalid {
                what: error.to_string(),
            })
        })?;
        outcome.map_err(|error| {
            StartError::Quantum(QuantumError::Invalid {
                what: format!("{error:?}"),
            })
        })?;
        Ok(hash)
    }
}

/// The two hashes of [`Executor::check_replay_for_tests`].
#[cfg(feature = "test-oracles")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayCheck {
    /// What the graph was recorded from.
    pub recorded: u64,
    /// What the stages would submit for the quantum now.
    pub now: u64,
}

#[cfg(feature = "test-oracles")]
impl ReplayCheck {
    pub fn matches(&self) -> bool {
        self.recorded != 0 && self.recorded == self.now
    }
}
