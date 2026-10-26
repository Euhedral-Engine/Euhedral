//! The stages of a quantum: one `Stage<Quantum>` per layer operation (`model/qwen38/Stages.java`).
//!
//! A stage is built once with what does not change from quantum to quantum (its weights, the buffer slots it reads
//! and writes, its layer). Its work is its [`Launch::launch`]: the kernel launches and copies of one quantum on a
//! submission it is given, and nothing else (no begin, no finish, no waits, no marker, no signal), so that a graph
//! capture can record it. The one generic [`run_eager`] is the stage for eager submission: it takes the lane the
//! graph chose, orders itself behind the markers of its predecessors on other lanes, calls `launch`, and returns the
//! marker of that work. It never waits.
//!
//! The geometry of the quantum (rows, start position, kind, the model's dimensions) comes from the [`Quantum`] it is
//! given; the buffers come from the quantum's workspace. A stage mutates nothing the quantum does not own: the
//! caches it reads are [`KvView`]s copied at admission, the GDN and scratch buffers are the sequence's shared
//! device part, and the rows to replay are a value of the quantum.
//!
//! What is not a stage's: the reservation of KV pages and the growth of page tables, the frontiers of the caches
//! (`appended`, `commit`), the upload of the input record, and the replay rows a verification leaves for the next
//! quantum. They belong to the sequence's owner, who sets them on the quantum at admission.

pub mod attention;
pub mod elementwise;
pub mod embed;
pub mod gdn;
pub mod linear;
pub mod mtp;
pub mod norm;
pub mod sampling;

use std::fmt::Display;
use std::sync::Arc;

use euhedral_graph::{LaneId, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{AllocationId, CudaError, CudaLane, DevicePtr, Submission, SubmitMode};
use thiserror::Error;

use crate::model::ModelConfig;
use crate::state::{KvView, SequenceDevice};
use crate::workspace::{BufferSlot, Workspace, WorkspaceError};
use attention::KvSource;
use sampling::Selection;

/// What a quantum is, which sets the routes of its stages (`Quantum.ExecutionKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantumKind {
    /// Rows of the prompt, in chunks.
    Prefill,
    /// One row of the sequence.
    Decode,
    /// Speculative verification: decode's topology over several rows, each computed bit for bit as one-row decode
    /// at its position (row-exact), with logits for every row.
    Verify,
    /// Drafting: rows of the MTP layer at MTP positions, writing only the MTP layer's cache.
    Draft,
}

/// Where a quantum's input record sits in the workspace: the start position of its rows. The record itself (the
/// token ids as 32-bit values, then the start position as a 64-bit value at the next 8-byte boundary) is uploaded
/// by the quantum's admission into the workspace's `TokenIds` slot (`Workspace.inputByteSize`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputRecordRef {
    start_position: u64,
}

impl InputRecordRef {
    /// The record of rows starting at `start_position`.
    pub fn new(start_position: u64) -> Self {
        Self { start_position }
    }

    /// The position of the quantum's first row.
    pub fn start_position(&self) -> u64 {
        self.start_position
    }

    /// Bytes from the start of the record to the start position: `rows` token ids, padded to 8 bytes.
    pub fn position_offset(rows: usize) -> u64 {
        (rows as u64 * 4).div_ceil(8) * 8
    }
}

/// A source of the rows an MTP draft quantum normalizes beside its embedding rows (`Quantum.draftSeedAddress`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftSeed {
    /// Hidden rows at this device address (the previous MTP row's hidden, for a recursive draft).
    Address(DevicePtr),
    /// The sequence's draft seed rows, from this row: the base model's post-final-norm hidden rows, which a
    /// catch-up reads after the quantum that wrote them.
    SeedRow(u32),
}

/// Where a draft quantum's selected token goes: the sequence's step tokens, so that the next quantum of the step
/// reads it on the device (`Quantum.writingDraftToken`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftTokenWrite {
    /// The slot of the step's tokens (1 and up; 0 is the step's current token).
    pub slot: usize,
    /// The draft head's row ids (int32), mapping a row of the head to its token; null when a row is the token.
    pub ids: DevicePtr,
}

/// The device lanes of the engine's `LanePool`, by lane id: the stage runs on the lane the graph chose, which is a
/// position in the pool.
#[derive(Clone, Debug)]
pub struct LaneTable {
    lanes: Arc<[Arc<CudaLane>]>,
}

impl LaneTable {
    /// The table of `lanes`, which must be in lane id order starting at 0, as the `LanePool` holds them.
    pub fn new(lanes: Vec<Arc<CudaLane>>) -> Result<Self, CudaError> {
        for (index, lane) in lanes.iter().enumerate() {
            if usize::from(lane.id().0) != index {
                return Err(CudaError::Argument(format!(
                    "lane {} stands at position {index} of the lane table; the table is in lane id order",
                    lane.id().0
                )));
            }
        }
        Ok(Self { lanes: lanes.into() })
    }

    /// The lane `id`.
    pub fn get(&self, id: LaneId) -> Option<&CudaLane> {
        self.lanes.get(usize::from(id.0)).map(|lane| &**lane)
    }
}

/// The sequence a quantum belongs to, as its stages read it: the sequence's shared device part (the GDN state and the
/// attention scratch), a view of each cache as it was at admission, and the rows each GDN layer replays first.
#[derive(Clone, Debug)]
pub struct SequenceRefs {
    device: Arc<SequenceDevice>,
    kv: Option<KvView>,
    mtp_kv: Option<KvView>,
    replay_rows: Vec<u32>,
}

impl SequenceRefs {
    /// The references of a quantum of the sequence whose shared device part is `device`, with no cache and nothing
    /// to replay.
    pub fn new(device: Arc<SequenceDevice>) -> Self {
        Self {
            device,
            kv: None,
            mtp_kv: None,
            replay_rows: Vec::new(),
        }
    }

    /// The base model's cache as it is at admission (`KvCache::view`, after the reservation).
    pub fn with_kv(mut self, view: KvView) -> Self {
        self.kv = Some(view);
        self
    }

    /// The MTP layer's cache as it is at admission.
    pub fn with_mtp_kv(mut self, view: KvView) -> Self {
        self.mtp_kv = Some(view);
        self
    }

    /// The rows each GDN layer replays from its checkpoint before its own rows, by model layer index: what the
    /// last verification ran past the rows it accepted. Layers beyond the vector replay nothing.
    pub fn with_replay_rows(mut self, rows: Vec<u32>) -> Self {
        self.replay_rows = rows;
        self
    }

    /// The sequence's shared device part.
    pub fn device(&self) -> &SequenceDevice {
        &self.device
    }

    /// The view of the cache `source`, if the quantum has one.
    pub fn kv(&self, source: KvSource) -> Option<&KvView> {
        match source {
            KvSource::Base => self.kv.as_ref(),
            KvSource::Mtp => self.mtp_kv.as_ref(),
        }
    }

    /// The rows the GDN layer `layer` replays first.
    pub fn replay_rows(&self, layer: usize) -> u32 {
        self.replay_rows.get(layer).copied().unwrap_or(0)
    }
}

/// Why a quantum was refused.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum QuantumError {
    #[error(transparent)]
    Workspace(#[from] WorkspaceError),
    #[error("a {kind:?} quantum runs row-exact: {expected}, not {given}")]
    RowExact {
        kind: QuantumKind,
        expected: bool,
        given: bool,
    },
}

/// One quantum, as the stages see it. Its fields are `pub(crate)` and are set through constructors, so that the
/// planning that builds quanta can add to them without breaking the stages.
#[derive(Debug)]
pub struct Quantum {
    pub(crate) kind: QuantumKind,
    pub(crate) rows: usize,
    pub(crate) input: InputRecordRef,
    /// The workspace the quantum's rows live in. Its view of `rows` rows was checked when the quantum was made.
    pub(crate) workspace: Arc<Workspace>,
    pub(crate) seq: SequenceRefs,
    pub(crate) mode: SubmitMode,
    pub(crate) config: Arc<ModelConfig>,
    pub(crate) lanes: LaneTable,
    /// Logits rows the quantum scores: the rows of its final norm, its head and its selection.
    pub(crate) logits_rows: usize,
    /// Whether the base rows' post-final-norm hidden also seed the sequence's drafting.
    pub(crate) seeds_draft: bool,
    pub(crate) draft_seed: Option<DraftSeed>,
    pub(crate) draft_token_write: Option<DraftTokenWrite>,
    /// The slot of the step's tokens this quantum's rows read their token ids from, instead of the input record.
    pub(crate) draft_token_read: Option<usize>,
    /// Whether the quantum writes its last input token to slot 0 of the step's tokens as the step's current token.
    pub(crate) writes_current_token: bool,
    /// The device words and the pinned readback the quantum's argmax fills.
    pub(crate) selection: Option<Arc<Selection>>,
}

impl Quantum {
    /// A quantum of `rows` rows of `kind` starting at `input`'s position, over `workspace` and `seq`, whose stages
    /// submit with `mode` on the lanes of `lanes`. It scores one row (decode, and the last row of a prefill); see
    /// [`Quantum::with_logits_rows`]. A verification is row-exact and nothing else is (as `Quantum` in Java): `mode`
    /// must say so. Refused when the workspace was sized for fewer rows.
    #[allow(
        clippy::too_many_arguments,
        reason = "the fields the plan fixes for a quantum; a builder would only rename them"
    )]
    pub fn new(
        kind: QuantumKind,
        rows: usize,
        input: InputRecordRef,
        workspace: Arc<Workspace>,
        seq: SequenceRefs,
        mode: SubmitMode,
        config: Arc<ModelConfig>,
        lanes: LaneTable,
    ) -> Result<Self, QuantumError> {
        workspace.view(rows as u64)?;
        let exact = kind == QuantumKind::Verify;
        if mode.row_exact != exact {
            return Err(QuantumError::RowExact {
                kind,
                expected: exact,
                given: mode.row_exact,
            });
        }
        let logits_rows = if kind == QuantumKind::Verify { rows } else { 1 };
        workspace.rows().check_logits(logits_rows as u64)?;
        Ok(Self {
            kind,
            rows,
            input,
            workspace,
            seq,
            mode,
            config,
            lanes,
            logits_rows,
            seeds_draft: false,
            draft_seed: None,
            draft_token_write: None,
            draft_token_read: None,
            writes_current_token: false,
            selection: None,
        })
    }

    /// Scores `rows` logits rows instead (0: none, a prefill chunk that is not the last).
    pub fn with_logits_rows(mut self, rows: usize) -> Result<Self, WorkspaceError> {
        self.workspace.rows().check_logits(rows as u64)?;
        self.logits_rows = rows;
        Ok(self)
    }

    /// The base rows' post-final-norm hidden also seed the sequence's drafting.
    pub fn seeding_draft(mut self) -> Self {
        self.seeds_draft = true;
        self
    }

    /// The rows an MTP draft quantum normalizes beside its embedding rows.
    pub fn with_draft_seed(mut self, seed: DraftSeed) -> Self {
        self.draft_seed = Some(seed);
        self
    }

    /// The selected token of a draft quantum goes to the step's tokens.
    pub fn with_draft_token_write(mut self, write: DraftTokenWrite) -> Self {
        self.draft_token_write = Some(write);
        self
    }

    /// The rows read their token ids from the step's tokens on the device, from `slot` (the tokens the quanta before
    /// this one wrote), not from the input record.
    pub fn reading_draft_tokens(mut self, slot: usize) -> Self {
        self.draft_token_read = Some(slot);
        self
    }

    /// The quantum's last input token is written to slot 0 of the step's tokens as the step's current token.
    pub fn writing_current_token(mut self) -> Self {
        self.writes_current_token = true;
        self
    }

    /// The device words and readback the quantum's argmax fills.
    pub fn with_selection(mut self, selection: Arc<Selection>) -> Self {
        self.selection = Some(selection);
        self
    }

    pub fn kind(&self) -> QuantumKind {
        self.kind
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The mode its submissions run in (row-exact exactly for a verification).
    pub fn mode(&self) -> SubmitMode {
        self.mode
    }

    pub fn start_position(&self) -> u64 {
        self.input.start_position
    }

    pub fn logits_rows(&self) -> usize {
        self.logits_rows
    }

    /// The allocation serial of layer `layer`'s page table in the cache `source` as of admission: part of the key
    /// of a captured graph that reads the table.
    pub fn table_generation(&self, source: KvSource, layer: usize) -> Option<AllocationId> {
        self.seq.kv(source)?.table_generation(layer)
    }

    /// The words and readback the quantum's argmax fills, if it selects.
    pub fn selection(&self) -> Option<&Arc<Selection>> {
        self.selection.as_ref()
    }

    /// The device address of `slot`.
    pub(crate) fn slot(&self, slot: BufferSlot) -> DevicePtr {
        self.workspace.buffer(slot).ptr()
    }

    /// The device address of the start position in the input record.
    pub(crate) fn position(&self) -> DevicePtr {
        self.slot(BufferSlot::TokenIds)
            .offset(InputRecordRef::position_offset(self.rows))
    }

    /// The epsilon of the norms, as the kernels take it.
    pub(crate) fn epsilon(&self) -> f32 {
        self.config.rms_norm_epsilon as f32
    }

    /// Rows as the launches take them.
    pub(crate) fn rows_u32(&self) -> Result<u32, StageError> {
        u32::try_from(self.rows).map_err(|_| fatal(format!("{} rows exceed a launch", self.rows)))
    }
}

/// A stage failure that retrying cannot fix.
pub(crate) fn fatal(why: impl Display) -> StageError {
    StageError::Fatal(why.to_string())
}

/// Narrows a size to the 32 bits a launch takes.
pub(crate) fn launch_u32(value: usize, what: &str) -> Result<u32, StageError> {
    u32::try_from(value).map_err(|_| fatal(format!("{what} {value} exceeds a launch")))
}

/// The launches of a stage for one quantum.
pub trait Launch: Send + Sync {
    /// Whether the stage has nothing to queue for `quantum` (a norm of no logits rows, a selection that scores
    /// nothing). An idle stage queues nothing and leaves no marker, so its successors wait for nothing.
    fn idle(&self, quantum: &Quantum) -> bool {
        let _ = quantum;
        false
    }

    /// Queues the stage's kernel launches and copies on `submission`, for `quantum`. Nothing else: no wait for a
    /// marker, no marker, no completion signal, no synchronization, no allocation, so the launches can be captured
    /// into a graph. It reads, and does not change, anything the quantum shares.
    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError>;

    /// Whether the marker of this stage's work must also write the lane's completion word, so that a completion
    /// edge after it is found by a poll of the lane. Eager submission finishes the submission signalled
    /// ([`run_eager`]); a graph launch is followed by the same finish on the submission that launched it.
    const SIGNALLED: bool = false;
}

/// The stage for eager submission: begins a submission of the quantum's mode on the lane the graph chose, waits for
/// the markers of the predecessors on other lanes, calls [`Launch::launch`], and finishes with the marker of the
/// work (signalled for a stage whose [`Launch::SIGNALLED`] is set).
///
/// A lane held by another submission answers [`StageError::Retry`], before anything was queued; every other failure
/// is fatal to the quantum.
pub fn run_eager<L: Launch>(stage: &L, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
    if stage.idle(quantum) {
        return Ok(Submitted::done());
    }
    let id = cx
        .lane
        .ok_or_else(|| fatal("a stage with device work was given no lane"))?;
    let lane = quantum
        .lanes
        .get(id)
        .ok_or_else(|| fatal(format!("lane {} is not in the lane table", id.0)))?;
    let submission = match Submission::begin(lane, quantum.mode) {
        Ok(submission) => submission,
        Err(CudaError::LaneBusy(_)) => return Err(StageError::Retry),
        Err(error) => return Err(fatal(error)),
    };
    for marker in cx.waits {
        submission.wait(*marker).map_err(fatal)?;
    }
    stage.launch(quantum, &submission)?;
    let marker = if L::SIGNALLED {
        submission.finish_signalled()
    } else {
        submission.finish()
    }
    .map_err(fatal)?;
    Ok(Submitted::marker(marker))
}

/// Makes `$stage` a `Stage<Quantum>` that runs eagerly ([`run_eager`]).
macro_rules! eager {
    ($($stage:ty),+ $(,)?) => {$(
        impl Stage<Quantum> for $stage {
            fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
                super::run_eager(self, quantum, cx)
            }
        }
    )+};
}
pub(crate) use eager;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_position_follows_the_token_ids_padded_to_eight_bytes() {
        assert_eq!(InputRecordRef::position_offset(1), 8);
        assert_eq!(InputRecordRef::position_offset(2), 8);
        assert_eq!(InputRecordRef::position_offset(3), 16);
        assert_eq!(InputRecordRef::position_offset(512), 2048);
    }

    #[test]
    fn a_quantum_is_what_a_stage_graph_runs() {
        fn runs_in_a_graph<Q: Send + Sync + 'static>() {}
        runs_in_a_graph::<Quantum>();
    }
}
