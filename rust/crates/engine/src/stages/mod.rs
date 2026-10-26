//! The stages of a quantum: one `Stage<Quantum>` per layer operation (`model/qwen38/Stages.java`).
//!
//! A stage is built once with what does not change from quantum to quantum (its weights, the buffer slots it reads
//! and writes, its layer), and runs once per quantum: it takes the lane the graph chose, orders itself behind the
//! markers of its predecessors on other lanes, queues its launches and returns the marker of that work. It never
//! waits. The geometry of the quantum (rows, start position, kind, the model's dimensions) comes from the
//! [`Quantum`] it is given; the buffers come from the quantum's workspace.
//!
//! What is not a stage's: the reservation of KV pages and the upload of the input record (the quantum's admission),
//! the frontiers of the caches (`appended`, `commit`: the quantum's admission and retirement), and the replay a
//! verification leaves for the next quantum (set by the sequence's owner once acceptance is known). A stage only
//! reads those.

pub mod attention;
pub mod elementwise;
pub mod gdn;
pub mod linear;
pub mod mtp;
pub mod norm;
pub mod sampling;

use std::fmt::Display;
use std::sync::Arc;

use euhedral_graph::{LaneId, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{CudaError, CudaLane, DevicePtr, Submission, SubmitMode};

use crate::model::ModelConfig;
use crate::state::SequenceState;
use crate::workspace::{BufferSlot, Workspace, WorkspaceError};
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

/// The sequence a quantum belongs to: its device state, which the stages read (the KV page tables, the GDN buffers,
/// the attention scratch). The frontiers and the replay rows change between quanta, under the sequence's owner.
#[derive(Clone, Debug)]
pub struct SequenceRefs {
    state: Arc<SequenceState>,
}

impl SequenceRefs {
    /// The references of a quantum of the sequence whose state is `state`.
    pub fn new(state: Arc<SequenceState>) -> Self {
        Self { state }
    }

    /// The sequence's device state.
    pub fn state(&self) -> &SequenceState {
        &self.state
    }
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
    /// The device words and the pinned readback the quantum's argmax fills.
    pub(crate) selection: Option<Arc<Selection>>,
}

impl Quantum {
    /// A quantum of `rows` rows of `kind` starting at `input`'s position, over `workspace` and `seq`, whose stages
    /// submit with `mode` on the lanes of `lanes`. It scores one row (decode, and the last row of a prefill); see
    /// [`Quantum::with_logits_rows`]. Refused when the workspace was sized for fewer rows.
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
    ) -> Result<Self, WorkspaceError> {
        workspace.view(rows as u64)?;
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

    pub fn start_position(&self) -> u64 {
        self.input.start_position
    }

    pub fn logits_rows(&self) -> usize {
        self.logits_rows
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

/// Runs `work` on the lane the graph chose, behind the markers its predecessors left on other lanes, and returns
/// the marker of what it queued. With `signalled` the marker's completion word is written too, which is what a
/// completion edge's successor is found by (`Submission::finish_signalled`).
///
/// A lane held by another submission answers [`StageError::Retry`], before anything was queued; every other
/// failure is fatal to the quantum.
pub(crate) fn submit(
    quantum: &Quantum,
    cx: &StageCx<'_>,
    signalled: bool,
    work: impl FnOnce(&Submission<'_>) -> Result<(), StageError>,
) -> Result<Submitted, StageError> {
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
    work(&submission)?;
    let marker = if signalled {
        submission.finish_signalled()
    } else {
        submission.finish()
    }
    .map_err(fatal)?;
    Ok(Submitted::marker(marker))
}

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
