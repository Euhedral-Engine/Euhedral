//! The stages of a quantum: one `Stage<Quantum>` per layer operation (`model/qwen38/Stages.java`).
//!
//! A stage is built once with what does not change from quantum to quantum (its weights, the buffer slots it reads
//! and writes, its layer). Its work is its [`Launch::launch`]: the kernel launches and copies of one quantum on the
//! [`KernelOps`] it is given, and nothing else (no begin, no finish, no waits, no marker, no signal), so that a
//! graph capture can record it. The one generic [`run_eager`] is the stage for eager submission: it takes the lane the
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
pub mod transfer;
pub mod unmarked;

use std::fmt::{self, Display};
use std::iter::repeat_n;
use std::mem;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use euhedral_gpu::cuda::CudaQueue;
use euhedral_gpu::{GpuError, Lane};
use euhedral_graph::{DeviceQueue, LaneId, LanePool, Marker, QuantumOutcome, Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::launch::Modes;
use euhedral_inference_cuda::launch::load::KernelTable;
use euhedral_inference_cuda::launch::ops::RusteeOps;
use euhedral_inference_cuda::{AllocationId, CudaLibrary, DevicePtr, KernelOps, LaneOps, PooledPinned, SubmitMode};
use smallvec::{SmallVec, smallvec};
use thiserror::Error;

use crate::error::EngineError;
use crate::model::ModelConfig;
use crate::quantum::Capture;
use crate::state::attention::DRAFT_TOKEN_SLOTS;
use crate::state::{FlightGuard, KvView, SequenceDevice, SequenceId};
use crate::workspace::{BufferSlot, Workspace, WorkspaceError, input_record_bytes};
use attention::KvSource;
use euhedral_inference_settings::KvFormat;
use sampling::{RowReadback, Selection, SelectionLease};
pub use unmarked::Unmarked;

/// What a quantum is, which sets the routes of its stages (`Quantum.ExecutionKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

/// The device lanes the stage graphs run on, by lane id: the typed pool of RustEE's shared lanes (the stage runs on the
/// lane the graph chose, which is a position in the pool), the kernels loaded for them, and the C library whose process
/// state the launches' modes come from.
#[derive(Clone, Debug)]
pub struct LaneTable {
    lanes: Arc<LanePool<CudaQueue>>,
    kernels: Arc<KernelTable>,
    library: Arc<CudaLibrary>,
}

impl LaneTable {
    /// The table of `lanes`, whose kernels `kernels` launch.
    pub fn new(lanes: Arc<LanePool<CudaQueue>>, kernels: Arc<KernelTable>, library: Arc<CudaLibrary>) -> Self {
        Self {
            lanes,
            kernels,
            library,
        }
    }

    /// The lane `id`.
    pub fn get(&self, id: LaneId) -> Option<&Arc<CudaQueue>> {
        (usize::from(id.0) < self.lanes.len()).then(|| self.lanes.lane(id))
    }

    /// The number of lanes.
    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    /// Whether the table has no lane.
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    /// The loaded kernels.
    pub fn kernels(&self) -> &Arc<KernelTable> {
        &self.kernels
    }

    /// The C library.
    pub fn library(&self) -> &Arc<CudaLibrary> {
        &self.library
    }

    /// The modes a submission of the table's lanes launches in: `row_exact` as the quantum says, exact numerics as the
    /// library has them selected for the process.
    pub fn modes(&self, row_exact: bool) -> Modes {
        Modes {
            row_exact,
            exact_numerics: self.library.exact_numerics(),
        }
    }

    /// Runs `work` on `lane` as one marked submission, with the table's kernels and the lane's stream-ordered
    /// allocation: what it enqueues, and what it allocates and frees, are ordered on `lane`, and the marker is the
    /// submission's. Nothing waits: a lane with no free marker answers [`SubmitError::Busy`] and `work` has not run.
    /// `work` that fails gives its marker up.
    pub fn submit_ops<R, E: Display>(
        &self,
        lane: &CudaQueue,
        row_exact: bool,
        work: impl FnOnce(&mut dyn LaneOps) -> Result<R, E>,
    ) -> Result<(R, Marker), SubmitError> {
        let home = Marker::new(DeviceQueue::id(lane), Lane::lane(lane), 0);
        let modes = self.modes(row_exact);
        let mut refused = None;
        let made = lane.submit_marked(|queue| {
            let mut ops = RusteeOps::new(queue, &self.kernels, modes).on_lane(&self.library, home);
            work(&mut ops).map_err(|error| {
                refused = Some(error.to_string());
                GpuError::Invalid("the work on the lane was refused".to_string())
            })
        });
        match made {
            Ok(made) => Ok(made),
            Err(GpuError::Busy) => Err(SubmitError::Busy),
            Err(error) => Err(refused.take().map_or(SubmitError::Lane(error), SubmitError::Work)),
        }
    }

    /// Runs `call`, which uses the C library's synchronous calls (a host copy, an allocation), as one marked submission
    /// of `lane`: the library's stream is the lane's for the call, so the call is ordered with the lane's work before
    /// and after it. For code that has no kernel to queue, such as a test reading a buffer back.
    pub fn submit_host<R>(&self, lane: &CudaQueue, call: impl FnOnce() -> R) -> Result<(R, Marker), SubmitError> {
        let home = Marker::new(DeviceQueue::id(lane), Lane::lane(lane), 0);
        let modes = self.modes(false);
        let made = lane.submit_marked(|queue| {
            let ops = RusteeOps::new(queue, &self.kernels, modes).on_lane(&self.library, home);
            ops.on_lent_stream(|_| call())
                .map_err(|error| GpuError::Invalid(error.to_string()))
        });
        made.map_err(|error| match error {
            GpuError::Busy => SubmitError::Busy,
            other => SubmitError::Lane(other),
        })
    }
}

/// Why a submission on a lane was not made.
#[derive(Debug, Error)]
pub enum SubmitError {
    /// The lane has no free marker: nothing was submitted, and the caller tries again once the lane has completed more.
    #[error("the lane has no free marker")]
    Busy,
    /// The work refused: its error.
    #[error("{0}")]
    Work(String),
    /// The lane failed.
    #[error("the lane failed: {0}")]
    Lane(GpuError),
}

impl From<SubmitError> for EngineError {
    fn from(error: SubmitError) -> Self {
        Self::Runtime(error.to_string())
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
    /// Counts the quantum as in flight on its sequence until it is dropped, so the sequence's owner knows when it may
    /// free the caches (`SequenceState::refs`).
    flight: Option<FlightGuard>,
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
            flight: None,
        }
    }

    /// The references with the quantum counted as in flight on its sequence until it is dropped.
    pub fn in_flight(mut self, guard: FlightGuard) -> Self {
        self.flight = Some(guard);
        self
    }

    /// Takes the in-flight guard out, so that its holder can let go of it when it is done with the quantum.
    pub(crate) fn take_flight(&mut self) -> Option<FlightGuard> {
        self.flight.take()
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

    /// Orders the quantum's stages that read a cache behind the work that reserved its rows, `marker`, for the views
    /// of both caches ([`KvView::ordered_after`]): the marker is kept until its work has completed.
    pub fn ordered_after(mut self, marker: Marker, lanes: &LanePool) -> Self {
        if let Some(view) = self.kv.take() {
            self.kv = Some(view.ordered_after(marker, lanes));
        }
        if let Some(view) = self.mtp_kv.take() {
            self.mtp_kv = Some(view.ordered_after(marker, lanes));
        }
        self
    }

    /// The rows each GDN layer replays, by model layer.
    pub fn replay_layers(&self) -> &[u32] {
        &self.replay_rows
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

/// One sequence's part of a quantum: its rows, where they start in the sequence and in the quantum's rows, and what its
/// stages read and fill. A quantum of one sequence has one segment; a batch has one per sequence.
#[derive(Debug)]
pub struct Segment {
    /// The segment's place among the quantum's segments: the input record keeps its start position in the slot of
    /// this number.
    pub(crate) index: usize,
    /// The sequence the segment's rows belong to, as its stages read it.
    pub(crate) refs: SequenceRefs,
    /// The sequence's number, for a segment of a batch ([`Quantum::batch`]); a quantum of one sequence does not name it.
    pub(crate) sequence: Option<SequenceId>,
    /// The position of the segment's first row in its sequence.
    pub(crate) start_position: u64,
    /// The segment's rows.
    pub(crate) rows: usize,
    /// The segment's first row among the quantum's rows.
    pub(crate) row_offset: usize,
    /// The device words and the pinned readback the segment's argmax fills.
    pub(crate) selection: Option<Arc<Selection>>,
    /// The lease on `selection` when it came from the sequence's pool: the words are the quantum's until it is
    /// dropped, after its owner read the keys.
    pub(crate) lease: Option<SelectionLease>,
    /// The pinned readback the segment's logits row is copied to when the host selects (a sampling request): the
    /// segment then has no selection. The owner's lease on it is held apart from the quantum.
    pub(crate) row_readback: Option<Arc<RowReadback>>,
    /// The slot of the step's tokens this segment's rows read their token ids from, instead of the input record.
    pub(crate) draft_token_read: Option<usize>,
    /// Whether the base rows' post-final-norm hidden also seed the sequence's drafting.
    pub(crate) seeds_draft: bool,
    /// The rows the segment commits, when its owner knows them before the quantum runs ([`Part::committing`]); else all.
    pub(crate) committed: Option<usize>,
    /// Set when the segment's request was cancelled: the segment's rows are then not committed ([`Part::cancelled_by`]).
    pub(crate) cancel: Option<Arc<AtomicBool>>,
}

impl Segment {
    /// The segment's rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The segment's first row among the quantum's rows.
    pub fn row_offset(&self) -> usize {
        self.row_offset
    }

    /// The position of the segment's first row in its sequence.
    pub fn start_position(&self) -> u64 {
        self.start_position
    }

    /// The segment of `rows` rows starting at `start_position`, whose rows are the quantum's first.
    fn new(refs: SequenceRefs, start_position: u64, rows: usize) -> Self {
        Self {
            index: 0,
            refs,
            sequence: None,
            start_position,
            rows,
            row_offset: 0,
            selection: None,
            lease: None,
            row_readback: None,
            draft_token_read: None,
            seeds_draft: false,
            committed: None,
            cancel: None,
        }
    }

    /// The segment's place among the quantum's segments.
    pub fn index(&self) -> usize {
        self.index
    }

    /// The words and readback the segment's argmax fills, if its rows select on the device.
    pub fn selection(&self) -> Option<&Arc<Selection>> {
        self.selection.as_ref()
    }

    /// The readback the segment's logits rows are copied to, if the host selects.
    pub fn row_readback(&self) -> Option<&Arc<RowReadback>> {
        self.row_readback.as_ref()
    }

    /// The sequence of the segment, when the quantum is a batch.
    pub fn sequence(&self) -> Option<SequenceId> {
        self.sequence
    }

    /// Whether the segment's request was cancelled.
    pub fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|flag| flag.load(Ordering::Acquire))
    }
}

/// One sequence's share of a batch ([`Quantum::batch`]): the rows it verifies, with what its segment of the quantum
/// reads and writes.
#[derive(Debug)]
pub struct Part {
    refs: SequenceRefs,
    sequence: SequenceId,
    start_position: u64,
    tokens: Vec<u32>,
    selection: Option<Arc<Selection>>,
    lease: Option<SelectionLease>,
    row_readback: Option<Arc<RowReadback>>,
    draft_token_read: Option<usize>,
    seeds_draft: bool,
    committed: Option<usize>,
    cancel: Option<Arc<AtomicBool>>,
}

impl Part {
    /// A part of `sequence`: its rows are `tokens`, from `start_position` on, over `refs`.
    pub fn new(refs: SequenceRefs, sequence: SequenceId, start_position: u64, tokens: Vec<u32>) -> Self {
        Self {
            refs,
            sequence,
            start_position,
            tokens,
            selection: None,
            lease: None,
            row_readback: None,
            draft_token_read: None,
            seeds_draft: false,
            committed: None,
            cancel: None,
        }
    }

    /// The part's rows select on the device, into `selection`.
    pub fn selecting(mut self, selection: Arc<Selection>) -> Self {
        self.selection = Some(selection);
        self
    }

    /// The part's rows select on the device, into a selection leased from its sequence's pool.
    pub fn selecting_leased(mut self, lease: SelectionLease) -> Self {
        self.selection = Some(Arc::clone(lease.selection()));
        self.lease = Some(lease);
        self
    }

    /// The part's logits rows are copied to `readback`, and the host selects.
    pub fn reading_rows(mut self, readback: Arc<RowReadback>) -> Self {
        self.row_readback = Some(readback);
        self
    }

    /// The part's rows read their token ids from the step's tokens from `slot` on ([`Quantum::reading_draft_tokens`]).
    pub fn reading_draft_tokens(mut self, slot: usize) -> Result<Self, QuantumError> {
        let end = slot + self.tokens.len();
        if end as u64 > DRAFT_TOKEN_SLOTS {
            return Err(QuantumError::DraftTokenSlots { slot, end });
        }
        if self.refs.device().scratch.draft().is_none() {
            return Err(QuantumError::NoDraftBuffers {
                what: "reading the step's tokens",
            });
        }
        self.draft_token_read = Some(slot);
        Ok(self)
    }

    /// The part's post-final-norm hidden rows also seed its sequence's drafting ([`Quantum::seeding_draft`]).
    pub fn seeding_draft(mut self) -> Result<Self, QuantumError> {
        let draft = self.refs.device().scratch.draft().ok_or(QuantumError::NoDraftBuffers {
            what: "seeding drafting",
        })?;
        if self.tokens.len() as u64 > draft.seed_capacity {
            return Err(QuantumError::DraftSeedRows {
                first: 0,
                rows: self.tokens.len(),
                capacity: draft.seed_capacity,
            });
        }
        self.seeds_draft = true;
        Ok(self)
    }

    /// The part commits `rows` of its rows, the prefix its owner knows it will accept, not all of them
    /// ([`Quantum::committed_rows`]).
    pub fn committing(mut self, rows: usize) -> Result<Self, QuantumError> {
        if rows > self.tokens.len() {
            return Err(QuantumError::Invalid {
                what: format!("{rows} committed rows of a part of {}", self.tokens.len()),
            });
        }
        self.committed = Some(rows);
        Ok(self)
    }

    /// The part's request is cancelled once `flag` is set: its segment then concludes as cancelled and commits
    /// nothing, and the other segments are not affected ([`Quantum::conclusion_for`]).
    pub fn cancelled_by(mut self, flag: Arc<AtomicBool>) -> Self {
        self.cancel = Some(flag);
        self
    }

    /// The sequence the part belongs to.
    pub fn sequence(&self) -> SequenceId {
        self.sequence
    }

    /// The part's rows.
    pub fn rows(&self) -> usize {
        self.tokens.len()
    }

    fn shape(&self) -> Shape {
        Shape {
            sequence: self.sequence,
            rows: self.tokens.len(),
            kv_format: self.refs.kv(KvSource::Base).map(KvView::format),
            logits: match (&self.selection, &self.row_readback) {
                (Some(_), None) => LogitsUse::Device,
                (None, Some(_)) => LogitsUse::Host,
                (None, None) => LogitsUse::None,
                (Some(_), Some(_)) => LogitsUse::Both,
            },
            selection: self.selection.as_ref().map(|selection| Arc::as_ptr(selection) as usize),
            readback: self
                .row_readback
                .as_ref()
                .map(|readback| Arc::as_ptr(readback) as usize),
        }
    }
}

/// How a part's logits rows leave the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LogitsUse {
    None,
    Device,
    Host,
    Both,
}

/// What [`lay_out`] needs to know of a part.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    pub sequence: SequenceId,
    pub rows: usize,
    pub kv_format: Option<KvFormat>,
    pub logits: LogitsUse,
    pub selection: Option<usize>,
    pub readback: Option<usize>,
}

/// The first row of each part of a batch, the total rows and whether the rows are scored, or why the parts cannot
/// be one quantum. `record_bytes` is what one input record of the workspace holds.
pub(crate) fn lay_out(shapes: &[Shape], record_bytes: u64) -> Result<(Vec<usize>, usize, bool), QuantumError> {
    if shapes.is_empty() {
        return Err(QuantumError::EmptyBatch);
    }
    let mut offsets = Vec::with_capacity(shapes.len());
    let mut total = 0usize;
    for (index, shape) in shapes.iter().enumerate() {
        if shape.rows == 0 {
            return Err(QuantumError::EmptyPart { part: index });
        }
        if shape.logits == LogitsUse::Both {
            return Err(QuantumError::Invalid {
                what: format!("part {index} both selects on the device and reads its rows back"),
            });
        }
        if let Some(first) = shapes[..index]
            .iter()
            .position(|other| other.sequence == shape.sequence)
        {
            return Err(QuantumError::DuplicateSequence {
                sequence: shape.sequence,
                first,
                second: index,
            });
        }
        offsets.push(total);
        total += shape.rows;
    }
    let first = shapes[0].logits;
    if let Some(other) = shapes.iter().position(|shape| shape.logits != first) {
        return Err(QuantumError::MixedLogits { first: 0, other });
    }
    let formats: Vec<(usize, KvFormat)> = shapes
        .iter()
        .enumerate()
        .filter_map(|(index, shape)| Some((index, shape.kv_format?)))
        .collect();
    if let Some(&(first_part, format)) = formats.first()
        && let Some(&(other, _)) = formats.iter().find(|(_, other)| *other != format)
    {
        return Err(QuantumError::KvFormatMismatch {
            first: first_part,
            other,
        });
    }
    for (index, shape) in shapes.iter().enumerate() {
        let shared = shapes[..index].iter().position(|other| {
            (shape.selection.is_some() && other.selection == shape.selection)
                || (shape.readback.is_some() && other.readback == shape.readback)
        });
        if let Some(first) = shared {
            return Err(QuantumError::SharedLogitsBuffer { first, second: index });
        }
    }
    let need = InputRecordRef::position_offset(total) + 8 * shapes.len() as u64;
    if need > record_bytes {
        return Err(QuantumError::InputRecord {
            need,
            have: record_bytes,
        });
    }
    Ok((offsets, total, first != LogitsUse::None))
}

/// Refuses parts that are one sequence under different ids: they still share its device buffers.
fn check_distinct_buffers<'a>(
    refs: impl Iterator<Item = &'a SequenceRefs>,
    shapes: &[Shape],
) -> Result<(), QuantumError> {
    let buffers: Vec<Vec<AllocationId>> = refs
        .map(|refs| {
            let mut ids = Vec::new();
            refs.device().generations(&mut ids);
            ids
        })
        .collect();
    for (second, ids) in buffers.iter().enumerate() {
        if let Some(first) = buffers[..second]
            .iter()
            .position(|other| other.iter().any(|id| ids.contains(id)))
        {
            return Err(QuantumError::DuplicateSequence {
                sequence: shapes[second].sequence,
                first,
                second,
            });
        }
    }
    Ok(())
}

/// How one segment of a batch ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The quantum ran and the segment's request still stands: its owner reads the rows and commits.
    Completed,
    /// The segment's request was cancelled (or the quantum was killed): nothing of it is committed.
    Cancelled,
    /// The quantum failed.
    Failed(String),
}

/// How a segment concludes when its quantum ended as `outcome` and the segment's request `cancelled` or not. A
/// member's cancel changes its own conclusion only.
pub(crate) fn conclude(outcome: &QuantumOutcome, cancelled: bool) -> Outcome {
    match outcome {
        QuantumOutcome::Completed if cancelled => Outcome::Cancelled,
        QuantumOutcome::Completed => Outcome::Completed,
        QuantumOutcome::Killed if cancelled => Outcome::Cancelled,
        // A member that still wanted its rows has a half-advanced state: the quantum stopped under it.
        QuantumOutcome::Killed => {
            Outcome::Failed("the quantum was killed before the segment's request was cancelled".to_string())
        }
        QuantumOutcome::Failed { stage, error } => Outcome::Failed(format!("stage {stage:?} failed: {error:?}")),
        QuantumOutcome::DeviceFailed { cause } => Outcome::Failed(format!("the device work failed: {cause}")),
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
    #[error("draft token slots {slot}..{end} are out of the {DRAFT_TOKEN_SLOTS} the step's tokens hold")]
    DraftTokenSlots { slot: usize, end: usize },
    #[error("a drafted token goes to slot 1 to {}, not {slot}; slot 0 is the step's current token", DRAFT_TOKEN_SLOTS - 1)]
    DraftTokenWriteSlot { slot: usize },
    #[error("only a draft quantum {what}, not a {kind:?} one")]
    NotDraft { what: &'static str, kind: QuantumKind },
    #[error("a draft quantum does not seed drafting itself")]
    DraftSeedsItself,
    #[error("the sequence has no draft buffers, which {what} needs")]
    NoDraftBuffers { what: &'static str },
    #[error("{rows} draft seed rows from row {first} exceed the {capacity} the sequence holds")]
    DraftSeedRows { first: u32, rows: usize, capacity: u64 },
    #[error("{given} token ids for {rows} rows")]
    TokenCount { given: usize, rows: usize },
    #[error("token id {id} at row {row} is outside the vocabulary of {vocabulary}")]
    TokenOutsideVocabulary { id: u32, row: usize, vocabulary: usize },
    #[error("a batch of no parts")]
    EmptyBatch,
    #[error("part {part} of the batch has no rows")]
    EmptyPart { part: usize },
    #[error("parts {first} and {second} of the batch are both sequence {}", sequence.0)]
    DuplicateSequence {
        sequence: SequenceId,
        first: usize,
        second: usize,
    },
    #[error(
        "part {other} of the batch scores its rows differently from part {first}: all select on the device, all read back, or none scores"
    )]
    MixedLogits { first: usize, other: usize },
    #[error("part {other} of the batch has a cache in another format than part {first}")]
    KvFormatMismatch { first: usize, other: usize },
    #[error("parts {first} and {second} of the batch score into the same buffer")]
    SharedLogitsBuffer { first: usize, second: usize },
    #[error("the batch's input record takes {need} bytes, and the workspace's records hold {have}")]
    InputRecord { need: u64, have: u64 },
    #[error("segment {segment} of a quantum of {segments}")]
    SegmentOutOfRange { segment: usize, segments: usize },
    #[error("{what}")]
    Invalid { what: String },
}

/// One quantum, as the stages see it. Its fields are `pub(crate)` and are set through constructors, so that the
/// planning that builds quanta can add to them without breaking the stages.
#[derive(Debug)]
pub struct Quantum {
    /// The quantum's own number, from which its home lane is taken ([`crate::quantum::home`]).
    pub(crate) id: u64,
    pub(crate) kind: QuantumKind,
    pub(crate) rows: usize,
    /// The workspace the quantum's rows live in. Its view of `rows` rows was checked when the quantum was made.
    pub(crate) workspace: Arc<Workspace>,
    /// The sequences' parts of the quantum, in row order: at least one, which only [`Quantum::new`] can build.
    segments: SmallVec<[Segment; 1]>,
    pub(crate) mode: SubmitMode,
    pub(crate) config: Arc<ModelConfig>,
    pub(crate) lanes: LaneTable,
    /// The latest marker of a failed submission on each lane: work with no marker of the quantum's recorded work
    /// behind it, which the home lane is ordered behind when the quantum is abandoned (see [`unmarked`]).
    pub(crate) unmarked: Unmarked,
    /// The first use of a builder for one sequence on a quantum of several: `validate` refuses the quantum.
    misuse: Option<String>,
    /// Logits rows the quantum scores: the rows of its final norm, its head and its selection.
    pub(crate) logits_rows: usize,
    pub(crate) draft_seed: Option<DraftSeed>,
    pub(crate) draft_token_write: Option<DraftTokenWrite>,
    /// Whether the quantum writes its last input token to slot 0 of the step's tokens as the step's current token.
    pub(crate) writes_current_token: bool,
    /// The input record the quantum's ids and start position are uploaded to (one of the workspace's records).
    pub(crate) record: usize,
    /// The selection of the quantum before it in a chain: the quantum's one row reads its token id from that
    /// selection's keys on the device, behind the quantum that wrote them ([`Quantum::token_from`]).
    pub(crate) token_from: Option<Arc<Selection>>,
    /// The token ids of the rows, the input record's first part ([`Quantum::with_tokens`]).
    pub(crate) tokens: Vec<u32>,
    /// The pinned copy of the input record, held until the quantum is dropped: its upload reads it.
    pub(crate) staging: Option<PooledPinned>,
    /// Whether the input record was queued for upload ([`Quantum::prepare`]); the marker of that work must then be
    /// recorded with [`Quantum::prepared_after`] before the quantum starts.
    pub(crate) uploaded: bool,
    /// The marker of the work that uploaded the input record: the quantum's first stage waits for it on the device.
    pub(crate) prepared: Option<Marker>,
    /// Set by the executor when the quantum runs as one body of the graph cache: its key, what its run holds, and how
    /// it ran.
    pub(crate) capture: Option<Capture>,
}

/// The next quantum's id.
static NEXT_QUANTUM: AtomicU64 = AtomicU64::new(0);

impl Quantum {
    /// The quantum's segments, in row order. Never empty.
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// The quantum's one segment: what the callers of a quantum of one sequence read.
    pub(crate) fn segment(&self) -> &Segment {
        assert_eq!(
            self.segments.len(),
            1,
            "a quantum of several sequences has no one segment"
        );
        // A quantum is built with one segment, and nothing removes it.
        &self.segments[0]
    }

    fn segment_mut(&mut self) -> &mut Segment {
        assert_eq!(
            self.segments.len(),
            1,
            "a quantum of several sequences has no one segment"
        );
        &mut self.segments[0]
    }

    /// The quantum's own number, distinct for every quantum made.
    pub fn id(&self) -> u64 {
        self.id
    }

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
            id: NEXT_QUANTUM.fetch_add(1, Ordering::Relaxed),
            kind,
            rows,
            workspace,
            segments: smallvec![Segment::new(seq, input.start_position(), rows)],
            mode,
            config,
            unmarked: Unmarked::new(lanes.len()),
            misuse: None,
            lanes,
            logits_rows,
            record: 0,
            token_from: None,
            draft_seed: None,
            draft_token_write: None,
            writes_current_token: false,
            tokens: Vec::new(),
            staging: None,
            uploaded: false,
            prepared: None,
            capture: None,
        })
    }

    /// One verification of several sequences at once: a row-exact quantum with one segment per part, its rows the
    /// parts' rows in order. The stages that read or write a sequence's state run once per segment; the linears and
    /// the norms run once over all rows, so the weights are read once. Refused when two parts are one sequence, when
    /// the parts do not all select on the device, all read back or all stay unscored, when their caches differ in
    /// format, when two parts score into one buffer, or when the workspace's input records or logits cannot hold
    /// the rows. A batch is not captured: it runs stage by stage.
    pub fn batch(
        parts: Vec<Part>,
        workspace: Arc<Workspace>,
        mode: SubmitMode,
        config: Arc<ModelConfig>,
        lanes: LaneTable,
    ) -> Result<Self, QuantumError> {
        let shapes: Vec<Shape> = parts.iter().map(Part::shape).collect();
        let record_bytes = input_record_bytes(workspace.rows().rows);
        let (offsets, rows, scored) = lay_out(&shapes, record_bytes)?;
        check_distinct_buffers(parts.iter().map(|part| &part.refs), &shapes)?;
        if !mode.row_exact {
            return Err(QuantumError::RowExact {
                kind: QuantumKind::Verify,
                expected: true,
                given: false,
            });
        }
        workspace.view(rows as u64)?;
        let logits_rows = if scored { rows } else { 0 };
        workspace.rows().check_logits(logits_rows as u64)?;
        let mut tokens = Vec::with_capacity(rows);
        let mut segments = SmallVec::new();
        for (index, (part, row_offset)) in parts.into_iter().zip(offsets).enumerate() {
            if let Some((row, id)) = part
                .tokens
                .iter()
                .enumerate()
                .find(|(_, id)| part.draft_token_read.is_none() && **id as usize >= config.vocab_size)
            {
                return Err(QuantumError::TokenOutsideVocabulary {
                    id: *id,
                    row: row_offset + row,
                    vocabulary: config.vocab_size,
                });
            }
            let part_rows = part.tokens.len();
            if part.draft_token_read.is_some() {
                tokens.extend(repeat_n(0u32, part_rows));
            } else {
                tokens.extend_from_slice(&part.tokens);
            }
            let mut segment = Segment::new(part.refs, part.start_position, part_rows);
            segment.index = index;
            segment.sequence = Some(part.sequence);
            segment.row_offset = row_offset;
            segment.selection = part.selection;
            segment.lease = part.lease;
            segment.row_readback = part.row_readback;
            segment.draft_token_read = part.draft_token_read;
            segment.seeds_draft = part.seeds_draft;
            segment.committed = part.committed;
            segment.cancel = part.cancel;
            segments.push(segment);
        }
        Ok(Self {
            id: NEXT_QUANTUM.fetch_add(1, Ordering::Relaxed),
            kind: QuantumKind::Verify,
            rows,
            workspace,
            segments,
            mode,
            config,
            unmarked: Unmarked::new(lanes.len()),
            misuse: None,
            lanes,
            logits_rows,
            record: 0,
            token_from: None,
            draft_seed: None,
            draft_token_write: None,
            writes_current_token: false,
            tokens,
            staging: None,
            uploaded: false,
            prepared: None,
            capture: None,
        })
    }

    /// How segment `segment` ended, when the quantum ended as `outcome`: a segment whose request was cancelled
    /// concludes as cancelled, and the other segments of the quantum conclude as they would alone.
    pub fn conclusion_for(&self, segment: usize, outcome: &QuantumOutcome) -> Result<Outcome, QuantumError> {
        Ok(conclude(outcome, self.segment_at(segment)?.cancelled()))
    }

    /// The rows segment `segment` commits when it completes: the rows its part named ([`Part::committing`]), else
    /// all of them. The owner commits only for a segment that concluded as [`Outcome::Completed`].
    pub fn committed_rows(&self, segment: usize) -> Result<usize, QuantumError> {
        let segment = self.segment_at(segment)?;
        Ok(segment.committed.unwrap_or(segment.rows))
    }

    /// Whether every segment's request was cancelled, so that nothing of the quantum is wanted: the only case in which
    /// the quantum as a whole may be stopped. The executor takes one `Arc<AtomicBool>` as its kill flag, which is no
    /// member's own flag: the pass that runs the batch owns a flag of its own and sets it when this turns true.
    pub fn stop_requested(&self) -> bool {
        self.segments.iter().all(Segment::cancelled)
    }

    /// Whether `members` (a verification of one sequence each, with the sequence's number) can be one batch: what
    /// [`Quantum::batch`] would refuse, found without consuming them, so that a refused pass runs its members alone.
    pub(crate) fn check_batch(members: &[(&Quantum, SequenceId)]) -> Result<(), QuantumError> {
        let Some((first, _)) = members.first() else {
            return Err(QuantumError::EmptyBatch);
        };
        let mut shapes = Vec::with_capacity(members.len());
        for (quantum, sequence) in members {
            quantum.require_single("a part of a batch")?;
            if quantum.kind != QuantumKind::Verify || !quantum.mode.row_exact {
                return Err(QuantumError::Invalid {
                    what: "only verifications are batched".to_string(),
                });
            }
            let segment = quantum.segment();
            if segment.draft_token_read.is_none() && quantum.tokens.len() != quantum.rows {
                return Err(QuantumError::TokenCount {
                    given: quantum.tokens.len(),
                    rows: quantum.rows,
                });
            }
            shapes.push(Shape {
                sequence: *sequence,
                rows: quantum.rows,
                kv_format: segment.refs.kv(KvSource::Base).map(KvView::format),
                logits: match (&segment.selection, &segment.row_readback) {
                    (Some(_), None) => LogitsUse::Device,
                    (None, Some(_)) => LogitsUse::Host,
                    (None, None) => LogitsUse::None,
                    (Some(_), Some(_)) => LogitsUse::Both,
                },
                selection: segment
                    .selection
                    .as_ref()
                    .map(|selection| Arc::as_ptr(selection) as usize),
                readback: segment
                    .row_readback
                    .as_ref()
                    .map(|readback| Arc::as_ptr(readback) as usize),
            });
        }
        let workspace = &first.workspace;
        let (_, rows, scored) = lay_out(&shapes, input_record_bytes(workspace.rows().rows))?;
        check_distinct_buffers(members.iter().map(|(quantum, _)| &quantum.segment().refs), &shapes)?;
        workspace.view(rows as u64)?;
        workspace.rows().check_logits(if scored { rows } else { 0 } as u64)?;
        Ok(())
    }

    /// The class of passes this verification can join: verifications of one class can be one batch, of different
    /// classes never (they differ in how their logits leave the device or in the key/value format).
    pub(crate) fn pass_class(&self) -> u64 {
        let Some(segment) = self.segments.first() else {
            return 0;
        };
        let logits = match (&segment.selection, &segment.row_readback) {
            (Some(_), None) => 1,
            (None, Some(_)) => 2,
            (None, None) => 3,
            (Some(_), Some(_)) => 4,
        };
        let format = match segment.refs.kv(KvSource::Base).map(KvView::format) {
            None => 0,
            Some(KvFormat::Nvfp4) => 1,
            Some(KvFormat::Bf16) => 2,
        };
        logits | format << 8
    }

    /// The part of a batch this verification of `sequence` is, which `cancel` (the request's flag) cancels.
    pub(crate) fn into_part(mut self, sequence: SequenceId, cancel: Arc<AtomicBool>) -> Result<Part, QuantumError> {
        self.require_single("a part of a batch")?;
        if self.kind != QuantumKind::Verify {
            return Err(QuantumError::Invalid {
                what: "only verifications are batched".to_string(),
            });
        }
        let rows = self.rows;
        let tokens = if self.segment().draft_token_read.is_some() {
            vec![0u32; rows]
        } else if self.tokens.len() == rows {
            mem::take(&mut self.tokens)
        } else {
            return Err(QuantumError::TokenCount {
                given: self.tokens.len(),
                rows,
            });
        };
        let Some(segment) = self.segments.pop() else {
            return Err(QuantumError::EmptyBatch);
        };
        Ok(Part {
            refs: segment.refs,
            sequence,
            start_position: segment.start_position,
            tokens,
            selection: segment.selection,
            lease: segment.lease,
            row_readback: segment.row_readback,
            draft_token_read: segment.draft_token_read,
            seeds_draft: segment.seeds_draft,
            committed: segment.committed,
            cancel: Some(cancel),
        })
    }

    /// Takes each segment's selection lease out, in segment order, so that its owner can give it back when it has
    /// read the keys: the leases of a batch's segments belong to different sequences, and the quantum is dropped by
    /// whichever of them lets go last.
    pub(crate) fn take_leases(&mut self) -> Vec<Option<SelectionLease>> {
        self.segments.iter_mut().map(|segment| segment.lease.take()).collect()
    }

    /// Takes each segment's in-flight guard out, in segment order: a pass holds one for each sequence, and each sequence's
    /// owner lets go of its own when it is done with its part, so that no sequence waits for another's to settle.
    pub fn take_flights(&mut self) -> Vec<Option<FlightGuard>> {
        self.segments
            .iter_mut()
            .map(|segment| segment.refs.take_flight())
            .collect()
    }

    /// Whether the input record's upload was queued ([`Quantum::prepare`]).
    pub fn is_prepared(&self) -> bool {
        self.uploaded
    }

    /// Whether the quantum has one segment, which the builders and accessors of one sequence need.
    fn single(&self) -> bool {
        self.segments.len() == 1
    }

    /// Refuses a builder of one sequence on a quantum of several: `Err` for those that can return it.
    fn require_single(&self, what: &'static str) -> Result<(), QuantumError> {
        if self.single() {
            Ok(())
        } else {
            Err(QuantumError::Invalid {
                what: format!(
                    "{what} is for a quantum of one sequence; this one has {}",
                    self.segments.len()
                ),
            })
        }
    }

    /// Notes a builder of one sequence used on a quantum of several, which cannot return the error: `validate`
    /// refuses the quantum.
    fn note_single(&mut self, what: &'static str) -> bool {
        if let Err(error) = self.require_single(what) {
            self.misuse.get_or_insert_with(|| error.to_string());
            return false;
        }
        true
    }

    /// The refusal of a builder that was used on a quantum of several sequences, if any.
    pub(crate) fn misuse(&self) -> Option<&str> {
        self.misuse.as_deref()
    }

    fn segment_at(&self, segment: usize) -> Result<&Segment, QuantumError> {
        self.segments.get(segment).ok_or(QuantumError::SegmentOutOfRange {
            segment,
            segments: self.segments.len(),
        })
    }

    /// Scores `rows` logits rows instead (0: none, a prefill chunk that is not the last).
    pub fn with_logits_rows(mut self, rows: usize) -> Result<Self, WorkspaceError> {
        if !self.note_single("scoring a number of logits rows") {
            return Ok(self);
        }
        self.workspace.rows().check_logits(rows as u64)?;
        self.logits_rows = rows;
        Ok(self)
    }

    /// The base rows' post-final-norm hidden also seed the sequence's drafting. Refused for a draft quantum, which
    /// does not seed itself, and for rows the sequence's draft seed rows cannot hold.
    pub fn seeding_draft(mut self) -> Result<Self, QuantumError> {
        self.require_single("seeding drafting")?;
        if self.kind == QuantumKind::Draft {
            return Err(QuantumError::DraftSeedsItself);
        }
        let draft = self
            .segment()
            .refs
            .device()
            .scratch
            .draft()
            .ok_or(QuantumError::NoDraftBuffers {
                what: "seeding drafting",
            })?;
        if self.rows as u64 > draft.seed_capacity {
            return Err(QuantumError::DraftSeedRows {
                first: 0,
                rows: self.rows,
                capacity: draft.seed_capacity,
            });
        }
        self.segment_mut().seeds_draft = true;
        Ok(self)
    }

    /// The rows an MTP draft quantum normalizes beside its embedding rows. Refused for a quantum that does not draft,
    /// and for seed rows of the sequence that its rows would run past.
    pub fn with_draft_seed(mut self, seed: DraftSeed) -> Result<Self, QuantumError> {
        self.require_draft("takes a draft seed")?;
        self.require_single("a draft seed")?;
        if let DraftSeed::SeedRow(first) = seed {
            let draft = self
                .segment()
                .refs
                .device()
                .scratch
                .draft()
                .ok_or(QuantumError::NoDraftBuffers {
                    what: "a seed from the sequence's rows",
                })?;
            if u64::from(first) + self.rows as u64 > draft.seed_capacity {
                return Err(QuantumError::DraftSeedRows {
                    first,
                    rows: self.rows,
                    capacity: draft.seed_capacity,
                });
            }
        }
        self.draft_seed = Some(seed);
        Ok(self)
    }

    /// The selected token of a draft quantum goes to the step's tokens. The slot is 1 or more (slot 0 is the step's
    /// current token) and below the slots the step's tokens hold (`Quantum.writingDraftToken`).
    pub fn with_draft_token_write(mut self, write: DraftTokenWrite) -> Result<Self, QuantumError> {
        self.require_draft("passes a drafted token on")?;
        if write.slot < 1 || write.slot as u64 >= DRAFT_TOKEN_SLOTS {
            return Err(QuantumError::DraftTokenWriteSlot { slot: write.slot });
        }
        self.require_draft_buffers("passing a drafted token on")?;
        self.draft_token_write = Some(write);
        Ok(self)
    }

    /// The rows read their token ids from the step's tokens on the device, from `slot` (the tokens the quanta before
    /// this one wrote), not from the input record. The slots `slot..slot + rows` are those the step's tokens hold
    /// (`Quantum.readingDraftTokens`).
    pub fn reading_draft_tokens(mut self, slot: usize) -> Result<Self, QuantumError> {
        let end = slot + self.rows;
        if end as u64 > DRAFT_TOKEN_SLOTS {
            return Err(QuantumError::DraftTokenSlots { slot, end });
        }
        self.require_draft_buffers("reading the step's tokens")?;
        self.segment_mut().draft_token_read = Some(slot);
        Ok(self)
    }

    /// The quantum's last input token is written to slot 0 of the step's tokens as the step's current token.
    pub fn writing_current_token(mut self) -> Result<Self, QuantumError> {
        self.require_draft_buffers("writing the step's current token")?;
        self.writes_current_token = true;
        Ok(self)
    }

    fn require_draft(&self, what: &'static str) -> Result<(), QuantumError> {
        if self.kind == QuantumKind::Draft {
            Ok(())
        } else {
            Err(QuantumError::NotDraft { what, kind: self.kind })
        }
    }

    fn require_draft_buffers(&self, what: &'static str) -> Result<(), QuantumError> {
        self.require_single(what)?;
        if self.segment().refs.device().scratch.draft().is_some() {
            Ok(())
        } else {
            Err(QuantumError::NoDraftBuffers { what })
        }
    }

    /// The input record the quantum's ids and start position go to (counted modulo the records the workspace holds):
    /// each quantum in flight reads a record of its own.
    pub fn with_input_record(mut self, record: usize) -> Self {
        self.record = record % self.workspace.records();
        self
    }

    /// The quantum's one row reads its token id from the keys of `selection`, the selection of the quantum before it in
    /// the chain, on the device: no id is uploaded, and the host need not have read the token. The quantum's first
    /// stage runs behind the earlier quantum's last (the chain's edge), which has written the keys.
    pub fn token_from(mut self, selection: Arc<Selection>) -> Result<Self, QuantumError> {
        self.require_single("a token chained from the quantum before")?;
        if self.rows != 1 || self.segment().draft_token_read.is_some() {
            return Err(QuantumError::Invalid {
                what: format!(
                    "a token chained from the quantum before is one row, read from the selection and not from the \
                     step's tokens; this quantum has {} rows",
                    self.rows
                ),
            });
        }
        self.token_from = Some(selection);
        Ok(self)
    }

    /// The device words and readback the quantum's argmax fills.
    pub fn with_selection(mut self, selection: Arc<Selection>) -> Self {
        if self.note_single("a selection of the quantum") {
            self.segment_mut().selection = Some(selection);
        }
        self
    }

    /// The quantum copies its logits row to `readback` inside its graph instead of selecting on the device: the host
    /// selects from the row once the quantum retired.
    pub fn with_row_readback(mut self, readback: Arc<RowReadback>) -> Self {
        if self.note_single("a row readback of the quantum") {
            self.segment_mut().row_readback = Some(readback);
        }
        self
    }

    /// The readback the quantum's logits row is copied to, if the host selects.
    pub fn row_readback(&self) -> Option<&Arc<RowReadback>> {
        self.segment_of_one()?.row_readback.as_ref()
    }

    /// The words and readback of a selection leased from the sequence's pool: the quantum holds the lease until it is
    /// dropped, so no other quantum is given the words before the owner has read the keys.
    pub fn with_selection_lease(mut self, lease: SelectionLease) -> Self {
        if self.note_single("a selection lease of the quantum") {
            let segment = self.segment_mut();
            segment.selection = Some(Arc::clone(lease.selection()));
            segment.lease = Some(lease);
        }
        self
    }

    /// The token ids of the quantum's rows, one per row, each in the vocabulary: the first part of its input record
    /// ([`Quantum::prepare`] uploads them).
    pub fn with_tokens(mut self, tokens: Vec<u32>) -> Result<Self, QuantumError> {
        if tokens.len() != self.rows {
            return Err(QuantumError::TokenCount {
                given: tokens.len(),
                rows: self.rows,
            });
        }
        if let Some((row, id)) = tokens
            .iter()
            .enumerate()
            .find(|(_, id)| **id as usize >= self.config.vocab_size)
        {
            return Err(QuantumError::TokenOutsideVocabulary {
                id: *id,
                row,
                vocabulary: self.config.vocab_size,
            });
        }
        self.tokens = tokens;
        Ok(self)
    }

    pub fn kind(&self) -> QuantumKind {
        self.kind
    }

    /// The markers of the quantum's failed submissions, which its home lane is ordered behind on abandonment.
    pub fn unmarked(&self) -> &Unmarked {
        &self.unmarked
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The mode its submissions run in (row-exact exactly for a verification).
    pub fn mode(&self) -> SubmitMode {
        self.mode
    }

    pub fn start_position(&self) -> u64 {
        self.segment().start_position
    }

    pub fn logits_rows(&self) -> usize {
        self.logits_rows
    }

    /// The allocation serial of layer `layer`'s page table in the cache `source` as of admission: part of the key
    /// of a captured graph that reads the table.
    pub fn table_generation(&self, source: KvSource, layer: usize) -> Option<AllocationId> {
        self.segment_of_one()?.refs.kv(source)?.table_generation(layer)
    }

    /// The words and readback the quantum's argmax fills, if it selects.
    pub fn selection(&self) -> Option<&Arc<Selection>> {
        self.segment_of_one()?.selection.as_ref()
    }

    /// The quantum's one segment, or none for a quantum of several sequences (read each through `segments`).
    fn segment_of_one(&self) -> Option<&Segment> {
        self.single().then(|| &self.segments[0])
    }

    /// The device address of `slot`; for the input record, the one this quantum uploads to.
    pub(crate) fn slot(&self, slot: BufferSlot) -> DevicePtr {
        if slot == BufferSlot::TokenIds {
            return self.workspace.input_record(self.record);
        }
        self.workspace.buffer(slot).ptr()
    }

    /// The device address of the start position of segment `index` in the input record: the positions follow the
    /// token ids, one slot per segment.
    pub(crate) fn position_of(&self, index: usize) -> DevicePtr {
        self.slot(BufferSlot::TokenIds)
            .offset(InputRecordRef::position_offset(self.rows) + 8 * index as u64)
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

    /// Markers, beyond those of its predecessors, that the stage's work must wait for on the device: the work that
    /// allocated the pages it reads (`KvView::ready`), when that ran on another lane. Waits are not part of a
    /// capture, so they are listed apart from [`Launch::launch`]; eager submission waits for them ([`run_eager`]).
    fn waits(&self, quantum: &Quantum) -> Vec<Marker> {
        let _ = quantum;
        Vec::new()
    }

    /// Queues the stage's kernel launches and copies on `kernels`, for `quantum`. Nothing else: no wait for a
    /// marker, no marker, no completion signal, no synchronization, no allocation, so the launches can be captured
    /// into a graph. It reads, and does not change, anything the quantum shares. A quantum for which the stage is
    /// [`Launch::idle`] queues nothing: calling `launch` on it is allowed (a capture does not ask) and does nothing.
    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError>;
}

/// [`Launch`] without its generics, for the stages a plan keeps as a list: what a graph capture calls, in order.
pub trait StageLaunch: Send + Sync {
    fn idle(&self, quantum: &Quantum) -> bool;
    fn waits(&self, quantum: &Quantum) -> Vec<Marker>;
    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError>;
}

impl<L: Launch> StageLaunch for L {
    fn idle(&self, quantum: &Quantum) -> bool {
        Launch::idle(self, quantum)
    }

    fn waits(&self, quantum: &Quantum) -> Vec<Marker> {
        Launch::waits(self, quantum)
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        Launch::launch(self, quantum, kernels)
    }
}

/// The stage for eager submission, standing alone: begins a submission of the quantum's mode on the lane the graph
/// chose, waits for the markers of the predecessors on other lanes, calls [`Launch::launch`], and finishes with the
/// marker of the work ([`run_stage`]).
///
/// A lane held by another submission answers [`StageError::Retry`], before anything was queued; every other failure
/// is fatal to the quantum.
pub fn run_eager<L: Launch>(stage: &L, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
    run_stage(stage, quantum, cx)
}

/// Runs a stage as [`run_eager`] does.
///
/// Every submission ends with a marker of its own on the lane ([`submit_on`]), which RustEE's stage graph records, so
/// that the quantum retires behind the latest marker of each lane it used. A stage with nothing to launch for the
/// quantum queues nothing and leaves no marker: the markers of its predecessors pass through it.
pub fn run_stage<L: StageLaunch + ?Sized>(
    stage: &L,
    quantum: &Quantum,
    cx: &StageCx<'_>,
) -> Result<Submitted, StageError> {
    if stage.idle(quantum) {
        // An idle stage queues nothing, so there is nothing to mark. A completion edge from it would fire at once; the
        // plan has only submission edges, which pass the stage's inputs through.
        return Ok(Submitted::done());
    }
    quantum.check_prepared()?;
    let id = cx
        .lane
        .ok_or_else(|| fatal("a stage with device work was given no lane"))?;
    let lane = quantum
        .lanes
        .get(id)
        .ok_or_else(|| fatal(format!("lane {} is not in the lane table", id.0)))?;
    let modes = quantum.lanes.modes(quantum.mode.row_exact);
    let table = quantum.lanes.kernels();
    let waits = cx.waits.iter().copied().chain(stage.waits(quantum));
    let marker = submit_on(&**lane, waits, &quantum.unmarked, |queue| {
        stage.launch(
            quantum,
            &mut RusteeOps::new(queue, table, modes).with_pdl(quantum.mode.pdl),
        )
    })?;
    Ok(Submitted::marker(marker))
}

/// Orders `lane` behind each of `waits` that another lane made, then runs `launch` on the lane in a submission that
/// ends with a marker of its own, which it returns.
///
/// The waits are enqueued first, because RustEE's marked submission claims its marker only when it begins and takes the
/// lane's waits before the claim. Nothing blocks: a lane whose marker ring is full of markers that have not completed
/// (or whose next slot an awaiter on another lane holds) answers [`StageError::Retry`], and then `launch` has not
/// run and no marker was made; the waits that were already enqueued stay, and the retry enqueues them again, which
/// orders the lane behind the same markers twice and changes nothing.
///
/// A launch that fails (or panics) has queued part of its work already, and RustEE records the marker of a failed
/// stage nowhere, so the marked submission is given an `Ok` that carries the launch's result and always publishes its
/// marker. When the launch failed the marker goes to `unmarked`, which orders the quantum's home lane behind it when the
/// quantum is abandoned, and the launch's error is the stage's; a panic is resumed once the marker is kept. If the submission itself fails after the launch ran, the launch's own
/// error (or panic) is still the stage's and only otherwise is the submission's error.
pub fn submit_on<L: Lane>(
    lane: &L,
    waits: impl IntoIterator<Item = Marker>,
    unmarked: &Unmarked,
    launch: impl FnOnce(&mut L::Enqueue<'_>) -> Result<(), StageError>,
) -> Result<Marker, StageError> {
    let own = lane.lane();
    for marker in waits {
        if marker.lane != own {
            lane.await_marker(marker).map_err(fatal)?;
        }
    }
    let mut ran = None;
    let made = lane.submit_marked(|queue| {
        ran = Some(catch_unwind(AssertUnwindSafe(|| launch(queue))));
        Ok(())
    });
    match (made, ran) {
        (Ok((_, marker)), Some(Ok(Ok(())))) => Ok(marker),
        (Ok((_, marker)), Some(Ok(Err(error)))) => {
            unmarked.record(marker);
            Err(error)
        }
        (Ok((_, marker)), Some(Err(panic))) => {
            unmarked.record(marker);
            resume_unwind(panic)
        }
        (Err(GpuError::Busy), None) => Err(StageError::Retry),
        // The submission itself failed after the launch ran: the launch's own result is the stage's.
        (Err(_), Some(Ok(Err(error)))) => Err(error),
        (Err(_), Some(Err(panic))) => resume_unwind(panic),
        (Err(error), _) => Err(fatal(error)),
        (Ok(_), _) => Err(fatal("a marked submission finished without running its launch")),
    }
}

/// Stages that run one after the other as one: the launches of each in order, in one submission.
///
/// Its work is the work of the stages alone in the same order on the same lane: a stage queues nothing of the quantum
/// it is idle for, and the markers a stage waits for are what the quantum holds, so they do not depend on what an
/// earlier stage of the chain did. The unit is idle when every stage is.
pub struct Chain {
    launches: Vec<Arc<dyn StageLaunch>>,
}

impl fmt::Debug for Chain {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Chain")
            .field("stages", &self.launches.len())
            .finish()
    }
}

impl Chain {
    /// The stages `launches`, in order.
    pub fn new(launches: Vec<Arc<dyn StageLaunch>>) -> Self {
        Self { launches }
    }
}

impl StageLaunch for Chain {
    fn idle(&self, quantum: &Quantum) -> bool {
        self.launches.iter().all(|launch| launch.idle(quantum))
    }

    fn waits(&self, quantum: &Quantum) -> Vec<Marker> {
        let mut waits: Vec<Marker> = Vec::new();
        let busy = self.launches.iter().filter(|launch| !launch.idle(quantum));
        for marker in busy.flat_map(|launch| launch.waits(quantum)) {
            if !waits.contains(&marker) {
                waits.push(marker);
            }
        }
        waits
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        // As a graph capture does, skip the stages with nothing to queue: a chain does not rely on each stage's own guard.
        for_each_busy(
            &self.launches,
            |launch| launch.idle(quantum),
            |launch| launch.launch(quantum, kernels),
        )
    }
}

/// Calls `launch` on each of `stages` that is not `idle`, in order, and stops at the first failure.
fn for_each_busy<S: ?Sized>(
    stages: &[Arc<S>],
    idle: impl Fn(&S) -> bool,
    mut launch: impl FnMut(&S) -> Result<(), StageError>,
) -> Result<(), StageError> {
    for stage in stages {
        if !idle(stage) {
            launch(stage)?;
        }
    }
    Ok(())
}

/// A stage of a plan's graph: runs as [`run_stage`], and when its quantum is abandoned orders the home lane behind the
/// work that failed submissions of the quantum left on other lanes ([`Unmarked`]).
pub struct Placed {
    launch: Arc<dyn StageLaunch>,
}

impl Placed {
    /// The stage `launch`.
    pub fn new(launch: Arc<dyn StageLaunch>) -> Self {
        Self { launch }
    }
}

impl fmt::Debug for Placed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Placed").finish_non_exhaustive()
    }
}

impl Stage<Quantum> for Placed {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        run_stage(&*self.launch, quantum, cx)
    }

    fn abandoned(&self, quantum: &Quantum, home: &(dyn DeviceQueue + 'static)) {
        quantum
            .unmarked
            .await_on(home, |lane| quantum.lanes.get(lane).map(|queue| queue.id()));
    }
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

    fn shape(sequence: usize, rows: usize) -> Shape {
        Shape {
            sequence: SequenceId(sequence),
            rows,
            kv_format: Some(KvFormat::Nvfp4),
            logits: LogitsUse::Device,
            selection: Some(0x1000 + sequence),
            readback: None,
        }
    }

    const RECORD: u64 = 4096;

    #[test]
    fn the_segments_of_a_batch_follow_one_another_in_row_order() {
        let (offsets, total, scored) = lay_out(&[shape(0, 5), shape(1, 8), shape(2, 3)], RECORD).unwrap();
        assert_eq!((offsets, total, scored), (vec![0, 5, 13], 16, true));
        let (offsets, total, _) = lay_out(&[shape(4, 1)], RECORD).unwrap();
        assert_eq!((offsets, total), (vec![0], 1), "one part is a single segment");
    }

    #[test]
    fn a_batch_of_unscored_parts_scores_no_rows() {
        let unscored = |sequence, rows| Shape {
            logits: LogitsUse::None,
            selection: None,
            ..shape(sequence, rows)
        };
        assert!(!lay_out(&[unscored(0, 2), unscored(1, 2)], RECORD).unwrap().2);
    }

    #[test]
    fn a_batch_refuses_what_cannot_be_laid_out() {
        let bf16 = Shape {
            kv_format: Some(KvFormat::Bf16),
            ..shape(1, 2)
        };
        let same_buffer = Shape {
            selection: shape(0, 2).selection,
            ..shape(1, 2)
        };
        let refusals = [
            (vec![], QuantumError::EmptyBatch),
            (vec![shape(0, 2), shape(1, 0)], QuantumError::EmptyPart { part: 1 }),
            (
                vec![shape(3, 2), shape(5, 2), shape(3, 1)],
                QuantumError::DuplicateSequence {
                    sequence: SequenceId(3),
                    first: 0,
                    second: 2,
                },
            ),
            (
                vec![shape(0, 2), bf16],
                QuantumError::KvFormatMismatch { first: 0, other: 1 },
            ),
            (
                vec![shape(0, 2), same_buffer],
                QuantumError::SharedLogitsBuffer { first: 0, second: 1 },
            ),
        ];
        for (parts, refusal) in refusals {
            assert_eq!(lay_out(&parts, RECORD), Err(refusal));
        }
    }

    #[test]
    fn a_batch_refuses_parts_that_score_differently() {
        let host = |sequence, rows| Shape {
            logits: LogitsUse::Host,
            selection: None,
            readback: Some(0x2000 + sequence),
            ..shape(sequence, rows)
        };
        assert_eq!(
            lay_out(&[shape(0, 2), host(1, 2)], RECORD),
            Err(QuantumError::MixedLogits { first: 0, other: 1 })
        );
        let unscored = Shape {
            logits: LogitsUse::None,
            selection: None,
            ..shape(2, 2)
        };
        assert_eq!(
            lay_out(&[host(0, 2), host(1, 2), unscored], RECORD),
            Err(QuantumError::MixedLogits { first: 0, other: 2 })
        );
        assert!(lay_out(&[host(0, 2), host(1, 2)], RECORD).is_ok());
    }

    #[test]
    fn a_batch_refuses_rows_whose_positions_do_not_fit_the_input_record() {
        // 24 ids are 96 bytes, then three positions: 120 bytes.
        let parts = [shape(0, 8), shape(1, 8), shape(2, 8)];
        assert!(lay_out(&parts, 120).is_ok());
        assert_eq!(
            lay_out(&parts, 119),
            Err(QuantumError::InputRecord { need: 120, have: 119 })
        );
    }

    #[test]
    fn a_cancelled_member_concludes_cancelled_and_the_others_complete() {
        let flags = [false, true, false];
        let conclusions: Vec<Outcome> = flags
            .iter()
            .map(|cancelled| conclude(&QuantumOutcome::Completed, *cancelled))
            .collect();
        assert_eq!(
            conclusions,
            vec![Outcome::Completed, Outcome::Cancelled, Outcome::Completed]
        );
    }

    #[test]
    fn a_failed_or_killed_quantum_concludes_every_segment_alike() {
        let killed = QuantumOutcome::Killed;
        assert!(
            matches!(conclude(&killed, false), Outcome::Failed(_)),
            "a member that still wanted its rows"
        );
        assert_eq!(conclude(&killed, true), Outcome::Cancelled);
        let failed = QuantumOutcome::DeviceFailed { cause: "lost".into() };
        for cancelled in [false, true] {
            assert!(matches!(conclude(&failed, cancelled), Outcome::Failed(why) if why.contains("lost")));
        }
    }

    #[test]
    fn a_chain_launches_its_busy_stages_in_order_and_stops_at_a_failure() {
        let stages: Vec<Arc<u32>> = (0..6).map(Arc::new).collect();
        let mut seen = Vec::new();
        for_each_busy(
            &stages,
            |stage| stage % 2 == 1,
            |stage| {
                seen.push(*stage);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(seen, vec![0, 2, 4], "idle stages queue nothing, the rest in order");
        let mut seen = Vec::new();
        let failed = for_each_busy(
            &stages,
            |_| false,
            |stage| {
                seen.push(*stage);
                if *stage == 2 { Err(fatal("refused")) } else { Ok(()) }
            },
        );
        assert!(matches!(failed, Err(StageError::Fatal(_))));
        assert_eq!(seen, vec![0, 1, 2], "nothing after the failing stage is launched");
    }

    #[test]
    fn a_quantum_is_what_a_stage_graph_runs() {
        fn runs_in_a_graph<Q: Send + Sync + 'static>() {}
        runs_in_a_graph::<Quantum>();
    }
}
