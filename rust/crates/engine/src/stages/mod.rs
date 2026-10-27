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
pub mod settle;

use std::fmt::{self, Display};
use std::sync::Arc;

use euhedral_graph::{LaneId, LanePool, Marker, Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{
    AllocationId, CudaError, CudaLane, CudaLibrary, DevicePtr, PooledPinned, Submission, SubmitMode,
};
use thiserror::Error;

use crate::model::ModelConfig;
use crate::quantum::Disposition;
use crate::state::attention::DRAFT_TOKEN_SLOTS;
use crate::state::{FlightGuard, KvView, SequenceDevice, StaleReservation};
use crate::workspace::{BufferSlot, INPUT_RECORDS, Workspace, WorkspaceError};
use attention::KvSource;
use sampling::{Selection, SelectionLease};
pub use settle::{LaneEnds, Settled, Settler, Settling};

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

    /// The number of lanes.
    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    /// Whether the table has no lane.
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
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
    /// of both caches ([`KvView::ordered_after`]): the marker is kept while its lane's event ring holds it, dropped
    /// when its work has completed, and refused when it is older than the ring and not reached.
    pub fn ordered_after(
        mut self,
        marker: Marker,
        lanes: &LanePool,
        library: &CudaLibrary,
    ) -> Result<Self, StaleReservation> {
        if let Some(view) = self.kv.take() {
            self.kv = Some(view.ordered_after(marker, lanes, library)?);
        }
        if let Some(view) = self.mtp_kv.take() {
            self.mtp_kv = Some(view.ordered_after(marker, lanes, library)?);
        }
        Ok(self)
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
    #[error("{what}")]
    Invalid { what: String },
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
    /// The latest marker the quantum submitted on each lane and the latest it signalled, which says whether it can
    /// retire if it ends early (see [`settle`]).
    pub(crate) ends: Arc<LaneEnds>,
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
    /// The input record the quantum's ids and start position are uploaded to (0 or 1; the workspace holds two).
    pub(crate) record: usize,
    /// The selection of the quantum before it in a chain: the quantum's one row reads its token id from that
    /// selection's keys on the device, behind the quantum that wrote them ([`Quantum::token_from`]).
    pub(crate) token_from: Option<Arc<Selection>>,
    /// The device words and the pinned readback the quantum's argmax fills.
    pub(crate) selection: Option<Arc<Selection>>,
    /// The lease on `selection` when it came from the sequence's pool: the words are the quantum's until it is
    /// dropped, after its owner read the keys.
    pub(crate) lease: Option<SelectionLease>,
    /// The token ids of the rows, the input record's first part ([`Quantum::with_tokens`]).
    pub(crate) tokens: Vec<u32>,
    /// The pinned copy of the input record, held until the quantum is dropped: its upload reads it.
    pub(crate) staging: Option<PooledPinned>,
    /// Whether the input record was queued for upload ([`Quantum::prepare`]); the marker of that work must then be
    /// recorded with [`Quantum::prepared_after`] before the quantum starts.
    pub(crate) uploaded: bool,
    /// The marker of the work that uploaded the input record: the quantum's first stage waits for it on the device.
    pub(crate) prepared: Option<Marker>,
    /// How the quantum runs: stage by stage, recording a graph, or replaying one.
    pub(crate) disposition: Disposition,
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
            ends: Arc::new(LaneEnds::new(lanes.len())),
            lanes,
            logits_rows,
            record: 0,
            token_from: None,
            seeds_draft: false,
            draft_seed: None,
            draft_token_write: None,
            draft_token_read: None,
            writes_current_token: false,
            selection: None,
            lease: None,
            tokens: Vec::new(),
            staging: None,
            uploaded: false,
            prepared: None,
            disposition: Disposition::Eager,
        })
    }

    /// Scores `rows` logits rows instead (0: none, a prefill chunk that is not the last).
    pub fn with_logits_rows(mut self, rows: usize) -> Result<Self, WorkspaceError> {
        self.workspace.rows().check_logits(rows as u64)?;
        self.logits_rows = rows;
        Ok(self)
    }

    /// The base rows' post-final-norm hidden also seed the sequence's drafting. Refused for a draft quantum, which
    /// does not seed itself, and for rows the sequence's draft seed rows cannot hold.
    pub fn seeding_draft(mut self) -> Result<Self, QuantumError> {
        if self.kind == QuantumKind::Draft {
            return Err(QuantumError::DraftSeedsItself);
        }
        let draft = self.seq.device().scratch.draft().ok_or(QuantumError::NoDraftBuffers {
            what: "seeding drafting",
        })?;
        if self.rows as u64 > draft.seed_capacity {
            return Err(QuantumError::DraftSeedRows {
                first: 0,
                rows: self.rows,
                capacity: draft.seed_capacity,
            });
        }
        self.seeds_draft = true;
        Ok(self)
    }

    /// The rows an MTP draft quantum normalizes beside its embedding rows. Refused for a quantum that does not draft,
    /// and for seed rows of the sequence that its rows would run past.
    pub fn with_draft_seed(mut self, seed: DraftSeed) -> Result<Self, QuantumError> {
        self.require_draft("takes a draft seed")?;
        if let DraftSeed::SeedRow(first) = seed {
            let draft = self.seq.device().scratch.draft().ok_or(QuantumError::NoDraftBuffers {
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
        self.draft_token_read = Some(slot);
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
        if self.seq.device().scratch.draft().is_some() {
            Ok(())
        } else {
            Err(QuantumError::NoDraftBuffers { what })
        }
    }

    /// The input record the quantum's ids and start position go to: 0 or 1, and the other one for the quantum admitted
    /// next in a chain.
    pub fn with_input_record(mut self, record: usize) -> Self {
        self.record = record % INPUT_RECORDS as usize;
        self
    }

    /// The quantum's one row reads its token id from the keys of `selection`, the selection of the quantum before it in
    /// the chain, on the device: no id is uploaded, and the host need not have read the token. The quantum's first
    /// stage runs behind the earlier quantum's last (the chain's edge), which has written the keys.
    pub fn token_from(mut self, selection: Arc<Selection>) -> Result<Self, QuantumError> {
        if self.rows != 1 || self.draft_token_read.is_some() {
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
        self.selection = Some(selection);
        self
    }

    /// The words and readback of a selection leased from the sequence's pool: the quantum holds the lease until it is
    /// dropped, so no other quantum is given the words before the owner has read the keys.
    pub fn with_selection_lease(mut self, lease: SelectionLease) -> Self {
        self.selection = Some(Arc::clone(lease.selection()));
        self.lease = Some(lease);
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

    /// The latest marker the quantum submitted on each lane, and the latest it signalled.
    pub fn lane_ends(&self) -> &Arc<LaneEnds> {
        &self.ends
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

    /// The device address of `slot`; for the input record, the one this quantum uploads to.
    pub(crate) fn slot(&self, slot: BufferSlot) -> DevicePtr {
        if slot == BufferSlot::TokenIds {
            return self.workspace.input_record(self.record);
        }
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

    /// Markers, beyond those of its predecessors, that the stage's work must wait for on the device: the work that
    /// allocated the pages it reads (`KvView::ready`), when that ran on another lane. Waits are not part of a
    /// capture, so they are listed apart from [`Launch::launch`]; eager submission waits for them ([`run_eager`]).
    fn waits(&self, quantum: &Quantum) -> Vec<Marker> {
        let _ = quantum;
        Vec::new()
    }

    /// Queues the stage's kernel launches and copies on `submission`, for `quantum`. Nothing else: no wait for a
    /// marker, no marker, no completion signal, no synchronization, no allocation, so the launches can be captured
    /// into a graph. It reads, and does not change, anything the quantum shares. A quantum for which the stage is
    /// [`Launch::idle`] queues nothing: calling `launch` on it is allowed (a capture does not ask) and does nothing.
    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError>;
}

/// [`Launch`] without its generics, for the stages a plan keeps as a list: what a graph capture calls, in order.
pub trait StageLaunch: Send + Sync {
    fn idle(&self, quantum: &Quantum) -> bool;
    fn waits(&self, quantum: &Quantum) -> Vec<Marker>;
    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError>;
}

impl<L: Launch> StageLaunch for L {
    fn idle(&self, quantum: &Quantum) -> bool {
        Launch::idle(self, quantum)
    }

    fn waits(&self, quantum: &Quantum) -> Vec<Marker> {
        Launch::waits(self, quantum)
    }

    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError> {
        Launch::launch(self, quantum, submission)
    }
}

/// The stage for eager submission, standing alone: begins a submission of the quantum's mode on the lane the graph
/// chose, waits for the markers of the predecessors on other lanes, calls [`Launch::launch`], and finishes with the
/// marker of the work, signalled ([`run_stage`]).
///
/// A lane held by another submission answers [`StageError::Retry`], before anything was queued; every other failure
/// is fatal to the quantum.
pub fn run_eager<L: Launch>(stage: &L, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
    // Standing alone, a stage has no lane to end: with nothing to launch it queues nothing.
    if stage.idle(quantum) {
        return Ok(Submitted::done());
    }
    run_stage(stage, quantum, cx, true)
}

/// Runs a stage as [`run_eager`] does, with its marker signalled or not.
///
/// A signalled marker also writes the lane's completion word, which is how the host finds that the marker is reached
/// ([`DeviceQueue::completed`](euhedral_graph::DeviceQueue::completed)). A graph's retirement boundary is the latest
/// marker of every lane the quantum used, so the last stage of each lane must signal, and the others need not: the
/// plan computes which those are ([`Plan::signalled`](crate::plan::Plan::signalled)). A quantum that ends before its
/// chains' ends have run is settled by [`Settling`]. A stage that has nothing to
/// launch for the quantum but is to signal still ends a submission on its lane, behind the markers it waits for, with
/// a signalled marker and no launch, so the lane's last marker is reachable whatever the quantum skips.
pub fn run_stage<L: StageLaunch + ?Sized>(
    stage: &L,
    quantum: &Quantum,
    cx: &StageCx<'_>,
    signal: bool,
) -> Result<Submitted, StageError> {
    let idle = stage.idle(quantum);
    if idle && !signal {
        return Ok(Submitted::done());
    }
    if !idle {
        quantum.check_prepared()?;
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
    let own = cx.waits.iter().copied();
    if idle {
        for marker in own {
            submission.wait(marker).map_err(fatal)?;
        }
    } else {
        for marker in own.chain(stage.waits(quantum)) {
            submission.wait(marker).map_err(fatal)?;
        }
        stage.launch(quantum, &submission)?;
    }
    let marker = if signal {
        submission.finish_signalled()
    } else {
        submission.finish()
    }
    .map_err(fatal)?;
    quantum.ends.record(marker, signal);
    Ok(Submitted::marker(marker))
}

/// A stage of a plan's graph: runs as [`run_stage`] with the signalling the plan gave it.
pub struct Placed {
    launch: Arc<dyn StageLaunch>,
    signal: bool,
}

impl Placed {
    /// The stage `launch`, whose marker is signalled when `signal`.
    pub fn new(launch: Arc<dyn StageLaunch>, signal: bool) -> Self {
        Self { launch, signal }
    }
}

impl fmt::Debug for Placed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Placed")
            .field("signal", &self.signal)
            .finish_non_exhaustive()
    }
}

impl Stage<Quantum> for Placed {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        run_stage(&*self.launch, quantum, cx, self.signal)
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

    #[test]
    fn a_quantum_is_what_a_stage_graph_runs() {
        fn runs_in_a_graph<Q: Send + Sync + 'static>() {}
        runs_in_a_graph::<Quantum>();
    }
}
