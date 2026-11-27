//! The owner of the engine's sequence, driven by the messages posted to its mailbox.
//!
//! One [`Owner`] holds everything a sequence changes between quanta: its caches' frontiers, the graphs it has recorded,
//! the quanta in flight and what the request has produced. It sits in a [`Mailbox`](super::mailbox::Mailbox), whose
//! claim one thread at a time holds. A frame that tells the owner something posts the message and then drives the
//! mailbox: it runs [`Owner::work`] (a turn) on the thread it is on if the claim is free, and the holder of the claim
//! sees the message if it is not. No source polls the owner. A turn takes in what arrived, concludes the quanta that
//! retired, admits the next ones and ends the request; the outcome is told to its caller after the claim is released.
//! [`Generation::generate`](super::Generation::generate) claims the owner to start a request, and offers one frame to
//! the `stages` lake for the first turn to run on a worker.
//!
//! What arrives is a [`Message`] in the owner's inbox: a quantum that retired (the executor's retirement callback posts
//! it), the host's selection of a sampled row, and the running of the work a parked quantum waited for on the owner's
//! lane. Nothing waits for them.
//!
//! **The chain.** A greedy request keeps two quanta in flight. The one admitted behind another runs behind it on the
//! device (the graph's external edge waits for its sink), and takes its token id from the selection the earlier one
//! wrote ([`Quantum::token_from`]), so the host need not have read it: there is no gap between the end of one step and
//! the start of the next. The cost is one quantum that may run for nothing, the one that follows a stop token or a
//! failure. When the request stops, the quanta in flight are told to stop (their kill switches are set): a quantum
//! that stopped part-way retires behind the work it queued, and its rows never become part of the sequence
//! (`discard`).
//!
//! **The records.** The owner reserves a quantum's cache rows and uploads its input record on its own lane
//! when it admits it. Each quantum in flight reads a record of its own: a greedy request alternates between two, so the
//! upload of the quantum behind never overwrites what the quantum ahead is still reading, and the quanta of a
//! speculative step take the workspace's records (one more than the draft depth) in order.
//!
//! **Speculation.** With the artifact's draft layer ([`Speculation::Artifact`]) the owner runs the prompt's chunks, each
//! followed by the MTP catch-up of its rows, and then steps (`mtp.rs`): the catch-up over the rows the last
//! verification committed, the recursive drafts and the verification of the current token and the drafts, admitted
//! together as one chain. The drafts pass their tokens to each other and to the verification on the device, so the
//! host reads nothing until the verification retires; it then reads the drafts and the verifier's tokens, accepts the
//! drafts that equal them, commits that many rows and tells the listener. Each quantum of a step reads an input record
//! of its own.

use std::any::Any;
use std::collections::VecDeque;
use std::fmt::{self, Display, Formatter};
use std::mem;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
#[cfg(feature = "test-oracles")]
use std::sync::OnceLock;
#[cfg(feature = "test-oracles")]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::frames::Frame;
use euhedral_data_structures::queues::{Consumer, Single};
use euhedral_gpu::Ran;
use euhedral_graph::{Marker, QuantumOutcome};
use euhedral_inference_cuda::{DevicePtr, SubmitMode};

use super::hidden::{Hidden, HiddenRows, HiddenSink, delivery_frame};
use super::listener::GenerationListener;
use super::mailbox::{Address, Turn};
use super::mtp::{Joining, Speculating, Verifying};
use super::order::Order;
use super::pool::Lease;
use super::request::{
    AdmissionError, CancelHandle, FinishReason, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
};
use super::sampled::{Answered, Guarded, LeasedRow, Prepared, Sampled, SampledRows, prepare_frame, sample_frame};
use super::teacher_forced::LogitsSink;
use crate::frames::{Kind, step};
use crate::handoff::Handoff;
use crate::model::ModelConfig;
use crate::prefix::{
    CopyJob, FIRST_COPY_SERIAL, Hit, NodeId, PrefixCacheHandle, PromptCheckpoints, RestoreError, Restored,
    SequenceAddresses, Speculation as CheckpointKind, Submit,
};
use crate::quantum::{Executor, StartError};
use crate::runtime::{Core, Units};
use crate::sampling::{SamplingConfig, SamplingError, TokenSampler};
use crate::stages::attention::KvSource;
use crate::stages::sampling::{RowReadbackLease, Selection, SelectionLease};
use crate::stages::{InputRecordRef, Outcome, Quantum, QuantumError, QuantumKind, SequenceRefs, SubmitError};
use crate::state::FlightGuard;
use crate::state::kv::KvRetained;
use crate::state::{Flight, SequenceId, SequenceState};
use crate::workspace::Workspace;

/// Decode quanta that start below this position overlap their launches with the quantum before them
/// (`Quantum.OVERLAP_MAX_START_POSITION`).
pub(super) const OVERLAP_MAX_START_POSITION: u64 = 1024;

/// Quanta of a greedy request in flight at once: the one running, and the one admitted behind it.
pub(crate) const GREEDY_DEPTH: usize = 2;

/// The input records a greedy or teacher-forced request alternates between (a speculative one gives each quantum of a
/// step its own).
const ALTERNATING_RECORDS: u64 = 2;

/// What an owner is told.
pub(crate) enum Message {
    /// A quantum retired.
    Retired(Box<Retired>),
    /// The hidden rows of a quantum reached pinned memory.
    HiddenCopied(Box<HiddenRows>),
    /// The sink of a hidden request is free again: a delivery to it ended.
    HiddenDelivered(Box<dyn HiddenSink>),
    /// The host selected the token of a sampled request (or could not).
    Sampled(Sampled),
    /// The host prepared the next mask of a constrained request (or could not).
    Prepared(Prepared),
    /// The work a parked quantum waited for on the owner's lane has run: the turn lets the quantum go.
    Reaped,
    /// A quantum the owner admitted did not start.
    StartFailed(Box<StartFailed>),
    /// A prefix-cache copy job of this sequence to admit (see [`copy_submit`]).
    Copy(Box<CopyRequest>),
    /// A copy job of this sequence ended and was reported.
    CopyEnded,
    /// The prefix cache answered the lookup of the request's prompt.
    Looked(Option<Hit>),
    /// The prefix cache answered the restore of the request's hit.
    Restored(Result<u32, RestoreError>),
    /// The prefix cache answered a capture with the node to capture under next.
    Captured(NodeId),
}

/// What a copy job of the prefix cache is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Purpose {
    /// The restore of a hit, which the request's first chunk follows.
    Restore,
    /// The capture of the state after a chunk.
    Capture,
}

/// A copy job and what it is for.
pub(crate) struct CopyRequest {
    job: CopyJob,
    kill: Arc<AtomicBool>,
    purpose: Purpose,
}

/// The `Submit` of a prefix-cache request on the sequence whose owner is at `address`: it hands the job to the owner,
/// which admits it behind the sequence's last device work and counts it in flight until it ends, so the sequence's
/// lease is not returned while the job still reads or writes the sequence's state. `kill` stops the job.
fn copy_submit(address: Arc<Address<Owner>>, kill: Arc<AtomicBool>, purpose: Purpose) -> Submit {
    Box::new(move |job| address.post(Message::Copy(Box::new(CopyRequest { job, kill, purpose }))))
}

impl fmt::Debug for Message {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Message::Retired(_) => formatter.write_str("Message::Retired"),
            Message::HiddenCopied(rows) => write!(formatter, "Message::HiddenCopied(position {})", rows.first_position),
            Message::HiddenDelivered(_) => formatter.write_str("Message::HiddenDelivered"),
            Message::Sampled(sampled) => write!(formatter, "Message::Sampled(step {})", sampled.step),
            Message::Prepared(prepared) => write!(formatter, "Message::Prepared(step {})", prepared.step),
            Message::Reaped => formatter.write_str("Message::Reaped"),
            Message::StartFailed(failed) => write!(formatter, "Message::StartFailed({})", failed.serial),
            Message::Copy(_) => formatter.write_str("Message::Copy"),
            Message::CopyEnded => formatter.write_str("Message::CopyEnded"),
            Message::Looked(hit) => write!(formatter, "Message::Looked({})", hit.as_ref().map_or(0, Hit::position)),
            Message::Restored(result) => write!(formatter, "Message::Restored({result:?})"),
            Message::Captured(_) => formatter.write_str("Message::Captured"),
        }
    }
}

/// What an owner keeps of a retired quantum until the work queued on its lane has run.
pub(crate) enum Kept {
    /// A quantum of the sequence alone.
    Whole(Box<Quantum>),
    /// A pass of several sequences, which every member's owner keeps (the last to let go frees its input record), and the
    /// selection lease of this member's part.
    Part {
        batch: Arc<Quantum>,
        _lease: Option<SelectionLease>,
        /// The part's count in flight on its sequence, which is its own: the sequence settles when this part is let go
        /// of, whatever the other members do.
        _flight: Option<FlightGuard>,
    },
}

impl Kept {
    /// The marker of the upload of the quantum's input record.
    fn prepared(&self) -> Option<Marker> {
        match self {
            Kept::Whole(quantum) => quantum.prepared,
            Kept::Part { batch, .. } => batch.prepared,
        }
    }
}

/// A retired quantum (or this sequence's part of a pass) and how it ended for this sequence.
pub(crate) struct Retired {
    /// The number of the quantum that ran, which for a part is the pass's.
    pub(crate) serial: u64,
    pub(crate) start: u64,
    pub(crate) kind: QuantumKind,
    pub(crate) rows: usize,
    /// The rows of the quantum this sequence commits when it completed.
    pub(crate) committed: usize,
    pub(crate) ran: Option<Ran>,
    /// Whether the quantum is a pass of several sequences, which the owner of the workspace was told retired before this.
    pub(crate) pass: bool,
    pub(crate) kept: Kept,
    pub(crate) outcome: QuantumOutcome,
}

impl Retired {
    /// A quantum of one sequence that retired.
    pub(crate) fn whole(quantum: Quantum, outcome: QuantumOutcome) -> Retired {
        Retired {
            serial: quantum.id(),
            start: quantum.start_position(),
            kind: quantum.kind(),
            rows: quantum.rows(),
            committed: quantum.rows(),
            ran: quantum.graph_run(),
            pass: false,
            kept: Kept::Whole(Box::new(quantum)),
            outcome,
        }
    }

    /// Segment `segment` of the pass `batch` that retired as `outcome`: how it ended for its own sequence (a member that
    /// was cancelled concludes cancelled, whatever the pass did), with the lease its owner gives back.
    pub(crate) fn part(
        batch: &Arc<Quantum>,
        segment: usize,
        lease: Option<SelectionLease>,
        flight: Option<FlightGuard>,
        outcome: &QuantumOutcome,
    ) -> Retired {
        let concluded = match batch.conclusion_for(segment, outcome) {
            Ok(Outcome::Completed) => QuantumOutcome::Completed,
            Ok(Outcome::Cancelled) => QuantumOutcome::Killed,
            Ok(Outcome::Failed(cause)) => QuantumOutcome::DeviceFailed { cause },
            Err(error) => QuantumOutcome::DeviceFailed {
                cause: error.to_string(),
            },
        };
        let (start, rows) = batch
            .segments()
            .get(segment)
            .map_or((0, 0), |part| (part.start_position(), part.rows()));
        Retired {
            serial: batch.id(),
            start,
            kind: batch.kind(),
            rows,
            committed: batch.committed_rows(segment).unwrap_or(0),
            ran: None,
            pass: true,
            kept: Kept::Part {
                batch: Arc::clone(batch),
                _lease: lease,
                _flight: flight,
            },
            outcome: concluded,
        }
    }
}

/// A quantum that did not start, and why.
pub(crate) struct StartFailed {
    /// The quantum's number ([`Flying::id`]).
    pub(crate) serial: u64,
    pub(crate) error: StartError,
}

/// The parts of the engine every request shares, made once at startup.
pub(crate) struct Shared {
    pub(crate) core: Arc<Core>,
    pub(crate) config: Arc<ModelConfig>,
    pub(crate) workspace: Arc<Workspace>,
    pub(crate) executor: Arc<Executor>,
    /// What starts the quanta of the engine's sequences, each behind the quanta it must follow.
    pub(crate) order: Arc<Order<Executor>>,
    /// Where frames that tell the owner something post it.
    pub(crate) address: Arc<Address<Owner>>,
    /// Rows of the longest prompt chunk.
    pub(crate) chunk: usize,
    /// Rows the context holds.
    pub(crate) capacity: u64,
    /// What speculation needs of the artifact; `None` for an artifact without the draft layer.
    pub(crate) drafts: Option<Drafts>,
    /// Drafts a step of the DFlash2 drafter makes and its verification checks; `None` for an artifact without the
    /// drafter. A request that samples or is constrained does not use it.
    pub(crate) dflash2: Option<usize>,
    /// The prefix cache, when the plan granted it memory and it opened.
    pub(crate) prefix: Option<PrefixCacheHandle>,
    /// Set when the engine stops: no request begins after it.
    pub(crate) closing: AtomicBool,
    /// A test hook: the owner panics at the end of the turn in which it has admitted this many quanta.
    #[cfg(feature = "test-oracles")]
    pub(crate) panic_after_admissions: AtomicU64,
    /// A test hook: a speculative request ends with one decode quantum that feeds its last token, so that the state of
    /// the sequence is the state greedy decode leaves after the same tokens (the replay of the last verification
    /// runs).
    #[cfg(feature = "test-oracles")]
    pub(crate) settle_after_speculation: AtomicBool,
    /// A test hook: told, from the owner, of each prefix-cache copy job once it is admitted (`true` for a restore).
    #[cfg(feature = "test-oracles")]
    pub(crate) copy_admitted: OnceLock<Box<dyn Fn(bool) + Send + Sync>>,
    /// A test hook: told, from the owner, once a restore's rows are reserved and before the cache is asked to plan it.
    #[cfg(feature = "test-oracles")]
    pub(crate) restore_reserved: OnceLock<Box<dyn Fn() + Send + Sync>>,
    /// A test hook: told, from the owner, of each quantum once it is admitted, with its start position.
    #[cfg(feature = "test-oracles")]
    pub(crate) quantum_admitted: OnceLock<Box<dyn Fn(u64) + Send + Sync>>,
}

/// What speculation needs of the artifact's draft layer.
pub(crate) struct Drafts {
    /// The token of each row of the draft head (`text/draft_head_token_ids`).
    pub(crate) token_ids: Arc<[u32]>,
    /// The same ids on the device, which a draft's selection maps its row through.
    pub(crate) device_token_ids: DevicePtr,
    /// Drafts a step makes and its verification checks.
    pub(crate) depth: usize,
}

/// A step of a request.
#[derive(Clone, Copy, Debug)]
pub(super) enum Step {
    /// `rows` prompt rows; `last` ends the prompt.
    Prefill { rows: usize, last: bool },
    /// One row of the sequence.
    Decode,
    /// The MTP catch-up (or the DFlash2 drafter's context update) of the `rows` prompt rows just prefilled.
    PromptCatchUp { rows: usize },
    /// A speculative step of `drafts` drafts: the catch-up (which makes the first), the recursive drafts and the
    /// verification of the current token and the drafts.
    Speculate { drafts: usize },
    /// A verification of `rows` forced rows of a teacher-forced run, which hands every row's logits to its sink.
    Forced { rows: usize },
    /// The decode that feeds the last token of a speculative request (a test hook).
    #[cfg_attr(
        not(feature = "test-oracles"),
        allow(dead_code, reason = "only the test hook asks for it")
    )]
    Settle,
}

/// Whom a request that never started is told, and why it did not.
pub(crate) struct Rejected {
    pub(crate) error: AdmissionError,
    audience: Audience,
}

impl Rejected {
    pub(crate) fn new(error: AdmissionError, audience: Audience) -> Rejected {
        Rejected { error, audience }
    }

    /// The outcome that tells the audience the request was refused.
    pub(crate) fn into_finished(self) -> Finished {
        let outcome = GenerationOutcome {
            reason: FinishReason::Failed(format!("the request was refused: {}", self.error)),
            prompt_tokens: 0,
            generated: 0,
            elapsed: Duration::ZERO,
            cached_prompt_tokens: 0,
        };
        match self.audience {
            Audience::Greedy(listener) => Finished::Greedy(listener, outcome),
            Audience::Forced(sink) => Finished::Forced(sink, outcome),
        }
    }
}

/// The listener or sink of a request.
pub(crate) enum Audience {
    Greedy(Box<dyn GenerationListener>),
    Forced(Box<dyn LogitsSink>),
}

impl Mode {
    /// The request's listener or sink, for a request that did not start.
    fn into_audience(self) -> Audience {
        match self {
            Mode::Greedy { listener } | Mode::Speculative { listener, .. } | Mode::Sampled { listener, .. } => {
                Audience::Greedy(listener)
            }
            Mode::Forced { sink, .. } => Audience::Forced(sink),
        }
    }

    /// How the request that ran in this mode ended.
    fn into_finished(self, outcome: GenerationOutcome) -> Finished {
        match self {
            Mode::Greedy { listener } | Mode::Speculative { listener, .. } | Mode::Sampled { listener, .. } => {
                Finished::Greedy(listener, outcome)
            }
            Mode::Forced { sink, .. } => Finished::Forced(sink, outcome),
        }
    }
}

/// What a request does with the rows it computes.
pub(super) enum Mode {
    /// Selects each token on the device and reports it.
    Greedy { listener: Box<dyn GenerationListener> },
    /// Drafts with the artifact's draft layer, verifies the drafts and reports the tokens it commits.
    Speculative {
        listener: Box<dyn GenerationListener>,
        state: Box<Speculating>,
    },
    /// Copies each step's logits row to the host, which selects the token there ([`Sampled`]): the token is then the
    /// input of the next decode. One quantum is in flight at a time, and the request does not speculate. A sampled
    /// request that speculates is a [`Mode::Speculative`] one whose state holds the samplers.
    Sampled {
        listener: Box<dyn GenerationListener>,
        state: Box<Sampling>,
    },
    /// Feeds the given tokens and hands each step's logits row to a sink.
    Forced {
        forced: Vec<u32>,
        sink: Box<dyn LogitsSink>,
        /// Rows of one step after the prompt: 1 decodes a row at a time, more verifies that many rows at once
        /// (`teacher_forced.rs`).
        rows: usize,
    },
}

/// What a sampled request keeps between its quanta.
pub(super) struct Sampling {
    /// The request's sampler, which a selection frame holds while it selects, and the selection the owner waits for.
    rows: SampledRows,
    /// The token the host selected, which the next decode feeds.
    next: Option<u32>,
    /// The request's constraint, when it has one.
    constraint: Option<Constraining>,
}

/// Where a constrained request's constraint is, and what waits for it. It is in one place at a time: with the owner
/// (`held`), in the frame that prepares it (`preparing`, the number of the preparation), or in a selection's frame
/// (the rows' join waits for that).
struct Constraining {
    held: Option<Guarded>,
    preparing: Option<u64>,
    /// A quantum that retired before the preparation ended: its row waits, with the position of the token it selects.
    deferred: Option<Deferred>,
}

struct Deferred {
    lease: RowReadbackLease,
    position: u64,
}

impl Constraining {
    fn new(constraint: Guarded) -> Constraining {
        Constraining {
            held: Some(constraint),
            preparing: None,
            deferred: None,
        }
    }
}

impl Mode {
    /// The samplers and the join of a request whose host selects.
    fn sampled_rows(&mut self) -> Option<&mut SampledRows> {
        match self {
            Mode::Sampled { state, .. } => Some(&mut state.rows),
            Mode::Speculative { state, .. } => state.rows_mut(),
            Mode::Greedy { .. } | Mode::Forced { .. } => None,
        }
    }
}

/// How a request's speculation is served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    /// No drafts: the device selects each token (greedy) or the host does (sampled, constrained, hidden rows).
    Plain,
    /// The artifact's MTP layer drafts `depth` tokens; the host selects with samplers if the request is sampled.
    Mtp(usize),
    /// The DFlash2 drafter drafts `depth` tokens of a greedy request.
    Dflash2(usize),
}

impl Route {
    fn depth(self) -> Option<usize> {
        match self {
            Route::Plain => None,
            Route::Mtp(depth) | Route::Dflash2(depth) => Some(depth),
        }
    }
}

/// The route of a request that asks for `speculation`. `host_selects`: the host selects the tokens (a sampled,
/// constrained or hidden-row request); `mtp` and `dflash2`: the draft depths of the artifact's drafters. A constrained
/// request never speculates. The DFlash2 drafter drafts a greedy request's tokens only, so any other request on a
/// DFlash2 artifact decodes plainly; the MTP drafts of a sampled request stay.
fn route(
    speculation: &Speculation,
    host_selects: bool,
    constrained: bool,
    mtp: Option<usize>,
    dflash2: Option<usize>,
) -> Result<Route, AdmissionError> {
    match speculation {
        _ if constrained => Ok(Route::Plain),
        Speculation::Off => Ok(Route::Plain),
        Speculation::Artifact => match (dflash2, mtp) {
            (Some(_), _) if host_selects => Ok(Route::Plain),
            (Some(depth), _) => Ok(Route::Dflash2(depth)),
            (None, Some(depth)) => Ok(Route::Mtp(depth)),
            (None, None) => Err(AdmissionError::SpeculationUnavailable),
        },
    }
}

/// The state of a request that speculates by `route`, or `None` for a request that does not.
fn speculating(route: Route, capacity: u64, prompt: usize, max_tokens: usize) -> Option<Speculating> {
    match route {
        Route::Plain => None,
        Route::Mtp(depth) => Some(Speculating::new(depth, capacity, prompt, max_tokens)),
        Route::Dflash2(depth) => Some(Speculating::new(depth, capacity, prompt, max_tokens).with_drafter()),
    }
}

/// What a request does with the prefix cache.
#[derive(Debug, PartialEq, Eq)]
enum Caching {
    /// It neither restores a checkpoint nor stores one: the drafter's context of a restored prefix would be missing.
    Skips,
    /// It uses checkpoints of `kind` (those that hold the draft's state too, when it drafts).
    Uses(Option<CheckpointKind>),
}

/// The use of the prefix cache by a speculative request.
fn speculative_caching(state: &Speculating) -> Caching {
    if state.drafter().is_some() {
        Caching::Skips
    } else {
        Caching::Uses(state.drafting().then_some(CheckpointKind::Mtp))
    }
}

/// Why a request that asks for its hidden states is refused: it speculates (its drafts' rows are not the rows of the
/// tokens it commits), or the artifact has no draft layer whose state the rows come from.
fn hidden_refusal(hidden: bool, speculation: &Speculation, drafts: bool) -> Option<AdmissionError> {
    if !hidden {
        return None;
    }
    if !matches!(speculation, Speculation::Off) {
        return Some(AdmissionError::HiddenWithSpeculation);
    }
    if !drafts {
        return Some(AdmissionError::HiddenUnavailable);
    }
    None
}

/// `count` samplers of `config` for rows of `vocabulary` logits, all made when the request begins.
fn samplers(config: SamplingConfig, vocabulary: usize, count: usize) -> Result<Vec<TokenSampler>, SamplingError> {
    (0..count).map(|_| TokenSampler::new(config, vocabulary)).collect()
}

/// What a quantum of the request is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Role {
    /// A prompt chunk; `last` ends the prompt.
    Prefill { last: bool },
    /// One row of the sequence.
    Decode,
    /// The MTP catch-up of a prompt chunk.
    PromptCatchUp,
    /// The catch-up of a step, which makes its first draft.
    CatchUp,
    /// The DFlash2 drafter's context update, of a prompt chunk's rows or of the rows a verification committed.
    Context,
    /// The DFlash2 drafter's block of a step, which makes its drafts.
    Block,
    /// A recursive draft of a step.
    Draft,
    /// The verification of a step, or of several forced rows.
    Verify,
    /// The decode that feeds the last token of a speculative request.
    Settle,
}

/// A quantum admitted and not yet concluded.
pub(super) struct Flying {
    /// The quantum's number.
    pub(super) id: u64,
    pub(super) role: Role,
    pub(super) kind: QuantumKind,
    pub(super) start: u64,
    pub(super) rows: usize,
    pub(super) last_chunk: bool,
    pub(super) selection: Option<Arc<Selection>>,
    /// The readback a sampled quantum's row is copied to, leased to the owner until the row has been selected from.
    pub(super) row: Option<RowReadbackLease>,
    pub(super) retained: KvRetained,
    pub(super) admitted: Instant,
    /// Set to stop the quantum when the request stops while it is in flight.
    pub(super) kill: Arc<AtomicBool>,
    /// When the owner found it retired.
    pub(super) retired_at: Option<Instant>,
    pub(super) retired: Option<Retired>,
    /// The marker of the reservation made for the quantum on the owner's lane: what `retained` is held for when the
    /// quantum never starts.
    pub(super) reserved: Marker,
    /// The chunk ends at a checkpoint: once it completed its state is captured.
    pub(super) checkpoint: bool,
}

/// A retired quantum the owner keeps until the work it queued on its own lane has run: a quantum that was stopped before
/// its stages waited for its input record's upload leaves that copy reading the pinned record it holds.
struct Parked {
    /// The owner-lane work that reads what is held; the entry goes once the lane has reached it.
    marker: Marker,
    // Dropped with the entry: the quantum (none for a restore's reservation that no quantum took) and the staging of
    // the page tables the reservation uploaded.
    _quantum: Option<Kept>,
    _retained: KvRetained,
}

/// The state of one request.
pub(super) struct Run {
    pub(super) mode: Mode,
    pub(super) prompt: Vec<u32>,
    /// The tokens or rows the request produces: `max_tokens`, or the forced tokens.
    pub(super) wanted: usize,
    pub(super) stop_tokens: Vec<u32>,
    /// Prompt tokens admitted.
    pub(super) offset: usize,
    /// The position the next quantum starts at.
    pub(super) position: u64,
    /// Quanta admitted: the input record of the next of a greedy request is `admitted % 2`.
    pub(super) admitted: u64,
    /// Quanta admitted since the request last had none in flight: the input record of the next of a speculative one.
    pub(super) cursor: usize,
    /// Quanta the request keeps in flight at most.
    pub(super) depth: usize,
    /// Decode quanta admitted.
    pub(super) decodes: usize,
    /// Tokens selected and reported, or rows handed to the sink.
    pub(super) generated: usize,
    pub(super) in_flight: VecDeque<Flying>,
    /// Whether the workspace's owner holds the last quantum admitted, which the next one is chained behind. Held only
    /// while another quantum may follow: a quantum retires only when its references are dropped.
    pub(super) chain: bool,
    /// The sequence the request runs on, and what orders its quanta.
    sequence: SequenceId,
    shared: Arc<Shared>,
    /// The selection of the latest quantum that selects.
    pub(super) previous: Option<Arc<Selection>>,
    pub(super) stopping: Option<FinishReason>,
    /// Set from any thread to end the request at the next quantum boundary.
    cancel: CancelHandle,
    /// The sink the request hands every row's hidden state to; `None` for a request that does not.
    pub(super) hidden: Option<Hidden>,
    /// Turns of the owner that panicked during this request.
    pub(super) panics: u32,
    pub(super) began: Instant,
    /// Rows the context holds.
    pub(super) capacity: u64,
    _units: Units,
    /// The request's dealings with the prefix cache; `None` for a request that does not use it.
    pub(super) checkpoints: Option<PromptCheckpoints>,
    /// The kill flag of the first chunk when a restore precedes it: the restore's, which a failed copy sets.
    first_kill: Option<Arc<AtomicBool>>,
    /// Whether the request has offered a step to the workspace's passes and not left them since.
    offered: bool,
    /// What the restore's reservation uploaded, kept until the first chunk retires.
    carried: KvRetained,
    /// The restore's reservation on the owner's lane, which reads `carried`; `None` once a chunk has taken both.
    carried_marker: Option<Marker>,
    /// The sequence's place in the pool, given back when the run is dropped.
    lease: Lease,
}

impl Run {
    /// The next step to admit, or `None` once the request has no more (or has been told to stop).
    pub(super) fn next_step(&self, chunk: usize) -> Option<Step> {
        if self.stopping.is_some() {
            return None;
        }
        if let Mode::Speculative { state, .. } = &self.mode {
            return state.next_step(self, chunk);
        }
        if self.offset < self.prompt.len() {
            let rows = chunk.min(self.prompt.len() - self.offset);
            return Some(Step::Prefill {
                rows,
                last: self.offset + rows == self.prompt.len(),
            });
        }
        if let Mode::Forced { rows, forced, .. } = &self.mode
            && *rows > 1
        {
            // The rows still to be fed: the last forced token is only predicted.
            let left = forced.len().saturating_sub(1 + self.decodes);
            let room = self.capacity.saturating_sub(self.position) as usize;
            let rows = left.min(*rows).min(room);
            return (rows > 0).then_some(Step::Forced { rows });
        }
        // Decode `k` takes token `k` and selects token `k + 1`: it is wanted while a token after it is.
        if self.wanted > 0 && self.decodes + 1 < self.wanted && self.position < self.capacity {
            Some(Step::Decode)
        } else {
            None
        }
    }

    /// The selections `step` takes, one for each quantum of it that selects a token.
    pub(super) fn selections(&self, step: Step) -> usize {
        match step {
            // A sampled request's rows go to the host: it takes a row readback (`Owner::admit_quantum`), no selection.
            Step::Prefill { last, .. } => {
                usize::from(last && self.wanted > 0 && !self.copies_rows(Role::Prefill { last }))
            }
            Step::Decode => usize::from(!self.copies_rows(Role::Decode)),
            Step::PromptCatchUp { .. } | Step::Settle | Step::Forced { .. } => 0,
            // The drafts are always selected on the device; the verification only when the host does not.
            Step::Speculate { drafts } => {
                let drafter = matches!(&self.mode, Mode::Speculative { state, .. } if state.drafter().is_some());
                // The drafter's block selects its drafts in one quantum.
                (if drafter { 1 } else { drafts }) + usize::from(!self.copies_rows(Role::Verify))
            }
        }
    }

    /// Whether the run is a teacher-forced one whose sink takes the final hidden state beside each row.
    pub(super) fn wants_forced_hidden(&self) -> bool {
        matches!(&self.mode, Mode::Forced { sink, .. } if sink.wants_hidden())
    }

    /// Whether the logits rows of a quantum of `role` go to the host: it selects the tokens of every quantum of a
    /// sampled request that does not speculate, and of the prompt's and the verifications' of one that does (its drafts
    /// stay greedy and on the device); a teacher-forced run's sink reads the rows of the quanta that score.
    pub(super) fn copies_rows(&self, role: Role) -> bool {
        match &self.mode {
            Mode::Sampled { .. } => true,
            Mode::Speculative { state, .. } => {
                state.samples() && matches!(role, Role::Prefill { .. } | Role::Decode | Role::Verify)
            }
            Mode::Forced { rows, .. } => forced_copies_rows(*rows, role),
            Mode::Greedy { .. } => false,
        }
    }

    /// Whether a frame holds the request's constraint to prepare its next mask. Admission does not wait for it (the
    /// mask is prepared while the decode runs); the end of the request does.
    fn preparing(&self) -> bool {
        matches!(&self.mode, Mode::Sampled { state, .. }
            if state.constraint.as_ref().is_some_and(|constraining| constraining.preparing.is_some()))
    }

    /// Whether the request waits for the host's selections of a quantum's rows.
    fn selecting(&self) -> bool {
        match &self.mode {
            Mode::Sampled { state, .. } => state.rows.waiting(),
            Mode::Speculative { state, .. } => state.selecting(),
            Mode::Greedy { .. } | Mode::Forced { .. } => false,
        }
    }

    /// The listener of a generation that reports tokens.
    pub(super) fn listener(&mut self) -> Option<&mut dyn GenerationListener> {
        match &mut self.mode {
            Mode::Greedy { listener } | Mode::Speculative { listener, .. } | Mode::Sampled { listener, .. } => {
                Some(listener.as_mut())
            }
            Mode::Forced { .. } => None,
        }
    }

    /// Ends the request with `reason` unless it already ends: no more steps. The quanta in flight are told to stop and
    /// are discarded when they retire: see the module documentation.
    pub(super) fn stop(&mut self, reason: FinishReason) {
        if self.stopping.is_none() {
            self.stopping = Some(reason);
            kill_all(self.in_flight.iter().map(|flying| &flying.kill), |kill| {
                self.shared.executor.killed(kill);
            });
            if let Some(checkpoints) = self.checkpoints.as_mut() {
                checkpoints.stop();
            }
        }
        self.end_chain();
        self.leave_passes();
    }

    /// The request offers no more steps to the workspace's passes: they stop waiting for it, and a step it still has
    /// pending is told back as not started.
    pub(super) fn leave_passes(&mut self) {
        if mem::take(&mut self.offered) {
            self.shared.order.leave(self.sequence);
        }
    }

    /// The prompt tokens a restore put in the sequence: 0 on a miss.
    fn cached_prompt_tokens(&self) -> usize {
        self.checkpoints
            .as_ref()
            .map_or(0, |checkpoints| checkpoints.cached() as usize)
    }

    /// Nothing is chained behind the last quantum admitted from now on: the workspace's owner lets go of it.
    pub(super) fn end_chain(&mut self) {
        if mem::take(&mut self.chain) {
            self.shared.order.release(self.sequence);
        }
    }
}

/// How a request ended, to be told to its caller once the sequence can take another.
pub(crate) enum Finished {
    Greedy(Box<dyn GenerationListener>, GenerationOutcome),
    Forced(Box<dyn LogitsSink>, GenerationOutcome),
    /// The marker a shutdown waits for: the owner is idle, and the mailbox sends it once every outcome before it was
    /// told.
    Idle(Arc<Handoff<()>>),
}

impl Finished {
    pub(crate) fn deliver(self) {
        match self {
            // Never told as an outcome: the mailbox holds it until no delivery runs (`Turn::into_barrier`).
            Finished::Idle(_) => {}
            Finished::Greedy(mut listener, outcome) => listener.finished(outcome),
            Finished::Forced(mut sink, outcome) => sink.finished(outcome),
        }
    }
}

/// What the owner decides about a quantum before it is built (`Owner::admit_quantum`).
pub(super) struct Admission {
    pub(super) role: Role,
    pub(super) kind: QuantumKind,
    pub(super) rows: usize,
    /// The position of the quantum's first row in the cache it appends to.
    pub(super) start: u64,
    /// The cache the quantum appends to; none for a quantum that appends to no cache (the drafter's).
    pub(super) source: Option<KvSource>,
    /// Whether its launches overlap with the quantum before it (programmatic dependent launch).
    pub(super) pdl: bool,
    /// The logits rows it scores; a quantum that scores any takes a selection.
    pub(super) logits: usize,
    /// The rows each GDN layer replays from its checkpoint before the quantum's own rows.
    pub(super) replay: Vec<u32>,
    /// Whether it ends the prompt.
    pub(super) last_chunk: bool,
    /// Whether another quantum is admitted behind it before it may retire.
    pub(super) more: bool,
}

/// The owner of the engine's sequence. See the module documentation.
pub(crate) struct Owner {
    pub(super) shared: Arc<Shared>,
    /// The sequence's number, which the workspace's owner orders its quanta by.
    pub(super) id: SequenceId,
    pub(super) sequence: SequenceState,
    inbox: Consumer<Message, Single>,
    /// Whether the sequence holds the state of a request that ended: the next request starts with a reset.
    used: bool,
    pub(super) run: Option<Run>,
    parked: Vec<Parked>,
    /// Outcomes to tell once the claim is released: a request's end, and the refusals of requests posted meanwhile.
    finished: VecDeque<Finished>,
    /// A shutdown's wait for the owner to be idle: sent after the last outcome is told.
    watch: Option<Arc<Handoff<()>>>,
    /// The number of the next selection a sampled request publishes; it never restarts, whatever the request.
    sample_steps: u64,
    /// The prefix cache's copy jobs of this sequence admitted and not yet ended. The sequence's lease is not returned,
    /// and the owner is not idle, while one is in flight: a capture reads the sequence's state and a restore writes it.
    copies: usize,
}

impl fmt::Debug for Owner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Owner").finish_non_exhaustive()
    }
}

/// Maps an error of a step to the text a failed request carries.
pub(super) fn why<E: Display>(what: &'static str) -> impl Fn(E) -> String {
    move |error| format!("{what}: {error}")
}

impl Owner {
    pub(crate) fn new(
        shared: Arc<Shared>,
        id: SequenceId,
        sequence: SequenceState,
        inbox: Consumer<Message, Single>,
    ) -> Owner {
        let address = Arc::clone(&shared.address);
        sequence.on_settled(move || address.post(Message::Reaped));
        Owner {
            shared,
            id,
            sequence,
            inbox,
            used: false,
            run: None,
            parked: Vec::new(),
            finished: VecDeque::new(),
            watch: None,
            sample_steps: 0,
            copies: 0,
        }
    }

    /// Gives up a request that began and whose first turn will not run, because no worker takes frames: nothing was
    /// admitted yet, so dropping it leaves nothing on the device. The listener hears nothing: the caller learns from
    /// the error it is given.
    pub(crate) fn withdraw(&mut self) {
        match self.run.take() {
            Some(run) if run.in_flight.is_empty() && may_end(self.parked.len(), self.sequence.flight()) => {}
            Some(run) => {
                log::error!("a request with quanta in flight cannot be withdrawn");
                self.run = Some(run);
            }
            None => {}
        }
    }

    /// The sequence, for a look at its state between requests.
    pub(crate) fn sequence(&self) -> &SequenceState {
        &self.sequence
    }

    /// Ends the sequence: frees its caches and the graphs recorded against them. The last release of the pinned host
    /// arena waits for the device, so this is the owner's teardown, after the last quantum retired.
    pub(crate) fn close(self) {
        let sequence = self.release();
        if let Err(busy) = sequence.close() {
            log::error!(
                "the sequence still has {} quanta in flight at shutdown: its device memory stays allocated until the \
                 process exits",
                busy.in_flight
            );
            mem::forget(busy);
        }
    }

    /// Gives the sequence back without freeing it: the graphs recorded against its caches are gone.
    pub(crate) fn release(self) -> SequenceState {
        let Owner { shared, sequence, .. } = self;
        shared.executor.clear_graphs();
        sequence
    }

    /// The test hook: the owner records no graphs from now on, and runs every quantum stage by stage.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn disable_graphs(&mut self) {
        self.shared.executor.set_captures(false);
    }

    /// Starts a greedy request, a sampled one, or one that speculates with the artifact's draft layer. `units` is what the engine's
    /// admission answered. A refused request gives its listener back.
    pub(crate) fn begin(
        &mut self,
        units: Result<Units, AdmissionError>,
        request: GenerationRequest,
        cancel: CancelHandle,
        lease: Lease,
    ) -> Result<(), Rejected> {
        self.begin_with(units, request, cancel, lease, None)
    }

    /// Starts a request, which hands the hidden state of every row it runs to `hidden` when there is a sink. Such a
    /// request has the host select its tokens (as a sampled one), runs one quantum at a time, does not speculate and
    /// does not use the prefix cache, whose restored rows would have no hidden state to hand out.
    pub(crate) fn begin_with(
        &mut self,
        units: Result<Units, AdmissionError>,
        request: GenerationRequest,
        cancel: CancelHandle,
        lease: Lease,
        hidden: Option<Box<dyn HiddenSink>>,
    ) -> Result<(), Rejected> {
        let GenerationRequest {
            prompt,
            max_tokens,
            stop_tokens,
            speculation,
            sampling,
            constraint,
            listener,
        } = request;
        let units = match units {
            Ok(units) => units,
            Err(error) => return Err(Rejected::new(error, Audience::Greedy(listener))),
        };
        if let Some(error) = hidden_refusal(hidden.is_some(), &speculation, self.shared.drafts.is_some()) {
            return Err(Rejected::new(error, Audience::Greedy(listener)));
        }
        let host_selects = hidden.is_some() || !sampling.is_greedy() || constraint.is_some();
        let route = match route(
            &speculation,
            host_selects,
            constraint.is_some(),
            self.shared.drafts.as_ref().map(|drafts| drafts.depth),
            self.shared.dflash2,
        ) {
            Ok(route) => route,
            Err(error) => return Err(Rejected::new(error, Audience::Greedy(listener))),
        };
        let mode = if host_selects {
            if self.sequence.row_readbacks.is_exhausted() {
                // Refused before the prompt prefills: the last chunk would find no row to copy its logits to.
                let error = AdmissionError::Engine("the sequence has no row readback to sample from".to_string());
                return Err(Rejected::new(error, Audience::Greedy(listener)));
            }
            let vocabulary = self.shared.config.vocab_size;
            // One sampler for each row a quantum selects from at once: a verification's rows, or one.
            let rows = route.depth().map_or(1, |depth| depth + 1);
            let samplers = match samplers(sampling, vocabulary, rows) {
                Ok(samplers) => SampledRows::new(samplers),
                Err(error) => {
                    let error = AdmissionError::Engine(format!("the sampler: {error}"));
                    return Err(Rejected::new(error, Audience::Greedy(listener)));
                }
            };
            match speculating(route, self.shared.capacity, prompt.len(), max_tokens as usize) {
                None => Mode::Sampled {
                    listener,
                    state: Box::new(Sampling {
                        rows: samplers,
                        next: None,
                        constraint: constraint.map(|constraint| Constraining::new(Guarded::new(constraint))),
                    }),
                },
                Some(state) => Mode::Speculative {
                    listener,
                    state: Box::new(state.sampled(samplers)),
                },
            }
        } else {
            match speculating(route, self.shared.capacity, prompt.len(), max_tokens as usize) {
                None => Mode::Greedy { listener },
                Some(state) => Mode::Speculative {
                    listener,
                    state: Box::new(state),
                },
            }
        };
        self.start_run(units, lease, mode, prompt, max_tokens as usize, stop_tokens, cancel)?;
        if let Some(run) = self.run.as_mut() {
            run.hidden = hidden.map(Hidden::new);
        }
        self.look_up_prompt();
        // The first mask is prepared while the prompt prefills.
        self.prepare_constraint();
        Ok(())
    }

    /// Asks the prefix cache for the longest checkpoint of the prompt, unless the request cannot use one (a teacher-forced
    /// one cannot). A request that drafts matches only checkpoints that hold the draft's state too (its MTP cache and
    /// the seed row), and stores such checkpoints. The prompt's first chunk waits for the answer
    /// ([`PromptCheckpoints::holds`]).
    fn look_up_prompt(&mut self) {
        let Some(cache) = self.shared.prefix.clone() else {
            return;
        };
        let Some(run) = self.run.as_mut() else { return };
        if run.hidden.is_some() {
            return;
        }
        let kind = match &run.mode {
            Mode::Greedy { .. } | Mode::Sampled { .. } => None,
            Mode::Speculative { state, .. } => match speculative_caching(state) {
                Caching::Skips => return,
                Caching::Uses(kind) => kind,
            },
            Mode::Forced { .. } => return,
        };
        let address = Arc::clone(&self.shared.address);
        let tokens: Arc<[u32]> = Arc::from(run.prompt.as_slice());
        run.checkpoints = Some(PromptCheckpoints::begin(cache, tokens, kind, move |hit| {
            address.post(Message::Looked(hit));
        }));
    }

    /// Starts a teacher-forced run. A refused run gives its sink back.
    pub(crate) fn begin_forced(
        &mut self,
        units: Result<Units, AdmissionError>,
        prefix: Vec<u32>,
        forced: Vec<u32>,
        step_rows: usize,
        sink: Box<dyn LogitsSink>,
        lease: Lease,
    ) -> Result<(), Rejected> {
        let units = match units {
            Ok(units) => units,
            Err(error) => return Err(Rejected::new(error, Audience::Forced(sink))),
        };
        if let Some(error) = self.forced_rows_refusal(step_rows, sink.as_ref()) {
            return Err(Rejected::new(error, Audience::Forced(sink)));
        }
        let vocabulary = self.shared.config.vocab_size;
        if let Some((index, id)) = forced.iter().enumerate().find(|(_, id)| **id as usize >= vocabulary) {
            let error = AdmissionError::TokenOutsideVocabulary {
                id: *id,
                index: prefix.len() + index,
                vocabulary,
            };
            return Err(Rejected::new(error, Audience::Forced(sink)));
        }
        // The last forced token is only predicted, never fed: the rows end one before.
        let rows = (prefix.len() + forced.len()).saturating_sub(1);
        if rows as u64 > self.shared.capacity {
            let error = AdmissionError::PromptTooLong {
                tokens: rows,
                capacity: self.shared.capacity as usize,
            };
            return Err(Rejected::new(error, Audience::Forced(sink)));
        }
        let wanted = forced.len();
        if wanted == 0 {
            return Err(Rejected::new(AdmissionError::EmptyPrompt, Audience::Forced(sink)));
        }
        self.start_run(
            units,
            lease,
            Mode::Forced {
                forced,
                sink,
                rows: step_rows,
            },
            prefix,
            wanted,
            Vec::new(),
            CancelHandle::default(),
        )
    }

    /// Ends a request that was refused after the caller was told it was accepted (it was posted while a turn held the
    /// claim): its audience hears a failure, after the claim is released, like any request that ended.
    pub(crate) fn reject(&mut self, rejected: Rejected) {
        self.finished.push_back(rejected.into_finished());
    }

    #[allow(clippy::too_many_arguments, reason = "what a request carries into its run")]
    fn start_run(
        &mut self,
        units: Units,
        lease: Lease,
        mode: Mode,
        prompt: Vec<u32>,
        wanted: usize,
        stop_tokens: Vec<u32>,
        cancel: CancelHandle,
    ) -> Result<(), Rejected> {
        // Busy means a request is running: an outcome that waits to be told does not hold the sequence.
        if self.run.is_some() {
            return Err(Rejected::new(AdmissionError::Busy, mode.into_audience()));
        }
        if prompt.is_empty() {
            return Err(Rejected::new(AdmissionError::EmptyPrompt, mode.into_audience()));
        }
        let vocabulary = self.shared.config.vocab_size;
        if let Some((index, id)) = prompt.iter().enumerate().find(|(_, id)| **id as usize >= vocabulary) {
            let error = AdmissionError::TokenOutsideVocabulary {
                id: *id,
                index,
                vocabulary,
            };
            return Err(Rejected::new(error, mode.into_audience()));
        }
        let capacity = self.shared.capacity;
        if prompt.len() as u64 > capacity {
            let error = AdmissionError::PromptTooLong {
                tokens: prompt.len(),
                capacity: capacity as usize,
            };
            return Err(Rejected::new(error, mode.into_audience()));
        }
        let depth = match mode {
            Mode::Greedy { .. } => GREEDY_DEPTH,
            Mode::Speculative { .. } => self.shared.order.records(),
            Mode::Forced { .. } | Mode::Sampled { .. } => 1,
        };
        self.run = Some(Run {
            mode,
            prompt,
            wanted,
            stop_tokens,
            offset: 0,
            position: 0,
            admitted: 0,
            cursor: 0,
            depth,
            decodes: 0,
            generated: 0,
            in_flight: VecDeque::new(),
            chain: false,
            sequence: self.id,
            shared: Arc::clone(&self.shared),
            previous: None,
            stopping: None,
            cancel,
            hidden: None,
            panics: 0,
            began: Instant::now(),
            capacity,
            _units: units,
            checkpoints: None,
            first_kill: None,
            offered: false,
            carried: KvRetained::default(),
            carried_marker: None,
            lease,
        });
        Ok(())
    }

    /// The selections no quantum holds. A step that selects waits for them: the quanta in flight retire, and are
    /// dropped, and then they are free.
    fn selections_free(&self) -> usize {
        self.sequence.selections.len() - self.sequence.selections.leased()
    }

    /// The next step the request can admit now: it has room for one and, when the step selects, the selections to give
    /// it. A step that needs the host to know a token (a decode in a speculative request, a speculative step) waits for
    /// the quanta in flight.
    fn admissible(&self) -> Option<Step> {
        let run = self.run.as_ref()?;
        // The tokens the host is selecting decide the next step.
        // A hidden request's next quantum overwrites the rows being copied, and waits for a sink that is far behind.
        let copying = run.hidden.as_ref().is_some_and(Hidden::full);
        if copying || run.selecting() || run.in_flight.len() >= run.depth {
            return None;
        }
        // The prefix cache first: the lookup's answer, the restore's job, or a capture of the state the next chunk would
        // overwrite.
        if run.checkpoints.as_ref().is_some_and(PromptCheckpoints::holds) {
            return None;
        }
        let next = match run.next_step(self.shared.chunk) {
            Some(next) => next,
            None => self.settle_step()?,
        };
        // A sampled decode feeds the token the host selected: it waits for it.
        if matches!((&run.mode, next), (Mode::Sampled { state, .. }, Step::Decode) if state.next.is_none()) {
            return None;
        }
        let alone = matches!(next, Step::Speculate { .. } | Step::Settle)
            || matches!((&run.mode, next), (Mode::Speculative { .. }, Step::Decode));
        if alone && !run.in_flight.is_empty() {
            return None;
        }
        (run.selections(next) <= self.selections_free()).then_some(next)
    }

    /// One turn of the owner: takes in what arrived, concludes the quanta that retired in order, admits the next ones
    /// and ends the request when nothing is left.
    pub(crate) fn work(&mut self) {
        while let Some(message) = self.inbox.poll() {
            self.receive(message);
        }
        self.reap();
        self.free_retired();
        self.conclude();
        self.admit_ready();
        #[cfg(feature = "test-oracles")]
        self.test_panic();
        self.finish_if_done();
    }

    /// The test hook: panics once the request has admitted the number of quanta the hook names.
    #[cfg(feature = "test-oracles")]
    fn test_panic(&self) {
        let admitted = self.run.as_ref().map_or(0, |run| run.admitted);
        let after = self.shared.panic_after_admissions.load(Ordering::Acquire);
        if admitted >= after && self.shared.panic_after_admissions.swap(u64::MAX, Ordering::AcqRel) == after {
            panic!("the owner's test hook fired after {admitted} admissions");
        }
    }

    /// A turn of the owner panicked: the request fails, the quanta in flight are stopped, and the turns that follow
    /// conclude them and end the request as any failed one. A turn that panics again ends the request here, leaking what
    /// was in flight, which can no longer be accounted for.
    pub(crate) fn panicked(&mut self, payload: &(dyn Any + Send)) {
        let message = payload
            .downcast_ref::<&str>()
            .map(|text| (*text).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a panic with no message".to_string());
        log::error!("the owner panicked: {message}");
        let Some(run) = self.run.as_mut() else { return };
        run.panics += 1;
        if run.panics < 2 {
            return self.fail(format!("the owner panicked: {message}"));
        }
        let Some(mut run) = self.run.take() else { return };
        run.leave_passes();
        log::error!(
            "the owner of sequence {} panicked again: the request ends with its quanta still in flight, and the sequence leaves the pool",
            self.id.0
        );
        let cached_prompt_tokens = run.cached_prompt_tokens();
        mem::forget(mem::take(&mut run.in_flight));
        mem::forget(mem::take(&mut run.carried));
        // The quanta can no longer be accounted for, so a next request on this sequence could race them: it is not given back.
        mem::forget(run.lease);
        if mem::take(&mut run.chain) {
            self.shared.order.forget(self.id);
        }
        self.used = true;
        let outcome = GenerationOutcome {
            reason: FinishReason::Failed(format!("the owner panicked: {message}")),
            prompt_tokens: run.prompt.len(),
            generated: run.generated,
            elapsed: run.began.elapsed(),
            cached_prompt_tokens,
        };
        self.finished.push_back(run.mode.into_finished(outcome));
    }

    fn receive(&mut self, message: Message) {
        match message {
            Message::Retired(retired) => {
                let retired = *retired;
                // A quantum the workspace waited for is done: the owner of the workspace starts what waits. A pass was
                // told to it before the members' owners were.
                if !retired.pass {
                    self.shared.order.retired(self.id, retired.serial);
                }
                let waiting = self.run.as_mut().and_then(|run| {
                    run.in_flight.iter_mut().find(|flying| {
                        flying.retired.is_none()
                            && flying.start == retired.start
                            && flying.kind == retired.kind
                            && flying.rows == retired.rows
                    })
                });
                match waiting {
                    Some(flying) => {
                        flying.retired_at = Some(Instant::now());
                        flying.retired = Some(retired);
                    }
                    None => {
                        log::error!(
                            "a {:?} quantum at {} retired that the owner did not admit",
                            retired.kind,
                            retired.start
                        );
                        self.park(retired.kept, KvRetained::default());
                    }
                }
            }
            Message::HiddenCopied(rows) => self.hidden_copied(*rows),
            Message::HiddenDelivered(sink) => self.hidden_delivered(sink),
            Message::Sampled(sampled) => self.deliver_sample(sampled),
            Message::Prepared(prepared) => self.deliver_prepared(prepared),
            // The turn's `reap` is what the message is for.
            Message::Reaped => {}
            Message::StartFailed(failed) => self.start_failed(*failed),
            Message::Copy(request) => self.admit_copy(*request),
            Message::CopyEnded => self.copies = self.copies.saturating_sub(1),
            Message::Looked(hit) => self.looked(hit),
            Message::Restored(result) => self.restored(result),
            Message::Captured(node) => {
                if let Some(checkpoints) = self.run.as_mut().and_then(|run| run.checkpoints.as_mut()) {
                    checkpoints.captured(node);
                }
            }
        }
    }

    /// The cache answered the lookup: a hit is restored into the sequence before the prompt's first chunk, a miss
    /// prefills the prompt from its start. A hit that arrives after the request was told to stop is dropped, which
    /// unpins it.
    fn looked(&mut self, hit: Option<Hit>) {
        let Some(run) = self.run.as_mut() else { return };
        let Some(checkpoints) = run.checkpoints.as_mut() else {
            return;
        };
        checkpoints.looked();
        let Some(hit) = hit else { return };
        if run.stopping.is_some() {
            return;
        }
        if let Err(why) = self.restore(hit) {
            self.fail(why);
        }
    }

    /// Restores `hit` into the sequence: reserves the cache pages up to its position (and resets the sequence if it holds
    /// an earlier request's state: the restore must not be reset after it), and asks the cache to copy the checkpoint into
    /// them. The prompt then resumes from the hit's position, its first chunk behind the restore's job.
    fn restore(&mut self, hit: Hit) -> Result<(), String> {
        let shared = Arc::clone(&self.shared);
        let core = &shared.core;
        let position = u64::from(hit.position());
        let reset = mem::take(&mut self.used);
        let drafting = self.drafts();
        // The draft's last row of the checkpoint was paired with the token the stored prompt had at the position: with
        // the same token here it is the row a cold run computes, and is restored with the others.
        let whole = drafting
            && hit.paired().is_some()
            && self
                .run
                .as_ref()
                .and_then(|run| run.prompt.get(position as usize).copied())
                == hit.paired();
        let mut retained = KvRetained::default();
        let (_, reserved) = core
            .lane_table()
            .submit_ops(core.owner_lane(), false, |lane| {
                if reset {
                    self.sequence.reset(lane).map_err(why("resetting the sequence"))?;
                }
                let cache = &mut self.sequence.kv;
                cache
                    .reserve(lane, &core.pool, 0, position, &mut retained)
                    .map_err(why("reserving the restored rows"))?;
                cache.appended(position).map_err(why("the cache's frontier"))?;
                // The rows are part of the sequence from here: the copy's job is ordered before everything that reads them.
                cache.commit(position).map_err(why("the cache's frontier"))?;
                if drafting {
                    // The draft's cache holds the rows below the checkpoint's last position, and the last too when the
                    // prompt has the token the checkpoint paired it with: otherwise that row is recomputed.
                    let rows = if whole { position } else { position - 1 };
                    let mtp = self.sequence.mtp_kv.as_mut().ok_or("the sequence has no MTP cache")?;
                    mtp.reserve(lane, &core.pool, 0, rows, &mut retained)
                        .map_err(why("reserving the restored draft rows"))?;
                    mtp.appended(rows).map_err(why("the draft cache's frontier"))?;
                    mtp.commit(rows).map_err(why("the draft cache's frontier"))?;
                }
                Ok::<(), String>(())
            })
            .map_err(why("the reservation of the restored rows"))?;
        let refs = self
            .sequence
            .refs(KvSource::Base)
            .map_err(why("the sequence's references"))?
            .ordered_after(reserved, core.lanes());
        if drafting {
            // The draft cache's reservation is claimed too; the references above hold its view.
            self.sequence
                .refs(KvSource::Mtp)
                .map_err(why("the draft cache's references"))?;
        }
        let addresses =
            SequenceAddresses::new(refs, &shared.config).ok_or("the sequence has no cache to restore into")?;
        let Some(run) = self.run.as_mut() else { return Ok(()) };
        let Some(checkpoints) = run.checkpoints.as_mut() else {
            return Ok(());
        };
        #[cfg(feature = "test-oracles")]
        if let Some(hook) = shared.restore_reserved.get() {
            hook();
        }
        let (submit, answer) = (Arc::clone(&shared.address), Arc::clone(&shared.address));
        let kill = checkpoints.restore(
            hit,
            self.id,
            Arc::new(addresses),
            move |kill| copy_submit(submit, kill, Purpose::Restore),
            move |result| answer.post(Message::Restored(result)),
        );
        run.first_kill = Some(kill);
        run.offset = position as usize;
        run.position = position;
        run.carried = retained;
        run.carried_marker = Some(reserved);
        if let Mode::Speculative { state, .. } = &mut run.mode
            && drafting
        {
            // Unless the last row came whole, the draft pairs the hidden row it was restored with and the token at the
            // position, before the chunks do.
            state.resumed_at(position as u32, whole);
        }
        Ok(())
    }

    /// Whether the request drafts: its checkpoints hold the draft's state.
    fn drafts(&self) -> bool {
        matches!(self.run.as_ref().map(|run| &run.mode), Some(Mode::Speculative { state, .. }) if state.drafting())
    }

    /// The cache answered the restore.
    fn restored(&mut self, result: Result<u32, RestoreError>) {
        let Some(run) = self.run.as_mut() else { return };
        let Some(checkpoints) = run.checkpoints.as_mut() else {
            return;
        };
        match checkpoints.restored(result) {
            Restored::Resumed(_) => {}
            Restored::Cancelled => run.stop(FinishReason::Cancelled),
            // Nothing was copied: the prompt prefills from its start. A request that stops has no use for the sequence.
            Restored::Refused(error) => {
                if run.stopping.is_some() {
                    return;
                }
                log::debug!("the prefix cache did not restore: {error}");
                run.offset = 0;
                run.position = 0;
                run.first_kill = None;
                if let Mode::Speculative { state, .. } = &mut run.mode {
                    state.start_over();
                }
                let draft = self.sequence.mtp_kv.as_mut().map_or(Ok(()), |cache| cache.truncate(0));
                if let Err(error) = self.sequence.kv.truncate(0).and(draft) {
                    self.fail(format!("starting the prompt over: {error}"));
                }
            }
            // The sequence holds a half-written state: the request ends with the error. The chunk behind the job was
            // stopped with it.
            Restored::Failed(error) => self.fail(format!("restoring the prompt's prefix: {error}")),
        }
    }

    /// Asks the cache to capture the state of a chunk that completed at a checkpoint, when it is due and the capture
    /// before it is answered (its node is the parent of this one).
    fn ask_capture(&mut self) {
        let Some(run) = self.run.as_mut().filter(|run| run.stopping.is_none()) else {
            return;
        };
        let Some(checkpoints) = run.checkpoints.as_mut().filter(|checkpoints| checkpoints.due()) else {
            return;
        };
        let (submit, answer) = (Arc::clone(&self.shared.address), Arc::clone(&self.shared.address));
        checkpoints.ask(
            self.id,
            move |kill| copy_submit(submit, kill, Purpose::Capture),
            move |node| answer.post(Message::Captured(node)),
        );
    }

    /// Admits a copy job of this sequence behind the sequence's last device work, and counts it until it ends. A job
    /// that cannot start is dropped, which reports it failed to the cache, and is not counted.
    ///
    /// A job that arrives once the request is told to stop is dropped: nothing is behind it, and the cache hears it did
    /// not run (a capture then publishes nothing). Otherwise the next chunk is to start behind the job, on the sequence's
    /// edge, which is what `more` asks of the order: a restore is always followed by the first chunk, and a capture by
    /// the next step of the request, if it has one. The chunk is admitted only after this ([`PromptCheckpoints::holds`]).
    fn admit_copy(&mut self, request: CopyRequest) {
        let CopyRequest { job, kill, purpose } = request;
        let shared = Arc::clone(&self.shared);
        let chunk = shared.chunk;
        let Some(run) = self.run.as_mut().filter(|run| run.stopping.is_none()) else {
            return;
        };
        let more = match purpose {
            Purpose::Restore => true,
            Purpose::Capture => run.next_step(chunk).is_some(),
        };
        self.copies += 1;
        if let Err(error) = shared.order.admit(self.id, &shared.executor, job, kill, false, more) {
            self.copies -= 1;
            log::error!("a prefix-cache copy did not start: {error}");
            return;
        }
        run.chain |= more;
        #[cfg(feature = "test-oracles")]
        if let Some(hook) = shared.copy_admitted.get() {
            hook(purpose == Purpose::Restore);
        }
        if let Some(checkpoints) = run.checkpoints.as_mut() {
            match purpose {
                Purpose::Restore => checkpoints.restore_submitted(),
                Purpose::Capture => checkpoints.capture_submitted(),
            }
        }
    }

    /// A quantum the owner admitted did not start: it will never retire, so it is no longer in flight, and the request
    /// fails.
    fn start_failed(&mut self, failed: StartFailed) {
        let StartFailed { serial, error } = failed;
        if serial >= FIRST_COPY_SERIAL {
            // A copy job that did not start was dropped, and so reported failed: no request fails with it.
            self.copies = self.copies.saturating_sub(1);
            log::error!("a prefix-cache copy did not start: {error}");
            return;
        }
        let Some(run) = self.run.as_mut() else { return };
        let stopping = run.stopping.is_some();
        self.unstart(serial);
        let Some(run) = self.run.as_mut() else { return };
        // A step withdrawn because the request ended is no failure of its own.
        if !matches!(error, StartError::Withdrawn) || !stopping {
            run.stop(FinishReason::Failed(format!("starting the quantum: {error}")));
        }
    }

    /// Forgets the quantum numbered `id`, which did not start. What its reservation uploaded stays held until the lane has
    /// run it.
    fn unstart(&mut self, id: u64) {
        let Some(run) = self.run.as_mut() else { return };
        let Some(at) = run.in_flight.iter().position(|flying| flying.id == id) else {
            return;
        };
        let Some(flying) = run.in_flight.remove(at) else { return };
        self.keep(flying.reserved, None, flying.retained);
    }

    /// Takes in the host's selection of one row of a sampled quantum. An answer for a selection the request no longer
    /// waits for is ignored; the last answer of the quantum joins them: a sampled request's token is reported and
    /// fed to the next decode, a speculative one's rows are resolved against its drafts. Nothing is reported once the
    /// request stopped.
    fn deliver_sample(&mut self, sampled: Sampled) {
        let Some(run) = self.run.as_mut() else { return };
        let Some(rows) = run.mode.sampled_rows() else {
            return;
        };
        let step = sampled.step;
        let mut sampled = sampled;
        let constraint = sampled.constraint.take();
        if rows.answer(sampled) == Answered::Stale {
            log::warn!("the answer of selection {step} is not the one awaited, and is ignored");
            return;
        }
        // The constraint comes back with the token it accepted.
        if let (Mode::Sampled { state, .. }, Some(constraint)) = (&mut run.mode, constraint)
            && let Some(constraining) = &mut state.constraint
        {
            constraining.held = Some(constraint);
        }
        let Some(rows) = run.mode.sampled_rows() else {
            return;
        };
        // A failure ends the request at once; the answers still to come are awaited before it ends (`selecting`).
        if let Some(why) = rows.failure() {
            let why = format!("selecting the token: {why}");
            return self.fail(why);
        }
        if rows.waiting() || run.stopping.is_some() {
            return;
        }
        match &run.mode {
            Mode::Sampled { .. } => self.sampled_token(),
            _ => self.joined_rows(),
        }
    }

    /// The host selected the token of a sampled request that does not speculate: reports it, ends the request at a stop
    /// token or its budget, and otherwise keeps it for the next decode to feed.
    fn sampled_token(&mut self) {
        let Some(run) = self.run.as_mut() else { return };
        let Mode::Sampled { state, .. } = &run.mode else {
            return;
        };
        let Some(token) = state.rows.tokens().first().copied() else {
            return;
        };
        run.generated += 1;
        if let Some(listener) = run.listener() {
            listener.token(token, Instant::now());
        }
        if run.stop_tokens.contains(&token) {
            run.stop(FinishReason::Stop);
        } else if run.generated >= run.wanted {
            run.stop(FinishReason::Length);
        } else if let Mode::Sampled { state, .. } = &mut run.mode {
            state.next = Some(token);
            // The decode that feeds the token runs while the next mask is prepared; with no room for it, no mask
            // is needed.
            if run.position < run.capacity {
                self.prepare_constraint();
            }
        }
    }

    /// Publishes the frame that prepares the constraint's next mask, if the request has one and holds it.
    fn prepare_constraint(&mut self) {
        let Some(Run {
            mode: Mode::Sampled { state, .. },
            ..
        }) = self.run.as_mut()
        else {
            return;
        };
        let Some(constraining) = &mut state.constraint else {
            return;
        };
        let Some(constraint) = constraining.held.take() else {
            return;
        };
        let step = self.sample_steps;
        self.sample_steps += 1;
        constraining.preparing = Some(step);
        let address = Arc::clone(&self.shared.address);
        let frame = prepare_frame(self.id.0, step, constraint, move |prepared| {
            address.post(Message::Prepared(prepared))
        });
        self.shared.core.completions().at(Instant::now(), frame);
    }

    /// Takes in the preparation of the constraint's next mask. A row that retired meanwhile is selected from now; a
    /// failure ends the request, which waits for no other answer from this frame. An answer for a preparation the
    /// request no longer waits for is ignored.
    fn deliver_prepared(&mut self, prepared: Prepared) {
        let Some(Run {
            mode: Mode::Sampled { state, .. },
            stopping,
            ..
        }) = self.run.as_mut()
        else {
            return;
        };
        let Some(constraining) = &mut state.constraint else {
            return;
        };
        if constraining.preparing != Some(prepared.step) {
            log::warn!(
                "the preparation {} is not the one awaited, and is ignored",
                prepared.step
            );
            return;
        }
        constraining.preparing = None;
        let constraint = match prepared.result {
            Ok(constraint) => constraint,
            Err(why) => {
                constraining.deferred = None;
                return self.fail(why);
            }
        };
        let Some(Deferred { lease, position }) = constraining.deferred.take() else {
            constraining.held = Some(constraint);
            return;
        };
        if stopping.is_some() {
            // The request ended while its row waited: the row goes back to its pool.
            return;
        }
        self.publish_samples(lease, 1, position, Some(constraint));
    }

    /// Frees, in stream order on the owner's lane, the buffers the sequence's caches let go of: pages that a request
    /// discarded and tables a longer one replaced. A buffer is handed over only when its last holder is gone, and
    /// every holder was a quantum that retired, so the device is done with it; the free is queued behind the lane's
    /// work and nothing waits. A lane with no free marker is tried again on the next turn, with the buffers still
    /// queued.
    fn free_retired(&mut self) {
        let core = &self.shared.core;
        if core.released.is_empty() {
            return;
        }
        let freed = core.lane_table().submit_ops(core.owner_lane(), false, |ops| {
            let mut first_error = None;
            core.released.drain(|buffer| {
                if let Err(error) = ops.free_async(buffer) {
                    first_error.get_or_insert(error);
                }
            });
            first_error.map_or(Ok(()), Err)
        });
        match freed {
            Ok(_) | Err(SubmitError::Busy) => {}
            Err(error) => self.fail(format!("freeing the released buffers: {error}")),
        }
    }

    /// Lets go of the quanta whose owner-lane work has run.
    fn reap(&mut self) {
        let lanes = self.shared.core.lanes();
        self.parked.retain(|parked| !lanes.reached(parked.marker));
    }

    /// Keeps `quantum` until the work queued on the owner's lane for it has run. A frame posts to the owner when it
    /// has, so that the turn that follows lets it go (and a request that waits for it ends). The marker is looked at
    /// once: reached, the quantum goes now; not, the frame is registered, and it fires even if the marker is reached
    /// the next instant.
    fn park(&mut self, kept: Kept, retained: KvRetained) {
        let Some(marker) = kept.prepared() else { return };
        self.keep(marker, Some(kept), retained);
    }

    /// Keeps `quantum` and `retained` until the lane has reached `marker`, as `park` does for a quantum.
    fn keep(&mut self, marker: Marker, quantum: Option<Kept>, retained: KvRetained) {
        let core = &self.shared.core;
        if core.lanes().reached(marker) {
            return;
        }
        self.parked.push(Parked {
            marker,
            _quantum: quantum,
            _retained: retained,
        });
        let address = Arc::clone(&self.shared.address);
        core.completions().after_marker(
            marker,
            step(Kind::Reaped, self.id.0, move || address.post(Message::Reaped)),
        );
    }

    /// Ends the request as cancelled if a cancel was asked for: one relaxed atomic load. `conclude` calls it at the top
    /// of every iteration, and runs at least one, so every turn loads the flag once before it admits anything and again
    /// after each listener call that may have cancelled. The quanta in flight are stopped and
    /// retire as when a stop token ends a request; no step is admitted after it.
    fn observe_cancel(&mut self) {
        if let Some(run) = self.run.as_mut().filter(|run| run.cancel.is_cancelled()) {
            run.stop(FinishReason::Cancelled);
        }
    }

    /// `told` and, when a shutdown waits and nothing runs any more, the signal that comes after them.
    fn then_idle(&mut self, mut told: Vec<Finished>) -> Vec<Finished> {
        if self.run.is_none()
            && self.parked.is_empty()
            && self.copies == 0
            && let Some(idle) = self.watch.take()
        {
            told.push(Finished::Idle(idle));
        }
        told
    }

    /// Ends the request with a failure.
    pub(super) fn fail(&mut self, why: String) {
        if let Some(run) = self.run.as_mut() {
            run.stop(FinishReason::Failed(why));
        }
    }

    /// Concludes the quanta that retired, oldest first: commits their rows, reads their tokens, tells the listener.
    fn conclude(&mut self) {
        loop {
            // Also the turn's one look when nothing retired; a cancel from a listener's call ends the reports of the
            // quanta that retired behind it.
            self.observe_cancel();
            let Some(run) = self.run.as_mut() else { return };
            if !run.in_flight.front().is_some_and(|flying| flying.retired.is_some()) {
                return;
            }
            let Some(mut flying) = run.in_flight.pop_front() else {
                return;
            };
            let Some(retired) = flying.retired.take() else {
                return;
            };
            // The quantum that feeds the last token of a speculative request runs after the request ended.
            let live = run.stopping.is_none() || flying.role == Role::Settle;
            match &retired.outcome {
                QuantumOutcome::Completed if live => {
                    self.completed(&mut flying, &retired);
                    if flying.checkpoint {
                        self.checkpoint_reached();
                    }
                }
                // Behind the quantum that ended the request: its rows never become part of the sequence.
                QuantumOutcome::Completed | QuantumOutcome::Killed if !live => {}
                QuantumOutcome::Killed => self.fail("a quantum was stopped".to_string()),
                QuantumOutcome::Failed { stage, error } => {
                    if live {
                        self.fail(format!(
                            "a {:?} quantum failed at its stage {}: {error:?}",
                            flying.kind, stage.0
                        ));
                    }
                }
                // The device failed under the quantum, at no stage of its own: the same failure as a stage's.
                QuantumOutcome::DeviceFailed { cause } => {
                    if live {
                        self.fail(format!("the device failed under a {:?} quantum: {cause}", flying.kind));
                    }
                }
                QuantumOutcome::Completed => {}
            }
            self.park(retired.kept, flying.retained);
        }
    }

    /// The chunk that ends at a checkpoint completed: its state is due to be captured, unless the request ended with it.
    fn checkpoint_reached(&mut self) {
        let Some(run) = self.run.as_mut().filter(|run| run.stopping.is_none()) else {
            return;
        };
        if let Some(checkpoints) = run.checkpoints.as_mut() {
            checkpoints.concluded();
        }
    }

    /// A quantum of the live request completed.
    fn completed(&mut self, flying: &mut Flying, retired: &Retired) {
        let now = Instant::now();
        if retired.committed != flying.rows {
            return self.fail(format!(
                "a {:?} quantum committed {} of its {} rows",
                flying.kind, retired.committed, flying.rows
            ));
        }
        // The rows of a quantum become part of its cache; a verification's accepted prefix does, once it is known.
        let rows = flying.rows as u64;
        let committed = match flying.role {
            Role::Verify => Ok(()),
            // The drafter's rows are in its rings, which no cache holds.
            Role::Context | Role::Block => Ok(()),
            Role::PromptCatchUp | Role::CatchUp | Role::Draft => match self.sequence.mtp_kv.as_mut() {
                Some(cache) => cache.commit(rows),
                None => return self.fail("a draft quantum retired on a sequence with no MTP cache".to_string()),
            },
            Role::Prefill { .. } | Role::Decode | Role::Settle => self.sequence.kv.commit(rows),
        };
        if let Err(error) = committed {
            return self.fail(format!("committing the quantum's rows: {error}"));
        }
        let Some(run) = self.run.as_mut() else { return };
        let ran = retired.ran;
        let event = QuantumEvent {
            kind: if flying.role == Role::Decode {
                QuantumKind::Decode
            } else {
                flying.kind
            },
            rows: flying.rows,
            start_position: flying.start,
            admitted: flying.admitted,
            retired: flying.retired_at.unwrap_or(now),
            replayed: ran == Some(Ran::Replayed),
            recorded: matches!(ran, Some(Ran::Recorded | Ran::Rerecorded)),
        };
        if flying.role != Role::Settle
            && let Some(listener) = run.listener()
        {
            listener.quantum(event);
            if flying.last_chunk {
                listener.prefill_done(now);
            }
        }
        if run.hidden.is_some()
            && matches!(flying.role, Role::Prefill { .. } | Role::Decode)
            && let Err(why) = self.copy_hidden(flying)
        {
            return self.fail(why);
        }
        let Some(run) = self.run.as_mut() else { return };
        match flying.role {
            Role::Settle => return,
            Role::PromptCatchUp | Role::Context => return,
            Role::Block => return self.blocked(flying),
            Role::CatchUp | Role::Draft => return self.drafted(flying),
            Role::Verify if matches!(run.mode, Mode::Forced { .. }) => return self.forced_rows_retired(flying),
            Role::Verify if run.copies_rows(Role::Verify) => return self.sample(flying),
            Role::Verify => return self.verified(flying),
            Role::Prefill { .. } | Role::Decode => {}
        }
        if run.wanted == 0 || (flying.selection.is_none() && flying.row.is_none()) {
            // A prompt with nothing to generate, or a chunk that is not the last.
            if run.wanted == 0 && flying.last_chunk {
                run.stop(FinishReason::Length);
            }
            return;
        }
        if let Mode::Forced { .. } = run.mode {
            return self.forced_step_retired(flying);
        }
        if run.copies_rows(flying.role) {
            return self.sample(flying);
        }
        let token = flying.selection.as_ref().and_then(|selection| selection.token(0));
        let Some(token) = token else {
            return self.fail("the head selected no token: every logit was NaN or negative infinity".to_string());
        };
        run.generated += 1;
        if let Some(listener) = run.listener() {
            listener.token(token, now);
        }
        if run.stop_tokens.contains(&token) {
            run.stop(FinishReason::Stop);
        } else if run.generated >= run.wanted {
            run.stop(FinishReason::Length);
        } else if let (Mode::Speculative { state, .. }, Role::Prefill { last: true }) = (&mut run.mode, flying.role) {
            // The prompt's last chunk selected the first token: the speculative steps go on from it.
            state.first_token(token, &run.prompt);
        }
    }

    /// Publishes the frames that select the tokens of a sampled quantum from the rows it copied: the quantum retired, so
    /// the rows are in the readback. A prompt chunk's or a decode's last row takes the position after it; row `j` of a
    /// verification (which feeds the current token and the drafts) takes the position `j + 1` after the current token's,
    /// the position of the token it selects. Each frame owns one row's sampler and a share of the readback's lease, and
    /// the frames run side by side: they are independent.
    fn sample(&mut self, flying: &mut Flying) {
        let Some(lease) = flying.row.take() else {
            return self.fail("a sampled quantum retired with no row readback".to_string());
        };
        let verifying = flying.role == Role::Verify;
        let count = if verifying { flying.rows } else { 1 };
        let first_position = if verifying {
            flying.start + 1
        } else {
            flying.start + flying.rows as u64
        };
        let Some(run) = self.run.as_mut() else { return };
        let mut constraint = None;
        if let Mode::Sampled { state, .. } = &mut run.mode
            && let Some(constraining) = &mut state.constraint
        {
            match constraining.held.take() {
                Some(held) => constraint = Some(held),
                None if constraining.preparing.is_none() => {
                    return self.fail("the constraint was lost".to_string());
                }
                None => {
                    // The mask is still being prepared: the row waits for it, and is selected from when it is.
                    debug_assert!(constraining.deferred.is_none(), "a row already waits for the mask");
                    constraining.deferred = Some(Deferred {
                        lease,
                        position: first_position,
                    });
                    return;
                }
            }
        }
        if let Mode::Speculative { state, .. } = &mut run.mode {
            state.joining(match flying.role {
                Role::Verify => Joining::Verification(Verifying::of(flying)),
                role => Joining::Token {
                    first: matches!(role, Role::Prefill { .. }),
                },
            });
        }
        self.publish_samples(lease, count, first_position, constraint);
    }

    /// Publishes the frames that select from the `count` rows of `lease`, the first at `first_position`. The constraint,
    /// when the request has one, goes with the (only) row.
    fn publish_samples(
        &mut self,
        lease: RowReadbackLease,
        count: usize,
        first_position: u64,
        mut constraint: Option<Guarded>,
    ) {
        // The steps are the owner's, not the request's: an answer that comes late from a request that ended never
        // matches a selection of the next one.
        let first_step = self.sample_steps;
        self.sample_steps += count as u64;
        let shared = &self.shared;
        let Some(run) = self.run.as_mut() else { return };
        let Some(rows) = run.mode.sampled_rows() else {
            return;
        };
        if let Err(why) = rows.begin(first_step, count) {
            return self.fail(why);
        }
        let lease = Arc::new(lease);
        for row in 0..count {
            let Some(sampler) = rows.sampler(row) else {
                // Checked by `begin`. The rows published answer for themselves and the join then fails the request;
                // with none published nothing answers, so it fails here.
                let why = "a row's sampler is in use".to_string();
                if !rows.abandon_from(row, why.clone()) {
                    self.fail(why);
                }
                return;
            };
            let address = Arc::clone(&shared.address);
            let frame = sample_frame(
                self.id.0,
                first_step + row as u64,
                first_position + row as u64,
                sampler,
                constraint.take(),
                LeasedRow::new(Arc::clone(&lease), row),
                move |sampled| address.post(Message::Sampled(sampled)),
            );
            shared.core.completions().at(Instant::now(), frame);
        }
    }

    /// Queues the copy of the rows a retired quantum seeded, and of its logits row when it selected, to pinned memory on
    /// the owner's lane. The next quantum is admitted once the lane has reached the copy ([`Owner::hidden_copied`]).
    fn copy_hidden(&mut self, flying: &Flying) -> Result<(), String> {
        // The seed rows are the sequence's own, whether the quantum ran alone or as a part of a pass: each segment
        // normalizes into its own sequence's rows.
        let seeds = self
            .sequence
            .device
            .scratch
            .draft()
            .ok_or_else(|| "the sequence has no draft buffers to hold the hidden rows".to_string())?
            .seed_rows
            .ptr();
        let core = &self.shared.core;
        let hidden_bytes = flying.rows * self.shared.config.hidden_size * 2;
        let logits_bytes = if flying.row.is_some() {
            self.shared.config.vocab_size * 2
        } else {
            0
        };
        let mut loan = core
            .pool
            .acquire(hidden_bytes + logits_bytes)
            .map_err(why("the pinned rows for the hidden state"))?;
        if logits_bytes > 0 {
            // The logits row is the one the quantum copied in-stream to its row readback, which only this request's
            // quantum wrote: the workspace's logits are shared by every sequence and may be another's by now.
            let row = flying
                .row
                .as_ref()
                .and_then(|lease| lease.readback().row(0))
                .ok_or_else(|| "a hidden request's quantum retired with no logits row in its readback".to_string())?;
            if row.len() != logits_bytes {
                return Err("the logits row's readback has the wrong size".to_string());
            }
            loan.buffer_mut().as_mut_slice()[hidden_bytes..hidden_bytes + logits_bytes].copy_from_slice(row);
        }
        // One device copy is queued, so a failure to queue it leaves nothing pending on the loan.
        let (_, marker) = core
            .lane_table()
            .submit_ops(core.owner_lane(), false, |kernels| {
                // SAFETY: the loan is held by the message that follows the copy, which is offered only once the lane has
                // reached the marker, and the next quantum, which overwrites the sources, is admitted after that message.
                unsafe { kernels.copy_device_to_readback(loan.buffer_mut(), 0, seeds, hidden_bytes) }
                    .map_err(why("queueing the hidden rows' copy"))
            })
            .map_err(why("the hidden rows' copy"))?;
        let rows = HiddenRows {
            first_position: flying.start,
            rows: flying.rows,
            hidden_bytes,
            logits_bytes,
            loan,
        };
        if let Some(hidden) = self.run.as_mut().and_then(|run| run.hidden.as_mut()) {
            hidden.copying = true;
        }
        let address = Arc::clone(&self.shared.address);
        let frame = step_frame(self.id.0, move || address.post(Message::HiddenCopied(Box::new(rows))));
        core.completions().after_marker(marker, frame);
        Ok(())
    }

    /// The rows of a quantum reached pinned memory: they go to the sink, as a frame, when it is free.
    fn hidden_copied(&mut self, rows: HiddenRows) {
        if let Some(hidden) = self.run.as_mut().and_then(|run| run.hidden.as_mut()) {
            hidden.arrived(rows);
        }
        self.deliver_hidden();
    }

    /// A delivery ended: the sink is back, and the rows that waited for it go.
    fn hidden_delivered(&mut self, sink: Box<dyn HiddenSink>) {
        if let Some(hidden) = self.run.as_mut().and_then(|run| run.hidden.as_mut()) {
            hidden.returned(sink);
        }
        self.deliver_hidden();
    }

    /// Publishes the frame that hands the oldest waiting rows to the sink, when the sink is free.
    fn deliver_hidden(&mut self) {
        let Some((sink, rows)) = self
            .run
            .as_mut()
            .and_then(|run| run.hidden.as_mut())
            .and_then(Hidden::next_delivery)
        else {
            return;
        };
        let address = Arc::clone(&self.shared.address);
        let frame = delivery_frame(self.id.0, address, sink, rows);
        self.shared.core.completions().at(Instant::now(), frame);
    }

    /// Admits quanta while the request has steps and room for them.
    fn admit_ready(&mut self) {
        let chunk = self.shared.chunk;
        self.ask_capture();
        while let Some(next) = self.admissible() {
            if let Err(why) = self.admit(next) {
                self.fail(why);
                return;
            }
        }
        if let Some(run) = self.run.as_mut() {
            let ended = run.next_step(chunk).is_none();
            if ended {
                // Nothing will be chained behind the last quantum: let it retire.
                run.end_chain();
            }
            // A hidden request that offers no step soon (its sink is far behind, or its context is full) must not hold
            // the passes of the other sequences, which wait for each member of the last pass to offer or leave.
            // Only with nothing in flight: a step still pending would be withdrawn.
            let sink_behind = run.hidden.as_ref().map(Hidden::behind);
            if passes_choice(sink_behind, ended, run.in_flight.len()) == PassesChoice::Leave {
                run.leave_passes();
            }
        }
    }

    /// Admits the quantum or quanta of `next`.
    fn admit(&mut self, next: Step) -> Result<(), String> {
        match next {
            Step::Prefill { .. } | Step::Decode => self.admit_row(next),
            Step::PromptCatchUp { rows } if self.dflash2() => self.admit_prompt_context(rows),
            Step::PromptCatchUp { rows } => self.admit_prompt_catch_up(rows),
            Step::Speculate { drafts } if self.dflash2() => self.admit_dflash2_speculation(drafts),
            Step::Speculate { drafts } => self.admit_speculation(drafts),
            Step::Settle => self.admit_settle(),
            Step::Forced { rows } => self.admit_forced_rows(rows),
        }
    }

    /// Admits a prompt chunk or a decode row of the request's own tokens.
    fn admit_row(&mut self, next: Step) -> Result<(), String> {
        let Some(run) = self.run.as_mut() else { return Ok(()) };
        let start = run.position;
        let speculative = matches!(run.mode, Mode::Speculative { .. });
        let drafting = matches!(&run.mode, Mode::Speculative { state, .. } if state.drafting());
        // The DFlash2 drafter reads the tap rows every base quantum writes, not the MTP draft's seed rows.
        let seeds = self.shared.dflash2.is_none() && drafting;
        let hidden = run.hidden.is_some();
        // A chunk that ends at a checkpoint has nothing chained behind it: it retires, and the state it leaves is captured
        // before the next chunk starts (see `PromptCheckpoints`).
        let end = run.offset + next_rows(next);
        // A chunk of a request that drafts is captured after its catch-up instead: the draft's cache is not whole before.
        // The prompt's last chunk of a speculative request is never captured: its first step follows at once.
        let capturable = match (&run.mode, next) {
            (Mode::Speculative { .. }, Step::Prefill { last, .. }) => !drafting && !last,
            _ => true,
        };
        let checkpoint = matches!(next, Step::Prefill { .. })
            && capturable
            && run
                .checkpoints
                .as_ref()
                .is_some_and(|checkpoints| checkpoints.wants(end));
        let admission = match next {
            Step::Prefill { rows, last } => Admission {
                role: Role::Prefill { last },
                kind: QuantumKind::Prefill,
                rows,
                start,
                source: Some(KvSource::Base),
                pdl: false,
                logits: usize::from(last && run.wanted > 0),
                replay: Vec::new(),
                last_chunk: last,
                // A greedy request chains the next row behind the prompt; a speculative one chains the next chunk (or the
                // catch-up of this one) and waits for the first token after the last chunk.
                more: if speculative {
                    !last && !checkpoint
                } else {
                    run.depth > 1 && !checkpoint
                },
            },
            Step::Decode => Admission {
                role: Role::Decode,
                kind: QuantumKind::Decode,
                rows: 1,
                start,
                source: Some(KvSource::Base),
                pdl: start < OVERLAP_MAX_START_POSITION,
                logits: 1,
                replay: match &mut run.mode {
                    Mode::Speculative { state, .. } => state.take_replay(&self.shared.config),
                    _ => Vec::new(),
                },
                last_chunk: false,
                more: !speculative && run.depth > 1,
            },
            _ => return Ok(()),
        };
        let tokens: Vec<u32> = match next {
            Step::Prefill { rows, .. } => run.prompt[run.offset..run.offset + rows].to_vec(),
            _ => match &mut run.mode {
                Mode::Forced { forced, .. } => vec![forced[run.decodes]],
                Mode::Sampled { state, .. } => vec![state.next.take().ok_or("a decode has no token to feed")?],
                Mode::Speculative { state, .. } => vec![state.current().ok_or("a decode has no token to feed")?],
                Mode::Greedy { .. } => Vec::new(),
            },
        };
        let chained = match (&run.mode, next) {
            (Mode::Greedy { .. }, Step::Decode) => Some(
                run.previous
                    .clone()
                    .ok_or_else(|| "a decode quantum has no selection before it".to_string())?,
            ),
            _ => None,
        };
        let refs = self.admit_quantum_taking(
            admission,
            move |quantum| {
                let quantum = match chained {
                    Some(previous) => quantum.token_from(previous)?,
                    None => quantum.with_tokens(tokens)?,
                };
                if hidden || (seeds && matches!(next, Step::Prefill { .. })) {
                    quantum.seeding_draft()
                } else {
                    Ok(quantum)
                }
            },
            checkpoint,
        )?;
        let Some(run) = self.run.as_mut() else { return Ok(()) };
        if let Some(refs) = refs
            && let Some(addresses) = SequenceAddresses::new(refs, &self.shared.config)
            && let Some(checkpoints) = run.checkpoints.as_mut()
        {
            checkpoints.expect(end, Arc::new(addresses));
            if let Some(flying) = run.in_flight.back_mut() {
                flying.checkpoint = true;
            }
        }
        run.position += match next {
            Step::Prefill { rows, .. } => rows as u64,
            _ => 1,
        };
        match next {
            Step::Prefill { rows, last } => {
                run.offset += rows;
                if let (Mode::Speculative { state, .. }, true) = (&mut run.mode, drafting) {
                    state.prefilled(rows, last)?;
                }
            }
            _ => run.decodes += 1,
        }
        Ok(())
    }

    /// Admits one quantum: reserves its rows, uploads its input record, and starts it behind the one before (the
    /// chain). `shape` sets what the quantum's role needs of it (its tokens, the rows that seed it, where its token
    /// goes); the owner's own bookkeeping of the request is the caller's.
    pub(super) fn admit_quantum(
        &mut self,
        admission: Admission,
        shape: impl FnOnce(Quantum) -> Result<Quantum, QuantumError>,
    ) -> Result<(), String> {
        self.admit_quantum_taking(admission, shape, false).map(drop)
    }

    /// [`admit_quantum`](Self::admit_quantum) that, when `take` is set, also returns a copy of the references the
    /// quantum was admitted with: what a capture of the state it leaves reads.
    pub(super) fn admit_quantum_taking(
        &mut self,
        admission: Admission,
        shape: impl FnOnce(Quantum) -> Result<Quantum, QuantumError>,
        take: bool,
    ) -> Result<Option<SequenceRefs>, String> {
        let shared = Arc::clone(&self.shared);
        let core = &shared.core;
        let Some(run) = self.run.as_mut() else { return Ok(None) };
        let Admission {
            role,
            kind,
            rows,
            start,
            source,
            pdl,
            logits,
            replay,
            last_chunk,
            more,
        } = admission;
        if run.in_flight.is_empty() {
            run.cursor = 0;
        }
        // A hidden request's decode runs as a one-row verification: row-exact, as the corpus's reference is, and offered
        // to the workspace's passes so that concurrent requests read the weights once. It stays a decode for the
        // request: its row commits at once, and its listener hears a decode.
        let hidden_decode = role == Role::Decode && run.hidden.is_some();
        let kind = if hidden_decode { QuantumKind::Verify } else { kind };
        // A step that is not a verification is not offered: the passes do not wait for the request.
        if !hidden_decode && matches!(role, Role::Decode | Role::Settle | Role::Prefill { .. }) {
            run.leave_passes();
        }

        // The reservation (and, for the first quantum of a request after another, the reset of the sequence) on the
        // owner's lane. Its marker orders the quantum's stages behind it.
        let mut retained = mem::take(&mut run.carried);
        run.carried_marker = None;
        let (_, reserved) = core
            .lane_table()
            .submit_ops(core.owner_lane(), false, |lane| {
                if run.admitted == 0 && self.used {
                    self.sequence.reset(lane).map_err(why("resetting the sequence"))?;
                    self.used = false;
                }
                let cache = match source {
                    None => return Ok(()),
                    Some(KvSource::Base) => &mut self.sequence.kv,
                    Some(KvSource::Mtp) => match self.sequence.mtp_kv.as_mut() {
                        Some(cache) => cache,
                        None => return Err("the sequence has no MTP cache".to_string()),
                    },
                };
                cache
                    .reserve(lane, &core.pool, start, rows as u64, &mut retained)
                    .map_err(why("reserving the cache's rows"))?;
                cache.appended(rows as u64).map_err(why("the cache's frontier"))?;
                Ok(())
            })
            .map_err(why("the reservation"))?;
        let refs = match source {
            Some(source) => self.sequence.refs(source),
            None => Ok(self.sequence.refs_reading()),
        }
        .map_err(why("the sequence's references"))?
        .ordered_after(reserved, core.lanes())
        .with_replay_rows(replay);
        let taken = take.then(|| refs.clone());
        let mode = SubmitMode {
            pdl,
            row_exact: kind == QuantumKind::Verify,
        };
        let record = match run.mode {
            Mode::Speculative { .. } => run.cursor,
            _ => (run.admitted % ALTERNATING_RECORDS) as usize,
        };
        let mut quantum = Quantum::new(
            kind,
            rows,
            InputRecordRef::new(start),
            Arc::clone(&shared.workspace),
            refs,
            mode,
            Arc::clone(&shared.config),
            core.lane_table().clone(),
        )
        .map_err(why("the quantum"))?
        .with_input_record(shared.order.record(self.id, record));
        quantum = shape(quantum).map_err(why("the quantum's inputs"))?;
        quantum = quantum
            .with_logits_rows(logits)
            .map_err(why("the quantum's logits rows"))?;
        let mut row = None;
        if logits > 0 {
            if run.copies_rows(role) {
                let readbacks = &self.sequence.row_readbacks;
                let lease = if run.wants_forced_hidden() {
                    readbacks.acquire_hidden()
                } else {
                    readbacks.acquire()
                }
                .map_err(why("the row readback"))?;
                quantum = quantum.with_row_readback(Arc::clone(lease.readback()));
                row = Some(lease);
            } else {
                quantum =
                    quantum.with_selection_lease(self.sequence.selections.acquire().map_err(why("the selection"))?);
            }
        }
        // Everything the executor would refuse is refused before the record is queued for upload: a quantum dropped
        // after that would free the pinned record under the copy.
        quantum.validate().map_err(why("the quantum"))?;
        // A verification is offered for a pass, whose start uploads its record (its own, or the pass's).
        let offered = role == Role::Verify || hidden_decode;
        if !offered {
            let (_, prepared) = core
                .lane_table()
                .submit_ops(core.owner_lane(), false, |kernels| {
                    quantum.prepare(&core.pool, kernels).map_err(why("the input record"))
                })
                .map_err(why("the input record's upload"))?;
            quantum.prepared_after(prepared);
        }

        let selection = quantum.selection().cloned();
        // The restore's flag goes to the first chunk. On a restore of a drafting request that has to re-pair the
        // checkpoint's last row, the first quantum is the one-row catch-up that precedes the chunk: it shares the flag
        // (clones it) and the chunk takes it, so a failed copy stops both.
        let kill = if kind == QuantumKind::Draft {
            run.first_kill.clone()
        } else {
            run.first_kill.take()
        }
        .unwrap_or_default();
        let admitted = Instant::now();
        let id = quantum.id();

        run.in_flight.push_back(Flying {
            id,
            role,
            kind,
            start,
            rows,
            last_chunk,
            selection: selection.clone(),
            row,
            retained,
            admitted,
            kill: Arc::clone(&kill),
            retired_at: None,
            retired: None,
            reserved,
            checkpoint: false,
        });
        if selection.is_some() {
            run.previous = selection;
        }
        // A quantum with nothing admitted behind it would never retire while the workspace's owner held it: without
        // `more` the owner keeps nothing of it.
        run.chain = more;
        run.admitted += 1;
        run.cursor += 1;
        // It starts behind its sequence's last quantum and the workspace's, now or in a later turn of the workspace's
        // owner; a start that fails is told back as a message.
        // A start that fails now admits nothing behind it: the caller fails the request.
        let started = if offered {
            run.offered = true;
            shared.order.offer(self.id, &shared.executor, quantum, kill)
        } else {
            shared.order.admit(self.id, &shared.executor, quantum, kill, true, more)
        };
        if let Err(error) = started {
            self.unstart(id);
            return Err(why("starting the quantum")(error));
        }
        #[cfg(feature = "test-oracles")]
        if let Some(hook) = shared.quantum_admitted.get() {
            hook(start);
        }
        Ok(taken)
    }

    /// Ends the request when it has nothing in flight and nothing more to admit.
    fn finish_if_done(&mut self) {
        let chunk = self.shared.chunk;
        let Some(run) = self.run.as_ref() else { return };
        // Nothing the prefix cache was asked is left unanswered (a restore writes the sequence, a capture reads it, and an
        // answer that arrives after the request is gone has nobody to tell), and a capture still due is made first.
        let cache_busy = run
            .checkpoints
            .as_ref()
            .is_some_and(|checkpoints| checkpoints.pending() > 0 || (run.stopping.is_none() && checkpoints.holds()));
        if self.copies > 0 || cache_busy {
            return;
        }
        let ended = run.stopping.is_some() || run.next_step(chunk).is_none();
        let delivering = run.hidden.as_ref().is_some_and(Hidden::busy);
        if !run.in_flight.is_empty() || delivering || run.selecting() || run.preparing() || !ended {
            return;
        }
        // A restore's reservation that no chunk took (the request ended first) has its table upload queued on the owner's
        // lane, reading the staging it holds.
        if let Some(run) = self.run.as_mut()
            && let Some(marker) = run.carried_marker.take()
        {
            let retained = mem::take(&mut run.carried);
            self.keep(marker, None, retained);
        }
        // Quanta that were stopped may still have work queued on the owner's lane.
        self.reap();
        // The last holder's drop of the sequence's guard posts to this owner (`Owner::new`), and the turn it starts comes
        // back here.
        if !may_end(self.parked.len(), self.sequence.flight()) {
            return;
        }
        let Some(mut run) = self.run.take() else { return };
        run.leave_passes();
        self.sequence.kv.discard();
        if let Some(mtp) = self.sequence.mtp_kv.as_mut() {
            mtp.discard();
        }
        self.used = true;
        let cached_prompt_tokens = run.cached_prompt_tokens();
        let reason = match run.stopping {
            Some(reason) => reason,
            None if run.generated >= run.wanted => FinishReason::Length,
            None => FinishReason::ContextFull,
        };
        let outcome = GenerationOutcome {
            reason,
            prompt_tokens: run.prompt.len(),
            generated: run.generated,
            elapsed: run.began.elapsed(),
            cached_prompt_tokens,
        };
        self.finished.push_back(run.mode.into_finished(outcome));
    }
}

/// Whether a request that has nothing more to run may end, and its sequence go back to the pool: none of its quanta is
/// parked, and nothing else that reads or writes the sequence is left, whoever holds it (a restore's references live on
/// the cache's side until its job is dropped). Until then the request is not over. `Flight::settled` is asked last: when
/// it says no, the drop of the last guard wakes the owner.
pub(super) fn may_end(parked: usize, flight: &Flight) -> bool {
    parked == 0 && flight.settled()
}

/// What a request does about the shared passes when it has no step to offer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PassesChoice {
    /// Keep offering to the passes.
    Offer,
    /// Leave the passes, so that the other sequences need not wait for this one.
    Leave,
}

/// Sets each kill flag and tells `killed` of it, also of a flag already set (a failed restore may have set it first, and
/// what waits on the flag must hear of every kill, not only of the one that set it).
fn kill_all<'a>(flags: impl Iterator<Item = &'a Arc<AtomicBool>>, killed: impl Fn(&Arc<AtomicBool>)) {
    for flag in flags {
        flag.store(true, Ordering::Release);
        killed(flag);
    }
}

/// Whether a request leaves the shared passes: only a hidden request (`sink_behind` is its sink's state, `None` for
/// any other request) whose sink is behind or that has no next step (`ended`), and only with nothing in flight, since
/// a step still pending would be withdrawn.
fn passes_choice(sink_behind: Option<bool>, ended: bool, in_flight: usize) -> PassesChoice {
    match sink_behind {
        Some(behind) if (behind || ended) && in_flight == 0 => PassesChoice::Leave,
        _ => PassesChoice::Offer,
    }
}

/// Whether a quantum of `role` in a teacher-forced run of `rows` rows a step copies its logits row to the run's readback.
/// The workspace's logits belong to the next sequence's quantum once this one retires, so the row is copied by the
/// quantum itself, never by a copy queued after its retirement.
fn forced_copies_rows(rows: usize, role: Role) -> bool {
    match role {
        Role::Prefill { last } => last,
        Role::Decode => true,
        Role::Verify => rows > 1,
        _ => false,
    }
}

/// The prompt rows a prefill step takes: 0 for any other.
fn next_rows(step: Step) -> usize {
    match step {
        Step::Prefill { rows, .. } => rows,
        _ => 0,
    }
}

/// A frame that runs `work` once, in the control kind.
fn step_frame(chain: usize, work: impl FnOnce() + Send + 'static) -> Box<dyn Frame> {
    step(Kind::Logits, chain, work)
}

impl Turn for Owner {
    type Message = Message;
    type Finished = Finished;

    fn turn(&mut self) {
        if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| self.work())) {
            self.panicked(payload.as_ref());
        }
    }

    fn take_finished(&mut self) -> Vec<Finished> {
        let told = self.finished.drain(..).collect();
        self.then_idle(told)
    }

    fn deliver(finished: Finished) {
        finished.deliver();
    }

    fn abandon(&mut self, why: &str) -> Vec<Finished> {
        // Two panics end a request at once, leaving what was in flight.
        let message = why.to_string();
        self.panicked(&message);
        self.panicked(&message);
        let told = self.finished.drain(..).collect();
        self.then_idle(told)
    }

    fn shut(&mut self, idle: Arc<Handoff<()>>) {
        if let Some(run) = self.run.as_mut() {
            run.stop(FinishReason::Cancelled);
        }
        // Always told through the outcomes' own path, after them: also with nothing running, because a turn that ended
        // the request may have its outcome in a list not yet told.
        if self.run.is_none() && self.parked.is_empty() && self.copies == 0 {
            self.finished.push_back(Finished::Idle(idle));
        } else {
            self.watch = Some(idle);
        }
    }

    fn into_barrier(finished: Finished) -> Result<Arc<Handoff<()>>, Finished> {
        match finished {
            Finished::Idle(idle) => Ok(idle),
            other => Err(other),
        }
    }

    fn idle(&self) -> bool {
        self.run.is_none() && self.parked.is_empty() && self.copies == 0 && self.finished.is_empty()
    }
}

#[cfg(test)]
mod passes_choice_tests {
    use super::*;

    #[test]
    fn a_sink_behind_with_nothing_in_flight_leaves() {
        assert_eq!(passes_choice(Some(true), false, 0), PassesChoice::Leave);
    }

    #[test]
    fn a_sink_behind_with_a_step_in_flight_stays() {
        assert_eq!(passes_choice(Some(true), false, 1), PassesChoice::Offer);
    }

    #[test]
    fn no_next_step_with_nothing_in_flight_leaves() {
        assert_eq!(passes_choice(Some(false), true, 0), PassesChoice::Leave);
    }

    #[test]
    fn no_next_step_with_a_step_in_flight_stays() {
        assert_eq!(passes_choice(Some(false), true, 2), PassesChoice::Offer);
    }

    #[test]
    fn a_sink_that_keeps_up_with_a_next_step_offers() {
        assert_eq!(passes_choice(Some(false), false, 0), PassesChoice::Offer);
    }

    #[test]
    fn a_request_without_a_hidden_sink_never_leaves() {
        for ended in [false, true] {
            assert_eq!(passes_choice(None, ended, 0), PassesChoice::Offer);
        }
    }
}

#[cfg(test)]
mod kill_tests {
    use std::cell::RefCell;

    use super::*;

    #[test]
    fn every_flag_is_set_and_told_also_one_that_was_set_already() {
        let (fresh, early) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(true)));
        let told = RefCell::new(Vec::new());
        kill_all([&fresh, &early].into_iter(), |flag| {
            told.borrow_mut().push(Arc::clone(flag))
        });
        assert!(fresh.load(Ordering::Acquire) && early.load(Ordering::Acquire));
        let told = told.into_inner();
        assert_eq!(told.len(), 2);
        assert!(Arc::ptr_eq(&told[0], &fresh) && Arc::ptr_eq(&told[1], &early));
    }
}

#[cfg(test)]
mod routing_tests {
    use super::*;

    const DEPTH: usize = 6;

    #[test]
    fn a_greedy_request_on_a_dflash2_artifact_drafts_with_the_drafter() {
        let route = route(&Speculation::Artifact, false, false, Some(3), Some(DEPTH)).unwrap();
        assert_eq!(route, Route::Dflash2(DEPTH));
        let state = speculating(route, 4096, 30, 128).unwrap();
        assert!(state.drafter().is_some());
        assert_eq!(
            speculative_caching(&state),
            Caching::Skips,
            "the drafter's context is not restored"
        );
    }

    #[test]
    fn a_sampled_or_constrained_request_on_a_dflash2_artifact_decodes_plainly() {
        // `host_selects`: a sampled request, a constrained one and a hidden-row one.
        for constrained in [false, true] {
            let route = route(&Speculation::Artifact, true, constrained, None, Some(DEPTH)).unwrap();
            assert_eq!(route, Route::Plain, "constrained: {constrained}");
            assert!(speculating(route, 4096, 30, 128).is_none());
        }
        let greedy_constrained = route(&Speculation::Artifact, false, true, None, Some(DEPTH)).unwrap();
        assert_eq!(greedy_constrained, Route::Plain);
        assert_eq!(
            route(&Speculation::Off, false, false, None, Some(DEPTH)).unwrap(),
            Route::Plain
        );
    }

    #[test]
    fn an_mtp_artifact_keeps_its_drafts_for_a_sampled_request_but_not_a_constrained_one() {
        assert_eq!(
            route(&Speculation::Artifact, true, false, Some(3), None).unwrap(),
            Route::Mtp(3)
        );
        assert_eq!(
            route(&Speculation::Artifact, false, false, Some(3), None).unwrap(),
            Route::Mtp(3)
        );
        assert_eq!(
            route(&Speculation::Artifact, true, true, Some(3), None).unwrap(),
            Route::Plain
        );
        let state = speculating(Route::Mtp(3), 4096, 30, 128).unwrap();
        assert!(state.drafter().is_none());
        assert_eq!(speculative_caching(&state), Caching::Uses(Some(CheckpointKind::Mtp)));
    }

    #[test]
    fn a_request_for_drafts_an_artifact_cannot_make_is_refused() {
        assert!(matches!(
            route(&Speculation::Artifact, false, false, None, None),
            Err(AdmissionError::SpeculationUnavailable)
        ));
    }

    #[test]
    fn a_forced_step_that_scores_copies_its_row_in_the_quantum() {
        for rows in [1, 4] {
            assert!(forced_copies_rows(rows, Role::Prefill { last: true }), "{rows} rows");
            assert!(forced_copies_rows(rows, Role::Decode), "{rows} rows");
        }
        assert!(forced_copies_rows(4, Role::Verify));
    }

    #[test]
    fn a_forced_quantum_that_scores_nothing_copies_nothing() {
        assert!(!forced_copies_rows(1, Role::Prefill { last: false }));
        assert!(!forced_copies_rows(1, Role::Settle));
        assert!(!forced_copies_rows(1, Role::Verify));
    }

    #[test]
    fn the_frames_of_two_sequences_are_never_on_one_chain() {
        let hash = |frame: Box<dyn Frame>| frame.header().id_hash();
        for kind in [Kind::Reaped, Kind::Logits] {
            assert_ne!(hash(step(kind, 0, || {})), hash(step(kind, 1, || {})), "{kind:?}");
        }
        assert_ne!(
            hash(step_frame(0, || {})),
            hash(step_frame(1, || {})),
            "the control frame"
        );
        assert_eq!(
            hash(step_frame(2, || {})),
            hash(step_frame(2, || {})),
            "one sequence keeps its chain"
        );
    }

    #[test]
    fn hidden_states_are_refused_with_speculation_or_without_a_draft_layer() {
        assert_eq!(hidden_refusal(false, &Speculation::Artifact, false), None);
        assert_eq!(hidden_refusal(true, &Speculation::Off, true), None);
        assert_eq!(
            hidden_refusal(true, &Speculation::Artifact, true),
            Some(AdmissionError::HiddenWithSpeculation)
        );
        assert_eq!(
            hidden_refusal(true, &Speculation::Off, false),
            Some(AdmissionError::HiddenUnavailable)
        );
    }
}
