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
//! it), the copy of a logits row to pinned memory (a teacher-forced run), and the running of the work a parked quantum
//! waited for on the owner's lane. Nothing waits for them.
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
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::frames::Frame;
use euhedral_data_structures::queues::{Consumer, Single};
use euhedral_gpu::Ran;
use euhedral_graph::QuantumOutcome;
use euhedral_inference_cuda::{DevicePtr, PooledPinned, SubmitMode};

use super::listener::GenerationListener;
use super::mailbox::{Address, Turn};
use super::mtp::{Joining, Speculating, Verifying};
use super::order::Order;
use super::request::{
    AdmissionError, CancelHandle, FinishReason, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
};
use super::sampled::{Answered, Guarded, LeasedRow, Prepared, Sampled, SampledRows, prepare_frame, sample_frame};
use super::teacher_forced::LogitsSink;
use crate::frames::{Kind, step};
use crate::handoff::Handoff;
use crate::model::ModelConfig;
use crate::quantum::{Executor, StartError};
use crate::runtime::{Core, Units};
use crate::sampling::{SamplingConfig, SamplingError, TokenSampler};
use crate::stages::attention::KvSource;
use crate::stages::sampling::{RowReadbackLease, Selection};
use crate::stages::{InputRecordRef, Quantum, QuantumError, QuantumKind, SubmitError};
use crate::state::kv::KvRetained;
use crate::state::{SequenceId, SequenceState};
use crate::workspace::{BufferSlot, Workspace};

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
    /// The logits row of a teacher-forced step reached pinned memory.
    Logits(LogitsCopied),
    /// The host selected the token of a sampled request (or could not).
    Sampled(Sampled),
    /// The host prepared the next mask of a constrained request (or could not).
    Prepared(Prepared),
    /// The work a parked quantum waited for on the owner's lane has run: the turn lets the quantum go.
    Reaped,
    /// A quantum the owner admitted did not start.
    StartFailed(Box<StartFailed>),
}

impl fmt::Debug for Message {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Message::Retired(_) => formatter.write_str("Message::Retired"),
            Message::Logits(copied) => write!(formatter, "Message::Logits(step {})", copied.step),
            Message::Sampled(sampled) => write!(formatter, "Message::Sampled(step {})", sampled.step),
            Message::Prepared(prepared) => write!(formatter, "Message::Prepared(step {})", prepared.step),
            Message::Reaped => formatter.write_str("Message::Reaped"),
            Message::StartFailed(failed) => write!(formatter, "Message::StartFailed({})", failed.serial),
        }
    }
}

/// A retired quantum and how it ended.
pub(crate) struct Retired {
    pub(crate) quantum: Quantum,
    pub(crate) outcome: QuantumOutcome,
}

/// A quantum that did not start, and why.
pub(crate) struct StartFailed {
    /// The quantum's number ([`Flying::id`]).
    pub(crate) serial: u64,
    pub(crate) error: StartError,
}

/// The logits row of step `step` in pinned memory.
pub(crate) struct LogitsCopied {
    pub(crate) step: usize,
    pub(crate) loan: PooledPinned,
}

/// The parts of the engine every request shares, made once at startup.
pub(crate) struct Shared {
    pub(crate) core: Arc<Core>,
    pub(crate) config: Arc<ModelConfig>,
    pub(crate) workspace: Arc<Workspace>,
    pub(crate) executor: Arc<Executor>,
    /// What starts the quanta of the engine's sequences, each behind the quanta it must follow.
    pub(crate) order: Order<Executor>,
    /// Where frames that tell the owner something post it.
    pub(crate) address: Arc<Address<Owner>>,
    /// Rows of the longest prompt chunk.
    pub(crate) chunk: usize,
    /// Rows the context holds.
    pub(crate) capacity: u64,
    /// What speculation needs of the artifact; `None` for an artifact without the draft layer.
    pub(crate) drafts: Option<Drafts>,
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
    /// The MTP catch-up of the `rows` prompt rows just prefilled.
    PromptCatchUp { rows: usize },
    /// A speculative step of `drafts` drafts: the catch-up (which makes the first), the recursive drafts and the
    /// verification of the current token and the drafts.
    Speculate { drafts: usize },
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
    /// A recursive draft of a step.
    Draft,
    /// The verification of a step.
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
    pub(super) retired: Option<(Quantum, QuantumOutcome)>,
}

/// A retired quantum the owner keeps until the work it queued on its own lane has run: a quantum that was stopped before
/// its stages waited for its input record's upload leaves that copy reading the pinned record it holds.
struct Parked {
    quantum: Quantum,
    // Dropped with the quantum: the staging of the page tables the reservation uploaded.
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
    /// A teacher-forced step's logits row is being copied.
    pub(super) waiting_logits: bool,
    /// Turns of the owner that panicked during this request.
    pub(super) panics: u32,
    pub(super) began: Instant,
    /// Rows the context holds.
    pub(super) capacity: u64,
    _units: Units,
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
            Step::PromptCatchUp { .. } | Step::Settle => 0,
            // The drafts are always selected on the device; the verification only when the host does not.
            Step::Speculate { drafts } => drafts + usize::from(!self.copies_rows(Role::Verify)),
        }
    }

    /// Whether the logits rows of a quantum of `role` go to the host, which selects the tokens: every quantum of a
    /// sampled request that does not speculate, and the prompt's and the verifications' of one that does. Its drafts
    /// stay greedy and on the device.
    pub(super) fn copies_rows(&self, role: Role) -> bool {
        match &self.mode {
            Mode::Sampled { .. } => true,
            Mode::Speculative { state, .. } => {
                state.samples() && matches!(role, Role::Prefill { .. } | Role::Decode | Role::Verify)
            }
            Mode::Greedy { .. } | Mode::Forced { .. } => false,
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
            for flying in &self.in_flight {
                flying.kill.store(true, Ordering::Release);
            }
        }
        self.end_chain();
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
    /// The cache the quantum appends to.
    pub(super) source: KvSource,
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
        }
    }

    /// Gives up a request that began and whose first turn will not run, because no worker takes frames: nothing was
    /// admitted yet, so dropping it leaves nothing on the device. The listener hears nothing: the caller learns from
    /// the error it is given.
    pub(crate) fn withdraw(&mut self) {
        match self.run.take() {
            Some(run) if run.in_flight.is_empty() => {}
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
        let Owner { shared, sequence, .. } = self;
        shared.executor.clear_graphs();
        if let Err(busy) = sequence.close() {
            log::error!(
                "the sequence still has {} quanta in flight at shutdown: its device memory stays allocated until the \
                 process exits",
                busy.in_flight
            );
            mem::forget(busy);
        }
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
        let mode = match speculation {
            // The host selects the tokens: its drafts, if it speculates, stay greedy.
            _ if !sampling.is_greedy() || constraint.is_some() => {
                if self.sequence.row_readbacks.is_exhausted() {
                    // Refused before the prompt prefills: the last chunk would find no row to copy its logits to.
                    let error = AdmissionError::Engine("the sequence has no row readback to sample from".to_string());
                    return Err(Rejected::new(error, Audience::Greedy(listener)));
                }
                let vocabulary = self.shared.config.vocab_size;
                let drafts = match speculation {
                    // A constrained request never speculates.
                    _ if constraint.is_some() => None,
                    Speculation::Off => None,
                    Speculation::Artifact => match self.shared.drafts.as_ref() {
                        Some(drafts) => Some(drafts.depth),
                        None => {
                            return Err(Rejected::new(
                                AdmissionError::SpeculationUnavailable,
                                Audience::Greedy(listener),
                            ));
                        }
                    },
                };
                // One sampler for each row a quantum selects from at once: a verification's rows, or one.
                let rows = drafts.map_or(1, |depth| depth + 1);
                let samplers = match samplers(sampling, vocabulary, rows) {
                    Ok(samplers) => SampledRows::new(samplers),
                    Err(error) => {
                        let error = AdmissionError::Engine(format!("the sampler: {error}"));
                        return Err(Rejected::new(error, Audience::Greedy(listener)));
                    }
                };
                match drafts {
                    None => Mode::Sampled {
                        listener,
                        state: Box::new(Sampling {
                            rows: samplers,
                            next: None,
                            constraint: constraint.map(|constraint| Constraining::new(Guarded::new(constraint))),
                        }),
                    },
                    Some(depth) => Mode::Speculative {
                        listener,
                        state: Box::new(
                            Speculating::new(depth, self.shared.capacity, prompt.len(), max_tokens as usize)
                                .sampled(samplers),
                        ),
                    },
                }
            }
            Speculation::Off => Mode::Greedy { listener },
            Speculation::Artifact => {
                let Some(drafts) = self.shared.drafts.as_ref() else {
                    return Err(Rejected::new(
                        AdmissionError::SpeculationUnavailable,
                        Audience::Greedy(listener),
                    ));
                };
                Mode::Speculative {
                    listener,
                    state: Box::new(Speculating::new(
                        drafts.depth,
                        self.shared.capacity,
                        prompt.len(),
                        max_tokens as usize,
                    )),
                }
            }
        };
        self.start_run(units, mode, prompt, max_tokens as usize, stop_tokens, cancel)?;
        // The first mask is prepared while the prompt prefills.
        self.prepare_constraint();
        Ok(())
    }

    /// Starts a teacher-forced run. A refused run gives its sink back.
    pub(crate) fn begin_forced(
        &mut self,
        units: Result<Units, AdmissionError>,
        prefix: Vec<u32>,
        forced: Vec<u32>,
        sink: Box<dyn LogitsSink>,
    ) -> Result<(), Rejected> {
        let units = match units {
            Ok(units) => units,
            Err(error) => return Err(Rejected::new(error, Audience::Forced(sink))),
        };
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
            Mode::Forced { forced, sink },
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

    fn start_run(
        &mut self,
        units: Units,
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
            waiting_logits: false,
            panics: 0,
            began: Instant::now(),
            capacity,
            _units: units,
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
        if run.waiting_logits || run.selecting() || run.in_flight.len() >= run.depth {
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
        log::error!("the owner panicked again: the request ends with its quanta still in flight");
        mem::forget(mem::take(&mut run.in_flight));
        if mem::take(&mut run.chain) {
            self.shared.order.forget(self.id);
        }
        self.used = true;
        let outcome = GenerationOutcome {
            reason: FinishReason::Failed(format!("the owner panicked: {message}")),
            prompt_tokens: run.prompt.len(),
            generated: run.generated,
            elapsed: run.began.elapsed(),
        };
        self.finished.push_back(run.mode.into_finished(outcome));
    }

    fn receive(&mut self, message: Message) {
        match message {
            Message::Retired(retired) => {
                let Retired { quantum, outcome } = *retired;
                // A quantum the workspace waited for is done: the owner of the workspace starts what waits.
                self.shared.order.retired(quantum.id());
                let waiting = self.run.as_mut().and_then(|run| {
                    run.in_flight.iter_mut().find(|flying| {
                        flying.retired.is_none()
                            && flying.start == quantum.start_position()
                            && flying.kind == quantum.kind()
                            && flying.rows == quantum.rows()
                    })
                });
                match waiting {
                    Some(flying) => {
                        flying.retired_at = Some(Instant::now());
                        flying.retired = Some((quantum, outcome));
                    }
                    None => {
                        log::error!(
                            "a {:?} quantum at {} retired that the owner did not admit",
                            quantum.kind(),
                            quantum.start_position()
                        );
                        self.park(quantum, KvRetained::default());
                    }
                }
            }
            Message::Logits(copied) => self.deliver_row(copied),
            Message::Sampled(sampled) => self.deliver_sample(sampled),
            Message::Prepared(prepared) => self.deliver_prepared(prepared),
            // The turn's `reap` is what the message is for.
            Message::Reaped => {}
            Message::StartFailed(failed) => self.start_failed(*failed),
        }
    }

    /// A quantum the owner admitted did not start: it will never retire, so it is no longer in flight, and the request
    /// fails.
    fn start_failed(&mut self, failed: StartFailed) {
        let StartFailed { serial, error } = failed;
        let Some(run) = self.run.as_mut() else { return };
        run.in_flight.retain(|flying| flying.id != serial);
        run.stop(FinishReason::Failed(format!("starting the quantum: {error}")));
    }

    /// Hands the logits row of a teacher-forced step to the sink.
    fn deliver_row(&mut self, copied: LogitsCopied) {
        let Some(run) = self.run.as_mut() else { return };
        let vocabulary = self.shared.config.vocab_size;
        if let Mode::Forced { forced, sink } = &mut run.mode {
            let all = copied.loan.buffer().as_slice();
            let bytes = &all[..vocabulary * 2];
            if let Some(next) = forced.get(copied.step) {
                if sink.wants_hidden() {
                    sink.hidden(
                        copied.step,
                        &all[vocabulary * 2..vocabulary * 2 + self.shared.config.hidden_size * 2],
                    );
                }
                sink.row(copied.step, *next, bytes);
            }
        }
        run.waiting_logits = false;
        run.generated += 1;
        if run.generated >= run.wanted {
            run.stop(FinishReason::Length);
        }
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
        let frame = prepare_frame(step, constraint, move |prepared| {
            address.post(Message::Prepared(prepared))
        });
        self.shared.core.completions.at(Instant::now(), frame);
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
        let freed = core.lane_table.submit_ops(core.owner_lane(), false, |ops| {
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
        let lanes = &self.shared.core.lanes;
        self.parked
            .retain(|parked| parked.quantum.prepared.is_some_and(|marker| !lanes.reached(marker)));
    }

    /// Keeps `quantum` until the work queued on the owner's lane for it has run. A frame posts to the owner when it
    /// has, so that the turn that follows lets it go (and a request that waits for it ends). The marker is looked at
    /// once: reached, the quantum goes now; not, the frame is registered, and it fires even if the marker is reached
    /// the next instant.
    fn park(&mut self, quantum: Quantum, retained: KvRetained) {
        let core = &self.shared.core;
        let Some(marker) = quantum.prepared.filter(|marker| !core.lanes.reached(*marker)) else {
            return;
        };
        self.parked.push(Parked {
            quantum,
            _retained: retained,
        });
        let address = Arc::clone(&self.shared.address);
        core.completions
            .after_marker(marker, step(Kind::Reaped, 0, move || address.post(Message::Reaped)));
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
            && let Some(idle) = self.watch.take()
        {
            told.push(Finished::Idle(idle));
        }
        told
    }

    /// Ends the request with a failure.
    fn fail(&mut self, why: String) {
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
            let Some((quantum, outcome)) = flying.retired.take() else {
                return;
            };
            // The quantum that feeds the last token of a speculative request runs after the request ended.
            let live = run.stopping.is_none() || flying.role == Role::Settle;
            match outcome {
                QuantumOutcome::Completed if live => self.completed(&mut flying, &quantum),
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
            self.park(quantum, flying.retained);
        }
    }

    /// A quantum of the live request completed.
    fn completed(&mut self, flying: &mut Flying, quantum: &Quantum) {
        let now = Instant::now();
        // The rows of a quantum become part of its cache; a verification's accepted prefix does, once it is known.
        let rows = flying.rows as u64;
        let committed = match flying.role {
            Role::Verify => Ok(()),
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
        let ran = quantum.graph_run();
        let event = QuantumEvent {
            kind: flying.kind,
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
        match flying.role {
            Role::Settle => return,
            Role::PromptCatchUp => return,
            Role::CatchUp | Role::Draft => return self.drafted(flying),
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
        if run.copies_rows(flying.role) {
            return self.sample(flying);
        }
        if let Mode::Forced { .. } = run.mode {
            let step = run.generated;
            run.waiting_logits = true;
            if let Err(why) = self.copy_logits(step, quantum, flying.rows) {
                self.fail(why);
            }
            return;
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
                first_step + row as u64,
                first_position + row as u64,
                sampler,
                constraint.take(),
                LeasedRow::new(Arc::clone(&lease), row),
                move |sampled| address.post(Message::Sampled(sampled)),
            );
            shared.core.completions.at(Instant::now(), frame);
        }
    }

    /// Queues the copy of the quantum's logits row to pinned memory, and of its last row's final hidden state when the
    /// sink wants it, and the message that says they have run. `rows` is the quantum's row count.
    fn copy_logits(&mut self, step: usize, quantum: &Quantum, rows: usize) -> Result<(), String> {
        let core = &self.shared.core;
        let bytes = self.shared.config.vocab_size * 2;
        let hidden_bytes = self.shared.config.hidden_size * 2;
        let wants_hidden =
            matches!(self.run.as_ref().map(|run| &run.mode), Some(Mode::Forced { sink, .. }) if sink.wants_hidden());
        let total = if wants_hidden { bytes + hidden_bytes } else { bytes };
        let mut loan = core.pool.acquire(total).map_err(why("the pinned row for the logits"))?;
        let (_, marker) = core
            .lane_table
            .submit_ops(core.owner_lane(), false, |kernels| {
                // SAFETY: the loan is held by the message that follows the copy, which is offered only once the lane
                // has reached the marker; nothing reads or writes the bytes before.
                unsafe {
                    kernels.copy_device_to_readback(loan.buffer_mut(), 0, quantum.slot(BufferSlot::Logits), bytes)
                }
                .map_err(why("queueing the logits row's copy"))?;
                if wants_hidden {
                    // SAFETY: as above; the hidden row follows the logits row in the loan.
                    unsafe {
                        kernels.copy_device_to_readback(
                            loan.buffer_mut(),
                            bytes,
                            quantum
                                .slot(BufferSlot::FinalHiddenState)
                                .offset(((rows.max(1) - 1) * hidden_bytes) as u64),
                            hidden_bytes,
                        )
                    }
                    .map_err(why("queueing the final hidden row's copy"))?;
                }
                Ok::<(), String>(())
            })
            .map_err(why("the logits row's copy"))?;
        let address = Arc::clone(&self.shared.address);
        let frame = step_frame(move || address.post(Message::Logits(LogitsCopied { step, loan })));
        core.completions.after_marker(marker, frame);
        Ok(())
    }

    /// Admits quanta while the request has steps and room for them.
    fn admit_ready(&mut self) {
        let chunk = self.shared.chunk;
        while let Some(next) = self.admissible() {
            if let Err(why) = self.admit(next) {
                self.fail(why);
                return;
            }
        }
        if let Some(run) = self.run.as_mut().filter(|run| run.next_step(chunk).is_none()) {
            // Nothing will be chained behind the last quantum: let it retire.
            run.end_chain();
        }
    }

    /// Admits the quantum or quanta of `next`.
    fn admit(&mut self, next: Step) -> Result<(), String> {
        match next {
            Step::Prefill { .. } | Step::Decode => self.admit_row(next),
            Step::PromptCatchUp { rows } => self.admit_prompt_catch_up(rows),
            Step::Speculate { drafts } => self.admit_speculation(drafts),
            Step::Settle => self.admit_settle(),
        }
    }

    /// Admits a prompt chunk or a decode row of the request's own tokens.
    fn admit_row(&mut self, next: Step) -> Result<(), String> {
        let Some(run) = self.run.as_mut() else { return Ok(()) };
        let start = run.position;
        let speculative = matches!(run.mode, Mode::Speculative { .. });
        let drafting = matches!(&run.mode, Mode::Speculative { state, .. } if state.drafting());
        let admission = match next {
            Step::Prefill { rows, last } => Admission {
                role: Role::Prefill { last },
                kind: QuantumKind::Prefill,
                rows,
                start,
                source: KvSource::Base,
                pdl: false,
                logits: usize::from(last && run.wanted > 0),
                replay: Vec::new(),
                last_chunk: last,
                // A greedy request chains the next row behind the prompt; a speculative one chains the next chunk (or the
                // catch-up of this one) and waits for the first token after the last chunk.
                more: if speculative { !last } else { run.depth > 1 },
            },
            Step::Decode => Admission {
                role: Role::Decode,
                kind: QuantumKind::Decode,
                rows: 1,
                start,
                source: KvSource::Base,
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
        self.admit_quantum(admission, move |quantum| {
            let quantum = match chained {
                Some(previous) => quantum.token_from(previous)?,
                None => quantum.with_tokens(tokens)?,
            };
            if drafting && matches!(next, Step::Prefill { .. }) {
                quantum.seeding_draft()
            } else {
                Ok(quantum)
            }
        })?;
        let Some(run) = self.run.as_mut() else { return Ok(()) };
        run.position += match next {
            Step::Prefill { rows, .. } => rows as u64,
            _ => 1,
        };
        match next {
            Step::Prefill { rows, last } => {
                run.offset += rows;
                if let (Mode::Speculative { state, .. }, true) = (&mut run.mode, drafting && !last) {
                    state.prefilled(rows);
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
        let shared = Arc::clone(&self.shared);
        let core = &shared.core;
        let Some(run) = self.run.as_mut() else { return Ok(()) };
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

        // The reservation (and, for the first quantum of a request after another, the reset of the sequence) on the
        // owner's lane. Its marker orders the quantum's stages behind it.
        let mut retained = KvRetained::default();
        let (_, reserved) = core
            .lane_table
            .submit_ops(core.owner_lane(), false, |lane| {
                if run.admitted == 0 && self.used {
                    self.sequence.reset(lane).map_err(why("resetting the sequence"))?;
                    self.used = false;
                }
                let cache = match source {
                    KvSource::Base => &mut self.sequence.kv,
                    KvSource::Mtp => match self.sequence.mtp_kv.as_mut() {
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
        let refs = self
            .sequence
            .refs(source)
            .map_err(why("the sequence's references"))?
            .ordered_after(reserved, &core.lanes)
            .with_replay_rows(replay);
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
            core.lane_table.clone(),
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
                let lease = self.sequence.row_readbacks.acquire().map_err(why("the row readback"))?;
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
        let (_, prepared) = core
            .lane_table
            .submit_ops(core.owner_lane(), false, |kernels| {
                quantum.prepare(&core.pool, kernels).map_err(why("the input record"))
            })
            .map_err(why("the input record's upload"))?;
        quantum.prepared_after(prepared);

        let selection = quantum.selection().cloned();
        let kill = Arc::new(AtomicBool::new(false));
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
        shared.order.admit(self.id, &shared.executor, quantum, kill, true, more);
        Ok(())
    }

    /// Ends the request when it has nothing in flight and nothing more to admit.
    fn finish_if_done(&mut self) {
        let chunk = self.shared.chunk;
        let Some(run) = self.run.as_ref() else { return };
        let ended = run.stopping.is_some() || run.next_step(chunk).is_none();
        if !run.in_flight.is_empty() || run.waiting_logits || run.selecting() || run.preparing() || !ended {
            return;
        }
        // Quanta that were stopped may still have work queued on the owner's lane.
        self.reap();
        if !self.parked.is_empty() {
            return;
        }
        let Some(run) = self.run.take() else { return };
        self.sequence.kv.discard();
        if let Some(mtp) = self.sequence.mtp_kv.as_mut() {
            mtp.discard();
        }
        self.used = true;
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
        };
        self.finished.push_back(run.mode.into_finished(outcome));
    }
}

/// A frame that runs `work` once, in the control kind.
fn step_frame(work: impl FnOnce() + Send + 'static) -> Box<dyn Frame> {
    step(Kind::Logits, 0, work)
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
        if self.run.is_none() && self.parked.is_empty() {
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
        self.run.is_none() && self.parked.is_empty() && self.finished.is_empty()
    }
}
