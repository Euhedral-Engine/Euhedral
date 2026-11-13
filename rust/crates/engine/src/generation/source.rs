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
//! that stopped part-way settles the lanes whose last marker it left unsignalled, so that it retires, and its rows
//! never become part of the sequence (`discard`).
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
use euhedral_graph::QuantumOutcome;
use euhedral_inference_cuda::{DevicePtr, KernelOps, PooledPinned, Submission, SubmitMode};

use super::listener::GenerationListener;
use super::mailbox::{Address, Turn};
use super::mtp::Speculating;
use super::request::{AdmissionError, FinishReason, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation};
use super::teacher_forced::LogitsSink;
use crate::frames::{Kind, step};
use crate::model::ModelConfig;
use crate::quantum::{Executor, GraphCache, Started};
use crate::runtime::{Core, Units};
use crate::stages::attention::KvSource;
use crate::stages::sampling::Selection;
use crate::stages::{InputRecordRef, Quantum, QuantumError, QuantumKind};
use crate::state::SequenceState;
use crate::state::kv::KvRetained;
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
    /// The work a parked quantum waited for on the owner's lane has run: the turn lets the quantum go.
    Reaped,
}

impl fmt::Debug for Message {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Message::Retired(_) => formatter.write_str("Message::Retired"),
            Message::Logits(copied) => write!(formatter, "Message::Logits(step {})", copied.step),
            Message::Reaped => formatter.write_str("Message::Reaped"),
        }
    }
}

/// A retired quantum and how it ended.
pub(crate) struct Retired {
    pub(crate) quantum: Quantum,
    pub(crate) outcome: QuantumOutcome,
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
    pub(crate) executor: Executor,
    /// Where frames that tell the owner something post it.
    pub(crate) address: Arc<Address<Owner>>,
    /// Rows of the longest prompt chunk.
    pub(crate) chunk: usize,
    /// Rows the context holds.
    pub(crate) capacity: u64,
    /// What speculation needs of the artifact; `None` for an artifact without the draft layer.
    pub(crate) drafts: Option<Drafts>,
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
            Mode::Greedy { listener } | Mode::Speculative { listener, .. } => Audience::Greedy(listener),
            Mode::Forced { sink, .. } => Audience::Forced(sink),
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
    /// Feeds the given tokens and hands each step's logits row to a sink.
    Forced {
        forced: Vec<u32>,
        sink: Box<dyn LogitsSink>,
    },
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
    pub(super) role: Role,
    pub(super) kind: QuantumKind,
    pub(super) start: u64,
    pub(super) rows: usize,
    pub(super) last_chunk: bool,
    pub(super) selection: Option<Arc<Selection>>,
    pub(super) retained: KvRetained,
    pub(super) admitted: Instant,
    pub(super) replayed: bool,
    pub(super) recorded: bool,
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
    /// The last quantum admitted, which the next one is chained behind. Held only while another quantum may follow:
    /// a quantum retires only when its reference is dropped.
    pub(super) chain: Option<Started>,
    /// The selection of the latest quantum that selects.
    pub(super) previous: Option<Arc<Selection>>,
    pub(super) stopping: Option<FinishReason>,
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
            Step::Prefill { last, .. } => usize::from(last && self.wanted > 0),
            Step::Decode => 1,
            Step::PromptCatchUp { .. } | Step::Settle => 0,
            Step::Speculate { drafts } => drafts + 1,
        }
    }

    /// The listener of a generation that reports tokens.
    pub(super) fn listener(&mut self) -> Option<&mut dyn GenerationListener> {
        match &mut self.mode {
            Mode::Greedy { listener } | Mode::Speculative { listener, .. } => Some(listener.as_mut()),
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
        self.chain = None;
    }
}

/// How a request ended, to be told to its caller once the sequence can take another.
pub(crate) enum Finished {
    Greedy(Box<dyn GenerationListener>, GenerationOutcome),
    Forced(Box<dyn LogitsSink>, GenerationOutcome),
}

impl Finished {
    pub(crate) fn deliver(self) {
        match self {
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
    pub(super) sequence: SequenceState,
    cache: GraphCache,
    inbox: Consumer<Message, Single>,
    /// Whether the sequence holds the state of a request that ended: the next request starts with a reset.
    used: bool,
    pub(super) run: Option<Run>,
    parked: Vec<Parked>,
    /// Outcomes to tell once the claim is released: a request's end, and the refusals of requests posted meanwhile.
    finished: VecDeque<Finished>,
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
        sequence: SequenceState,
        cache: GraphCache,
        inbox: Consumer<Message, Single>,
    ) -> Owner {
        Owner {
            shared,
            sequence,
            cache,
            inbox,
            used: false,
            run: None,
            parked: Vec::new(),
            finished: VecDeque::new(),
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
        let Owner { sequence, cache, .. } = self;
        drop(cache);
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
        self.cache = GraphCache::disabled();
    }

    /// Starts a greedy request, or one that speculates with the artifact's draft layer. `units` is what the engine's
    /// admission answered. A refused request gives its listener back.
    pub(crate) fn begin(
        &mut self,
        units: Result<Units, AdmissionError>,
        request: GenerationRequest,
    ) -> Result<(), Rejected> {
        let GenerationRequest {
            prompt,
            max_tokens,
            stop_tokens,
            speculation,
            listener,
        } = request;
        let units = match units {
            Ok(units) => units,
            Err(error) => return Err(Rejected::new(error, Audience::Greedy(listener))),
        };
        let mode = match speculation {
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
        self.start_run(units, mode, prompt, max_tokens as usize, stop_tokens)
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
        self.start_run(units, Mode::Forced { forced, sink }, prefix, wanted, Vec::new())
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
            Mode::Speculative { .. } => self.shared.workspace.records(),
            Mode::Forced { .. } => 1,
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
            chain: None,
            previous: None,
            stopping: None,
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
        if run.waiting_logits || run.in_flight.len() >= run.depth {
            return None;
        }
        let next = match run.next_step(self.shared.chunk) {
            Some(next) => next,
            None => self.settle_step()?,
        };
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
        mem::forget(run.chain.take());
        self.used = true;
        let outcome = GenerationOutcome {
            reason: FinishReason::Failed(format!("the owner panicked: {message}")),
            prompt_tokens: run.prompt.len(),
            generated: run.generated,
            elapsed: run.began.elapsed(),
        };
        self.finished.push_back(match run.mode {
            Mode::Greedy { listener } | Mode::Speculative { listener, .. } => Finished::Greedy(listener, outcome),
            Mode::Forced { sink, .. } => Finished::Forced(sink, outcome),
        });
    }

    fn receive(&mut self, message: Message) {
        match message {
            Message::Retired(retired) => {
                let Retired { quantum, outcome } = *retired;
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
            // The turn's `reap` is what the message is for.
            Message::Reaped => {}
        }
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

    /// Ends the request with a failure.
    fn fail(&mut self, why: String) {
        if let Some(run) = self.run.as_mut() {
            run.stop(FinishReason::Failed(why));
        }
    }

    /// Concludes the quanta that retired, oldest first: commits their rows, reads their tokens, tells the listener.
    fn conclude(&mut self) {
        loop {
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
                QuantumOutcome::Completed if live => self.completed(&flying, &quantum),
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
    fn completed(&mut self, flying: &Flying, quantum: &Quantum) {
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
        let event = QuantumEvent {
            kind: flying.kind,
            rows: flying.rows,
            start_position: flying.start,
            admitted: flying.admitted,
            retired: flying.retired_at.unwrap_or(now),
            replayed: flying.replayed,
            recorded: flying.recorded,
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
            Role::Verify => return self.verified(flying),
            Role::Prefill { .. } | Role::Decode => {}
        }
        if run.wanted == 0 || flying.selection.is_none() {
            // A prompt with nothing to generate, or a chunk that is not the last.
            if run.wanted == 0 && flying.last_chunk {
                run.stop(FinishReason::Length);
            }
            return;
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
        let marker = {
            let submission =
                Submission::begin(core.owner_lane(), SubmitMode::default()).map_err(why("the owner's lane"))?;
            let mut kernels = &submission;
            // SAFETY: the loan is held by the message that follows the copy, which is offered only once the lane has
            // reached the marker; nothing reads or writes the bytes before.
            unsafe { kernels.copy_device_to_readback(loan.buffer_mut(), 0, quantum.slot(BufferSlot::Logits), bytes) }
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
            submission.finish_signalled().map_err(why("the logits row's copy"))?
        };
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
            run.chain = None;
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
            _ => match &run.mode {
                Mode::Forced { forced, .. } => vec![forced[run.decodes]],
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
        let reserved = {
            let submission =
                Submission::begin(core.owner_lane(), SubmitMode::default()).map_err(why("the owner's lane"))?;
            if run.admitted == 0 && self.used {
                self.sequence
                    .reset(&submission)
                    .map_err(why("resetting the sequence"))?;
                self.used = false;
            }
            let cache = match source {
                KvSource::Base => &mut self.sequence.kv,
                KvSource::Mtp => self
                    .sequence
                    .mtp_kv
                    .as_mut()
                    .ok_or_else(|| "the sequence has no MTP cache".to_string())?,
            };
            let mut kernels = &submission;
            cache
                .reserve_with(&submission, &mut kernels, &core.pool, start, rows as u64, &mut retained)
                .map_err(why("reserving the cache's rows"))?;
            cache.appended(rows as u64).map_err(why("the cache's frontier"))?;
            submission.finish_signalled().map_err(why("the reservation"))?
        };
        let refs = self
            .sequence
            .refs(source)
            .map_err(why("the sequence's references"))?
            .ordered_after(reserved, &core.lanes, &core.library)
            .map_err(why("ordering the quantum behind its reservation"))?
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
        .with_input_record(record);
        quantum = shape(quantum).map_err(why("the quantum's inputs"))?;
        quantum = quantum
            .with_logits_rows(logits)
            .map_err(why("the quantum's logits rows"))?;
        if logits > 0 {
            quantum = quantum.with_selection_lease(self.sequence.selections.acquire().map_err(why("the selection"))?);
        }
        // Everything the executor would refuse is refused before the record is queued for upload: a quantum dropped
        // after that would free the pinned record under the copy.
        quantum.validate().map_err(why("the quantum"))?;
        let prepared = {
            let submission =
                Submission::begin(core.owner_lane(), SubmitMode::default()).map_err(why("the owner's lane"))?;
            quantum
                .prepare(&core.pool, &submission)
                .map_err(why("the input record"))?;
            submission
                .finish_signalled()
                .map_err(why("the input record's upload"))?
        };
        quantum.prepared_after(prepared);

        let selection = quantum.selection().cloned();
        let kill = Arc::new(AtomicBool::new(false));
        let admitted = Instant::now();
        let started = shared
            .executor
            .start(&mut self.cache, quantum, Arc::clone(&kill), run.chain.as_ref())
            .map_err(why("starting the quantum"))?;

        run.in_flight.push_back(Flying {
            role,
            kind,
            start,
            rows,
            last_chunk,
            selection: selection.clone(),
            retained,
            admitted,
            replayed: started.replayed,
            recorded: started.recording,
            kill,
            retired_at: None,
            retired: None,
        });
        if selection.is_some() {
            run.previous = selection;
        }
        // A quantum with nothing admitted behind it would never retire while its reference is held.
        run.chain = more.then_some(started);
        run.admitted += 1;
        run.cursor += 1;
        Ok(())
    }

    /// Ends the request when it has nothing in flight and nothing more to admit.
    fn finish_if_done(&mut self) {
        let chunk = self.shared.chunk;
        let Some(run) = self.run.as_ref() else { return };
        let ended = run.stopping.is_some() || run.next_step(chunk).is_none();
        if !run.in_flight.is_empty() || run.waiting_logits || !ended {
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
        self.finished.push_back(match run.mode {
            Mode::Greedy { listener } | Mode::Speculative { listener, .. } => Finished::Greedy(listener, outcome),
            Mode::Forced { sink, .. } => Finished::Forced(sink, outcome),
        });
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
        self.finished.drain(..).collect()
    }

    fn deliver(finished: Finished) {
        finished.deliver();
    }

    fn abandon(&mut self, why: &str) -> Vec<Finished> {
        // Two panics end a request at once, leaving what was in flight.
        let message = why.to_string();
        self.panicked(&message);
        self.panicked(&message);
        self.finished.drain(..).collect()
    }

    fn idle(&self) -> bool {
        self.run.is_none() && self.parked.is_empty() && self.finished.is_empty()
    }
}
