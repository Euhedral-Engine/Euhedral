//! The generation source: the owner of the engine's sequence, polled by the lattice.
//!
//! One [`Owner`] holds everything a sequence changes between quanta: its caches' frontiers, the graphs it has recorded,
//! the quanta in flight and what the request has produced. It is a baton ([`Baton`]): whoever takes it is its only
//! user, and it goes back when they are done. Three things take it:
//!
//! - [`GenerationSource`], a `LatticeSource` the lattice's workers poll while a request runs. A poll takes the baton,
//!   asks whether the owner has anything to do (a message in its inbox, a quantum to admit, an ending), and if it has,
//!   publishes an [`OwnerFrame`] that carries the owner into the `stages` lake. It does no other work.
//! - The [`OwnerFrame`], which runs [`Owner::work`] on a worker: it takes in what arrived, concludes the quanta that
//!   retired, admits the next ones and ends the request, then hands the baton back.
//! - [`Generation::generate`](super::Generation::generate), which takes the baton to start a request.
//!
//! What arrives is a [`Message`] in the owner's inbox: a quantum that retired (any worker's retirement frame offers it),
//! and, for a teacher-forced run, the copy of a logits row to pinned memory. Nothing waits for them.
//!
//! **The chain.** A greedy request keeps two quanta in flight. The one admitted behind another runs behind it on the
//! device (the graph's external edge waits for its sink), and takes its token id from the selection the earlier one
//! wrote ([`Quantum::token_from`]), so the host need not have read it: there is no gap between the end of one step and
//! the start of the next. The cost is one quantum that may run for nothing, the one that follows a stop token or a
//! failure. It runs to its end and its rows never become part of the sequence (`discard`). It is not stopped with its
//! kill switch: a quantum killed while its stages are being submitted leaves a last marker on a lane that no signalled
//! marker follows (only the ends of the lanes' chains signal), the lane's completion word never reaches the quantum's
//! retirement boundary, and the quantum never retires, which would keep the request from ending.
//!
//! **The records.** The owner reserves a quantum's cache rows and uploads its input record on its own lane
//! when it admits it. Consecutive quanta use the two input records in turn, so the upload of the quantum behind never
//! overwrites what the quantum ahead is still reading.
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
#[cfg(feature = "test-oracles")]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Instant;

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::{LatticeReceiver, LatticeSource};
use euhedral_data_structures::queues::{Consumer, Multi, Producer};
use euhedral_graph::{Lake, QuantumOutcome};
use euhedral_inference_cuda::{DevicePtr, PooledPinned, Submission, SubmitMode, ops};

use super::baton::Baton;
use super::listener::GenerationListener;
use super::mtp::Speculating;
use super::request::{AdmissionError, FinishReason, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation};
use super::teacher_forced::LogitsSink;
use crate::frames::{Kind, id_hash, step};
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
}

impl fmt::Debug for Message {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Message::Retired(_) => formatter.write_str("Message::Retired"),
            Message::Logits(copied) => write!(formatter, "Message::Logits(step {})", copied.step),
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
    pub(crate) producer: Producer<Message, Multi>,
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

    /// Ends the request with `reason` unless it already ends: no more steps. The quanta in flight run to their end and are discarded: see the module documentation.
    pub(super) fn stop(&mut self, reason: FinishReason) {
        if self.stopping.is_none() {
            self.stopping = Some(reason);
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
    inbox: Consumer<Message, Multi>,
    /// Whether the sequence holds the state of a request that ended: the next request starts with a reset.
    used: bool,
    pub(super) run: Option<Run>,
    parked: Vec<Parked>,
    finished: Option<Finished>,
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
        inbox: Consumer<Message, Multi>,
    ) -> Owner {
        Owner {
            shared,
            sequence,
            cache,
            inbox,
            used: false,
            run: None,
            parked: Vec::new(),
            finished: None,
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

    /// Starts a greedy request, or one that speculates with the artifact's draft layer.
    pub(crate) fn begin(&mut self, units: Units, request: GenerationRequest) -> Result<(), AdmissionError> {
        let mode = match request.speculation {
            Speculation::Off => Mode::Greedy {
                listener: request.listener,
            },
            Speculation::Artifact => {
                let Some(drafts) = self.shared.drafts.as_ref() else {
                    return Err(AdmissionError::SpeculationUnavailable);
                };
                Mode::Speculative {
                    listener: request.listener,
                    state: Box::new(Speculating::new(
                        drafts.depth,
                        self.shared.capacity,
                        request.prompt.len(),
                        request.max_tokens as usize,
                    )),
                }
            }
        };
        self.start_run(
            units,
            mode,
            request.prompt,
            request.max_tokens as usize,
            request.stop_tokens,
        )
    }

    /// Starts a teacher-forced run.
    pub(crate) fn begin_forced(
        &mut self,
        units: Units,
        prefix: Vec<u32>,
        forced: Vec<u32>,
        sink: Box<dyn LogitsSink>,
    ) -> Result<(), AdmissionError> {
        let vocabulary = self.shared.config.vocab_size;
        if let Some((index, id)) = forced.iter().enumerate().find(|(_, id)| **id as usize >= vocabulary) {
            return Err(AdmissionError::TokenOutsideVocabulary {
                id: *id,
                index: prefix.len() + index,
                vocabulary,
            });
        }
        // The last forced token is only predicted, never fed: the rows end one before.
        let rows = (prefix.len() + forced.len()).saturating_sub(1);
        if rows as u64 > self.shared.capacity {
            return Err(AdmissionError::PromptTooLong {
                tokens: rows,
                capacity: self.shared.capacity as usize,
            });
        }
        let wanted = forced.len();
        if wanted == 0 {
            return Err(AdmissionError::EmptyPrompt);
        }
        self.start_run(units, Mode::Forced { forced, sink }, prefix, wanted, Vec::new())
    }

    fn start_run(
        &mut self,
        units: Units,
        mode: Mode,
        prompt: Vec<u32>,
        wanted: usize,
        stop_tokens: Vec<u32>,
    ) -> Result<(), AdmissionError> {
        if self.run.is_some() || self.finished.is_some() {
            return Err(AdmissionError::Busy);
        }
        if prompt.is_empty() {
            return Err(AdmissionError::EmptyPrompt);
        }
        let vocabulary = self.shared.config.vocab_size;
        if let Some((index, id)) = prompt.iter().enumerate().find(|(_, id)| **id as usize >= vocabulary) {
            return Err(AdmissionError::TokenOutsideVocabulary {
                id: *id,
                index,
                vocabulary,
            });
        }
        let capacity = self.shared.capacity;
        if prompt.len() as u64 > capacity {
            return Err(AdmissionError::PromptTooLong {
                tokens: prompt.len(),
                capacity: capacity as usize,
            });
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

    /// Whether the owner has something to do: a message to take in, a quantum to admit, an ending to make, or parked
    /// quanta whose work on the owner's lane may have run.
    pub(crate) fn needs_attention(&self) -> bool {
        if !self.inbox.is_empty() || !self.parked.is_empty() {
            return true;
        }
        let Some(run) = &self.run else {
            return false;
        };
        let admits = self.admissible().is_some();
        let ends = run.in_flight.is_empty()
            && !run.waiting_logits
            && (run.stopping.is_some() || run.next_step(self.shared.chunk).is_none())
            && self.settle_step().is_none();
        admits || ends
    }

    /// Takes the outcome of a request that ended, for its caller to be told once the owner is back with the engine.
    pub(crate) fn take_finished(&mut self) -> Option<Finished> {
        self.finished.take()
    }

    /// One turn of the owner, on a worker: takes in what arrived, concludes the quanta that retired in order, admits
    /// the next ones and ends the request when nothing is left.
    pub(crate) fn work(&mut self) {
        let inbox = self.inbox.clone();
        inbox.drain_all(|message| self.receive(message));
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
        self.finished = Some(match run.mode {
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
        }
    }

    /// Hands the logits row of a teacher-forced step to the sink.
    fn deliver_row(&mut self, copied: LogitsCopied) {
        let Some(run) = self.run.as_mut() else { return };
        let vocabulary = self.shared.config.vocab_size;
        if let Mode::Forced { forced, sink } = &mut run.mode {
            let bytes = &copied.loan.buffer().as_slice()[..vocabulary * 2];
            if let Some(next) = forced.get(copied.step) {
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

    fn park(&mut self, quantum: Quantum, retained: KvRetained) {
        self.parked.push(Parked {
            quantum,
            _retained: retained,
        });
        self.reap();
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
            if let Err(why) = self.copy_logits(step, quantum) {
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

    /// Queues the copy of the quantum's logits row to pinned memory and the message that says it has run.
    fn copy_logits(&mut self, step: usize, quantum: &Quantum) -> Result<(), String> {
        let core = &self.shared.core;
        let bytes = self.shared.config.vocab_size * 2;
        let mut loan = core.pool.acquire(bytes).map_err(why("the pinned row for the logits"))?;
        let marker = {
            let submission =
                Submission::begin(core.owner_lane(), SubmitMode::default()).map_err(why("the owner's lane"))?;
            // SAFETY: the loan is held by the message that follows the copy, which is offered only once the lane has
            // reached the marker; nothing reads or writes the bytes before.
            unsafe {
                ops::copy_device_to_readback(
                    &submission,
                    loan.buffer_mut(),
                    0,
                    quantum.slot(BufferSlot::Logits),
                    bytes,
                )
            }
            .map_err(why("queueing the logits row's copy"))?;
            submission.finish_signalled().map_err(why("the logits row's copy"))?
        };
        let producer = self.shared.producer.clone();
        let frame = step_frame(move || {
            if producer.offer(Message::Logits(LogitsCopied { step, loan })).is_err() {
                log::error!("the owner's inbox refused a logits row");
            }
        });
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
                // A greedy request chains the next row behind the prompt; a speculative one catches up the chunk first
                // and waits for the first token after the last.
                more: if speculative { drafting && !last } else { run.depth > 1 },
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
            cache
                .reserve(&submission, &core.pool, start, rows as u64, &mut retained)
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
        // A kill switch that is never set: see the module documentation.
        let kill = Arc::new(AtomicBool::new(false));
        let admitted = Instant::now();
        let started = shared
            .executor
            .start(&mut self.cache, quantum, kill, run.chain.as_ref())
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
        self.finished = Some(match run.mode {
            Mode::Greedy { listener } | Mode::Speculative { listener, .. } => Finished::Greedy(listener, outcome),
            Mode::Forced { sink, .. } => Finished::Forced(sink, outcome),
        });
    }
}

/// A frame that runs `work` once, in the control kind.
fn step_frame(work: impl FnOnce() + Send + 'static) -> Box<dyn Frame> {
    step(Kind::Logits, 0, work)
}

/// The lattice source of a request: a worker that polls it takes the owner and, if the owner has something to do,
/// publishes the frame that does it.
pub(crate) struct GenerationSource {
    /// The owner, while no frame holds it.
    baton: Baton<Box<Owner>>,
    /// Where the owner goes when the request has ended.
    home: Arc<Baton<Box<Owner>>>,
    stages: Arc<Lake>,
    downstream: OnceLock<Arc<dyn LatticeReceiver>>,
    complete: AtomicBool,
    this: Weak<GenerationSource>,
}

impl GenerationSource {
    /// A source over `owner`, which returns to `home` when its request has ended.
    pub(crate) fn new(owner: Box<Owner>, home: Arc<Baton<Box<Owner>>>, stages: Arc<Lake>) -> Arc<GenerationSource> {
        let baton = Baton::new();
        baton.put(owner);
        Arc::new_cyclic(|this| GenerationSource {
            baton,
            home,
            stages,
            downstream: OnceLock::new(),
            complete: AtomicBool::new(false),
            this: this.clone(),
        })
    }

    /// Takes the owner back from a source that was not attached.
    pub(crate) fn take_owner(&self) -> Option<Box<Owner>> {
        self.baton.take()
    }

    /// The poll: nothing here waits or computes.
    fn poll(self: &Arc<Self>) {
        if self.complete.load(Ordering::Acquire) {
            return;
        }
        let Some(owner) = self.baton.take() else { return };
        if !owner.needs_attention() {
            self.baton.put(owner);
            return;
        }
        let frame = OwnerFrame {
            header: FrameHeader::new(id_hash(Kind::Owner, 0)),
            source: Arc::clone(self),
            owner: Some(owner),
        };
        // A lake that takes no frame drops it, and the frame's drop gives the owner back.
        let _ = self.stages.offer(Box::new(frame));
    }
}

impl LatticeSource for GenerationSource {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        if self.complete.load(Ordering::Acquire) {
            downstream.on_complete();
            return;
        }
        let _ = self.downstream.set(Arc::clone(&downstream));
        downstream.add_upstream(self);
    }

    fn pull(
        &self,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        _demand: u64,
    ) -> u64 {
        // The frames go to the lake, not to this puller.
        self.poll_arc();
        0
    }

    fn request(&self, demand: u64) {
        if demand > 0 {
            self.poll_arc();
        }
    }

    fn complete(&self) {
        if !self.complete.swap(true, Ordering::AcqRel)
            && let Some(downstream) = self.downstream.get()
        {
            downstream.on_complete();
        }
    }

    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}

impl GenerationSource {
    /// The poll from a `&self`: the frames it publishes hold the source, which a `Weak` upgrade provides.
    fn poll_arc(&self) {
        if let Some(this) = self.this.upgrade() {
            this.poll();
        }
    }
}

/// The owner on its way to do one turn of work in the `stages` lake.
struct OwnerFrame {
    header: FrameHeader,
    source: Arc<GenerationSource>,
    owner: Option<Box<Owner>>,
}

impl Frame for OwnerFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(owner) = self.owner.as_mut()
            && let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| owner.work()))
        {
            owner.panicked(payload.as_ref());
        }
        Ok(())
    }
}

impl Drop for OwnerFrame {
    fn drop(&mut self) {
        let Some(mut owner) = self.owner.take() else { return };
        match owner.take_finished() {
            None => self.source.baton.put(owner),
            Some(finished) => {
                // The sequence is free again before its caller is told, so that the caller may start the next request.
                self.source.home.put(owner);
                self.source.complete();
                finished.deliver();
            }
        }
    }
}
