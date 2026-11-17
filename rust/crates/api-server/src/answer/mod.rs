//! The answer to one planned request: admission, the generation, the pipeline and one JSON reply, or a stream of
//! server-sent events.
//!
//! A request is an [`Entry`] of the admission and has an owner that is touched by one turn at a time (see
//! [`owner`]). Whoever has news posts a command to the owner's inbox and publishes the frame that drives it: the
//! admission (the entry was started, refused or interrupted), the listener of the generation (text, the end) and the
//! transport (the client went away). Nothing runs inside such a call but the hand-off. The owner decides; what it
//! decides is carried out as frames of their own: the cancellation of a generation, the withdrawal of a queued
//! request, and the conclusion, which builds the reply, sends it, records the request and closes the generation and
//! releases its slot, in that order, so that the slot is freed exactly once on every path.
//!
//! The frames of a request after planning are the start (open the generation, launch it), the turns of the owner (one
//! per hand-off), and the conclusion. A streamed request has the same frames. Its listener leaves the settled text in
//! a relay (one wake for all that waits, so a fast generation cannot fill the inbox) and the owner turns it into events
//! (see [`stream`]), which it writes through the transport one piece at a time: the continuation of a write posts the
//! command that starts the next one, and the events that come meanwhile are held and sent together. What is held is
//! bounded; a client that does not read is cut off and its generation cancelled. The head goes out as soon as the
//! generation is open; an error after it is an event of the stream, and the conclusion then records the request
//! without a reply of its own.

mod owner;
mod reply;
mod stream;

use std::fmt::{self, Debug, Formatter};
use std::mem::take;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, Ordering, fence};
use std::sync::{Arc, Weak};
use std::time::Instant;

use euhedral_core::frames::Frame;
use euhedral_core::ingest::QueueIngestSink;
use euhedral_data_structures::queues::{MpscQueue, Multi, Producer};
use euhedral_data_structures::sync::ClaimCell;
use euhedral_inference_api_protocol::pipeline::{Piece, PipelineError, PipelineOptions, TextPipeline};
use euhedral_inference_api_protocol::{ApiError, Surface};
use euhedral_net::{Responder, no_work};
use log::{error, info, warn};

#[cfg(feature = "test-support")]
use self::owner::Fault;
use self::owner::{Batch, Command, Effect, Ending, Opened, Owner, Relay, Settled};
use self::reply::{Completed, body};
use self::stream::Encoder;
use crate::admission::{Admission, Counts, Entry, Interruption, Slot};
use crate::backend::{Backend, BackendError, Generation, GenerationResult, TextListener};
use crate::frames::{once, once_or_refused};
use crate::limits::Limits;
use crate::metrics::{Api, ConstraintFailure, FinishReason, Generation as Finished, Metrics, Outcome};
use crate::planning::{Endpoint, Planned};
use crate::router::{error_response, json};

/// Turns one drive runs before it hands the rest to a new frame.
const TURNS_PER_DRIVE: usize = 4;

/// Slots in a chunk of the inbox.
const COMMANDS_PER_CHUNK: usize = 64;

/// The stages' instants of a request, in nanoseconds since the answerer began (the first instant is 1, so that 0 is
/// "never").
#[derive(Clone, Copy, Debug)]
pub(crate) struct Timing {
    pub(crate) queued_ns: u64,
    pub(crate) started_ns: u64,
    pub(crate) first_token_ns: u64,
}

/// How a request is answered.
pub(crate) enum Verdict {
    /// The generation ended with an answer to give.
    Answer(Completed),
    /// The answer is an error.
    Error {
        error: ApiError,
        outcome: Outcome,
        /// For the log only.
        log: Option<String>,
    },
    /// The answer went out as a stream; what is left is to record it.
    Streamed(Completed),
    /// The answer was sent already.
    Silent,
}

/// The end of a request: the answer, and what is given back after it.
pub(crate) struct Conclusion {
    pub(crate) responder: Option<Responder>,
    pub(crate) verdict: Verdict,
    /// Closed after the answer: only a generation that ended.
    pub(crate) generation: Option<Arc<dyn Generation>>,
    /// Released last.
    pub(crate) slot: Option<Slot>,
    pub(crate) timing: Timing,
}

/// Runs `work`; a panic is logged and goes no further, so that what follows it still runs.
fn guarded(work: impl FnOnce()) {
    if catch_unwind(AssertUnwindSafe(work)).is_err() {
        error!(target: "euhedral.requests", "a step of a request's end panicked");
    }
}

/// The metrics' api of an endpoint.
pub fn api_of(endpoint: Endpoint) -> Api {
    match endpoint {
        Endpoint::ChatCompletions => Api::ChatCompletions,
        Endpoint::Messages => Api::Messages,
        Endpoint::Responses => Api::Responses,
    }
}

struct Shared {
    backend: Arc<dyn Backend>,
    metrics: Arc<Metrics>,
    admission: Admission,
    /// Weak: a frame queued in the sink holds this state.
    sink: Weak<QueueIngestSink>,
    planned: Planned,
    surface: Surface,
    api: Api,
    epoch: Instant,
    inbox: Producer<Command, Multi>,
    /// A stream's relay of settled text from the listener to the owner.
    relay: Option<(Producer<Batch, Multi>, Arc<Relay>)>,
    /// The most settled text a stream holds.
    held_limit: usize,
    owner: ClaimCell<Owner>,
    /// A command was posted that no turn has taken yet.
    pending: AtomicBool,
    /// The request ended or was interrupted: a start that has not opened its generation does not.
    cut: AtomicBool,
    me: Weak<Shared>,
}

impl Shared {
    /// Tells the owner `command` and publishes the frame that runs its turns. Never waits.
    fn post(&self, command: Command) {
        if self.inbox.offer(command).is_err() {
            error!(target: "euhedral.requests", "{} {}: the request's inbox refused a command", self.api.label(), self.planned.id);
            return;
        }
        // The command is written before the claim is tried: see the admission's module documentation.
        self.pending.store(true, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        self.schedule();
    }

    fn schedule(&self) {
        if let Some(shared) = self.me.upgrade() {
            self.publish(once_or_refused(move |_| shared.drive()));
        }
    }

    /// Offers a frame to the sink; with the sink gone the frame is dropped, which runs its work in place.
    fn publish(&self, frame: Box<dyn Frame>) {
        if let Some(sink) = self.sink.upgrade() {
            drop(sink.offer(frame));
        }
    }

    fn drive(self: &Arc<Self>) {
        for _ in 0..TURNS_PER_DRIVE {
            let Some(mut claim) = self.owner.try_claim() else {
                return;
            };
            self.pending.swap(false, Ordering::SeqCst);
            let mut effects = Vec::new();
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| claim.turn(&mut effects))) {
                error!(target: "euhedral.requests", "{} {}: the request's turn panicked", self.api.label(), self.planned.id);
                drop(payload);
                // What the turn decided before the panic still stands; the rest is salvaged.
                if catch_unwind(AssertUnwindSafe(|| claim.salvage(&mut effects))).is_err() {
                    error!(target: "euhedral.requests", "{} {}: salvaging the request failed", self.api.label(), self.planned.id);
                }
                #[cfg(feature = "test-support")]
                if claim.loses_the_wake() {
                    self.pending.store(false, Ordering::SeqCst);
                }
                // Commands the turn did not reach are still queued, and the post that announced them may have been
                // taken as seen: another turn must drain them, or the generation and the slot are never freed.
                self.pending.store(true, Ordering::SeqCst);
            }
            // A cancel never waits and is made while the claim is held, so it comes before any close of the
            // generation, which only a conclusion (carried out after the claim) does.
            let (cancels, effects): (Vec<Effect>, Vec<Effect>) = effects
                .into_iter()
                .partition(|effect| matches!(effect, Effect::CancelGeneration(_)));
            let mut panic = None;
            for effect in cancels {
                if let Err(payload) = catch_unwind(AssertUnwindSafe(|| self.carry_out(effect))) {
                    panic.get_or_insert(payload);
                }
            }
            claim.release_seq_cst();
            for effect in effects {
                if let Err(payload) = catch_unwind(AssertUnwindSafe(|| self.carry_out(effect))) {
                    panic.get_or_insert(payload);
                }
            }
            if let Some(payload) = panic {
                self.schedule();
                resume_unwind(payload);
            }
            if !self.pending.load(Ordering::SeqCst) {
                return;
            }
        }
        self.schedule();
    }

    fn carry_out(self: &Arc<Self>, effect: Effect) {
        match effect {
            Effect::Withdraw(handle) => self.admission.cancel(handle),
            Effect::CancelGeneration(generation) => generation.cancel(),
            Effect::Conclude(conclusion) => {
                let shared = self.clone();
                self.publish(once_or_refused(move |_| conclusion.run(&shared)));
            }
            Effect::OpenStream(responder, head) => {
                let shared = self.clone();
                responder.stream(
                    head,
                    Box::new(move |opened| {
                        let opened = opened.inspect(|stream| {
                            let gone = Arc::downgrade(&shared);
                            stream.on_cancel(once(move || {
                                if let Some(shared) = gone.upgrade() {
                                    shared.client_gone();
                                }
                            }));
                        });
                        shared.post(Command::Head(opened));
                        no_work()
                    }),
                );
            }
            Effect::Write(stream, bytes) => {
                let shared = self.clone();
                stream.send(
                    bytes,
                    Box::new(move |sent| {
                        shared.post(Command::Sent(sent));
                        no_work()
                    }),
                );
            }
            Effect::End(stream) => stream.end(),
        }
    }

    fn nanos(&self) -> u64 {
        u64::try_from(Instant::now().saturating_duration_since(self.epoch).as_nanos())
            .unwrap_or(u64::MAX)
            .saturating_add(1)
    }

    fn client_gone(&self) {
        self.cut.store(true, Ordering::Release);
        self.post(Command::ClientGone);
    }
}

impl Entry for Shared {
    fn start(&self, slot: Slot) {
        if self.cut.load(Ordering::Acquire) {
            return self.post(Command::Begun(slot, Opened::Skipped));
        }
        let output = self.planned.output.clone();
        let opened = catch_unwind(AssertUnwindSafe(|| {
            self.backend.open_generation(self.planned.config, output)
        }))
        .unwrap_or_else(|_| Err(BackendError::Failed("opening the generation panicked".to_owned())));
        let generation = match opened {
            Ok(generation) => generation,
            Err(error) => return self.post(Command::Begun(slot, Opened::Failed(error))),
        };
        self.post(Command::Begun(slot, Opened::Generation(generation.clone())));
        let Some(shared) = self.me.upgrade() else {
            return;
        };
        let listener = Box::new(Listener {
            shared,
            generation: Some(generation.clone()),
            pipeline: TextPipeline::new(&PipelineOptions {
                reasoning: self.planned.output.reasoning,
                stops: self.planned.request.stops.clone(),
                tools: self.planned.request.tools.pipeline_options(),
            }),
            parts: Vec::new(),
            failure: None,
            first_token_ns: 0,
            panicked: false,
            done: false,
            handoff: self.relay.as_ref().map(|(texts, relay)| Handoff {
                texts: texts.clone(),
                relay: relay.clone(),
                limit: self.held_limit,
                overran: false,
            }),
            announced: false,
        });
        // A panic drops the listener, which reports the end.
        let launched = catch_unwind(AssertUnwindSafe(|| {
            generation.generate(self.planned.prompt.clone(), self.planned.max_new_tokens, listener);
        }));
        if launched.is_err() {
            error!(target: "euhedral.requests", "{} {}: starting the generation panicked", self.api.label(), self.planned.id);
        }
        self.post(Command::Launched);
    }

    fn refuse(&self, error: ApiError) {
        self.cut.store(true, Ordering::Release);
        self.post(Command::Refused(error));
    }

    fn interrupt(&self, why: Interruption) {
        self.cut.store(true, Ordering::Release);
        self.post(Command::Interrupt(why));
    }
}

/// A prompt that holds this text makes the owner's turn panic when the generation is launched (feature `test-support`),
/// to test that such a request still ends and frees its slot.
#[cfg(feature = "test-support")]
pub const OWNER_FAULT_MARKER: &str = "owner-turn-fault";

/// Like [`OWNER_FAULT_MARKER`], but the turn waits for [`release_owner_fault`] before it panics, so that the end of
/// the generation is queued behind the command it panics on.
#[cfg(feature = "test-support")]
pub const OWNER_FAULT_LATE_MARKER: &str = "owner-turn-fault-late";

#[cfg(feature = "test-support")]
static OWNER_FAULT_RELEASED: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "test-support")]
static OWNER_FAULT_HELD: AtomicBool = AtomicBool::new(false);

/// Whether an owner turn is held by [`OWNER_FAULT_LATE_MARKER`] (it has handled the launch).
#[cfg(feature = "test-support")]
pub fn owner_fault_is_held() -> bool {
    OWNER_FAULT_HELD.load(Ordering::Acquire)
}

/// Lets the owner turns held by [`OWNER_FAULT_LATE_MARKER`] panic.
#[cfg(feature = "test-support")]
pub fn release_owner_fault() {
    OWNER_FAULT_RELEASED.store(true, Ordering::Release);
}

/// The generation's ear. It reads the text through the pipeline, because its calls are serial and come in order: when
/// the pipeline decides the output is complete (a stop sequence, the call limit, a malformed call) it cancels the
/// generation before it returns, so what the backend counts is what the answer holds. Then it tells the owner.
struct Listener {
    /// Strong: the answer must be settled and the generation closed even when the request was abandoned meanwhile.
    shared: Arc<Shared>,
    /// Dropped at the end, so that a backend holding the listener and the generation holds no cycle.
    generation: Option<Arc<dyn Generation>>,
    pipeline: TextPipeline,
    parts: Vec<Piece>,
    /// The message of the malformed tool call that ended the output.
    failure: Option<String>,
    first_token_ns: u64,
    /// The pipeline panicked: the request ends with a server error.
    panicked: bool,
    /// `finished` was called.
    done: bool,
    /// The reply is a stream: what the pipeline settles is handed to the owner as it comes.
    handoff: Option<Handoff>,
    /// The first text was handed over, with the prompt tokens the prefix cache restored.
    announced: bool,
}

/// The listener's side of the relay to the owner of a stream.
struct Handoff {
    texts: Producer<Batch, Multi>,
    relay: Arc<Relay>,
    /// The most text that may wait for the owner.
    limit: usize,
    /// The limit was passed and the owner told.
    overran: bool,
}

impl Listener {
    fn cancel(&self) {
        if let Some(generation) = &self.generation {
            guarded(|| generation.cancel());
        }
    }

    fn push(&mut self, text: &str) {
        if self.first_token_ns == 0 {
            self.first_token_ns = self.shared.nanos();
        }
        match self.pipeline.push(text) {
            Ok(pieces) => self.parts.extend(pieces),
            Err(PipelineError::MalformedToolCall { message, emitted }) => {
                self.parts.extend(emitted);
                self.failure = Some(message);
            }
        }
    }

    /// Hands the owner what the pipeline settled so far.
    fn hand_over(&mut self) {
        let cached = (!self.announced).then(|| {
            self.announced = true;
            self.generation
                .as_ref()
                .map_or(0, |generation| generation.cached_prompt_tokens())
        });
        let pieces = take(&mut self.parts);
        if cached.is_none() && pieces.is_empty() {
            return;
        }
        let Some(handoff) = &mut self.handoff else {
            return;
        };
        let batch = Batch { pieces, cached };
        let weight = batch.weight();
        let unread = handoff.relay.unread.fetch_add(weight, Ordering::SeqCst) + weight;
        if handoff.texts.offer(batch).is_err() {
            handoff.relay.unread.fetch_sub(weight, Ordering::SeqCst);
            error!(target: "euhedral.requests", "{} {}: the text relay refused a batch", self.shared.api.label(), self.shared.planned.id);
            return;
        }
        if !handoff.relay.wake.swap(true, Ordering::SeqCst) {
            self.shared.post(Command::Text);
        }
        if unread > handoff.limit && !handoff.overran {
            handoff.overran = true;
            self.cancel();
            self.shared.post(Command::Overrun);
        }
    }

    /// The text after the final flush, and how the answer ended.
    fn settle(&mut self, result: &GenerationResult) -> Result<Settled, String> {
        if !self.pipeline.call_limit_reached() {
            let settled = if result.stop_token_reached && !self.pipeline.stop_matched() {
                self.pipeline.finish()
            } else {
                self.pipeline.finish_truncated()
            };
            match settled {
                Ok(pieces) => self.parts.extend(pieces),
                Err(error) => return Err(error.message().to_owned()),
            }
        }
        let reason = if self.pipeline.stop_matched() {
            FinishReason::StopSequence
        } else if self.pipeline.call_limit_reached() {
            FinishReason::ToolCalls
        } else if !result.stop_token_reached {
            FinishReason::Length
        } else if self.pipeline.tool_calls() > 0 {
            FinishReason::ToolCalls
        } else {
            FinishReason::End
        };
        Ok(Settled {
            parts: take(&mut self.parts),
            reason,
            stop_sequence: self.pipeline.matched_stop().map(str::to_owned),
        })
    }

    fn end(&mut self, result: Result<GenerationResult, BackendError>) -> Ending {
        let engine_cancelled = !(self.pipeline.stop_matched() || self.pipeline.call_limit_reached())
            && self
                .generation
                .as_ref()
                .is_some_and(|generation| generation.is_cancelled());
        let mut settled = None;
        let mut result = result;
        if let Ok(done) = &result
            && self.failure.is_none()
            && !engine_cancelled
        {
            let done = done.clone();
            match catch_unwind(AssertUnwindSafe(|| self.settle(&done))) {
                Ok(outcome) => settled = Some(outcome),
                Err(_) => self.panicked = true,
            }
        }
        if self.panicked {
            result = Err(BackendError::Failed("the text pipeline panicked".to_owned()));
        }
        Ending {
            result,
            failure: self.failure.take(),
            engine_cancelled,
            settled,
            first_token_ns: self.first_token_ns,
        }
    }
}

impl TextListener for Listener {
    fn text(&mut self, text: &str) {
        if self.done || self.panicked || self.failure.is_some() || self.pipeline.ended() {
            return;
        }
        if catch_unwind(AssertUnwindSafe(|| self.push(text))).is_err() {
            error!(target: "euhedral.requests", "{} {}: the text pipeline panicked", self.shared.api.label(), self.shared.planned.id);
            self.panicked = true;
        }
        if self.handoff.is_some() {
            self.hand_over();
        }
        if self.panicked || self.failure.is_some() || self.pipeline.ended() {
            self.cancel();
        }
    }

    fn prefilled(&mut self) {
        if self.handoff.is_some() && !self.done {
            self.shared.post(Command::Prefilled);
        }
    }

    fn finished(&mut self, result: Result<GenerationResult, BackendError>) {
        if self.done {
            return;
        }
        self.done = true;
        let ending = self.end(result);
        self.generation = None;
        self.shared.post(Command::Finished(ending));
    }
}

impl Drop for Listener {
    /// A backend that lets go of its listener without the end still ends the request.
    fn drop(&mut self) {
        if !self.done {
            self.finished(Err(BackendError::Failed(
                "the backend dropped the generation's listener".to_owned(),
            )));
        }
    }
}

impl Conclusion {
    /// Builds and sends the answer, records the request, closes the generation and releases the slot.
    fn run(self, shared: &Shared) {
        let Conclusion {
            responder,
            verdict,
            generation,
            slot,
            timing,
        } = self;
        let ended_ns = shared.nanos();
        let surface = shared.surface;
        let (response, completed) = match verdict {
            Verdict::Silent => (None, None),
            Verdict::Streamed(done) => (None, Some(Record::Answered(done))),
            Verdict::Error { error, outcome, log } => (
                Some(error_response(&error, surface)),
                Some(Record::Failed(error, outcome, log)),
            ),
            Verdict::Answer(done) => {
                let built = catch_unwind(AssertUnwindSafe(|| {
                    body(&shared.planned, shared.backend.model_id(), &done)
                }))
                .unwrap_or_else(|_| Err(ApiError::server_error()));
                match built {
                    Ok(bytes) => (Some(json(200, bytes)), Some(Record::Answered(done))),
                    Err(error) => (
                        Some(error_response(&error, surface)),
                        Some(Record::Failed(error, Outcome::Failed, None)),
                    ),
                }
            }
        };
        if let (Some(responder), Some(response)) = (responder, response) {
            guarded(|| responder.send(response));
        }
        match completed {
            Some(Record::Answered(done)) => guarded(|| shared.record_answer(&done, timing, ended_ns)),
            Some(Record::Failed(error, outcome, log)) => {
                guarded(|| shared.record_failure(&error, outcome, log.as_deref()));
            }
            None => {}
        }
        if let Some(generation) = generation {
            guarded(|| generation.close());
        }
        if let Some(slot) = slot {
            slot.release();
        }
    }
}

/// What a conclusion records.
enum Record {
    Answered(Completed),
    Failed(ApiError, Outcome, Option<String>),
}

impl Shared {
    fn record_failure(&self, error: &ApiError, outcome: Outcome, log: Option<&str>) {
        let api = self.api;
        self.metrics.request(api, outcome);
        match error.code {
            Some("invalid_tool_call") => self.metrics.constrained_failure(ConstraintFailure::InvalidToolCall),
            Some("invalid_structured_output") => {
                self.metrics
                    .constrained_failure(ConstraintFailure::InvalidStructuredOutput);
            }
            _ => {}
        }
        let id = &self.planned.id;
        match (log, outcome) {
            (Some(text), Outcome::Failed) => {
                error!(target: "euhedral.requests", "{} {id}: {text}", api.label());
            }
            (Some(text), _) => warn!(target: "euhedral.requests", "{} {id}: {text}", api.label()),
            (None, Outcome::Failed) => {
                warn!(target: "euhedral.requests", "{} {id}: answered with {}", api.label(), error.message);
            }
            (None, _) => {}
        }
    }

    fn record_answer(&self, done: &Completed, timing: Timing, ended_ns: u64) {
        let api = self.api;
        let result = &done.result;
        let prompt = self.planned.prompt.token_count() as u64;
        let reasoning = self.planned.output.reasoning.then_some(result.reasoning_tokens as u64);
        let finished = Finished {
            api,
            reason: done.reason,
            prompt_tokens: prompt,
            cached_prompt_tokens: result.cached_prompt_tokens as u64,
            completion_tokens: result.completion_tokens as u64,
            reasoning_tokens: reasoning,
            queued_ns: timing.queued_ns,
            started_ns: timing.started_ns,
            first_token_ns: timing.first_token_ns,
            ended_ns,
        };
        self.metrics.request(api, Outcome::Success);
        self.metrics.generation(&finished, Metrics::wall_clock_ms());
        self.metrics.tool_calls(api, done.calls() as u64);
        let first = timing.first_token_ns != 0;
        let first_token_s = if first {
            timing.first_token_ns.saturating_sub(timing.started_ns) as f64 / 1e9
        } else {
            0.0
        };
        let decode_ns = ended_ns.saturating_sub(timing.first_token_ns);
        let rate = if first && result.completion_tokens >= 2 && decode_ns > 0 {
            (result.completion_tokens - 1) as f64 * 1e9 / decode_ns as f64
        } else {
            0.0
        };
        let reasoned = reasoning.map_or_else(String::new, |tokens| format!(" ({tokens} reasoning)"));
        info!(
            target: "euhedral.requests",
            "{} {}: prompt {} tokens ({} restored from the prefix cache), {} generated{}, first token {:.3} s, {:.1} tok/s, {}",
            api.label(),
            self.planned.id,
            prompt,
            result.cached_prompt_tokens,
            result.completion_tokens,
            reasoned,
            first_token_s,
            rate,
            done.reason.label(),
        );
    }
}

/// Answers planned requests: admits them, runs their generations and sends one JSON reply or one event stream each. Cheap to clone.
#[derive(Clone)]
pub struct Answerer {
    backend: Arc<dyn Backend>,
    metrics: Arc<Metrics>,
    admission: Admission,
    sink: Arc<QueueIngestSink>,
    epoch: Instant,
    held_limit: usize,
}

impl Debug for Answerer {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("Answerer")
            .field("admission", &self.admission)
            .finish_non_exhaustive()
    }
}

impl Answerer {
    /// An answerer over `backend`'s generations admitted by `admission`; its frames go through `sink`, which must be
    /// attached to the lattice and outlive the requests.
    pub fn new(
        backend: Arc<dyn Backend>,
        metrics: Arc<Metrics>,
        admission: Admission,
        sink: Arc<QueueIngestSink>,
    ) -> Answerer {
        Answerer {
            backend,
            metrics,
            admission,
            sink,
            epoch: Instant::now(),
            held_limit: Limits::default().max_held_bytes,
        }
    }

    /// The most bytes of a streamed reply that are held while a write is in flight; past it the generation is
    /// cancelled and the stream cut.
    pub fn with_held_limit(mut self, bytes: usize) -> Answerer {
        self.held_limit = bytes;
        self
    }

    /// How many generations run and wait.
    pub fn counts(&self) -> Counts {
        self.admission.counts()
    }

    /// Records a request that was refused while it was planned.
    pub fn planning_failed(&self, endpoint: Endpoint, error: &ApiError) {
        let api = api_of(endpoint);
        let outcome = match error.status {
            400..=499 => Outcome::Invalid,
            503 => Outcome::Rejected,
            _ => Outcome::Failed,
        };
        self.metrics.request(api, outcome);
        if error.code == Some("unsupported_schema") {
            self.metrics.schema_rejected(api);
        }
    }

    /// Takes a planned request to its answer, a whole reply or a stream: it is offered to the admission at `arrived` (the
    /// request limit counts from there) and `reply` is answered when it ends, or never if the client is gone. Returns
    /// at once.
    pub fn answer(&self, planned: Planned, reply: Responder, surface: Surface, arrived: Instant) {
        let api = api_of(planned.endpoint);
        let queued_ns = u64::try_from(arrived.saturating_duration_since(self.epoch).as_nanos())
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        #[cfg(feature = "test-support")]
        let fault = if planned.prompt.text.contains(OWNER_FAULT_LATE_MARKER) {
            Fault::AfterFinish
        } else if planned.prompt.text.contains(OWNER_FAULT_MARKER) {
            Fault::Now
        } else {
            Fault::None
        };
        let texts = planned
            .request
            .stream
            .then(|| MpscQueue::<Batch>::new(COMMANDS_PER_CHUNK));
        let relay = texts
            .as_ref()
            .map(|queue| (queue.producer(), Arc::new(Relay::default())));
        let queue = MpscQueue::<Command>::new(COMMANDS_PER_CHUNK);
        let inbox = queue.producer();
        let consumer = queue.consumer();
        let shared = Arc::new_cyclic(|me: &Weak<Shared>| Shared {
            backend: self.backend.clone(),
            metrics: self.metrics.clone(),
            admission: self.admission.clone(),
            sink: Arc::downgrade(&self.sink),
            planned,
            surface,
            api,
            epoch: self.epoch,
            inbox,
            relay,
            held_limit: self.held_limit,
            owner: ClaimCell::new(Owner::new(consumer, reply, queued_ns, self.epoch)),
            pending: AtomicBool::new(false),
            cut: AtomicBool::new(false),
            me: me.clone(),
        });
        // Nothing else holds the claim yet: the request was not offered.
        if let Some(mut claim) = shared.owner.try_claim() {
            #[cfg(feature = "test-support")]
            {
                claim.fault = fault;
            }
            let gone = Arc::downgrade(&shared);
            if let (Some(texts), Some((_, relay))) = (texts, &shared.relay) {
                claim.stream(
                    Encoder::new(&shared.planned, self.backend.model_id()),
                    self.held_limit,
                    texts.consumer(),
                    relay.clone(),
                );
            }
            claim.watch(once(move || {
                if let Some(shared) = gone.upgrade() {
                    shared.client_gone();
                }
            }));
            claim.release_seq_cst();
        }
        let handle = self.admission.offer_at(shared.clone(), arrived);
        shared.post(Command::Handle(handle));
    }
}
