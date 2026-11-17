//! The owner of one request: the state that is touched by one turn at a time and decides what happens next.
//!
//! The owner is a state machine over [`Command`]s. A turn takes the commands posted so far, in order, and updates the
//! state; what it decides to do (cancel the generation, answer, close and release) is returned as [`Effect`]s that
//! the caller carries out once the claim is given back, each as a frame of its own. A turn does the text pipeline's
//! work for the text it was handed and nothing else: building and sending the reply is the effect's.
//!
//! The commands come from four places, none of which waits: admission (the handle of the offer, the start and its
//! slot, a refusal, an interruption), the listener of the generation (text, the end), the transport (the client
//! went away) and the start itself (the generation was launched). An interruption is decided before it is delivered,
//! so it may arrive before the generation was launched: the owner holds the cancellation until then and applies it
//! when the start reports. An answer is sent once: the first of the early answers (the client left, the time limit
//! passed, the shutdown limit passed) or the answer of the finished generation.

use std::mem::{replace, take};
use std::sync::Arc;
use std::time::Instant;

use euhedral_core::frames::Frame;
use euhedral_inference_api_protocol::ApiError;
use euhedral_inference_api_protocol::json::JsonValue;
use euhedral_inference_api_protocol::pipeline::Piece;
use euhedral_net::{PieceSent, ReplyStream, Responder, Response, StreamError};

use super::Conclusion;
use super::Timing;
use super::Verdict;
use super::reply::Completed;
use super::stream::{Encoder, head};
use crate::admission::{Handle, Interruption, Slot};
use crate::backend::{BackendError, Generation, GenerationResult};
use crate::metrics::{FinishReason, Outcome};
use euhedral_data_structures::queues::{Consumer, Single};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// What the start found when it tried to open the generation.
pub(super) enum Opened {
    Generation(Arc<dyn Generation>),
    Failed(BackendError),
    /// The request was cut before the start: nothing was opened.
    Skipped,
}

pub(super) enum Command {
    /// The name the admission gave the offer.
    Handle(Handle),
    /// The start ran: the slot it was given and what it opened.
    Begun(Slot, Opened),
    /// `generate` returned.
    Launched,
    /// The generation ended and the listener settled the text.
    Finished(Ending),
    /// The entry never ran and never will.
    Refused(ApiError),
    Interrupt(Interruption),
    /// The client went away.
    ClientGone,
    /// The listener left text in the relay (a streamed request only).
    Text,
    /// The listener holds more settled text than the stream may hold: the client does not read.
    Overrun,
    /// A prefill quantum retired (a streamed request only).
    Prefilled,
    /// The head of the stream was written, or could not be.
    Head(Result<ReplyStream, StreamError>),
    /// A piece of the stream was written, or the stream failed.
    Sent(Result<PieceSent, StreamError>),
}

/// The pieces one call of the listener settled. The first batch of a stream says how many prompt tokens the prefix
/// cache restored.
pub(super) struct Batch {
    pub(super) pieces: Vec<Piece>,
    pub(super) cached: Option<usize>,
}

impl Batch {
    /// The bytes of text the batch holds.
    pub(super) fn weight(&self) -> usize {
        self.pieces
            .iter()
            .map(|piece| match piece {
                Piece::Reasoning(text) | Piece::Text(text) => text.len(),
                Piece::ToolCall(call) => call.name.len() + call.arguments.len(),
            })
            .sum()
    }
}

/// What the listener and the owner of a stream agree on: the listener leaves its batches in a queue and wakes the owner
/// once for all that wait, so a generation faster than the owner never fills the owner's inbox.
#[derive(Default)]
pub(super) struct Relay {
    /// A `Text` command was posted and the owner has not begun to read the queue.
    pub(super) wake: AtomicBool,
    /// The bytes in the queue.
    pub(super) unread: AtomicUsize,
}

/// Where the body of a streamed reply is.
enum Wire {
    /// The head was not sent: an answer is still a whole reply.
    Unsent,
    /// The head is being written.
    Opening,
    /// Nothing is being written.
    Idle(ReplyStream),
    /// A piece is being written.
    Writing,
    /// Ended, cut or failed.
    Over,
}

/// The state of a streamed reply. The events are held in `held` and written in order, one piece at a time: the write
/// that returns starts the next.
struct Streaming {
    encoder: Encoder,
    wire: Wire,
    held: Vec<u8>,
    /// The buffer of the last write, empty, for the next one.
    spare: Vec<u8>,
    /// Nothing more is appended; the body ends once the held events are written.
    terminal: bool,
    /// The body is cut where it is: the client is gone or does not read.
    cut: bool,
    quanta: usize,
    /// The most text, in bytes of what the generation said, that may arrive while one write is in flight.
    max_held: usize,
    /// The text that arrived since the write in flight began.
    arrived: usize,
    /// The batches the listener left.
    texts: Consumer<Batch, Single>,
    relay: Arc<Relay>,
}

/// What the text pipeline settled when the generation ended: made by the listener, which is where the text is read, so
/// that it holds what was true at the moment the generation was cancelled.
pub(super) struct Ending {
    pub(super) result: Result<GenerationResult, BackendError>,
    /// The message of the malformed tool call that ended the output.
    pub(super) failure: Option<String>,
    /// Nobody in this request cancelled, so the engine did.
    pub(super) engine_cancelled: bool,
    /// The text after the final flush; `None` when the end was not an answer.
    pub(super) settled: Option<Result<Settled, String>>,
    pub(super) first_token_ns: u64,
}

/// The answer's text and how it ended.
pub(super) struct Settled {
    pub(super) parts: Vec<Piece>,
    pub(super) reason: FinishReason,
    pub(super) stop_sequence: Option<String>,
}

/// A panic a test makes the owner's turn raise when the generation is launched.
#[cfg(feature = "test-support")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Fault {
    None,
    Now,
    AfterFinish,
}

/// What a turn decided, carried out once the claim is back.
pub(super) enum Effect {
    /// Take the entry out of the admission's queue.
    Withdraw(Handle),
    CancelGeneration(Arc<dyn Generation>),
    Conclude(Conclusion),
    /// Write the head of the stream.
    OpenStream(Responder, Response),
    /// Write a piece.
    Write(ReplyStream, Vec<u8>),
    /// End the body.
    End(ReplyStream),
}

/// The outcome a refusal of admission is recorded with.
pub(super) fn outcome_of(error: &ApiError) -> Outcome {
    match error.code {
        Some("timeout" | "client_closed") => Outcome::Cancelled,
        _ if error.status == 503 => Outcome::Rejected,
        _ => Outcome::Failed,
    }
}

pub(super) struct Owner {
    pub(super) inbox: Consumer<Command, Single>,
    responder: Option<Responder>,
    handle: Option<Handle>,
    slot: Option<Slot>,
    generation: Option<Arc<dyn Generation>>,
    started: bool,
    launched: bool,
    finished: bool,
    /// The generation is to be cancelled as soon as it was launched.
    cancel_wanted: bool,
    cancel_sent: bool,
    /// The client left before the admission's handle was known.
    withdraw_wanted: bool,
    /// The server is stopping: the answer is the interruption.
    shutdown: bool,
    /// The answer was sent (or handed to an effect) already.
    answered: bool,
    abandoned: bool,
    /// Nothing more happens: the request is over.
    over: bool,
    /// A turn panicked: only what frees the generation and the slot is still done.
    broken: bool,
    /// A fault to raise at the launch, to test the recovery from a panic in a turn.
    #[cfg(feature = "test-support")]
    pub(super) fault: Fault,
    timing: Timing,
    epoch: Instant,
    streaming: Option<Streaming>,
}

impl Owner {
    pub(super) fn new(inbox: Consumer<Command, Single>, responder: Responder, queued_ns: u64, epoch: Instant) -> Owner {
        Owner {
            inbox,
            responder: Some(responder),
            handle: None,
            slot: None,
            generation: None,
            started: false,
            launched: false,
            finished: false,
            cancel_wanted: false,
            cancel_sent: false,
            withdraw_wanted: false,
            shutdown: false,
            answered: false,
            abandoned: false,
            over: false,
            broken: false,
            #[cfg(feature = "test-support")]
            fault: Fault::None,
            timing: Timing {
                queued_ns,
                started_ns: 0,
                first_token_ns: 0,
            },
            epoch,
            streaming: None,
        }
    }

    /// Makes the reply a stream of `encoder`'s events, holding at most `max_held` bytes while a write is in flight.
    pub(super) fn stream(
        &mut self,
        encoder: Encoder,
        max_held: usize,
        texts: Consumer<Batch, Single>,
        relay: Arc<Relay>,
    ) {
        self.streaming = Some(Streaming {
            encoder,
            wire: Wire::Unsent,
            held: Vec::new(),
            spare: Vec::new(),
            terminal: false,
            cut: false,
            quanta: 0,
            max_held,
            arrived: 0,
            texts,
            relay,
        });
    }

    /// The head of the stream was handed to the transport.
    fn head_sent(&self) -> bool {
        self.streaming.as_ref().is_some_and(|s| !matches!(s.wire, Wire::Unsent))
    }

    /// Asks the transport to run `frame` if the client goes away before the answer is sent.
    pub(super) fn watch(&self, frame: Box<dyn Frame>) {
        if let Some(responder) = &self.responder {
            responder.on_cancel(frame);
        }
    }

    fn now_ns(&self) -> u64 {
        u64::try_from(Instant::now().saturating_duration_since(self.epoch).as_nanos())
            .unwrap_or(u64::MAX)
            .saturating_add(1)
    }

    pub(super) fn turn(&mut self, effects: &mut Vec<Effect>) {
        while let Some(command) = self.inbox.poll() {
            // The wire outlives the request: what was held is written after the generation ended.
            if matches!(command, Command::Head(_) | Command::Sent(_)) {
                self.wire_event(command, effects);
                continue;
            }
            if self.over {
                continue;
            }
            if self.broken {
                self.salvaging(command, effects);
                continue;
            }
            match command {
                Command::Handle(handle) => self.named(handle, effects),
                Command::Begun(slot, opened) => self.begun(slot, opened, effects),
                Command::Launched => {
                    self.launched = true;
                    #[cfg(feature = "test-support")]
                    self.raise_fault();
                    self.apply_cancel(effects);
                }
                Command::Finished(ending) => self.ended(ending, effects),
                Command::Refused(error) => self.refused(error, effects),
                Command::Interrupt(why) => self.interrupted(why, effects),
                // Once the head is out the client is found gone by a failed write; the transport's cancel flag is
                // also raised by a client that only half-closed, which still reads its stream.
                Command::ClientGone if self.head_sent() => {}
                Command::ClientGone => self.gone(effects),
                Command::Text => self.settled_text(),
                Command::Overrun => self.too_slow(
                    effects,
                    "the owner does not keep up: the settled text waiting for it passed the limit",
                ),
                Command::Prefilled => self.prefilled(),
                Command::Head(_) | Command::Sent(_) => {}
            }
        }
        self.pump(effects);
    }

    /// The head or a piece was written, or the stream failed.
    fn wire_event(&mut self, command: Command, effects: &mut Vec<Effect>) {
        let Some(s) = &mut self.streaming else {
            return;
        };
        let lost = match command {
            Command::Head(Ok(stream)) => {
                s.wire = Wire::Idle(stream);
                false
            }
            Command::Sent(Ok(sent)) => {
                s.spare = sent.piece;
                s.wire = Wire::Idle(sent.stream);
                false
            }
            _ => {
                s.wire = Wire::Over;
                true
            }
        };
        if lost {
            // The client cannot be written to: what is held goes nowhere, and the generation is not wanted.
            self.gone(effects);
        }
    }

    /// Settled text of a streamed reply becomes events.
    fn settled_text(&mut self) {
        let Some(s) = &mut self.streaming else {
            return;
        };
        // Cleared before the queue is read: a batch left after this read posts the next wake.
        s.relay.wake.store(false, Ordering::SeqCst);
        while let Some(batch) = s.texts.poll() {
            let weight = batch.weight();
            s.relay.unread.fetch_sub(weight, Ordering::SeqCst);
            if self.answered || s.terminal || matches!(s.wire, Wire::Unsent) {
                continue;
            }
            s.arrived += weight;
            if let Some(cached) = batch.cached {
                s.encoder.start(cached, &mut s.held);
            }
            for piece in &batch.pieces {
                s.encoder.piece(piece, &mut s.held);
            }
        }
    }

    /// A prefill quantum retired: after the first, and until something else is written, the stream says it is alive.
    fn prefilled(&mut self) {
        if self.answered {
            return;
        }
        let Some(s) = &mut self.streaming else {
            return;
        };
        let seen = s.quanta;
        s.quanta += 1;
        if seen > 0 && !s.terminal && !s.encoder.started() && !matches!(s.wire, Wire::Unsent) {
            s.encoder.keep_alive(&mut s.held);
        }
    }

    /// The client is gone: the body goes no further.
    fn cut_stream(&mut self) {
        if let Some(s) = &mut self.streaming {
            s.cut = true;
            s.terminal = true;
            s.held.clear();
        }
    }

    /// Writes the held events if the wire is free, ends the body when all is written, and cancels the generation of a
    /// client that does not read.
    fn pump(&mut self, effects: &mut Vec<Effect>) {
        let Some(s) = &mut self.streaming else {
            return;
        };
        let mut slow = false;
        match replace(&mut s.wire, Wire::Over) {
            Wire::Idle(stream) => {
                if s.cut {
                    drop(stream);
                } else if !s.held.is_empty() {
                    let bytes = replace(&mut s.held, take(&mut s.spare));
                    s.arrived = 0;
                    s.wire = Wire::Writing;
                    effects.push(Effect::Write(stream, bytes));
                } else if s.terminal {
                    effects.push(Effect::End(stream));
                } else {
                    s.wire = Wire::Idle(stream);
                }
            }
            wire @ (Wire::Opening | Wire::Writing) => {
                // Once the end is queued nothing more arrives: what is held is all that is left.
                slow = !s.cut && !s.terminal && s.arrived > s.max_held;
                s.wire = wire;
            }
            wire @ (Wire::Unsent | Wire::Over) => s.wire = wire,
        }
        if slow {
            self.too_slow(
                effects,
                "the client does not read: the text that arrived during one write passed the limit",
            );
        }
    }

    /// The client does not read: the body is cut and the generation cancelled.
    fn too_slow(&mut self, effects: &mut Vec<Effect>, why: &str) {
        if self.streaming.as_ref().is_none_or(|s| s.cut || s.terminal) {
            return;
        }
        self.cut_stream();
        self.cancel(effects);
        self.answer_early(
            ApiError::client_closed(),
            Outcome::Cancelled,
            Some(why.to_owned()),
            None,
            effects,
        );
    }

    /// Hands the head to the transport once the generation is open.
    fn open_stream(&mut self, effects: &mut Vec<Effect>) {
        if self.answered {
            return;
        }
        let Some(s) = &mut self.streaming else {
            return;
        };
        if !matches!(s.wire, Wire::Unsent) {
            return;
        }
        if let Some(responder) = self.responder.take() {
            s.wire = Wire::Opening;
            effects.push(Effect::OpenStream(responder, head()));
        }
    }

    /// Appends the in-stream error of a failure after the head, unless the body was cut.
    fn stream_error(&mut self, error: &ApiError) {
        if let Some(s) = &mut self.streaming
            && !matches!(s.wire, Wire::Unsent)
            && !s.terminal
        {
            s.encoder.error(error, &mut s.held);
            s.terminal = true;
        }
    }

    /// Panics as the test asked: at once, or once the test has let the generation end (so the end is queued behind).
    #[cfg(feature = "test-support")]
    #[allow(
        clippy::disallowed_methods,
        reason = "test hook: holds a turn so that a queued end is ordered behind it"
    )]
    fn raise_fault(&self) {
        use std::sync::atomic::Ordering;
        use std::thread::sleep;
        use std::time::Duration;
        match self.fault {
            Fault::None => {}
            Fault::Now => panic!("a fault raised in a turn"),
            Fault::AfterFinish => {
                super::OWNER_FAULT_HELD.store(true, Ordering::Release);
                let end = Instant::now() + Duration::from_secs(10);
                while !super::OWNER_FAULT_RELEASED.load(Ordering::Acquire) && Instant::now() < end {
                    sleep(Duration::from_millis(1));
                }
                // The frame the queued end published has run and found the claim held by now.
                sleep(Duration::from_millis(200));
                panic!("a fault raised in a turn, after the end was queued");
            }
        }
    }

    /// Whether the fault models a wake-up lost to the race between a claim and a post.
    #[cfg(feature = "test-support")]
    pub(super) fn loses_the_wake(&self) -> bool {
        self.fault == Fault::AfterFinish
    }

    /// A turn panicked: answers with a server error if nothing was answered, asks the generation to stop and, if it
    /// ended already, hands it and the slot to the conclusion. From now on [`salvaging`](Owner::salvaging) takes the
    /// commands.
    pub(super) fn salvage(&mut self, effects: &mut Vec<Effect>) {
        self.salvage_turn(effects);
        self.pump(effects);
    }

    fn salvage_turn(&mut self, effects: &mut Vec<Effect>) {
        self.broken = true;
        if self.over {
            return;
        }
        self.cancel_wanted = true;
        self.apply_cancel(effects);
        if self.finished {
            self.conclude_silently(effects);
            return;
        }
        // A request that has not started and is answered now must not be opened later.
        if !self.started {
            if let Some(handle) = self.handle.take() {
                effects.push(Effect::Withdraw(handle));
            } else {
                self.withdraw_wanted = true;
            }
        }
        // The slot stays with the generation: it is released when the generation has ended and is closed.
        self.answer_early(ApiError::server_error(), Outcome::Failed, None, None, effects);
    }

    /// What a broken owner does with a command: only what frees the generation and the slot.
    fn salvaging(&mut self, command: Command, effects: &mut Vec<Effect>) {
        match command {
            Command::Begun(slot, opened) => {
                self.started = true;
                match opened {
                    Opened::Generation(generation) => {
                        self.generation = Some(generation);
                        if !self.abandoned {
                            self.slot = Some(slot);
                        }
                    }
                    Opened::Failed(_) | Opened::Skipped => self.over = true,
                }
            }
            Command::Launched => {
                self.launched = true;
                self.apply_cancel(effects);
            }
            Command::Finished(_) => {
                self.finished = true;
                self.conclude_silently(effects);
            }
            Command::Interrupt(Interruption::Abandoned) => {
                self.abandoned = true;
                self.slot = None;
                self.cancel_wanted = true;
                self.apply_cancel(effects);
            }
            Command::Interrupt(_) | Command::ClientGone => {
                self.cancel_wanted = true;
                self.apply_cancel(effects);
            }
            Command::Handle(handle) => self.named(handle, effects),
            Command::Refused(_)
            | Command::Text
            | Command::Overrun
            | Command::Prefilled
            | Command::Head(_)
            | Command::Sent(_) => {}
        }
    }

    /// Hands the generation and the slot to a conclusion that sends nothing more.
    fn conclude_silently(&mut self, effects: &mut Vec<Effect>) {
        let verdict = if self.answered {
            Verdict::Silent
        } else {
            self.answered = true;
            self.stream_error(&ApiError::server_error());
            Verdict::Error {
                error: ApiError::server_error(),
                outcome: Outcome::Failed,
                log: Some("the request's owner failed".to_owned()),
            }
        };
        effects.push(Effect::Conclude(Conclusion {
            responder: self.responder.take(),
            verdict,
            generation: self.generation.take(),
            slot: self.slot.take(),
            timing: self.timing,
        }));
        self.over = true;
    }

    fn named(&mut self, handle: Handle, effects: &mut Vec<Effect>) {
        self.handle = Some(handle);
        if self.withdraw_wanted && !self.started {
            self.withdraw_wanted = false;
            effects.push(Effect::Withdraw(handle));
        }
    }

    /// Sends `error` as the answer, unless an answer went out.
    fn answer_early(
        &mut self,
        error: ApiError,
        outcome: Outcome,
        log: Option<String>,
        slot: Option<Slot>,
        effects: &mut Vec<Effect>,
    ) {
        if self.answered {
            return drop(slot);
        }
        self.answered = true;
        self.stream_error(&error);
        effects.push(Effect::Conclude(Conclusion {
            responder: self.responder.take(),
            verdict: Verdict::Error { error, outcome, log },
            generation: None,
            slot,
            timing: self.timing,
        }));
    }

    /// Cancels the generation once it is launched and not over.
    fn apply_cancel(&mut self, effects: &mut Vec<Effect>) {
        if !self.cancel_wanted || self.cancel_sent || !self.launched || self.finished {
            return;
        }
        if let Some(generation) = &self.generation {
            self.cancel_sent = true;
            effects.push(Effect::CancelGeneration(generation.clone()));
        }
    }

    fn cancel(&mut self, effects: &mut Vec<Effect>) {
        self.cancel_wanted = true;
        self.apply_cancel(effects);
    }

    fn begun(&mut self, slot: Slot, opened: Opened, effects: &mut Vec<Effect>) {
        self.started = true;
        self.timing.started_ns = self.now_ns();
        match opened {
            Opened::Generation(generation) => {
                self.generation = Some(generation);
                if self.abandoned {
                    drop(slot);
                } else {
                    self.slot = Some(slot);
                    self.open_stream(effects);
                }
            }
            Opened::Failed(error) => {
                let (error, outcome) = open_failure(&error);
                self.answer_early(error, outcome, None, Some(slot), effects);
                // The slot of an answered request goes with the answer; one that was answered earlier is freed now.
                self.over = true;
            }
            Opened::Skipped => {
                if self.shutdown {
                    self.answer_early(ApiError::shutting_down(), Outcome::Rejected, None, Some(slot), effects);
                } else {
                    drop(slot);
                }
                self.over = true;
            }
        }
    }

    fn refused(&mut self, error: ApiError, effects: &mut Vec<Effect>) {
        let outcome = outcome_of(&error);
        self.answer_early(error, outcome, None, None, effects);
        self.over = true;
    }

    fn gone(&mut self, effects: &mut Vec<Effect>) {
        // The generation may be opened or be opening while the client leaves: it is cancelled once launched, always.
        self.cancel(effects);
        self.cut_stream();
        self.answer_early(ApiError::client_closed(), Outcome::Cancelled, None, None, effects);
        if !self.started {
            if let Some(handle) = self.handle.take() {
                effects.push(Effect::Withdraw(handle));
            } else {
                self.withdraw_wanted = true;
            }
        }
    }

    fn interrupted(&mut self, why: Interruption, effects: &mut Vec<Effect>) {
        match why {
            Interruption::Cancelled => self.gone(effects),
            Interruption::TimedOut => {
                self.cancel(effects);
                self.answer_early(ApiError::timeout(), Outcome::Cancelled, None, None, effects);
            }
            Interruption::Shutdown => {
                self.shutdown = true;
                self.cancel(effects);
            }
            Interruption::Abandoned => {
                self.abandoned = true;
                self.cancel(effects);
                let slot = self.slot.take();
                self.answer_early(
                    ApiError::interrupted_by_shutdown(),
                    Outcome::Rejected,
                    None,
                    slot,
                    effects,
                );
            }
        }
    }

    /// The generation ended: decides the answer and hands the generation and the slot to the conclusion.
    fn ended(&mut self, ending: Ending, effects: &mut Vec<Effect>) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.settled_text();
        self.timing.first_token_ns = ending.first_token_ns;
        // `answered` is set only once the verdict exists: a panic in it still gets the error answer from the salvage.
        let verdict = if self.answered {
            Verdict::Silent
        } else {
            let verdict = self.verdict(ending);
            self.answered = true;
            match &verdict {
                Verdict::Error { error, .. } => self.stream_error(error),
                _ => {
                    if let Some(s) = &mut self.streaming
                        && !matches!(s.wire, Wire::Unsent)
                    {
                        s.terminal = true;
                    }
                }
            }
            verdict
        };
        effects.push(Effect::Conclude(Conclusion {
            responder: self.responder.take(),
            verdict,
            generation: self.generation.take(),
            slot: self.slot.take(),
            timing: self.timing,
        }));
        self.over = true;
    }

    fn verdict(&mut self, ending: Ending) -> Verdict {
        let failed = |error: ApiError, log: Option<String>| Verdict::Error {
            error,
            outcome: Outcome::Failed,
            log,
        };
        if self.shutdown {
            return Verdict::Error {
                error: ApiError::interrupted_by_shutdown(),
                outcome: Outcome::Rejected,
                log: None,
            };
        }
        let result = match ending.result {
            Ok(result) => result,
            Err(error) => return failed(ApiError::server_error(), Some(error.to_string())),
        };
        if let Some(message) = &ending.failure {
            return failed(ApiError::invalid_tool_call(message), None);
        }
        if ending.engine_cancelled {
            return Verdict::Error {
                error: ApiError::unavailable("Generation was interrupted by engine shutdown."),
                outcome: Outcome::Rejected,
                log: None,
            };
        }
        match ending.settled {
            Some(Ok(settled)) if self.head_sent() => self.streamed(settled, result),
            Some(Ok(settled)) => Verdict::Answer(Completed {
                parts: settled.parts,
                reason: settled.reason,
                stop_sequence: settled.stop_sequence,
                result,
                streamed_calls: 0,
            }),
            Some(Err(message)) => failed(ApiError::invalid_tool_call(&message), None),
            None => failed(ApiError::server_error(), Some("the text was not settled".to_owned())),
        }
    }
}

impl Owner {
    /// The last events of a streamed answer: the final flush of the pipeline and the finish, or the error of a
    /// structured answer that is not one JSON document.
    fn streamed(&mut self, settled: Settled, result: GenerationResult) -> Verdict {
        let Some(s) = &mut self.streaming else {
            return Verdict::Silent;
        };
        if s.cut {
            return Verdict::Silent;
        }
        s.encoder.start(result.cached_prompt_tokens, &mut s.held);
        for piece in &settled.parts {
            s.encoder.piece(piece, &mut s.held);
        }
        if result.stop_token_reached
            && s.encoder.calls() == 0
            && let Some(answer) = s.encoder.answer()
            && !answer.trim().is_empty()
            && JsonValue::parse_distinct(answer).is_none()
        {
            return Verdict::Error {
                error: ApiError::invalid_structured_output(),
                outcome: Outcome::Failed,
                log: None,
            };
        }
        s.encoder
            .finish(settled.reason, settled.stop_sequence.as_deref(), &result, &mut s.held);
        let calls = s.encoder.calls();
        Verdict::Streamed(Completed {
            parts: Vec::new(),
            reason: settled.reason,
            stop_sequence: settled.stop_sequence,
            result,
            streamed_calls: calls,
        })
    }
}

/// The answer and the outcome of a generation the backend could not open.
fn open_failure(error: &BackendError) -> (ApiError, Outcome) {
    let error = match error {
        BackendError::Unavailable(reason) => ApiError::engine_unavailable(reason),
        BackendError::Unenforceable(_) | BackendError::Failed(_) => ApiError::server_error(),
    };
    let outcome = if error.status == 503 {
        Outcome::Rejected
    } else {
        Outcome::Failed
    };
    (error, outcome)
}
