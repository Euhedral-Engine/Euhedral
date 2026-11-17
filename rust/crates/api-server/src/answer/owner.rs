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

use std::mem::take;
use std::sync::Arc;
use std::time::Instant;

use euhedral_core::frames::Frame;
use euhedral_inference_api_protocol::ApiError;
use euhedral_inference_api_protocol::pipeline::{Piece, PipelineError, PipelineOptions, TextPipeline};
use euhedral_net::Responder;

use super::Conclusion;
use super::Timing;
use super::Verdict;
use super::reply::Completed;
use crate::admission::{Handle, Interruption, Slot};
use crate::backend::{BackendError, Generation, GenerationResult};
use crate::metrics::{FinishReason, Outcome};
use euhedral_data_structures::queues::{Consumer, Single};

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
    Text(String),
    Finished(Result<GenerationResult, BackendError>),
    /// The entry never ran and never will.
    Refused(ApiError),
    Interrupt(Interruption),
    /// The client went away.
    ClientGone,
}

/// What a turn decided, carried out once the claim is back.
pub(super) enum Effect {
    /// Take the entry out of the admission's queue.
    Withdraw(Handle),
    CancelGeneration(Arc<dyn Generation>),
    Conclude(Conclusion),
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
    pipeline: TextPipeline,
    parts: Vec<Piece>,
    /// The message of the malformed tool call that ended the output.
    failure: Option<String>,
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
    timing: Timing,
    epoch: Instant,
}

impl Owner {
    pub(super) fn new(
        inbox: Consumer<Command, Single>,
        responder: Responder,
        options: &PipelineOptions,
        queued_ns: u64,
        epoch: Instant,
    ) -> Owner {
        Owner {
            inbox,
            responder: Some(responder),
            handle: None,
            slot: None,
            generation: None,
            pipeline: TextPipeline::new(options),
            parts: Vec::new(),
            failure: None,
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
            timing: Timing {
                queued_ns,
                started_ns: 0,
                first_token_ns: 0,
            },
            epoch,
        }
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
            if self.over {
                continue;
            }
            match command {
                Command::Handle(handle) => self.named(handle, effects),
                Command::Begun(slot, opened) => self.begun(slot, opened, effects),
                Command::Launched => {
                    self.launched = true;
                    self.apply_cancel(effects);
                }
                Command::Text(text) => self.text(&text, effects),
                Command::Finished(result) => self.ended(result, effects),
                Command::Refused(error) => self.refused(error, effects),
                Command::Interrupt(why) => self.interrupted(why, effects),
                Command::ClientGone => self.gone(effects),
            }
        }
    }

    fn named(&mut self, handle: Handle, effects: &mut Vec<Effect>) {
        self.handle = Some(handle);
        if self.withdraw_wanted && !self.started {
            self.withdraw_wanted = false;
            effects.push(Effect::Withdraw(handle));
        }
    }

    /// Sends `error` as the answer, unless an answer went out.
    fn answer_early(&mut self, error: ApiError, outcome: Outcome, slot: Option<Slot>, effects: &mut Vec<Effect>) {
        if self.answered {
            return drop(slot);
        }
        self.answered = true;
        effects.push(Effect::Conclude(Conclusion {
            responder: self.responder.take(),
            verdict: Verdict::Error {
                error,
                outcome,
                log: None,
            },
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
                }
            }
            Opened::Failed(error) => {
                let (error, outcome) = open_failure(&error);
                self.answer_early(error, outcome, Some(slot), effects);
                // The slot of an answered request goes with the answer; one that was answered earlier is freed now.
                self.over = true;
            }
            Opened::Skipped => {
                if self.shutdown {
                    self.answer_early(
                        ApiError::interrupted_by_shutdown(),
                        Outcome::Rejected,
                        Some(slot),
                        effects,
                    );
                } else {
                    drop(slot);
                }
                self.over = true;
            }
        }
    }

    fn text(&mut self, text: &str, effects: &mut Vec<Effect>) {
        if self.finished || self.answered || self.failure.is_some() || self.pipeline.ended() {
            return;
        }
        if self.timing.first_token_ns == 0 {
            self.timing.first_token_ns = self.now_ns();
        }
        match self.pipeline.push(text) {
            Ok(pieces) => self.parts.extend(pieces),
            Err(PipelineError::MalformedToolCall { message, emitted }) => {
                self.parts.extend(emitted);
                self.failure = Some(message);
            }
        }
        if self.failure.is_some() || self.pipeline.ended() {
            self.cancel(effects);
        }
    }

    fn refused(&mut self, error: ApiError, effects: &mut Vec<Effect>) {
        let outcome = outcome_of(&error);
        self.answer_early(error, outcome, None, effects);
        self.over = true;
    }

    fn gone(&mut self, effects: &mut Vec<Effect>) {
        self.answer_early(ApiError::client_closed(), Outcome::Cancelled, None, effects);
        if self.started {
            self.cancel(effects);
        } else if let Some(handle) = self.handle.take() {
            effects.push(Effect::Withdraw(handle));
        } else {
            self.withdraw_wanted = true;
        }
    }

    fn interrupted(&mut self, why: Interruption, effects: &mut Vec<Effect>) {
        match why {
            Interruption::Cancelled => self.gone(effects),
            Interruption::TimedOut => {
                self.answer_early(ApiError::timeout(), Outcome::Cancelled, None, effects);
                self.cancel(effects);
            }
            Interruption::Shutdown => {
                self.shutdown = true;
                self.cancel(effects);
            }
            Interruption::Abandoned => {
                self.abandoned = true;
                let slot = self.slot.take();
                self.answer_early(ApiError::interrupted_by_shutdown(), Outcome::Rejected, slot, effects);
                self.cancel(effects);
            }
        }
    }

    /// The generation ended: decides the answer and hands the generation and the slot to the conclusion.
    fn ended(&mut self, result: Result<GenerationResult, BackendError>, effects: &mut Vec<Effect>) {
        if self.finished {
            return;
        }
        self.finished = true;
        let engine_cancelled = !(self.pipeline.stop_matched() || self.pipeline.call_limit_reached())
            && self
                .generation
                .as_ref()
                .is_some_and(|generation| generation.is_cancelled());
        let verdict = if self.answered {
            Verdict::Silent
        } else {
            self.answered = true;
            self.verdict(result, engine_cancelled)
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

    fn verdict(&mut self, result: Result<GenerationResult, BackendError>, engine_cancelled: bool) -> Verdict {
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
        let result = match result {
            Ok(result) => result,
            Err(BackendError::Unavailable(reason)) => {
                return Verdict::Error {
                    error: ApiError::unavailable("Generation was interrupted by engine shutdown."),
                    outcome: Outcome::Rejected,
                    log: Some(format!("generation interrupted: {reason}")),
                };
            }
            Err(error) => return failed(ApiError::server_error(), Some(error.to_string())),
        };
        if let Some(message) = &self.failure {
            return failed(ApiError::invalid_tool_call(message), None);
        }
        if engine_cancelled {
            return Verdict::Error {
                error: ApiError::unavailable("Generation was interrupted by engine shutdown."),
                outcome: Outcome::Rejected,
                log: None,
            };
        }
        if !self.pipeline.call_limit_reached() {
            let settled = if result.stop_token_reached && !self.pipeline.stop_matched() {
                self.pipeline.finish()
            } else {
                self.pipeline.finish_truncated()
            };
            match settled {
                Ok(pieces) => self.parts.extend(pieces),
                Err(error) => return failed(ApiError::invalid_tool_call(error.message()), None),
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
        Verdict::Answer(Completed {
            parts: take(&mut self.parts),
            reason,
            stop_sequence: self.pipeline.matched_stop().map(str::to_owned),
            result,
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
