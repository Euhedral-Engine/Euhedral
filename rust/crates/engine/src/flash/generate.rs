//! The owner of a Flash-Next sequence's generation (`Session.java`, the prompt and decode loops): it cuts the prompt
//! into chunks, starts one quantum at a time through a [`Stepper`], and on each retirement commits the quantum's rows,
//! selects the next token (on the device for a greedy request without a constraint, on the host otherwise, through
//! [`TokenSampler`] and [`TokenConstraint`] as the dense path does) and reports it to the listener.
//!
//! Role: the owner is the only holder of the request. It runs inside a [`Mailbox`](crate::generation::Mailbox): the
//! retirement of a quantum is posted as a [`Message`] and the owner's turn takes it in. Call sequence: [`FlashOwner::begin`]
//! admits a request and starts its first quantum; each [`Message::Retired`] commits that quantum and starts the next
//! until the request ends, when its outcome waits in [`Turn::take_finished`] to be told once the claim is free.
//!
//! One quantum is in flight at a time: the next chunk, and the token the decode feeds, depend on the one before it.
//! Departure from Java, which runs a prompt as one graph of chunks: each chunk is its own quantum, started at the
//! retirement of the one before.

use std::collections::VecDeque;
use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;
use std::time::Instant;

use euhedral_data_structures::queues::{Consumer, Single};

use super::chunks::{PromptChunk, prompt_chunks};
use crate::generation::{
    AdmissionError, CancelHandle, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent,
    Speculation, TokenConstraint, Turn,
};
use crate::handoff::Handoff;
use crate::sampling::{TokenFilter, TokenSampler};
use crate::stages::QuantumKind;

/// How the head's row of a quantum reaches the owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Score {
    /// The quantum scores no row (a prompt chunk that is not the last).
    None,
    /// The argmax runs on the device; the owner reads the key ([`Stepper::selected`]).
    Device,
    /// The row is copied to the host; the owner selects from it ([`Stepper::row`]).
    Host,
}

/// What one quantum is: the rows it feeds and how its last row is read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub tokens: Vec<u32>,
    pub score: Score,
    /// The first quantum of a request: the sequence returns to position 0 before it.
    pub fresh: bool,
}

/// The device side of one sequence, as the owner uses it.
pub trait Stepper: Send + 'static {
    /// Starts the quantum of `step` at the sequence's position. Its retirement is posted as [`Message::Retired`]
    /// (also when it failed); a refusal here means nothing was started and nothing will be posted.
    fn begin(&mut self, step: &Step) -> Result<(), String>;
    /// The token the retired quantum's argmax chose; `None` when no logit could be selected.
    fn selected(&self) -> Option<u32>;
    /// The BF16 bytes of the retired quantum's logits row.
    fn row(&self) -> Option<&[u8]>;
    /// Publishes the retired quantum's rows, with the n-gram context after them.
    fn commit(&mut self, next_context: &[i32]) -> Result<(), String>;
    /// Drops what a failed quantum reserved.
    fn discard(&mut self);
}

/// What the owner is told about a quantum.
#[derive(Debug)]
pub struct Retired {
    /// The context after the quantum's rows, or why the quantum failed.
    pub result: Result<Box<[i32]>, String>,
    pub at: Instant,
}

/// What is posted to the owner.
#[derive(Debug)]
pub enum Message {
    Retired(Retired),
}

/// What the owner was built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub vocabulary: usize,
    /// The positions the sequence holds.
    pub capacity: u32,
    /// The rows of a prompt chunk.
    pub chunk: u32,
}

/// A request that was refused, with whom to tell.
pub struct Rejected {
    pub error: AdmissionError,
    listener: Box<dyn GenerationListener>,
}

impl Rejected {
    fn into_finished(self) -> Finished {
        let reason = FinishReason::Failed(format!("the request was refused: {}", self.error));
        Finished::Outcome(self.listener, outcome(reason, 0, 0, Instant::now()))
    }
}

/// How a request ended, to be told once the sequence can take another.
pub enum Finished {
    Outcome(Box<dyn GenerationListener>, GenerationOutcome),
    /// The marker a shutdown waits for.
    Idle(Arc<Handoff<()>>),
}

fn outcome(reason: FinishReason, prompt_tokens: usize, generated: usize, began: Instant) -> GenerationOutcome {
    GenerationOutcome {
        reason,
        prompt_tokens,
        generated,
        elapsed: began.elapsed(),
        cached_prompt_tokens: 0,
    }
}

/// The quantum in flight.
struct Flight {
    rows: u32,
    start: u32,
    score: Score,
    /// Whether it is the prompt's last chunk.
    last_chunk: bool,
    prefill: bool,
    admitted: Instant,
}

/// One request's state.
struct Run {
    prompt_tokens: usize,
    chunks: Vec<PromptChunk>,
    prompt: Vec<u32>,
    /// The chunks committed.
    done: usize,
    wanted: u32,
    stop_tokens: Vec<u32>,
    sampler: TokenSampler,
    constraint: Option<Box<dyn TokenConstraint>>,
    listener: Box<dyn GenerationListener>,
    cancel: CancelHandle,
    began: Instant,
    generated: u32,
    /// The positions committed.
    position: u32,
    /// The token the next decode feeds.
    pending: Option<u32>,
    flight: Option<Flight>,
    stopping: Option<FinishReason>,
}

impl Run {
    fn host_selects(&self) -> bool {
        !self.sampler.is_greedy() || self.constraint.is_some()
    }
}

/// The owner of one Flash-Next sequence.
pub struct FlashOwner<S: Stepper> {
    stepper: S,
    inbox: Consumer<Message, Single>,
    limits: Limits,
    run: Option<Run>,
    finished: VecDeque<Finished>,
    /// The shutdown waiting for the sequence to be idle.
    watch: Option<Arc<Handoff<()>>>,
}

impl<S: Stepper> FlashOwner<S> {
    pub fn new(stepper: S, inbox: Consumer<Message, Single>, limits: Limits) -> Self {
        Self {
            stepper,
            inbox,
            limits,
            run: None,
            finished: VecDeque::new(),
            watch: None,
        }
    }

    /// Admits `request` and starts its first quantum. Refuses it, with the listener to tell, when a request runs, the
    /// prompt is empty, too long or outside the vocabulary, the request speculates (no drafter is loaded) or the
    /// sampler cannot be made.
    pub fn begin(&mut self, request: GenerationRequest, cancel: CancelHandle) -> Result<(), Rejected> {
        let GenerationRequest {
            prompt,
            max_tokens,
            stop_tokens,
            speculation,
            sampling,
            constraint,
            listener,
        } = request;
        let refuse = |error, listener| Err(Rejected { error, listener });
        if self.run.is_some() {
            return refuse(AdmissionError::Busy, listener);
        }
        if prompt.is_empty() {
            return refuse(AdmissionError::EmptyPrompt, listener);
        }
        if let Some((index, id)) = prompt
            .iter()
            .enumerate()
            .find(|(_, id)| **id as usize >= self.limits.vocabulary)
        {
            let error = AdmissionError::TokenOutsideVocabulary {
                id: *id,
                index,
                vocabulary: self.limits.vocabulary,
            };
            return refuse(error, listener);
        }
        if prompt.len() > self.limits.capacity as usize {
            let error = AdmissionError::PromptTooLong {
                tokens: prompt.len(),
                capacity: self.limits.capacity as usize,
            };
            return refuse(error, listener);
        }
        if speculation == Speculation::Artifact {
            return refuse(AdmissionError::SpeculationUnavailable, listener);
        }
        let sampler = match TokenSampler::new(sampling, self.limits.vocabulary) {
            Ok(sampler) => sampler,
            Err(error) => return refuse(AdmissionError::Engine(error.to_string()), listener),
        };
        let chunks = match prompt_chunks(prompt.len(), self.limits.chunk as usize) {
            Ok(chunks) => chunks,
            Err(error) => return refuse(AdmissionError::Engine(error.to_string()), listener),
        };
        self.run = Some(Run {
            prompt_tokens: prompt.len(),
            chunks,
            prompt,
            done: 0,
            wanted: max_tokens,
            stop_tokens,
            sampler,
            constraint,
            listener,
            cancel,
            began: Instant::now(),
            generated: 0,
            position: 0,
            pending: None,
            flight: None,
            stopping: None,
        });
        self.advance();
        Ok(())
    }

    /// Tells the audience of a request that was refused after it was posted.
    pub fn reject(&mut self, rejected: Rejected) {
        self.finished.push_back(rejected.into_finished());
    }

    /// Starts the next quantum of the request, or ends it.
    fn advance(&mut self) {
        let Some(run) = self.run.as_mut() else { return };
        if run.flight.is_some() {
            return;
        }
        if run.stopping.is_none() && run.cancel.is_cancelled() {
            run.stopping = Some(FinishReason::Cancelled);
        }
        if let Some(reason) = run.stopping.take() {
            return self.end(reason);
        }
        let host = run.host_selects();
        let fresh = run.done == 0 && run.position == 0;
        let (tokens, start, last_chunk, prefill, score) = match run.chunks.get(run.done) {
            Some(chunk) => {
                let scored = chunk.scores && run.wanted > 0;
                let score = match (scored, host) {
                    (false, _) => Score::None,
                    (true, true) => Score::Host,
                    (true, false) => Score::Device,
                };
                (
                    run.prompt[chunk.start..chunk.start + chunk.rows].to_vec(),
                    chunk.start as u32,
                    chunk.scores,
                    true,
                    score,
                )
            }
            None => {
                let Some(token) = run.pending else {
                    return self.fail("a decode with no token to feed".to_owned());
                };
                if run.position >= self.limits.capacity {
                    return self.end(FinishReason::ContextFull);
                }
                (
                    vec![token],
                    run.position,
                    false,
                    false,
                    if host { Score::Host } else { Score::Device },
                )
            }
        };
        if score == Score::Host
            && let Some(constraint) = run.constraint.as_mut()
            && let Err(error) = constraint.prepare()
        {
            return self.fail(format!("the constraint could not prepare the next selection: {error}"));
        }
        let rows = tokens.len() as u32;
        let step = Step { tokens, score, fresh };
        run.flight = Some(Flight {
            rows,
            start,
            score,
            last_chunk,
            prefill,
            admitted: Instant::now(),
        });
        if let Err(why) = self.stepper.begin(&step) {
            if let Some(run) = self.run.as_mut() {
                run.flight = None;
            }
            self.fail(why);
        }
    }

    /// A quantum retired.
    fn retired(&mut self, retired: Retired) {
        let Some(run) = self.run.as_mut() else { return };
        let Some(flight) = run.flight.take() else { return };
        let next_context = match retired.result {
            Ok(context) => context,
            Err(why) => return self.fail(why),
        };
        if let Err(why) = self.stepper.commit(&next_context) {
            return self.fail(format!("the quantum's rows were not committed: {why}"));
        }
        run.position += flight.rows;
        if flight.prefill {
            run.done += 1;
        }
        run.listener.quantum(QuantumEvent {
            kind: if flight.prefill {
                QuantumKind::Prefill
            } else {
                QuantumKind::Decode
            },
            rows: flight.rows as usize,
            start_position: u64::from(flight.start),
            admitted: flight.admitted,
            retired: retired.at,
            replayed: false,
            recorded: false,
        });
        if flight.score == Score::None {
            if flight.last_chunk {
                // A prompt with nothing to generate.
                run.stopping = Some(FinishReason::Length);
            }
            return self.advance();
        }
        if flight.last_chunk {
            run.listener.prefill_done(retired.at);
        }
        let selected = match flight.score {
            Score::Device => self
                .stepper
                .selected()
                .ok_or_else(|| "the head selected no token: every logit was NaN or negative infinity".to_owned()),
            _ => select_on_host(&self.stepper, run),
        };
        let token = match selected {
            Ok(token) => token,
            Err(why) => return self.fail(why),
        };
        run.generated += 1;
        run.listener.token(token, Instant::now());
        if run.stop_tokens.contains(&token) {
            run.stopping = Some(FinishReason::Stop);
        } else if run.generated >= run.wanted {
            run.stopping = Some(FinishReason::Length);
        } else {
            run.pending = Some(token);
        }
        self.advance();
    }

    fn fail(&mut self, why: String) {
        self.stepper.discard();
        self.end(FinishReason::Failed(why));
    }

    fn end(&mut self, reason: FinishReason) {
        let Some(run) = self.run.take() else { return };
        let told = outcome(reason, run.prompt_tokens, run.generated as usize, run.began);
        self.finished.push_back(Finished::Outcome(run.listener, told));
    }

    /// The idle marker, once nothing runs and every outcome was handed on.
    fn then_idle(&mut self, mut told: Vec<Finished>) -> Vec<Finished> {
        if self.run.is_none()
            && let Some(idle) = self.watch.take()
        {
            told.push(Finished::Idle(idle));
        }
        told
    }
}

/// Selects from the retired quantum's host row, then commits the token to the constraint.
fn select_on_host<S: Stepper>(stepper: &S, run: &mut Run) -> Result<u32, String> {
    let bytes = stepper
        .row()
        .ok_or_else(|| "the logits row is not in the readback".to_owned())?;
    let filter = run
        .constraint
        .as_deref()
        .map(|constraint| constraint as &dyn TokenFilter);
    let token = run
        .sampler
        .select_bf16(bytes, u64::from(run.position), filter)
        .map_err(|error| error.to_string())?;
    if let Some(constraint) = run.constraint.as_mut() {
        constraint
            .accept(token)
            .map_err(|error| format!("the constraint refused token {token}: {error}"))?;
    }
    Ok(token)
}

impl<S: Stepper> Turn for FlashOwner<S> {
    type Message = Message;
    type Finished = Finished;

    fn turn(&mut self) {
        while let Some(Message::Retired(retired)) = self.inbox.poll() {
            self.retired(retired);
        }
        // A cancel is seen at a quantum boundary: one in flight retires first.
        self.advance();
    }

    fn take_finished(&mut self) -> Vec<Finished> {
        let told = self.finished.drain(..).collect();
        self.then_idle(told)
    }

    fn abandon(&mut self, why: &str) -> Vec<Finished> {
        if let Some(run) = self.run.as_mut() {
            run.flight = None;
        }
        self.fail(why.to_owned());
        let told = self.finished.drain(..).collect();
        self.then_idle(told)
    }

    fn deliver(finished: Finished) {
        if let Finished::Outcome(mut listener, outcome) = finished {
            listener.finished(outcome);
        }
    }

    fn shut(&mut self, idle: Arc<Handoff<()>>) {
        if let Some(run) = self.run.as_mut() {
            run.stopping.get_or_insert(FinishReason::Cancelled);
        }
        self.advance();
        if self.run.is_none() {
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
        self.run.is_none() && self.finished.is_empty()
    }
}

impl Debug for Rejected {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Rejected")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl Debug for Finished {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Outcome(_, outcome) => formatter.debug_tuple("Outcome").field(outcome).finish(),
            Self::Idle(_) => formatter.write_str("Idle"),
        }
    }
}

impl<S: Stepper> Debug for FlashOwner<S> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FlashOwner")
            .field("limits", &self.limits)
            .field("running", &self.run.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
