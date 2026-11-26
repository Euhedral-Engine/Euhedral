//! The run: up to `batch` prompts are answered at once, each as one hidden request of the engine, and an answer that
//! ends is appended to the corpus from a host lake frame; the end of an answer or of a write starts the next prompt,
//! so nothing waits.
//!
//! The run is a source of its own work, driven by what the engine and the host lake tell it. The engine's sinks and
//! listeners (one pair per prompt, called on frames of the engine's lattice) and the write frames post a [`Record`]
//! to the run's inbox and then [`drive`](Generation::drive) it: whoever finds the claim on the state free runs a turn
//! of it, taking in every record posted so far; a thread that finds the claim held goes on, and the holder sees its
//! record (the same hand-over as the engine's mailbox: a poster writes first and tries second, a holder clears
//! `pending` before its turn and loads it after releasing the claim, with a `SeqCst` fence on each side).
//!
//! Who touches what: the state, the lanes' buffers and the writer belong to whoever holds the claim. A lane is one
//! prompt in flight: its rows arrive in order from the sink, its tokens from the listener and its end last, after
//! every row and token, so a lane is complete when its end is taken in. The turn that completes it builds the
//! sequence, queues it for the writer (one append at a time: the files are appended in order) and starts the next
//! prompt on the lane.
//!
//! A failure (an engine refusal, a failed answer, a record that does not add up, a failed write) cancels the answers
//! in flight, discards the queued ones and ends the run with the first failure once every answer has ended and no
//! write is under way.

use std::collections::VecDeque;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering, fence};
use std::time::Instant;

use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_data_structures::sync::ClaimCell;
use euhedral_graph::{Lake, UnitGuard};
use euhedral_inference_engine::generation::{
    CancelHandle, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, HiddenSink, QuantumEvent,
    Speculation,
};
use euhedral_inference_engine::sampling::{SamplingConfig, TokenSampler};
use euhedral_inference_engine::{EngineRuntime, Handoff};
use euhedral_inference_settings::SamplingSettings;

use super::{GenerateError, Prompt, answers_greedily, seed};
use crate::corpus::{Sequence, Writer};
use crate::step::{Kind, step};

/// Slots in a chunk of the run's inbox.
const INBOX_CHUNK: usize = 64;
/// Bytes of a hidden value (BF16).
const VALUE_BYTES: usize = 2;

/// What answers a prompt: the engine, or a stand-in in a test.
pub trait Answerer: Send + Sync + 'static {
    /// Starts the answer: `sink` hears the hidden rows of every row the request runs and `request`'s listener hears
    /// its tokens and its end. An error means the request did not start, and the sink and the listener are dropped
    /// without a call.
    fn answer(&self, request: GenerationRequest, sink: Box<dyn HiddenSink>, cancel: CancelHandle)
    -> Result<(), String>;
}

/// A boxed sink as the engine's generic parameter.
struct Forward(Box<dyn HiddenSink>);

impl HiddenSink for Forward {
    fn rows(&mut self, first_position: u64, rows: usize, hidden: &[u8], logits: Option<&[u8]>) {
        self.0.rows(first_position, rows, hidden, logits);
    }
}

impl Answerer for EngineRuntime {
    fn answer(
        &self,
        request: GenerationRequest,
        sink: Box<dyn HiddenSink>,
        cancel: CancelHandle,
    ) -> Result<(), String> {
        self.generate_hidden(request, Forward(sink), cancel)
            .map_err(|error| error.to_string())
    }
}

/// How the prompts are answered.
#[derive(Clone, Debug)]
pub struct Options {
    /// The most tokens of an answer the model runs; the answer holds one more, which it chose and never ran.
    pub max_tokens: u32,
    /// The share of prompts answered greedily.
    pub greedy_fraction: f64,
    /// The prompts answered at once.
    pub batch: usize,
    /// The sampling of the prompts that are not answered greedily.
    pub sampling: SamplingSettings,
    /// The tokens that end an answer.
    pub stops: Vec<u32>,
    /// The vocabulary: the width of a logits row.
    pub vocabulary: usize,
}

/// What a run wrote.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// The answers appended to the corpus.
    pub answered: usize,
    /// Their answer tokens, the chosen-and-never-run last of each included.
    pub tokens: usize,
}

/// What the engine's sinks and listeners and the write frames tell the run.
enum Record {
    /// Rows of a lane: `rows` rows from position `first`, and, from a sampled lane that selected a token, the greedy
    /// choice at the last of them.
    Rows {
        lane: usize,
        first: u64,
        rows: usize,
        hidden: Vec<u8>,
        choice: Option<u32>,
    },
    Token {
        lane: usize,
        id: u32,
    },
    Ended {
        lane: usize,
        reason: FinishReason,
        generated: usize,
    },
    /// A sink could not make sense of what it was given.
    Fault {
        lane: usize,
        why: String,
    },
    /// An append ended: the writer is back.
    Written {
        writer: Writer,
        result: io::Result<()>,
        answer: usize,
    },
}

/// A prompt in flight.
struct Lane {
    id: String,
    sampled: bool,
    prompt: usize,
    /// The prompt, then the answer so far.
    tokens: Vec<u32>,
    /// The greedy choices a sampled lane passed over, one for each token of the answer.
    choices: Vec<u32>,
    /// The hidden rows so far, BF16.
    hidden: Vec<u8>,
    rows: u64,
    cancel: CancelHandle,
}

struct State {
    /// What answers; let go of once nothing more starts, so a caller that then stops the engine holds its last handle.
    engine: Option<Arc<dyn Answerer>>,
    /// The corpus, while no append is under way.
    writer: Option<Writer>,
    inbox: Consumer<Record, Single>,
    prompts: VecDeque<Prompt>,
    lanes: Vec<Option<Lane>>,
    /// Answers that wait for the writer.
    ready: VecDeque<Sequence>,
    writing: bool,
    /// The host lake takes frames only while a unit is admitted.
    unit: Option<UnitGuard>,
    failure: Option<GenerateError>,
    summary: Summary,
    done: Option<Arc<Handoff<Result<Summary, GenerateError>>>>,
}

impl Drop for State {
    fn drop(&mut self) {
        if let Some(done) = self.done.take() {
            done.send(Err(GenerateError::Dropped));
        }
    }
}

/// The run. See the module documentation.
struct Generation {
    host: Arc<Lake>,
    options: Options,
    state: ClaimCell<State>,
    inbox: Producer<Record, Multi>,
    /// Set by every post after its offer, cleared by the holder before its turn.
    pending: AtomicBool,
}

/// Starts answering `prompts` and appending the answers to `writer`. Returns at once; the hand-off is sent when the run
/// ends, with what it wrote or why it stopped. The caller waits for it on a thread outside the lattice.
pub fn start(
    host: &Arc<Lake>,
    engine: Arc<dyn Answerer>,
    writer: Writer,
    prompts: Vec<Prompt>,
    options: Options,
) -> Arc<Handoff<Result<Summary, GenerateError>>> {
    let done = Handoff::new();
    let unit = match host.admit() {
        Ok(unit) => unit,
        Err(error) => {
            done.send(Err(GenerateError::Corpus(format!(
                "the host lake would not take work: {error}"
            ))));
            return done;
        }
    };
    let queue = MpscQueue::<Record>::new(INBOX_CHUNK);
    let generation = Arc::new(Generation {
        host: Arc::clone(host),
        state: ClaimCell::new(State {
            engine: Some(engine),
            writer: Some(writer),
            inbox: queue.consumer(),
            prompts: prompts.into(),
            lanes: (0..options.batch.max(1)).map(|_| None).collect(),
            ready: VecDeque::new(),
            writing: false,
            unit: Some(unit),
            failure: None,
            summary: Summary::default(),
            done: Some(Arc::clone(&done)),
        }),
        options,
        inbox: queue.producer(),
        pending: AtomicBool::new(true),
    });
    generation.drive();
    done
}

impl Generation {
    /// Tells the run `record` and runs its turns if no one else is running them. Never waits.
    fn post(self: &Arc<Self>, record: Record) {
        // The inbox is unbounded: an offer is never refused.
        drop(self.inbox.offer(record));
        // The record is written before the claim is tried: see the module documentation.
        self.pending.store(true, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        self.drive();
    }

    /// Runs turns while the claim is free and records wait. A caller that finds the claim held returns: the holder
    /// loads `pending` after it releases the claim. The loop ends when `pending` is clear: each pass takes in what
    /// was posted before it, and posts stop when the answers end.
    fn drive(self: &Arc<Self>) {
        loop {
            let Some(mut claim) = self.state.try_claim() else {
                return;
            };
            self.pending.swap(false, Ordering::SeqCst);
            claim.turn(self);
            claim.release_seq_cst();
            if !self.pending.load(Ordering::SeqCst) {
                return;
            }
        }
    }
}

impl State {
    /// Takes in the records, hands a finished answer to the writer, starts prompts on the free lanes and ends the run
    /// when it is over.
    fn turn(&mut self, this: &Arc<Generation>) {
        while let Some(record) = self.inbox.poll() {
            self.take(record);
        }
        self.write_next(this);
        self.start_prompts(this);
        self.settle();
    }

    fn take(&mut self, record: Record) {
        match record {
            Record::Rows {
                lane,
                first,
                rows,
                hidden,
                choice,
            } => {
                let Some(held) = self.lanes[lane].as_mut() else {
                    return;
                };
                if first != held.rows {
                    let why = format!(
                        "prompt {}: rows resumed at position {first}, expected {}",
                        held.id, held.rows
                    );
                    return self.fail(GenerateError::Inconsistent(why));
                }
                held.rows += rows as u64;
                held.hidden.extend_from_slice(&hidden);
                held.choices.extend(choice);
            }
            Record::Token { lane, id } => {
                if let Some(held) = self.lanes[lane].as_mut() {
                    held.tokens.push(id);
                }
            }
            Record::Ended {
                lane,
                reason,
                generated,
            } => {
                if let Some(held) = self.lanes[lane].take() {
                    self.conclude(held, reason, generated);
                }
            }
            Record::Fault { lane, why } => {
                let id = self.lanes[lane].as_ref().map_or("?", |held| held.id.as_str());
                self.fail(GenerateError::Engine(format!("prompt {id}: {why}")));
            }
            Record::Written { writer, result, answer } => {
                self.writing = false;
                self.writer = Some(writer);
                match result {
                    Ok(()) => {
                        self.summary.answered += 1;
                        self.summary.tokens += answer;
                    }
                    Err(error) => self.fail(error.into()),
                }
            }
        }
    }

    /// A lane's answer has ended: checks that what was heard adds up and queues the sequence for the writer.
    fn conclude(&mut self, lane: Lane, reason: FinishReason, generated: usize) {
        if self.failure.is_some() {
            return;
        }
        let Lane {
            id,
            sampled,
            prompt,
            tokens,
            choices,
            hidden,
            rows,
            ..
        } = lane;
        if !matches!(reason, FinishReason::Stop | FinishReason::Length) {
            return self.fail(GenerateError::Engine(format!(
                "prompt {id}: the answer ended as {reason:?}"
            )));
        }
        let answer = tokens.len() - prompt;
        if generated != answer || rows + 1 != tokens.len() as u64 || (sampled && choices.len() != answer) {
            let why = format!(
                "prompt {id}: the engine reported {generated} answer tokens, {answer} were heard, {rows} rows ran for \
                 {} tokens, {} choices were made",
                tokens.len(),
                choices.len()
            );
            return self.fail(GenerateError::Inconsistent(why));
        }
        // A greedy answer's labels are its tokens; a sampled one's are what greedy decoding chose at each of them.
        let labels = if sampled {
            tokens[..prompt].iter().copied().chain(choices).collect()
        } else {
            tokens.clone()
        };
        self.ready.push_back(Sequence {
            id,
            tokens,
            labels,
            prompt,
            sampled,
            hidden,
        });
    }

    /// Starts the append of the oldest finished answer, if the writer is free.
    fn write_next(&mut self, this: &Arc<Generation>) {
        if self.writing || self.failure.is_some() {
            return;
        }
        let Some(writer) = self.writer.take() else {
            return;
        };
        let Some(sequence) = self.ready.pop_front() else {
            self.writer = Some(writer);
            return;
        };
        let answer = sequence.tokens.len() - sequence.prompt;
        let next = Arc::clone(this);
        let frame = step(Kind::Corpus, move || {
            let mut writer = writer;
            let result = writer.append(&sequence);
            next.post(Record::Written { writer, result, answer });
        });
        if this.host.offer(frame).is_err() {
            return self.fail(GenerateError::Corpus(
                "the host lake would not take the write".to_owned(),
            ));
        }
        self.writing = true;
    }

    /// Starts prompts while a lane is free.
    fn start_prompts(&mut self, this: &Arc<Generation>) {
        while self.failure.is_none() {
            let Some(index) = self.lanes.iter().position(Option::is_none) else {
                return;
            };
            let (Some(engine), Some(prompt)) = (self.engine.clone(), self.prompts.pop_front()) else {
                return;
            };
            let Prompt { id, tokens } = prompt;
            let (seed, options) = (seed(&id), &this.options);
            let sampled = !answers_greedily(seed, options.greedy_fraction);
            let cancel = CancelHandle::default();
            let started = request(this, index, sampled, seed, &tokens)
                .and_then(|(request, sink)| engine.answer(request, sink, cancel.clone()));
            match started {
                Ok(()) => {
                    let prompt = tokens.len();
                    self.lanes[index] = Some(Lane {
                        id,
                        sampled,
                        prompt,
                        tokens,
                        choices: Vec::new(),
                        hidden: Vec::new(),
                        rows: 0,
                        cancel,
                    });
                }
                Err(why) => self.fail(GenerateError::Engine(format!("prompt {id}: {why}"))),
            }
        }
    }

    /// The first failure ends what is to come: answers in flight are cancelled, queued ones dropped.
    fn fail(&mut self, error: GenerateError) {
        if self.failure.is_none() {
            self.failure = Some(error);
        }
        self.prompts.clear();
        self.ready.clear();
        for lane in self.lanes.iter().flatten() {
            lane.cancel.cancel();
        }
    }

    /// Ends the run once nothing more starts and everything started has ended.
    fn settle(&mut self) {
        if self.done.is_none() || !(self.failure.is_some() || self.prompts.is_empty()) {
            return;
        }
        self.engine = None;
        if self.writing || !self.ready.is_empty() || self.lanes.iter().any(Option::is_some) {
            return;
        }
        let result = self.failure.take().map_or(Ok(self.summary), Err);
        self.writer = None;
        self.unit = None;
        if let Some(done) = self.done.take() {
            done.send(result);
        }
    }
}

/// The request and the sink of the prompt `tokens` on lane `lane`.
fn request(
    this: &Arc<Generation>,
    lane: usize,
    sampled: bool,
    seed: u64,
    tokens: &[u32],
) -> Result<(GenerationRequest, Box<dyn HiddenSink>), String> {
    let options = &this.options;
    let sampling = if sampled {
        let SamplingSettings {
            temperature,
            top_k,
            top_p,
        } = options.sampling;
        SamplingConfig::new(temperature, top_k, top_p, seed, false).map_err(|error| error.to_string())?
    } else {
        SamplingConfig::greedy(seed)
    };
    let chooser = sampled
        .then(|| TokenSampler::new(SamplingConfig::greedy(seed), options.vocabulary))
        .transpose()
        .map_err(|error| error.to_string())?;
    let request = GenerationRequest {
        prompt: tokens.to_vec(),
        // The answer ends with a token it chose and never ran: one more than the tokens it runs.
        max_tokens: options.max_tokens.saturating_add(1),
        stop_tokens: options.stops.clone(),
        speculation: Speculation::Off,
        sampling,
        constraint: None,
        listener: Box::new(Listener {
            run: Arc::clone(this),
            lane,
        }),
    };
    let sink = Sink {
        run: Arc::clone(this),
        lane,
        chooser,
    };
    Ok((request, Box::new(sink)))
}

/// Hears the hidden rows of a lane.
struct Sink {
    run: Arc<Generation>,
    lane: usize,
    /// Makes the greedy choice of a sampled lane from each logits row; a greedy lane's choices are its tokens.
    chooser: Option<TokenSampler>,
}

impl HiddenSink for Sink {
    fn rows(&mut self, first_position: u64, rows: usize, hidden: &[u8], logits: Option<&[u8]>) {
        let lane = self.lane;
        let choice = match (self.chooser.as_mut(), logits) {
            (Some(chooser), Some(row)) => match chooser.select_bf16(row, first_position + rows as u64, None) {
                Ok(id) => Some(id),
                Err(error) => {
                    return self.run.post(Record::Fault {
                        lane,
                        why: error.to_string(),
                    });
                }
            },
            _ => None,
        };
        debug_assert_eq!(hidden.len() % VALUE_BYTES, 0);
        self.run.post(Record::Rows {
            lane,
            first: first_position,
            rows,
            hidden: hidden.to_vec(),
            choice,
        });
    }
}

/// Hears the tokens and the end of a lane's answer.
struct Listener {
    run: Arc<Generation>,
    lane: usize,
}

impl GenerationListener for Listener {
    fn prefill_done(&mut self, _at: Instant) {}

    fn token(&mut self, id: u32, _at: Instant) {
        self.run.post(Record::Token { lane: self.lane, id });
    }

    fn quantum(&mut self, _event: QuantumEvent) {}

    fn finished(&mut self, outcome: GenerationOutcome) {
        self.run.post(Record::Ended {
            lane: self.lane,
            reason: outcome.reason,
            generated: outcome.generated,
        });
    }
}
