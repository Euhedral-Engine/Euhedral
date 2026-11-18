//! Generation: prefill, greedy decode and teacher-forced decoding on the engine's sequences.
//!
//! [`EngineRuntime::generate`](crate::EngineRuntime::generate) starts a greedy request and
//! [`EngineRuntime::teacher_forced`](crate::EngineRuntime::teacher_forced) a teacher-forced run, each on an idle
//! sequence from the pool. Both are carried by the sequence's owner (one for each sequence), which sits in a [`Mailbox`](mailbox::Mailbox): the frames that tell it something (a quantum
//! retired, a logits row arrived, a parked quantum's lane work ran) post the message and run the owner's turn on their
//! own worker if no other turn runs. The caller is told through its listener or sink, on a worker, and waits on a
//! hand-off of its own.

mod acceptance;
mod constraint;
mod draft_length;
mod listener;
mod mailbox;
mod mtp;
mod order;
mod pool;
mod request;
mod sampled;
mod source;
mod teacher_forced;

use std::fmt::{self, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::generics::FrameSink;
use euhedral_data_structures::queues::MpscQueue;
use euhedral_graph::{DEFAULT_RETRY_BACKSTOP, GraphRuntime, RetryCounts};
use euhedral_inference_cuda::CudaLibrary;

use crate::error::EngineError;
use crate::formats::{Nvfp4Linear, TrellisLinear};
use crate::frames::{Kind, step};
use crate::model::{LayerIndex, ModelConfig, Role, Weights};
use crate::plan::{Formats, Plan, View};
use crate::quantum::Executor;
use crate::residency::planner::ResidencyPlan;
use crate::runtime::{Core, STAGE_RING_QUANTA, Units};
use crate::state::{SequenceId, SequenceState};
use crate::workspace::Workspace;
use crate::workspace::{input_records, input_records_for};
pub use acceptance::{Accepted, accept, accept_tokens};
pub use constraint::{ConstraintError, ReasoningConstraint, TokenConstraint};
pub use draft_length::{DraftLength, DraftLengthError, MAX_DRAFTS, MAX_VERIFIED};
pub use listener::GenerationListener;
pub(crate) use mailbox::OPERATION_BOUND;
use mailbox::{Address, Mailbox, Spawn};
use order::{Order, Reply};
use pool::{Lease, SequencePool};
pub use request::{
    AdmissionError, CancelHandle, FinishReason, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
    Verification,
};
pub(crate) use source::GREEDY_DEPTH;
use source::{Audience, Drafts, Message, Owner, Rejected, Retired, Shared, StartFailed};
pub use teacher_forced::LogitsSink;

/// The engine's admission of a request, as the answer `Owner::begin` takes.
fn admit(core: &Core) -> Result<Units, AdmissionError> {
    core.admit().map_err(|error| AdmissionError::Engine(error.to_string()))
}

/// Why a request is refused once the owner is closed.
fn closed_error() -> AdmissionError {
    AdmissionError::Engine("the engine is closed".to_string())
}

/// The owner a request may begin on: none once the engine stops, so that the request is refused as one that found the
/// owner closed.
fn unless_closing<'a, T>(closing: &AtomicBool, owner: Option<&'a mut T>) -> Option<&'a mut T> {
    owner.filter(|_| !closing.load(Ordering::SeqCst))
}

/// A request that found the owner closed.
fn closed(audience: Audience) -> Rejected {
    Rejected::new(closed_error(), audience)
}

/// What the engine keeps for generating: the plan's graphs on the runtime's lanes, and the owner of the sequence.
pub(crate) struct Generation {
    shared: Arc<Shared>,
    /// The owner, and the messages posted to it.
    mailbox: Arc<Mailbox<Owner>>,
}

impl fmt::Debug for Generation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Generation").finish_non_exhaustive()
    }
}

impl Generation {
    /// Builds the plan of the loaded model and the graphs of its views, and makes `sequence`, numbered `id`, their
    /// owner's. `address` is where the owner will be (the workspace's owner tells it of quanta that did not start), and
    /// `order` is the engine's, shared by every sequence; `sequences` is how many sequences the engine runs, which the
    /// lanes' rings must hold the quanta of. A refused plan gives the sequence back. `draft_ids` are the token ids of
    /// the draft head's rows, which an artifact with a draft layer needs to speculate.
    #[allow(
        clippy::too_many_arguments,
        reason = "the startup's state, and where this sequence sits in the engine"
    )]
    fn new(
        core: &Arc<Core>,
        config: &Arc<ModelConfig>,
        residency: &ResidencyPlan,
        weights: &Weights,
        workspace: &Arc<Workspace>,
        sequence: SequenceState,
        id: SequenceId,
        address: Arc<Address<Owner>>,
        order: &Arc<Order<Executor>>,
        sequences: usize,
        draft_ids: Option<Arc<[u32]>>,
    ) -> Result<Generation, Box<(EngineError, SequenceState)>> {
        let capacity = sequence.kv.planned_rows().min(residency.max_context_tokens as u64);
        // An unbounded queue, whose chunk holds the messages the protocol has in flight: the quanta in flight retire into
        // it, a teacher-forced step posts one row, and a parked quantum posts that its lane work ran. Offering to it
        // cannot fail, so no message is ever dropped.
        let queue = MpscQueue::<Message>::new(GREEDY_DEPTH + 4);
        let (producer, inbox) = (queue.producer(), queue.consumer());
        let executor = match Self::executor(core, config, residency, weights, &address, sequences) {
            Ok(executor) => executor,
            Err(error) => return Err(Box::new((error, sequence))),
        };
        let drafts = match (draft_ids, weights.get(LayerIndex::GLOBAL, Role::DraftTokenIds)) {
            (Some(token_ids), Some(device)) if residency.draft_depth > 0 && sequence.mtp_kv.is_some() => Some(Drafts {
                token_ids,
                device_token_ids: device.ptr,
                depth: residency.draft_depth.min(MAX_VERIFIED),
            }),
            _ => None,
        };
        let shared = Arc::new(Shared {
            core: Arc::clone(core),
            config: Arc::clone(config),
            workspace: Arc::clone(workspace),
            executor: Arc::new(executor),
            order: Arc::clone(order),
            address: Arc::clone(&address),
            chunk: residency.rows.rows as usize,
            capacity,
            drafts,
            closing: AtomicBool::new(false),
            #[cfg(feature = "test-oracles")]
            panic_after_admissions: AtomicU64::new(u64::MAX),
            #[cfg(feature = "test-oracles")]
            settle_after_speculation: AtomicBool::new(false),
        });
        let mailbox = Arc::new(Mailbox::new(
            Owner::new(Arc::clone(&shared), id, sequence, inbox),
            producer,
            Self::spawner(core),
        ));
        address.set(&mailbox);
        Ok(Generation { shared, mailbox })
    }

    /// A frame that drives an owner's mailbox from a thread that must not run its turn, or after a frame's turns are
    /// used: unordered (a random routing hash), so that it may run on any worker, and so that no two owners' frames
    /// share a chain in the lake.
    fn spawner(core: &Arc<Core>) -> Spawn {
        let stages = Arc::clone(&core.stages);
        let seeds = AtomicU64::new(0);
        Box::new(move |work| {
            let mut frame = step(Kind::Owner, 0, work);
            frame.header_mut().randomize_hash(seeds.fetch_add(1, Ordering::Relaxed));
            stages.offer(frame).is_ok()
        })
    }

    /// The stage graphs of the model's plan, whose quanta retire into the owner's mailbox.
    fn executor(
        core: &Arc<Core>,
        config: &Arc<ModelConfig>,
        residency: &ResidencyPlan,
        weights: &Weights,
        address: &Arc<Address<Owner>>,
        sequences: usize,
    ) -> Result<Executor, EngineError> {
        let library: &Arc<CudaLibrary> = &core.library;
        let formats = Formats::new(
            Arc::new(TrellisLinear::from_settings(
                &core.settings.kernels.trellis,
                library.device().sm_count,
            )),
            Arc::new(Nvfp4Linear::new(library)),
        );
        let plan = Plan::build(config, weights, residency, &formats)
            .map_err(|error| EngineError::Runtime(format!("the execution plan: {error}")))?;
        if plan.fan_out_width() > core.compute_lanes() {
            return Err(EngineError::Runtime(format!(
                "the plan's graphs fan out {} wide and the engine made {} lanes for them",
                plan.fan_out_width(),
                core.compute_lanes()
            )));
        }
        let stages_of = |view: View| plan.stages(view).map_or(0, <[_]>::len);
        let biggest = [
            View::Decode,
            View::PrefillChunk,
            View::Verify { rows: 2 },
            View::Draft { rows: 1 },
        ]
        .into_iter()
        .map(stages_of)
        .max()
        .unwrap_or(0);
        // A greedy request keeps a few quanta in flight; a speculative step keeps its catch-up, drafts and verification,
        // and the owner prepares one more behind them.
        let step = if plan.drafts() {
            stages_of(View::Verify {
                rows: plan.verified_rows(),
            }) + stages_of(View::Draft { rows: 1 }) * plan.verified_rows()
                + biggest
        } else {
            0
        };
        // Every sequence keeps that many in flight on the same lanes.
        let needed = (biggest * STAGE_RING_QUANTA).max(step) * sequences.max(1);
        if needed > core.stage_ring {
            return Err(EngineError::Runtime(format!(
                "the quanta in flight of {sequences} sequences record {needed} markers on a lane and the lanes' rings of \
                 {} events do not hold them",
                core.stage_ring
            )));
        }
        let runtime = GraphRuntime {
            stages: Arc::clone(&core.stages) as Arc<dyn FrameSink>,
            completions: Arc::clone(&core.completions),
            lanes: Arc::clone(&core.lanes),
            retries: Arc::new(RetryCounts::default()),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        };
        let retiring = Arc::clone(address);
        Executor::new(
            Arc::new(plan),
            &runtime,
            core.compute_lanes(),
            Arc::clone(&core.gpu),
            core.settings.kernels.graphs.captures_per_view as usize,
            move |quantum, outcome| {
                // Runs the owner's turn on this worker when no other turn runs.
                retiring.post(Message::Retired(Box::new(Retired { quantum, outcome })));
            },
        )
        .map_err(|error| EngineError::Runtime(format!("the stage graphs: {error:?}")))
    }

    /// Starts `request` on this sequence, which `lease` holds until the request ends.
    fn generate(&self, request: GenerationRequest, cancel: CancelHandle, lease: Lease) -> Result<(), AdmissionError> {
        let core = Arc::clone(&self.shared.core);
        self.start(move |owner| match owner {
            Some(owner) => owner.begin(admit(&core), request, cancel, lease),
            None => Err(closed(Audience::Greedy(request.listener))),
        })
    }

    /// Starts a teacher-forced run on this sequence.
    fn teacher_forced(
        &self,
        prefix: Vec<u32>,
        forced: Vec<u32>,
        sink: Box<dyn LogitsSink>,
        lease: Lease,
    ) -> Result<(), AdmissionError> {
        let core = Arc::clone(&self.shared.core);
        self.start(move |owner| match owner {
            Some(owner) => owner.begin_forced(admit(&core), prefix, forced, sink, lease),
            None => Err(closed(Audience::Forced(sink))),
        })
    }

    /// Starts the request `begin` makes. Whether it is admitted was decided by the pool that gave the sequence, never by
    /// whether a turn holds the claim. With the claim free the answer is the return value, and
    /// the first turn runs on a worker, never on the caller's thread. With the claim held the request is posted to the
    /// owner and this returns `Ok` at once (no wait: a listener calls this from a worker); a request the owner then
    /// refuses has its listener or sink told, as a failed outcome, after the claim is released.
    fn start(
        &self,
        begin: impl FnOnce(Option<&mut Owner>) -> Result<(), Rejected> + Send + 'static,
    ) -> Result<(), AdmissionError> {
        // A request that begins once the engine stops is refused as a closed one: the stop's cancel is an operation on
        // the owner, so whatever begins after it sees the flag.
        let shared = Arc::clone(&self.shared);
        let begin = move |owner: Option<&mut Owner>| begin(unless_closing(&shared.closing, owner));
        let mut begin = Some(begin);
        let inline = self
            .mailbox
            .try_with(|owner| begin.take().map_or(Ok(()), |begin| begin(Some(owner))));
        match inline {
            Ok(Some(started)) => {
                let started = started.map_err(|rejected| rejected.error);
                if started.is_ok() && !self.mailbox.nudge() {
                    // No worker will take frames: the request would never run. Give it up; applied whoever holds the
                    // claim.
                    self.mailbox.tell(Owner::withdraw);
                    return Err(AdmissionError::Engine("the engine is shutting down".to_string()));
                }
                started
            }
            // The owner is closed: the caller learns from the error.
            Ok(None) => Err(closed_error()),
            Err(_) => {
                let Some(begin) = begin else {
                    return Err(closed_error());
                };
                self.mailbox.defer(move |owner| match owner {
                    Some(owner) => {
                        if let Err(rejected) = begin(Some(owner)) {
                            owner.reject(rejected);
                        }
                        Vec::new()
                    }
                    // Closed while the request waited: its audience is told, once, with the other outcomes.
                    None => begin(None).err().map(Rejected::into_finished).into_iter().collect(),
                });
                Ok(())
            }
        }
    }

    /// The test hook: the owner panics at the end of the turn in which a request has admitted `quanta` quanta.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn panic_after_admissions(&self, quanta: u64) {
        self.shared.panic_after_admissions.store(quanta, Ordering::Release);
    }

    /// The test hook: a speculative request ends with a decode quantum that feeds its last token.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn settle_after_speculation(&self, settle: bool) {
        self.shared.settle_after_speculation.store(settle, Ordering::Release);
    }

    /// The test hook: the engine records no graphs from now on. Only while no request runs.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn disable_graphs(&self) -> bool {
        self.mailbox.ask(true, Owner::disable_graphs).is_some()
    }

    /// Looks at the sequence while no request holds it: `None` only while one does (or the engine is closed), never
    /// because a turn holds the claim at the moment. Waits at most the mailbox's bound: not for a lattice worker.
    pub(crate) fn inspect<R: Send + 'static>(
        &self,
        read: impl FnOnce(&SequenceState) -> R + Send + 'static,
    ) -> Option<R> {
        self.mailbox.ask(true, move |owner| read(owner.sequence()))
    }

    /// Ends the sequence. Nothing may be running: with a request in flight the sequence stays allocated, and says so.
    /// `false` then.
    fn close(&self) -> bool {
        match self.mailbox.close() {
            Some(owner) => {
                owner.close();
                true
            }
            None => {
                log::error!("a generation is running at shutdown: the sequence's device memory stays allocated");
                false
            }
        }
    }

    /// Takes the sequence back from an owner that has run nothing.
    fn take_sequence(&self) -> Option<SequenceState> {
        self.mailbox.close().map(Owner::release)
    }
}

/// The engine's sequences, each with an owner of its own, the pool that says which of them is idle, and the one order
/// they share on the workspace.
pub(crate) struct Generations {
    generations: Vec<Generation>,
    pool: SequencePool,
    order: Arc<Order<Executor>>,
    /// A test hook: the next request admitted arms the owner's panic after this many admissions.
    #[cfg(feature = "test-oracles")]
    panic_next: AtomicU64,
}

impl fmt::Debug for Generations {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Generations")
            .field("sequences", &self.generations.len())
            .finish_non_exhaustive()
    }
}

impl Generations {
    /// Builds an owner and the graphs of the model's plan for each of `sequences`, and the one order they share. A
    /// refused plan gives every sequence back.
    pub(crate) fn new(
        core: &Arc<Core>,
        config: &Arc<ModelConfig>,
        residency: &ResidencyPlan,
        weights: &Weights,
        workspace: &Arc<Workspace>,
        sequences: Vec<SequenceState>,
        draft_ids: Option<Arc<[u32]>>,
    ) -> Result<Generations, Box<(EngineError, Vec<SequenceState>)>> {
        assert!(
            workspace.records() as u64 >= input_records_for(residency.draft_depth, sequences.len()),
            "the workspace holds input records for every sequence"
        );
        // The owners' addresses exist before the owners do: the workspace's owner tells each of its sequence's quanta
        // that did not start.
        let addresses: Vec<Arc<Address<Owner>>> = sequences.iter().map(|_| Address::new()).collect();
        let replies: Vec<Reply> = addresses
            .iter()
            .map(|address| {
                let told = Arc::clone(address);
                Box::new(move |serial, error| {
                    told.post(Message::StartFailed(Box::new(StartFailed { serial, error })));
                }) as Reply
            })
            .collect();
        let order = Arc::new(Order::new(
            replies,
            input_records(residency.draft_depth) as usize,
            Generation::spawner(core),
        ));
        let count = sequences.len();
        let mut generations: Vec<Generation> = Vec::with_capacity(count);
        let mut waiting = sequences.into_iter().zip(addresses).enumerate();
        while let Some((index, (sequence, address))) = waiting.next() {
            let built = Generation::new(
                core,
                config,
                residency,
                weights,
                workspace,
                sequence,
                SequenceId(index),
                address,
                &order,
                count,
                draft_ids.clone(),
            );
            match built {
                Ok(generation) => generations.push(generation),
                Err(refused) => {
                    let (error, sequence) = *refused;
                    let mut back: Vec<SequenceState> =
                        generations.iter().filter_map(Generation::take_sequence).collect();
                    back.push(sequence);
                    back.extend(waiting.map(|(_, (sequence, _))| sequence));
                    drop(order.close());
                    return Err(Box::new((error, back)));
                }
            }
        }
        Ok(Generations {
            pool: SequencePool::new(count),
            generations,
            order,
            #[cfg(feature = "test-oracles")]
            panic_next: AtomicU64::new(u64::MAX),
        })
    }

    /// The number of sequences.
    pub(crate) fn sequences(&self) -> usize {
        self.generations.len()
    }

    /// Takes an idle sequence, or refuses with `Busy` when every sequence runs a generation.
    fn take(&self) -> Result<(&Generation, Lease), AdmissionError> {
        let (id, lease) = self.pool.take().ok_or(AdmissionError::Busy)?;
        let generation = &self.generations[id.0];
        #[cfg(feature = "test-oracles")]
        {
            let quanta = self.panic_next.swap(u64::MAX, Ordering::AcqRel);
            if quanta != u64::MAX {
                generation.panic_after_admissions(quanta);
            }
        }
        Ok((generation, lease))
    }

    /// Starts `request` on an idle sequence.
    pub(crate) fn generate(&self, request: GenerationRequest, cancel: CancelHandle) -> Result<(), AdmissionError> {
        let (generation, lease) = self.take()?;
        generation.generate(request, cancel, lease)
    }

    /// Starts a teacher-forced run on an idle sequence.
    pub(crate) fn teacher_forced(
        &self,
        prefix: Vec<u32>,
        forced: Vec<u32>,
        sink: Box<dyn LogitsSink>,
    ) -> Result<(), AdmissionError> {
        let (generation, lease) = self.take()?;
        generation.teacher_forced(prefix, forced, sink, lease)
    }

    /// The test hook: the owner of the sequence the next request takes panics at the end of the turn in which that
    /// request has admitted `quanta` quanta.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn panic_after_admissions(&self, quanta: u64) {
        self.panic_next.store(quanta, Ordering::Release);
    }

    /// The test hook: a speculative request ends with a decode quantum that feeds its last token.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn settle_after_speculation(&self, settle: bool) {
        for generation in &self.generations {
            generation.settle_after_speculation(settle);
        }
    }

    /// The test hook: no sequence records graphs from now on. Only while no request runs.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn disable_graphs(&self) -> bool {
        let disabled: Vec<bool> = self.generations.iter().map(Generation::disable_graphs).collect();
        disabled.into_iter().all(|disabled| disabled)
    }

    /// Looks at sequence `index` while no request holds it. `None` while one does, or when there is no such sequence.
    pub(crate) fn inspect<R: Send + 'static>(
        &self,
        index: usize,
        read: impl FnOnce(&SequenceState) -> R + Send + 'static,
    ) -> Option<R> {
        self.generations.get(index)?.inspect(read)
    }

    /// Cancels every request that runs and waits at most `bound` in all until each ended and its listener was told its
    /// last call (a request that ends in a stop is told `Cancelled`). Refuses new requests from now on. Runs while the
    /// lattice still works. `false` when the bound passed with a request still running, and the caller must not free
    /// what it uses; called from inside a turn it returns `false` at once. For the engine's owner, outside the lattice.
    pub(crate) fn stop(&self, bound: Duration) -> bool {
        // Every owner is told to stop before any is waited for, so that the requests end side by side.
        let deadline = Instant::now() + bound;
        let idle: Vec<_> = self
            .generations
            .iter()
            .map(|generation| {
                generation.shared.closing.store(true, Ordering::SeqCst);
                generation.mailbox.stopping()
            })
            .collect();
        let stopped = idle.into_iter().fold(true, |all, idle| {
            let ended = idle.is_some_and(|idle| idle.wait(deadline.saturating_duration_since(Instant::now())).is_ok());
            all && ended
        });
        if !stopped {
            log::error!("a generation did not end within {bound:?} of the cancel at shutdown");
        }
        stopped
    }

    /// Ends the sequences, then the workspace's order, which they use. Nothing may be running: a sequence with a
    /// request in flight stays allocated, says so, and keeps the order's pins.
    pub(crate) fn close(&self) {
        let closed: Vec<bool> = self.generations.iter().map(Generation::close).collect();
        if closed.into_iter().all(|closed| closed) {
            // What the workspace's owner still pins goes with it.
            drop(self.order.close());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_request_begins_on_an_owner_once_the_engine_stops() {
        let closing = AtomicBool::new(false);
        let mut owner = 7;
        assert_eq!(unless_closing(&closing, Some(&mut owner)).copied(), Some(7));
        closing.store(true, Ordering::SeqCst);
        assert!(unless_closing(&closing, Some(&mut owner)).is_none());
        assert!(unless_closing::<i32>(&closing, None).is_none());
    }
}
