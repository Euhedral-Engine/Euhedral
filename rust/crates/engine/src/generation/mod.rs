//! Generation: prefill, greedy decode and teacher-forced decoding of the engine's sequence.
//!
//! [`EngineRuntime::generate`](crate::EngineRuntime::generate) starts a greedy request and
//! [`EngineRuntime::teacher_forced`](crate::EngineRuntime::teacher_forced) a teacher-forced run. Both are carried by the
//! sequence's owner, which sits in a [`Mailbox`](mailbox::Mailbox): the frames that tell it something (a quantum
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
mod request;
mod sampled;
mod source;
mod teacher_forced;

use std::fmt::{self, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use euhedral_core::generics::FrameSink;
use euhedral_data_structures::queues::MpscQueue;
use euhedral_graph::{DEFAULT_RETRY_BACKSTOP, GraphRuntime, RetryCounts};
use euhedral_inference_cuda::CudaLibrary;

use crate::error::EngineError;
use crate::formats::{Nvfp4Linear, TrellisLinear};
use crate::frames::{Kind, step};
use crate::model::{LayerIndex, ModelConfig, Role, Weights};
use crate::plan::{Formats, Plan, View};
use crate::prefix::{CacheSpec, Copier, open_prefix_cache};
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
pub use mailbox::Spawn;
pub(crate) use mailbox::{Address, Mailbox, Turn};
use order::Order;
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
    /// Builds the plan of the loaded model and the graphs of its views, and makes `sequence` their owner's. A refused
    /// plan gives the sequence back. `draft_ids` are the token ids of the draft head's rows, which an artifact with a
    /// draft layer needs to speculate.
    pub(crate) fn new(
        core: &Arc<Core>,
        config: &Arc<ModelConfig>,
        residency: &ResidencyPlan,
        weights: &Weights,
        workspace: &Arc<Workspace>,
        sequence: SequenceState,
        draft_ids: Option<Arc<[u32]>>,
    ) -> Result<Generation, Box<(EngineError, SequenceState)>> {
        let capacity = sequence.kv.planned_rows().min(residency.max_context_tokens as u64);
        // An unbounded queue, whose chunk holds the messages the protocol has in flight: the quanta in flight retire into
        // it, a teacher-forced step posts one row, and a parked quantum posts that its lane work ran. Offering to it
        // cannot fail, so no message is ever dropped.
        let queue = MpscQueue::<Message>::new(GREEDY_DEPTH + 4);
        let (producer, inbox) = (queue.producer(), queue.consumer());
        let address = Address::new();
        let prefix = open_prefix_cache(
            CacheSpec {
                library: &core.library,
                config,
                format: residency.kv_format,
                settings: core.settings.prefix_cache,
                granted: residency.prefix_cache_bytes,
                chunk: residency.rows.rows as u32,
                device_tokens: u32::try_from(residency.kv_device_tokens).unwrap_or(u32::MAX),
            },
            Self::spawner(core),
        );
        let executor = match Self::executor(core, config, residency, weights, &address, prefix.is_some()) {
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
        // The engine's one workspace is ordered by one owner. Its starts that fail are told to the sequence's owner, whose
        // address exists before the owner does.
        let id = SequenceId(0);
        let told = Arc::clone(&address);
        let replies: Vec<order::Reply> = vec![Box::new(move |serial, error| {
            told.post(Message::StartFailed(Box::new(StartFailed { serial, error })));
        })];
        assert!(
            workspace.records() as u64 >= input_records_for(residency.draft_depth, residency.sequences),
            "the workspace holds input records for every sequence"
        );
        let order = Order::new(
            replies,
            input_records(residency.draft_depth) as usize,
            Self::spawner(core),
        );
        let shared = Arc::new(Shared {
            core: Arc::clone(core),
            config: Arc::clone(config),
            workspace: Arc::clone(workspace),
            executor: Arc::new(executor),
            order,
            address: Arc::clone(&address),
            chunk: residency.rows.rows as usize,
            capacity,
            drafts,
            prefix,
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
        copies: bool,
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
        let needed = (biggest * STAGE_RING_QUANTA).max(step);
        if needed > core.stage_ring {
            return Err(EngineError::Runtime(format!(
                "the quanta in flight record {needed} markers on a lane and the lanes' rings of {} events do not hold \
                 them",
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
        let executor = Executor::new(
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
        .map_err(|error| EngineError::Runtime(format!("the stage graphs: {error:?}")))?;
        if !copies {
            return Ok(executor);
        }
        let copier = Copier::new(
            &runtime,
            core.lane_table.clone(),
            core.transfer_lane_id(0),
            SequenceId(0),
        )
        .map_err(|error| EngineError::Runtime(format!("the prefix cache's copy graph: {error:?}")))?;
        Ok(executor.with_copier(copier))
    }

    /// Starts `request` on the engine's sequence.
    pub(crate) fn generate(&self, request: GenerationRequest, cancel: CancelHandle) -> Result<(), AdmissionError> {
        let core = Arc::clone(&self.shared.core);
        self.start(move |owner| match owner {
            Some(owner) => owner.begin(admit(&core), request, cancel),
            None => Err(closed(Audience::Greedy(request.listener))),
        })
    }

    /// Starts a teacher-forced run on the engine's sequence.
    pub(crate) fn teacher_forced(
        &self,
        prefix: Vec<u32>,
        forced: Vec<u32>,
        sink: Box<dyn LogitsSink>,
    ) -> Result<(), AdmissionError> {
        let core = Arc::clone(&self.shared.core);
        self.start(move |owner| match owner {
            Some(owner) => owner.begin_forced(admit(&core), prefix, forced, sink),
            None => Err(closed(Audience::Forced(sink))),
        })
    }

    /// Starts the request `begin` makes. Whether it is admitted depends on the owner's state alone: `Busy` means a
    /// request is running, never that a turn holds the claim. With the claim free the answer is the return value, and
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

    /// Cancels the request that runs, if any, and waits at most `bound` until it ended and its listener was told its
    /// last call (a request that ends in a stop is told `Cancelled`). Refuses new requests from now on. Runs while the
    /// lattice still works: the request ends on its workers. `false` when the bound passed with a request still
    /// running, and the caller must not free what it uses. Called from inside a listener's `token`, `quantum` or
    /// `prefill_done` it returns `false` at once (the owner's turn that makes the call is what would end the request).
    /// Called from inside `finished` it cannot be told apart, and waits the bound for that very call to return, so it
    /// returns `false` after the bound. For the engine's owner, outside the lattice.
    pub(crate) fn stop(&self, bound: Duration) -> bool {
        self.shared.closing.store(true, Ordering::SeqCst);
        let stopped = self.mailbox.stop(bound);
        if !stopped {
            log::error!("a generation did not end within {bound:?} of the cancel at shutdown");
        }
        stopped
    }

    /// Ends the sequence. Nothing may be running: with a request in flight the sequence stays allocated, and says so.
    pub(crate) fn close(&self) {
        match self.mailbox.close() {
            Some(owner) => {
                owner.close();
                if let Some(cache) = &self.shared.prefix {
                    cache.close();
                }
                // What the workspace's owner still pins goes with it.
                drop(self.shared.order.close());
            }
            None => log::error!("a generation is running at shutdown: the sequence's device memory stays allocated"),
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
