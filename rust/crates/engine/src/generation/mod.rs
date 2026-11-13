//! Generation: prefill, greedy decode and teacher-forced decoding of the engine's sequence.
//!
//! [`EngineRuntime::generate`](crate::EngineRuntime::generate) starts a greedy request and
//! [`EngineRuntime::teacher_forced`](crate::EngineRuntime::teacher_forced) a teacher-forced run. Both are carried by the
//! sequence's owner, which sits in a [`Mailbox`](mailbox::Mailbox): the frames that tell it something (a quantum
//! retired, a logits row arrived, a parked quantum's lane work ran) post the message and run the owner's turn on their
//! own worker if no other turn runs. The caller is told through its listener or sink, on a worker, and waits on a
//! hand-off of its own.

mod acceptance;
mod draft_length;
mod listener;
mod mailbox;
mod mtp;
mod request;
mod source;
mod teacher_forced;

use std::fmt::{self, Formatter};
use std::sync::Arc;
#[cfg(feature = "test-oracles")]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};

use euhedral_core::generics::FrameSink;
use euhedral_data_structures::queues::MpscQueue;
use euhedral_graph::{DEFAULT_RETRY_BACKSTOP, GraphRuntime, RetryCounts};
use euhedral_inference_cuda::CudaLibrary;

use crate::error::EngineError;
use crate::formats::{Nvfp4Linear, TrellisLinear};
use crate::frames::{Kind, step};
use crate::model::{LayerIndex, ModelConfig, Role, Weights};
use crate::plan::{Formats, Plan, View};
use crate::quantum::{Executor, GraphCache};
use crate::residency::planner::ResidencyPlan;
use crate::runtime::{Core, STAGE_RING_QUANTA, Units};
use crate::state::SequenceState;
use crate::workspace::Workspace;
pub use acceptance::{Accepted, accept};
pub use draft_length::{DraftLength, DraftLengthError, MAX_DRAFTS, MAX_VERIFIED};
pub use listener::GenerationListener;
use mailbox::{Address, Mailbox, Spawn};
pub use request::{
    AdmissionError, FinishReason, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation, Verification,
};
pub(crate) use source::GREEDY_DEPTH;
use source::{Drafts, Message, Owner, Retired, Shared};
pub use teacher_forced::LogitsSink;

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
        let executor = match Self::executor(core, config, residency, weights, &address) {
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
            executor,
            address: Arc::clone(&address),
            chunk: residency.rows.rows as usize,
            capacity,
            drafts,
            #[cfg(feature = "test-oracles")]
            panic_after_admissions: AtomicU64::new(u64::MAX),
            #[cfg(feature = "test-oracles")]
            settle_after_speculation: AtomicBool::new(false),
        });
        let cache = GraphCache::with_capacity(core.settings.kernels.graphs.captures_per_view as usize);
        // A frame that drives the owner from a thread that must not run its turn, or after a frame's turns are used:
        // unordered, so that it may run on any worker.
        let stages = Arc::clone(&core.stages);
        let seeds = AtomicU64::new(0);
        let spawn: Spawn = Box::new(move |work| {
            let mut frame = step(Kind::Owner, 0, work);
            frame.header_mut().randomize_hash(seeds.fetch_add(1, Ordering::Relaxed));
            stages.offer(frame).is_ok()
        });
        let mailbox = Arc::new(Mailbox::new(
            Owner::new(Arc::clone(&shared), sequence, cache, inbox),
            producer,
            spawn,
        ));
        address.set(&mailbox);
        Ok(Generation { shared, mailbox })
    }

    /// The stage graphs of the model's plan, whose quanta retire into the owner's mailbox.
    fn executor(
        core: &Arc<Core>,
        config: &Arc<ModelConfig>,
        residency: &ResidencyPlan,
        weights: &Weights,
        address: &Arc<Address<Owner>>,
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
        Executor::new(
            Arc::new(plan),
            &runtime,
            core.capture_lane(),
            move |quantum, outcome| {
                // Runs the owner's turn on this worker when no other turn runs.
                retiring.post(Message::Retired(Box::new(Retired { quantum, outcome })));
            },
        )
        .map_err(|error| EngineError::Runtime(format!("the stage graphs: {error:?}")))
    }

    /// Starts `request` on the engine's sequence.
    pub(crate) fn generate(&self, request: GenerationRequest) -> Result<(), AdmissionError> {
        self.run(|owner, units| owner.begin(units, request))
    }

    /// Starts a teacher-forced run on the engine's sequence.
    pub(crate) fn teacher_forced(
        &self,
        prefix: Vec<u32>,
        forced: Vec<u32>,
        sink: Box<dyn LogitsSink>,
    ) -> Result<(), AdmissionError> {
        self.run(|owner, units| owner.begin_forced(units, prefix, forced, sink))
    }

    /// Claims the owner, if no turn or other claim holds it, and starts the request `begin` makes. The first turn runs
    /// on a worker, never on the caller's thread.
    fn run(&self, begin: impl FnOnce(&mut Owner, Units) -> Result<(), AdmissionError>) -> Result<(), AdmissionError> {
        let core = &self.shared.core;
        let started = self
            .mailbox
            .with(|owner| {
                let units = core
                    .admit()
                    .map_err(|error| AdmissionError::Engine(error.to_string()))?;
                begin(owner, units)
            })
            .ok_or(AdmissionError::Busy)?;
        if started.is_ok() && !self.mailbox.nudge() {
            // No worker will take frames: the request would never run. Give it up.
            self.mailbox.with(Owner::withdraw);
            return Err(AdmissionError::Engine("the engine is shutting down".to_string()));
        }
        started
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
        self.mailbox.with_idle(Owner::disable_graphs).is_some()
    }

    /// Looks at the sequence while no request holds it.
    pub(crate) fn inspect<R>(&self, read: impl FnOnce(&SequenceState) -> R) -> Option<R> {
        self.mailbox.with_idle(|owner| read(owner.sequence()))
    }

    /// Ends the sequence. Nothing may be running: with a request in flight the sequence stays allocated, and says so.
    pub(crate) fn close(&self) {
        match self.mailbox.close() {
            Some(owner) => owner.close(),
            None => log::error!("a generation is running at shutdown: the sequence's device memory stays allocated"),
        }
    }
}
