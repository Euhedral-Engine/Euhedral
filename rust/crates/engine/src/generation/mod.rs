//! Generation: prefill, greedy decode and teacher-forced decoding of the engine's sequence.
//!
//! [`EngineRuntime::generate`](crate::EngineRuntime::generate) starts a greedy request and
//! [`EngineRuntime::teacher_forced`](crate::EngineRuntime::teacher_forced) a teacher-forced run. Both are carried by a
//! [`GenerationSource`](source::GenerationSource), a lattice source that the workers poll while the request runs and
//! that publishes the frames of the sequence's owner into the `stages` lake; the caller is told through its listener or
//! sink, on a worker, and waits on a latch of its own.

mod baton;
mod listener;
mod request;
mod source;
mod teacher_forced;

use std::fmt::{self, Formatter};
use std::sync::Arc;

use euhedral_core::generics::FrameSink;
use euhedral_core::generics::LatticeSource;
use euhedral_data_structures::queues::{MpmcQueue, Multi, Producer};
use euhedral_graph::{GraphRuntime, RetryCounts};
use euhedral_inference_cuda::CudaLibrary;

use crate::error::EngineError;
use crate::formats::{Nvfp4Linear, TrellisLinear};
use crate::model::{ModelConfig, Weights};
use crate::plan::{Formats, Plan, View};
use crate::quantum::{Executor, GraphCache};
use crate::residency::planner::ResidencyPlan;
use crate::runtime::{Core, STAGE_RING, STAGE_RING_QUANTA, Units};
use crate::state::SequenceState;
use crate::workspace::Workspace;
use baton::Baton;
pub use listener::GenerationListener;
pub use request::{AdmissionError, FinishReason, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation};
use source::{GenerationSource, Message, Owner, Retired, Shared};
pub use teacher_forced::LogitsSink;

/// What the engine keeps for generating: the plan's graphs on the runtime's lanes, and the owner of the sequence.
pub(crate) struct Generation {
    shared: Arc<Shared>,
    /// The owner, while no request holds it.
    home: Arc<Baton<Box<Owner>>>,
}

impl fmt::Debug for Generation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Generation").finish_non_exhaustive()
    }
}

impl Generation {
    /// Builds the plan of the loaded model and the graphs of its views, and makes `sequence` their owner's. A refused
    /// plan gives the sequence back.
    pub(crate) fn new(
        core: &Arc<Core>,
        config: &Arc<ModelConfig>,
        residency: &ResidencyPlan,
        weights: &Weights,
        workspace: &Arc<Workspace>,
        sequence: SequenceState,
    ) -> Result<Generation, Box<(EngineError, SequenceState)>> {
        let capacity = sequence.kv.planned_rows().min(residency.max_context_tokens as u64);
        let queue = MpmcQueue::<Message>::new(256);
        let (producer, inbox) = (queue.producer(), queue.consumer());
        let executor = match Self::executor(core, config, residency, weights, &producer) {
            Ok(executor) => executor,
            Err(error) => return Err(Box::new((error, sequence))),
        };
        let shared = Arc::new(Shared {
            core: Arc::clone(core),
            config: Arc::clone(config),
            workspace: Arc::clone(workspace),
            executor,
            producer,
            chunk: residency.rows.rows as usize,
            capacity,
        });
        let cache = GraphCache::with_capacity(core.settings.kernels.graphs.captures_per_view as usize);
        let home = Arc::new(Baton::new());
        home.put(Box::new(Owner::new(Arc::clone(&shared), sequence, cache, inbox)));
        Ok(Generation { shared, home })
    }

    /// The stage graphs of the model's plan, whose quanta retire into the owner's inbox.
    fn executor(
        core: &Arc<Core>,
        config: &Arc<ModelConfig>,
        residency: &ResidencyPlan,
        weights: &Weights,
        inbox: &Producer<Message, Multi>,
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
        let stages = [
            View::Decode,
            View::PrefillChunk,
            View::Verify { rows: 2 },
            View::Draft { rows: 1 },
        ]
        .iter()
        .filter_map(|view| plan.stages(*view).map(<[_]>::len))
        .max()
        .unwrap_or(0);
        if stages * STAGE_RING_QUANTA > STAGE_RING {
            return Err(EngineError::Runtime(format!(
                "a quantum of {stages} stages records a marker per stage and the lanes' rings of {STAGE_RING} events \
                 do not hold the markers of {STAGE_RING_QUANTA} quanta"
            )));
        }
        let runtime = GraphRuntime {
            stages: Arc::clone(&core.stages) as Arc<dyn FrameSink>,
            completions: Arc::clone(&core.completions),
            lanes: Arc::clone(&core.lanes),
            retries: Arc::new(RetryCounts::default()),
            retry_backstop: euhedral_graph::DEFAULT_RETRY_BACKSTOP,
        };
        let retiring = inbox.clone();
        Executor::new(
            Arc::new(plan),
            &runtime,
            core.capture_lane(),
            move |quantum, outcome| {
                if retiring
                    .offer(Message::Retired(Box::new(Retired { quantum, outcome })))
                    .is_err()
                {
                    log::error!("the owner's inbox refused a retired quantum");
                }
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

    fn run(&self, begin: impl FnOnce(&mut Owner, Units) -> Result<(), AdmissionError>) -> Result<(), AdmissionError> {
        let core = &self.shared.core;
        let mut owner = self.home.take().ok_or(AdmissionError::Busy)?;
        let units = match core.admit() {
            Ok(units) => units,
            Err(error) => {
                self.home.put(owner);
                return Err(AdmissionError::Engine(error.to_string()));
            }
        };
        if let Err(error) = begin(&mut owner, units) {
            self.home.put(owner);
            return Err(error);
        }
        let source = GenerationSource::new(owner, Arc::clone(&self.home), Arc::clone(&core.stages));
        if let Err(error) = core.lattice.add_upstream(Arc::clone(&source) as Arc<dyn LatticeSource>) {
            if let Some(owner) = source.take_owner() {
                self.home.put(owner);
            }
            return Err(AdmissionError::Engine(format!(
                "the lattice would not take the request: {error:?}"
            )));
        }
        Ok(())
    }

    /// Looks at the sequence while no request holds it.
    pub(crate) fn inspect<R>(&self, read: impl FnOnce(&SequenceState) -> R) -> Option<R> {
        let owner = self.home.take()?;
        let out = read(owner.sequence());
        self.home.put(owner);
        Some(out)
    }

    /// Ends the sequence. Nothing may be running: with a request in flight the sequence stays allocated, and says so.
    pub(crate) fn close(&self) {
        match self.home.take() {
            Some(owner) => owner.close(),
            None => log::error!("a generation is running at shutdown: the sequence's device memory stays allocated"),
        }
    }
}
