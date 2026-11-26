//! The scorer: rows from any number of runs, scored in groups on one lane.
//!
//! A row is a [`Job`]: the row, the chunk it belongs to and its step. Any thread offers one with [`Scorer::submit`]; the
//! lane scores up to a group at a time and each job's values go to its chunk's slots. Nothing waits and nothing locks.
//!
//! The lane and the consumers of the two queues (the jobs, and the lane coming back from a finished group) are state
//! one thread at a time may touch: a [`ClaimCell`]. A thread that loses the claim has published what it wanted first
//! and counted it in `signals`; the holder reads `signals` before it looks and again after it lets go, and looks once
//! more if it moved, so what was published is found by the holder or by its publisher's own next claim. A queue that
//! has a slot claimed but not yet published hides what is behind it, so a look that finds nothing is not a reason to
//! look again until `signals` moves: the publisher of that slot counts it when it lands.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use euhedral_core::generics::LatticeSource;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_data_structures::sync::ClaimCell;
use euhedral_gpu::cuda::CudaGpu;
use euhedral_gpu::{Gpu, StreamOptions};
use euhedral_graph::{Completions, LanePool};
use euhedral_inference_engine::EngineRuntime;
use euhedral_inference_engine::runtime::open_device;

use super::chunk::ChunkScore;
use super::device::{Finished, Geometry, ScoreError, ScoreKernels, ScoreLane, ScoreRow};
use super::summary::FIELDS;
use crate::kernels::{ModuleOptions, Modules};

/// The jobs and lanes a queue holds per chunk of storage.
const CHUNK: usize = 64;

/// One row of one chunk.
pub(crate) struct Job {
    pub(crate) chunk: Arc<ChunkScore>,
    pub(crate) step: usize,
    pub(crate) row: ScoreRow,
}

struct Hub {
    /// The lane, while it is here: it is away while a group runs.
    lane: Option<ScoreLane>,
    jobs: Consumer<Job, Single>,
    returned: Consumer<ScoreLane, Single>,
}

/// Scores rows on a lane, in groups.
pub struct Scorer {
    group: usize,
    hub: ClaimCell<Hub>,
    jobs: Producer<Job, Multi>,
    returned: Producer<ScoreLane, Multi>,
    signals: AtomicUsize,
    /// Why the lane can score no more, once a group has failed: its staging may still be written by that group.
    poisoned: OnceLock<String>,
    /// The device and the lane pool stay open for the lane.
    _open: Option<(Arc<CudaGpu>, Arc<LanePool>)>,
}

impl Debug for Scorer {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("Scorer")
            .field("group", &self.group)
            .finish_non_exhaustive()
    }
}

impl Scorer {
    /// A scorer over `lane`.
    pub fn new(lane: ScoreLane) -> Arc<Scorer> {
        Scorer::with(lane, None)
    }

    fn with(lane: ScoreLane, open: Option<(Arc<CudaGpu>, Arc<LanePool>)>) -> Arc<Scorer> {
        let group = lane.geometry().group;
        let (jobs, returned) = (MpscQueue::<Job>::new(CHUNK), MpscQueue::<ScoreLane>::new(CHUNK));
        Arc::new(Scorer {
            group,
            hub: ClaimCell::new(Hub {
                lane: Some(lane),
                jobs: jobs.consumer(),
                returned: returned.consumer(),
            }),
            jobs: jobs.producer(),
            returned: returned.producer(),
            signals: AtomicUsize::new(0),
            poisoned: OnceLock::new(),
            _open: open,
        })
    }

    /// Opens a lane of its own on the engine's device, compiles the score kernel and has the engine's lattice deliver
    /// the lane's retirements. Waits for the compiler: call it at startup, once the engine is ready.
    pub fn open(runtime: &EngineRuntime, geometry: Geometry) -> Result<Arc<Scorer>, ScoreError> {
        let library = runtime.library();
        let gpu = open_device(library).map_err(|error| ScoreError::Request(error.to_string()))?;
        let options = ModuleOptions::beside(library).map_err(|error| ScoreError::Request(error.to_string()))?;
        let mut modules = Modules::open(&gpu, options)?;
        let kernels = ScoreKernels::load(&mut modules)?;
        let (pool, mut streams) = gpu.open_lanes(&["score"], StreamOptions::default())?;
        let stream = streams
            .pop()
            .ok_or_else(|| ScoreError::Request("the device opened no lane".to_owned()))?;
        let completions = Completions::new(&pool);
        runtime
            .lattice()
            .add_upstream(Arc::clone(&completions) as Arc<dyn LatticeSource>)
            .map_err(|error| ScoreError::Request(format!("the lattice would not take the completions: {error:?}")))?;
        let lane = ScoreLane::new(stream, kernels, library, geometry, completions)?;
        Ok(Scorer::with(lane, Some((gpu, pool))))
    }

    /// Offers a row to the scorer. Never waits.
    pub(crate) fn submit(self: &Arc<Self>, job: Job) {
        if let Err(job) = self.jobs.offer(job) {
            job.chunk.fail("the scorer's queue refused a row".to_owned());
            return;
        }
        self.signals.fetch_add(1, Ordering::SeqCst);
        self.pump();
    }

    /// Ends the scorer at shutdown, once no run is left: waits for the lane's stream and frees its staging. A lane that
    /// is away with a group (a run was abandoned) is left to the process's end.
    pub fn close(&self) {
        let lane = self.hub.try_claim().and_then(|mut hub| {
            let lane = hub.lane.take();
            lane.or_else(|| hub.returned.poll())
        });
        if let Some(lane) = lane {
            lane.close();
        }
    }

    /// Starts a group if the lane is here and a row is waiting.
    fn pump(self: &Arc<Self>) {
        loop {
            let seen = self.signals.load(Ordering::SeqCst);
            let Some(mut hub) = self.hub.try_claim() else { return };
            if hub.lane.is_none() {
                hub.lane = hub.returned.poll();
            }
            let mut taken = Vec::new();
            if hub.lane.is_some() {
                while taken.len() < self.group {
                    match hub.jobs.poll() {
                        Some(job) => taken.push(job),
                        None => break,
                    }
                }
            }
            let lane = if taken.is_empty() { None } else { hub.lane.take() };
            hub.release_seq_cst();
            match lane {
                Some(lane) => {
                    if !self.start(lane, taken) {
                        return;
                    }
                }
                None if self.signals.load(Ordering::SeqCst) == seen => return,
                None => {}
            }
        }
    }

    /// Starts `jobs` on `lane`. Returns true when the lane went back at once, so the caller looks again.
    fn start(self: &Arc<Self>, lane: ScoreLane, jobs: Vec<Job>) -> bool {
        if let Some(why) = self.poisoned.get() {
            for job in &jobs {
                job.chunk.fail(why.clone());
            }
            self.give_back(lane);
            return true;
        }
        let (rows, owners): (Vec<ScoreRow>, Vec<(Arc<ChunkScore>, usize)>) =
            jobs.into_iter().map(|job| (job.row, (job.chunk, job.step))).unzip();
        let held: Vec<Arc<ChunkScore>> = owners.iter().map(|(chunk, _)| Arc::clone(chunk)).collect();
        let scorer = Arc::clone(self);
        let done = Box::new(move |finished: Finished| {
            let Finished { lane, scratch, result } = finished;
            match result {
                Ok(values) => {
                    for ((chunk, step), row) in owners.iter().zip(values.as_chunks::<FIELDS>().0) {
                        chunk.store(*step, row);
                    }
                }
                Err(error) => {
                    let why = format!("scoring a group of rows: {error}");
                    let _ = scorer.poisoned.set(why.clone());
                    for (chunk, _) in &owners {
                        chunk.fail(why.clone());
                    }
                }
            }
            scorer.give_back(lane);
            scorer.pump();
            // The rows' chunks end when the last of these lets go, after the lane is back.
            drop(owners);
            drop(scratch);
        });
        match lane.score(rows, done) {
            Ok(()) => false,
            Err(refused) => {
                let why = format!("scoring a group of rows: {}", refused.error);
                let _ = self.poisoned.set(why.clone());
                for chunk in &held {
                    chunk.fail(why.clone());
                }
                self.give_back(refused.lane);
                true
            }
        }
    }

    fn give_back(&self, lane: ScoreLane) {
        // The queue is unbounded: it refuses only when it is closed, with the scorer.
        if self.returned.offer(lane).is_ok() {
            self.signals.fetch_add(1, Ordering::SeqCst);
        }
    }
}
