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

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test: the pump is the worker, a failed unwrap is the failure, and the reports are read after the chunks end"
)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::thread;
    use std::time::{Duration, Instant};

    use euhedral_gpu::cuda::CudaModule;
    use euhedral_inference_cuda::CudaLibrary;
    use euhedral_inference_engine::generation::{FinishReason, GenerationOutcome};

    use super::*;

    const VOCABULARY: usize = 64;
    const RECORD_BYTES: usize = 2 * (VOCABULARY + 4);
    const STEPS: usize = 7;

    type Reports = Arc<Mutex<Vec<Result<Vec<f64>, String>>>>;

    fn row(target: u32) -> ScoreRow {
        let logits = (0..VOCABULARY)
            .flat_map(|at| ((((at as f32) * 0.25).to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let mut record = vec![0u8; RECORD_BYTES];
        record[..4].copy_from_slice(&(1.0f32 / 64.0).to_bits().to_le_bytes());
        record[4..8].copy_from_slice(&(-8.0f32).to_bits().to_le_bytes());
        for at in 0..VOCABULARY {
            record[8 + 2 * at..10 + 2 * at].copy_from_slice(&((at * 3) as u16).to_le_bytes());
        }
        ScoreRow { logits, record, target }
    }

    fn chunk(reports: &Reports) -> Arc<ChunkScore> {
        let sink = Arc::clone(reports);
        let chunk = ChunkScore::new(STEPS, Arc::new(move |verdict| sink.lock().unwrap().push(verdict)));
        chunk.finished(GenerationOutcome {
            reason: FinishReason::Length,
            prompt_tokens: 0,
            generated: STEPS,
            elapsed: Duration::ZERO,
            cached_prompt_tokens: 0,
        });
        chunk
    }

    /// Pumps the lane's retirements until `reports` holds `count` verdicts.
    fn pump(completions: &Arc<Completions>, reports: &Reports, count: usize) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while reports.lock().unwrap().len() < count {
            completions.reap();
            completions.pull(
                &mut |mut frame| {
                    frame.execute().unwrap();
                    frame.do_finally();
                },
                &mut |_| false,
                u64::MAX,
            );
            assert!(
                Instant::now() < deadline,
                "the chunks never ended: a group's retirement did not arrive"
            );
        }
    }

    #[test]
    #[ignore = "needs a GPU and the bundled NVRTC"]
    fn rows_from_two_threads_all_land_through_one_lane_and_a_failed_group_poisons_it() {
        let path = test_support::gradle_property("euhedral.cuda.library")
            .expect("set euhedral.cuda.library (EUHEDRAL_TEST_CUDA_LIBRARY) to the engine library");
        let library = CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads");
        let gpu = open_device(&library).expect("the device opens");
        let mut modules = Modules::<CudaModule>::open(&gpu, ModuleOptions::beside(&library).unwrap()).unwrap();
        let kernels = ScoreKernels::load(&mut modules).unwrap();
        let (pool, mut streams) = gpu.open_lanes(&["score"], StreamOptions::default()).unwrap();
        let completions = Completions::new(&pool);
        // A group of three: the fourteen rows below are at least five groups, so the lane goes away and comes back.
        let geometry = Geometry {
            vocabulary: VOCABULARY,
            record_bytes: RECORD_BYTES,
            group: 3,
        };
        let lane = ScoreLane::new(streams.remove(0), kernels, &library, geometry, Arc::clone(&completions)).unwrap();
        let scorer = Scorer::with(lane, Some((Arc::clone(&gpu), pool)));

        let reports: Reports = Arc::default();
        let chunks = [chunk(&reports), chunk(&reports)];
        thread::scope(|scope| {
            for chunk in &chunks {
                let (scorer, chunk) = (&scorer, Arc::clone(chunk));
                scope.spawn(move || {
                    for step in 0..STEPS {
                        scorer.submit(Job {
                            chunk: Arc::clone(&chunk),
                            step,
                            row: row(5),
                        });
                    }
                });
            }
        });
        drop(chunks);
        pump(&completions, &reports, 2);
        {
            let reports = reports.lock().unwrap();
            let first = reports[0].as_ref().expect("every row of the first chunk scored");
            assert_eq!(first.len(), STEPS * FIELDS);
            // The same row everywhere: every position has the same five values, whichever group it rode in.
            assert!(first.chunks(FIELDS).all(|values| values == &first[..FIELDS]));
            assert_eq!(reports[1].as_ref().unwrap(), first, "both chunks scored the same rows");
        }

        // A row the lane refuses poisons it: its chunk fails, and the next row fails at once with the same reason.
        reports.lock().unwrap().clear();
        let bad = chunk(&reports);
        scorer.submit(Job {
            chunk: Arc::clone(&bad),
            step: 0,
            row: row(u32::try_from(VOCABULARY).unwrap()),
        });
        drop(bad);
        pump(&completions, &reports, 1);
        let refused = reports.lock().unwrap()[0].clone().unwrap_err();
        let later = chunk(&reports);
        scorer.submit(Job {
            chunk: Arc::clone(&later),
            step: 0,
            row: row(5),
        });
        drop(later);
        pump(&completions, &reports, 2);
        assert_eq!(
            reports.lock().unwrap()[1],
            Err(refused),
            "a poisoned scorer fails later rows with its reason"
        );
        scorer.close();
    }
}
