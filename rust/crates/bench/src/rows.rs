//! The work done on the logits rows of one teacher-forced run, as frames.
//!
//! The engine hands each step's logits row to a [`LogitsSink`] on a worker, and the sink must not wait. [`RowSink`]
//! copies the row, starts the read of the step's reference record with `AsyncIo::read_into` when the task has one, and
//! returns. The read's completion is a frame that offers the step's work to the `host` lake, one frame per step with
//! its own id hash, so the steps spread over the CPUs. Each frame stores its step's values in its own slots of the
//! run's shared state. Nothing waits and nothing locks: the state is read-only to the frames except for the atomic
//! slots, and the run ends when the last frame lets go of it. Its `Drop` hands the values (or the first fault) to the
//! hand-off the caller waits on, wherever that frame ran.

use std::fmt;
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_graph::{Lake, UnitGuard};
use euhedral_hashing::hasher_api;
use euhedral_inference_engine::generation::{FinishReason, GenerationOutcome, LogitsSink};
use euhedral_inference_engine::{EngineRuntime, Handoff};
use euhedral_io::{AsyncIo, IoDoneOwned, OwnedFile, RawFile};

use crate::config::ConfigError;

/// What a task is given for one step.
#[derive(Debug, Default)]
pub struct RowInput {
    pub step: usize,
    /// The forced token the row predicts.
    pub forced: u32,
    /// The row's BF16 logits, little endian.
    pub logits: Vec<u8>,
    /// The final hidden state's BF16 values, when the task wants it.
    pub hidden: Vec<u8>,
    /// The reference record of the step, when the task names a source.
    pub reference: Vec<u8>,
}

/// Where the reference records of a run are: a file, the offset of the first, and the bytes of each.
#[derive(Clone, Debug)]
pub struct ReferenceRows {
    pub file: Arc<OwnedFile>,
    pub first_offset: u64,
    pub row_bytes: usize,
}

/// The work of a run: what is computed from each step's row.
pub trait RowTask: Send + Sync + 'static {
    /// The float64 values a step produces.
    fn fields(&self) -> usize;

    /// Whether [`RowInput::hidden`] is wanted.
    fn wants_hidden(&self) -> bool {
        false
    }

    /// The reference records to read for each step, when the task needs them.
    fn reference(&self) -> Option<&ReferenceRows> {
        None
    }

    /// Computes the values of one step into `out` (`fields()` long). Runs on a worker, in a frame.
    fn compute(&self, input: &RowInput, out: &mut [f64]);
}

/// The values of a run, `steps * fields` of them, step after step.
pub type RunValues = Result<Vec<f64>, String>;

/// The shared state of a run: the task, the slots, and the way back to the caller.
struct Run {
    task: Arc<dyn RowTask>,
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
    fields: usize,
    slots: Vec<AtomicU64>,
    /// Which steps stored their values: a frame that panics or is dropped leaves its step unset.
    computed: Vec<AtomicBool>,
    fault: OnceLock<String>,
    done: Option<Arc<Handoff<RunValues>>>,
    /// Held until the run ends: the host lake takes frames only while a unit is admitted.
    _unit: UnitGuard,
}

impl Run {
    fn fail(&self, why: String) {
        let _ = self.fault.set(why);
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        if let Some(done) = self.done.take() {
            let missing = self
                .computed
                .iter()
                .position(|computed| !computed.load(Ordering::Acquire));
            let result = match (self.fault.take(), missing) {
                (Some(why), _) => Err(why),
                (None, Some(step)) => Err(format!(
                    "step {step} has no values: its row never arrived, or its computation panicked or was dropped"
                )),
                (None, None) => Ok(self
                    .slots
                    .iter()
                    .map(|slot| f64::from_bits(slot.load(Ordering::Relaxed)))
                    .collect()),
            };
            done.send(result);
        }
    }
}

/// What a frame of this module is: the kind goes into the id hash with the step.
#[derive(Clone, Copy)]
enum Kind {
    /// A reference record arrived.
    Arrived = 1,
    /// A step's values are computed.
    Computed = 2,
}

/// A frame that runs a closure once; its id hash is distinct per kind and step.
struct Work {
    header: FrameHeader,
    work: Option<Box<dyn FnOnce() + Send>>,
}

fn frame(kind: Kind, step: usize, work: impl FnOnce() + Send + 'static) -> Box<dyn Frame> {
    Box::new(Work {
        header: FrameHeader::new(hasher_api::mix(((kind as u64) << 32) | step as u64 | 1 << 63)),
        work: Some(Box::new(work)),
    })
}

impl Frame for Work {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(work) = self.work.take() {
            work();
        }
        Ok(())
    }
}

impl fmt::Debug for Work {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Work").finish_non_exhaustive()
    }
}

#[cfg(unix)]
fn raw_file(file: &OwnedFile) -> RawFile {
    use std::os::fd::AsRawFd;

    file.as_raw_fd()
}

#[cfg(windows)]
fn raw_file(file: &OwnedFile) -> RawFile {
    use std::os::windows::io::AsRawHandle;

    file.as_raw_handle()
}

/// Computes one step and stores its values.
fn compute(run: &Run, input: &RowInput) {
    let mut out = vec![0.0; run.fields];
    run.task.compute(input, &mut out);
    for (slot, value) in run.slots[input.step * run.fields..].iter().zip(out) {
        slot.store(value.to_bits(), Ordering::Relaxed);
    }
    run.computed[input.step].store(true, Ordering::Release);
}

/// What a run's engine side hands over: the outcome of the generation.
pub type EngineHandoff = Arc<Handoff<GenerationOutcome>>;

/// What a run's sink hands over: the values it collected.
pub type ValuesHandoff = Arc<Handoff<RunValues>>;

/// The sink of one run. The engine calls it on workers, one call at a time.
pub struct RowSink {
    run: Arc<Run>,
    hidden: Vec<u8>,
    steps: usize,
    finished: Option<Arc<Handoff<GenerationOutcome>>>,
}

impl fmt::Debug for RowSink {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("RowSink").finish_non_exhaustive()
    }
}

impl RowSink {
    /// A sink for a run of `steps` steps of `task`. `io` reads the task's reference records; the steps' work runs in
    /// `host`. The first hand-off is sent with the engine's outcome when its run ends, the second with the values once every
    /// step has been computed. Admits a unit on `host` for the run.
    pub fn new(
        io: Arc<AsyncIo>,
        host: Arc<Lake>,
        task: Arc<dyn RowTask>,
        steps: usize,
    ) -> Result<(RowSink, EngineHandoff, ValuesHandoff), ConfigError> {
        let unit = host
            .admit()
            .map_err(|error| ConfigError::Invalid(format!("the host lake would not take work: {error}")))?;
        let fields = task.fields();
        let (finished, done) = (Handoff::new(), Handoff::new());
        let (engine, values) = (Arc::clone(&finished), Arc::clone(&done));
        let run = Arc::new(Run {
            task,
            io,
            host,
            fields,
            slots: (0..steps * fields).map(|_| AtomicU64::new(0)).collect(),
            computed: (0..steps).map(|_| AtomicBool::new(false)).collect(),
            fault: OnceLock::new(),
            done: Some(done),
            _unit: unit,
        });
        Ok((
            RowSink {
                run,
                hidden: Vec::new(),
                steps,
                finished: Some(finished),
            },
            engine,
            values,
        ))
    }
}

/// Offers the computation of `input` to the host lake.
fn offer(run: Arc<Run>, input: RowInput) {
    let host = Arc::clone(&run.host);
    let step = input.step;
    let held = Arc::clone(&run);
    let work = frame(Kind::Computed, step, move || compute(&held, &input));
    if host.offer(work).is_err() {
        run.fail(format!("the host lake refused the work of step {step}"));
    }
}

impl Drop for RowSink {
    /// A sink dropped before the engine finished its run reports the run as failed, so the caller never waits for it.
    fn drop(&mut self) {
        if let Some(finished) = self.finished.take() {
            finished.send(GenerationOutcome {
                reason: FinishReason::Failed("the sink was dropped before the run finished".to_string()),
                prompt_tokens: 0,
                generated: 0,
                elapsed: Duration::ZERO,
            });
        }
    }
}

impl LogitsSink for RowSink {
    fn wants_hidden(&self) -> bool {
        self.run.task.wants_hidden()
    }

    fn hidden(&mut self, _step: usize, hidden: &[u8]) {
        self.hidden = hidden.to_vec();
    }

    fn row(&mut self, step: usize, forced: u32, logits: &[u8]) {
        if step >= self.steps {
            self.run
                .fail(format!("row {step} arrived in a run of {} steps", self.steps));
            return;
        }
        let mut input = RowInput {
            step,
            forced,
            logits: logits.to_vec(),
            hidden: mem::take(&mut self.hidden),
            reference: Vec::new(),
        };
        let run = Arc::clone(&self.run);
        let Some(source) = run.task.reference() else {
            offer(run, input);
            return;
        };
        let offset = source.first_offset + (step * source.row_bytes) as u64;
        let wanted = source.row_bytes;
        let arrived = Arc::clone(&run);
        let then = Box::new(move |done: IoDoneOwned<Vec<u8>>| {
            frame(Kind::Arrived, step, move || match (done.result, done.buf) {
                (Ok(read), Some(buffer)) if read == wanted => {
                    input.reference = buffer;
                    offer(Arc::clone(&arrived), input);
                }
                (Ok(read), _) => arrived.fail(format!(
                    "the reference record of step {step} has {read} of {wanted} bytes"
                )),
                (Err(error), _) => arrived.fail(format!("reading the reference record of step {step}: {error}")),
            })
        });
        // The file is held by the task, which the run holds until its last frame is gone.
        if let Err(error) = run.io.read_into(raw_file(&source.file), offset, vec![0; wanted], then) {
            run.fail(format!(
                "starting the read of the reference record of step {step}: {error}"
            ));
        }
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        if let Some(finished) = self.finished.take() {
            finished.send(outcome);
        }
    }
}

/// Starts a teacher-forced run of `task` on `runtime`. Returns the hand-off of the engine's outcome and the hand-off of the
/// values, which is set once the engine has finished and every step is computed. The caller waits for them.
pub fn start(
    runtime: &Arc<EngineRuntime>,
    prefix: Vec<u32>,
    forced: Vec<u32>,
    task: Arc<dyn RowTask>,
) -> Result<(EngineHandoff, ValuesHandoff), ConfigError> {
    let (sink, engine, values) =
        RowSink::new(Arc::clone(runtime.io()), Arc::clone(runtime.host()), task, forced.len())?;
    runtime
        .teacher_forced(prefix, forced, Box::new(sink))
        .map_err(|error| ConfigError::Invalid(format!("the run was refused: {error}")))?;
    Ok((engine, values))
}
