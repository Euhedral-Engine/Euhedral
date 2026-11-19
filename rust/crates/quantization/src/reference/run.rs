//! Driving the chunks of a reference: each chunk is forced through the model, its rows are encoded and written from
//! frames of the host lake as they arrive, and when all of them are written and durable the chunk is recorded
//! complete. The next chunk starts from the continuation of that record, so nothing waits.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use euhedral_graph::{Lake, UnitGuard};
use euhedral_inference_engine::generation::{FinishReason, GenerationOutcome, LogitsSink};
use euhedral_inference_engine::{EngineRuntime, Handoff};
use euhedral_io::AsyncIo;

use super::partial::{Partial, RowFile};
use super::{ReferenceError, encode_record};
use crate::kld::Reference;
use crate::rows::{Kind, frame};

/// What starts the teacher-forced run of a chunk: the engine, or a stand-in in a test.
pub trait Forcer: Send + Sync + 'static {
    /// Starts the run: `prefix` is prefilled and the logits row that predicts each token of `forced` goes to `sink`,
    /// whose methods run on the lattice. An error means the run did not start, and the sink is dropped.
    fn force(&self, prefix: Vec<u32>, forced: Vec<u32>, sink: Box<dyn LogitsSink>) -> Result<(), String>;
}

impl Forcer for EngineRuntime {
    fn force(&self, prefix: Vec<u32>, forced: Vec<u32>, sink: Box<dyn LogitsSink>) -> Result<(), String> {
        self.teacher_forced(prefix, forced, sink)
            .map_err(|error| error.to_string())
    }
}

/// How far a run goes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Stops once this many chunks are complete, leaving the partial file for a later run to carry on from.
    pub stop_after: Option<usize>,
}

/// How a run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Written {
    /// The chunks that are complete.
    pub completed: usize,
    /// Whether the reference is finished: the partial file was renamed to the output.
    pub finished: bool,
}

type Outcome = Result<Written, ReferenceError>;

/// Writes `reference` (its tokens a chunk each, its context and vocabulary) to `out`, carrying on from the partial file a stopped run left.
/// Returns at once; the hand-off is set when the run ends, with how far it got or why it stopped. The caller waits for
/// it, on a thread outside the lattice.
pub fn write(
    io: &Arc<AsyncIo>,
    host: &Arc<Lake>,
    forcer: Arc<dyn Forcer>,
    out: &Path,
    reference: Reference,
    options: Options,
) -> Arc<Handoff<Outcome>> {
    let done = Handoff::new();
    let writing = Writing {
        host: Arc::clone(host),
        forcer: Some(forcer),
        partial: None,
        options,
        done: Some(Arc::clone(&done)),
    };
    Partial::open(
        io,
        host,
        out,
        reference,
        Box::new(move |opened| match opened {
            Ok(partial) => writing.opened(partial),
            Err(error) => writing.end(Err(error)),
        }),
    );
    done
}

/// The run between chunks. It always ends: a run that is dropped before it ends reports that.
struct Writing {
    host: Arc<Lake>,
    /// Let go of before the run reports its end, so a caller that then stops the engine holds its last handle.
    forcer: Option<Arc<dyn Forcer>>,
    partial: Option<Partial>,
    options: Options,
    done: Option<Arc<Handoff<Outcome>>>,
}

impl Drop for Writing {
    fn drop(&mut self) {
        self.forcer = None;
        if let Some(done) = self.done.take() {
            done.send(Err(ReferenceError::Engine(
                "the run was dropped before it ended".to_owned(),
            )));
        }
    }
}

impl Writing {
    fn end(mut self, result: Outcome) {
        self.forcer = None;
        if let Some(done) = self.done.take() {
            done.send(result);
        }
    }

    fn opened(mut self, partial: Partial) {
        self.partial = Some(partial);
        self.next();
    }

    fn gone() -> ReferenceError {
        ReferenceError::Engine("the partial file is gone".to_owned())
    }

    fn next(self) {
        let Some((completed, chunks)) = self.partial.as_ref().map(|p| (p.completed(), p.layout().chunks())) else {
            return self.end(Err(Self::gone()));
        };
        if completed == chunks {
            self.seal();
        } else if self.options.stop_after.is_some_and(|stop| completed >= stop) {
            self.end(Ok(Written {
                completed,
                finished: false,
            }));
        } else {
            self.start(completed);
        }
    }

    fn seal(mut self) {
        let Some(partial) = self.partial.take() else {
            return self.end(Err(Self::gone()));
        };
        let completed = partial.completed();
        partial.finish(Box::new(move |sealed| {
            self.end(sealed.map(|()| Written {
                completed,
                finished: true,
            }));
        }));
    }

    /// Forces chunk `chunk`.
    fn start(self, chunk: usize) {
        let unit = match self.host.admit() {
            Ok(unit) => unit,
            Err(error) => {
                return self.end(Err(ReferenceError::Engine(format!(
                    "the host lake would not take work: {error}"
                ))));
            }
        };
        let Some((layout, rows)) = self.partial.as_ref().map(|p| (Arc::clone(p.layout()), p.rows())) else {
            return self.end(Err(Self::gone()));
        };
        let (prefix, forced) = (layout.prefix(chunk).to_vec(), layout.forced(chunk).to_vec());
        let Some(forcer) = self.forcer.clone() else {
            return self.end(Err(ReferenceError::Engine("the engine is gone".to_owned())));
        };
        let run = Arc::new(ChunkRun {
            host: Arc::clone(&self.host),
            rows: Some(rows),
            chunk,
            forced: forced.clone(),
            written: (0..forced.len()).map(|_| AtomicBool::new(false)).collect(),
            fault: OnceLock::new(),
            outcome: OnceLock::new(),
            writing: Some(self),
            _unit: unit,
        });
        let sink = ReferenceSink {
            run: Arc::clone(&run),
            row_bytes: 2 * layout.vocabulary,
        };
        let started = forcer.force(prefix, forced, Box::new(sink));
        drop(forcer);
        if let Err(why) = started {
            run.fail(format!("chunk {chunk} did not start: {why}"));
        }
    }

    /// A chunk ended: record it complete when it is whole, and go on to the next.
    fn chunk_ended(mut self, chunk: usize, verdict: Result<(), ReferenceError>) {
        if let Err(error) = verdict {
            return self.end(Err(error));
        }
        let Some(mark) = self.partial.as_ref().map(Partial::mark) else {
            return self.end(Err(Self::gone()));
        };
        mark.complete(
            chunk + 1,
            Box::new(move |recorded| match recorded {
                Ok(()) => {
                    if let Some(partial) = self.partial.as_mut() {
                        partial.set_completed(chunk + 1);
                    }
                    self.next();
                }
                Err(error) => self.end(Err(error)),
            }),
        );
    }
}

/// The shared state of one chunk's run. The chunk ends when the last holder lets go: the sink, and every frame that
/// encodes a row or waits for its write.
struct ChunkRun {
    host: Arc<Lake>,
    /// Let go of first when the chunk ends, so the file has no other handle than the run's own.
    rows: Option<RowFile>,
    chunk: usize,
    forced: Vec<u32>,
    written: Vec<AtomicBool>,
    fault: OnceLock<String>,
    outcome: OnceLock<GenerationOutcome>,
    writing: Option<Writing>,
    /// Held until the chunk ends: the host lake takes frames only while a unit is admitted.
    _unit: UnitGuard,
}

impl ChunkRun {
    fn fail(&self, why: String) {
        let _ = self.fault.set(why);
    }

    fn verdict(&self) -> Result<(), ReferenceError> {
        if let Some(why) = self.fault.get() {
            return Err(ReferenceError::Engine(why.clone()));
        }
        let steps = self.written.len();
        let Some(outcome) = self.outcome.get() else {
            return Err(ReferenceError::Engine(
                "the engine ended the chunk without an outcome".to_owned(),
            ));
        };
        if outcome.reason != FinishReason::Length || outcome.generated != steps {
            return Err(ReferenceError::Engine(format!(
                "the run ended with {:?} after {} of {steps} steps",
                outcome.reason, outcome.generated
            )));
        }
        match self.written.iter().position(|written| !written.load(Ordering::Acquire)) {
            Some(step) => Err(ReferenceError::Engine(format!(
                "record {step} of chunk {} was not written",
                self.chunk
            ))),
            None => Ok(()),
        }
    }
}

impl Drop for ChunkRun {
    fn drop(&mut self) {
        drop(self.rows.take());
        if let Some(writing) = self.writing.take() {
            let verdict = self.verdict();
            writing.chunk_ended(self.chunk, verdict);
        }
    }
}

/// The engine's side of a chunk: each row it hands over is encoded and written from a frame of the host lake.
struct ReferenceSink {
    run: Arc<ChunkRun>,
    row_bytes: usize,
}

impl LogitsSink for ReferenceSink {
    fn row(&mut self, step: usize, forced: u32, logits: &[u8]) {
        let run = &self.run;
        if step >= run.written.len() || run.forced[step] != forced {
            run.fail(format!(
                "row {step} of chunk {} does not follow the forced tokens",
                run.chunk
            ));
            return;
        }
        if logits.len() != self.row_bytes {
            run.fail(format!(
                "row {step} has {} bytes, a row of the vocabulary has {}",
                logits.len(),
                self.row_bytes
            ));
            return;
        }
        let logits = logits.to_vec();
        let held = Arc::clone(run);
        let work = frame(Kind::Computed, step, move || {
            let record = encode_record(&logits);
            let Some(rows) = held.rows.as_ref() else {
                return;
            };
            let done = Arc::clone(&held);
            rows.write(
                held.chunk,
                step,
                record,
                Box::new(move |written| match written {
                    Ok(()) => done.written[step].store(true, Ordering::Release),
                    Err(error) => done.fail(error.to_string()),
                }),
            );
        });
        if run.host.offer(work).is_err() {
            run.fail(format!("the host lake refused the encoding of row {step}"));
        }
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        let _ = self.run.outcome.set(outcome);
    }
}
