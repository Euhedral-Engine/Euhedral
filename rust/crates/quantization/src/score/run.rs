//! A scoring run: the chunks of a reference forced through the engine a few at a time, each row scored against its
//! reference record, the values of every chunk handed over once all have ended.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use euhedral_inference_engine::generation::{GenerationOutcome, LogitsSink};
use euhedral_inference_engine::{EngineRuntime, Handoff};
use euhedral_io::{AsyncIo, IoDoneOwned, OwnedFile};

use super::chunk::ChunkScore;
use super::device::ScoreRow;
use super::scorer::{Job, Scorer};
use crate::kld::Reference;
use crate::rows::{Kind, frame, raw_file};

/// What the engine is asked to force: the part of the engine a run uses.
pub trait RowForcer: Send + Sync + 'static {
    /// Starts a teacher-forced run of `forced` after `prefix`, `rows` rows to a verification, the rows going to `sink`.
    fn force_rows(
        &self,
        prefix: Vec<u32>,
        forced: Vec<u32>,
        rows: usize,
        sink: Box<dyn LogitsSink>,
    ) -> Result<(), String>;
}

impl RowForcer for EngineRuntime {
    fn force_rows(
        &self,
        prefix: Vec<u32>,
        forced: Vec<u32>,
        rows: usize,
        sink: Box<dyn LogitsSink>,
    ) -> Result<(), String> {
        self.teacher_forced_rows(prefix, forced, rows, sink)
            .map_err(|error| error.to_string())
    }
}

/// What a run hands over: the values of every chunk, in chunk order, or why it stopped.
pub type Scored = Result<Vec<Vec<f64>>, String>;

/// Where a run reads from and sends to.
pub struct Source {
    pub forcer: Arc<dyn RowForcer>,
    pub io: Arc<AsyncIo>,
    pub scorer: Arc<Scorer>,
    pub reference: Reference,
    /// The reference file, open for its records.
    pub records: OwnedFile,
    /// The rows of a verification.
    pub rows: usize,
}

impl Debug for Source {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Source").finish_non_exhaustive()
    }
}

struct Drive {
    /// Let go before the result is sent, so that nothing of the run holds the engine once the caller wakes.
    forcer: Option<Arc<dyn RowForcer>>,
    io: Arc<AsyncIo>,
    scorer: Arc<Scorer>,
    reference: Reference,
    /// The reference file, open for its records.
    records: OwnedFile,
    rows: usize,
    next: AtomicUsize,
    results: Vec<OnceLock<Vec<f64>>>,
    failure: OnceLock<String>,
    done: Arc<Handoff<Scored>>,
}

/// Scores every chunk of `source.reference`, `batch` chunks at a time. Returns at once; the hand-off is set when the
/// run has ended: every chunk is scored, or one failed and the chunks in flight have ended. The run ends when the last
/// of its frames lets go of it, wherever that is.
pub fn score(source: Source, batch: usize) -> Arc<Handoff<Scored>> {
    let done = Handoff::new();
    let Source {
        forcer,
        io,
        scorer,
        reference,
        records,
        rows,
    } = source;
    let chunks = reference.chunks();
    let drive = Arc::new(Drive {
        forcer: Some(forcer),
        io,
        scorer,
        results: (0..chunks).map(|_| OnceLock::new()).collect(),
        reference,
        records,
        rows,
        next: AtomicUsize::new(0),
        failure: OnceLock::new(),
        done: Arc::clone(&done),
    });
    for _ in 0..batch.clamp(1, chunks.max(1)) {
        drive.start_next();
    }
    done
}

impl Drive {
    fn start_next(self: &Arc<Self>) {
        if self.failure.get().is_some() {
            return;
        }
        let index = self.next.fetch_add(1, Ordering::SeqCst);
        let reference = &self.reference;
        let Some(forcer) = self.forcer.as_ref().filter(|_| index < reference.chunks()) else {
            return;
        };
        let drive = Arc::clone(self);
        let chunk = ChunkScore::new(reference.records(), Arc::new(move |result| drive.ended(index, result)));
        let sink = ScoreSink {
            chunk: Arc::clone(&chunk),
            drive: Arc::clone(self),
            index,
        };
        let started = forcer.force_rows(
            reference.prefix(index).to_vec(),
            reference.forced(index).to_vec(),
            self.rows,
            Box::new(sink),
        );
        if let Err(why) = started {
            // The sink is gone with the refusal; this is the last holder, so the verdict names why.
            chunk.fail(format!("chunk {index} was refused: {why}"));
        }
    }

    fn ended(self: &Arc<Self>, index: usize, result: Result<Vec<f64>, String>) {
        match result {
            Ok(values) => {
                let _ = self.results[index].set(values);
            }
            Err(why) => {
                let _ = self.failure.set(why);
            }
        }
        self.start_next();
    }
}

impl Drop for Drive {
    fn drop(&mut self) {
        drop(self.forcer.take());
        let scored = match self.failure.take() {
            Some(why) => Err(why),
            None => self
                .results
                .iter_mut()
                .enumerate()
                .map(|(index, slot)| slot.take().ok_or_else(|| format!("chunk {index} has no values")))
                .collect(),
        };
        self.done.send(scored);
    }
}

/// The sink of one chunk: it copies each row, reads the row's reference record and offers the pair to the scorer. The
/// engine calls it on workers, one call at a time.
struct ScoreSink {
    chunk: Arc<ChunkScore>,
    drive: Arc<Drive>,
    index: usize,
}

impl LogitsSink for ScoreSink {
    fn row(&mut self, step: usize, forced: u32, logits: &[u8]) {
        let drive = &self.drive;
        let (steps, wanted) = (self.chunk.steps(), drive.reference.row_bytes());
        if step >= steps || logits.len() != 2 * drive.reference.vocabulary {
            self.chunk.fail(format!(
                "row {step} of {} bytes arrived in a run of {steps} steps",
                logits.len()
            ));
            return;
        }
        let logits = logits.to_vec();
        let offset = drive.reference.row_offset(self.index, step);
        let (chunk, drive) = (Arc::clone(&self.chunk), Arc::clone(&self.drive));
        let then = Box::new(move |done: IoDoneOwned<Vec<u8>>| {
            frame(Kind::Arrived, step, move || match (done.result, done.buf) {
                (Ok(read), Some(record)) if read == wanted => drive.scorer.submit(Job {
                    chunk,
                    step,
                    row: ScoreRow {
                        logits,
                        record,
                        target: forced,
                    },
                }),
                (Ok(read), _) => chunk.fail(format!(
                    "the reference record of step {step} has {read} of {wanted} bytes"
                )),
                (Err(error), _) => chunk.fail(format!("reading the reference record of step {step}: {error}")),
            })
        });
        // The file is held by the drive, which the read's frame holds until it has run.
        let read = self
            .drive
            .io
            .read_into(raw_file(&self.drive.records), offset, vec![0; wanted], then);
        if let Err(error) = read {
            self.chunk.fail(format!(
                "starting the read of the reference record of step {step}: {error}"
            ));
        }
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        self.chunk.finished(outcome);
    }
}
