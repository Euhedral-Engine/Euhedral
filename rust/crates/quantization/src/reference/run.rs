//! Driving the chunks of a reference: up to `batch` chunks are forced through the model at once, `rows` forced tokens
//! to a pass; each chunk's rows are encoded and written from frames of the host lake as they arrive. The records sit
//! at fixed offsets, so chunks may end in any order, but they are concluded in chunk order (a [`Sequencer`]): the
//! count of complete chunks in the file advances only over the contiguous prefix of whole chunks, once their rows are
//! durable, so a resume after a crash never skips a chunk. Concluding a chunk starts the next one, so nothing waits.

use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use euhedral_data_structures::sync::ClaimCell;
use euhedral_graph::{Lake, Sequencer, Slot, Ticket, UnitGuard};
use euhedral_inference_engine::generation::{FinishReason, GenerationOutcome, LogitsSink};
use euhedral_inference_engine::{EngineRuntime, Handoff};
use euhedral_io::AsyncIo;

use super::partial::{Mark, Partial, RowFile};
use super::{ReferenceError, encode_record};
use crate::kld::Reference;
use crate::rows::{Kind, frame};

/// What starts the teacher-forced run of a chunk: the engine, or a stand-in in a test.
pub trait Forcer: Send + Sync + 'static {
    /// Starts the run: `prefix` is prefilled and the logits row that predicts each token of `forced` goes to `sink`,
    /// whose methods run on the lattice; up to `rows` forced tokens are fed in one pass. An error means the run did
    /// not start, and the sink is dropped.
    fn force(&self, prefix: Vec<u32>, forced: Vec<u32>, rows: usize, sink: Box<dyn LogitsSink>) -> Result<(), String>;
}

impl Forcer for EngineRuntime {
    fn force(&self, prefix: Vec<u32>, forced: Vec<u32>, rows: usize, sink: Box<dyn LogitsSink>) -> Result<(), String> {
        self.teacher_forced_rows(prefix, forced, rows, sink)
            .map_err(|error| error.to_string())
    }
}

/// The rows of a pass when a run does not say.
pub const DEFAULT_ROWS: usize = 8;
/// The chunks in flight when a run does not say.
pub const DEFAULT_BATCH: usize = 4;

/// How far a run goes, and how wide.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Stops once this many chunks are complete, leaving the partial file for a later run to carry on from.
    pub stop_after: Option<usize>,
    /// Forced tokens fed to the model in one pass.
    pub rows: usize,
    /// Chunks forced at once: the run holds at most this many chunks that are not yet concluded.
    pub batch: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            stop_after: None,
            rows: DEFAULT_ROWS,
            batch: DEFAULT_BATCH,
        }
    }
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
    let (rows, batch, stop_after) = (options.rows.max(1), options.batch.max(1), options.stop_after);
    let run = Arc::new_cyclic(|this: &Weak<Run>| {
        let concluding = Weak::clone(this);
        Run {
            host: Arc::clone(host),
            rows,
            batch,
            stop_after,
            order: Sequencer::ordered(move |ended| {
                if let Some(run) = concluding.upgrade() {
                    run.conclude(ended);
                }
            }),
            more: ClaimCell::new(None),
            partial: ClaimCell::new(None),
            mark: ClaimCell::new(None),
            done: ClaimCell::new(Some(Arc::clone(&done))),
            failure: ClaimCell::new(None),
            failed: AtomicBool::new(false),
            broken: AtomicBool::new(false),
            ended: AtomicBool::new(false),
            marking: AtomicBool::new(false),
            limit: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            concluded: AtomicUsize::new(0),
            prefix: AtomicUsize::new(0),
            recorded: AtomicUsize::new(0),
            chunks: AtomicUsize::new(0),
        }
    });
    Partial::open(
        io,
        host,
        out,
        reference,
        Box::new(move |opened| match opened {
            Ok(partial) => run.opened(partial, forcer),
            Err(error) => run.end(Err(error)),
        }),
    );
    done
}

/// What a chunk reports when it ends: which, and whether it is whole.
type Ended = (usize, Result<(), ReferenceError>);

/// What starting a chunk needs besides its position.
struct More {
    forcer: Arc<dyn Forcer>,
    rows: RowFile,
    layout: Arc<Reference>,
}

/// The run: shared by the chunks in flight and by every write that records a count. It always ends: a run that is
/// dropped before it ends reports that.
///
/// Who touches what. The chunks are concluded one at a time and in order by the sequencer, so `more` is touched only
/// there (after the first chunks start). The count is recorded by one write at a time, and whoever holds `marking`
/// owns `mark`. The run ends once, on whichever thread finds everything settled, and that thread owns `partial` and
/// `failure`.
struct Run {
    host: Arc<Lake>,
    rows: usize,
    batch: usize,
    stop_after: Option<usize>,
    order: Arc<Sequencer<Arc<Ticket<Ended>>>>,
    /// What the chunks that start after the first ones need; let go of when none will, before the run reports its end,
    /// so a caller that then stops the engine holds its last handle.
    more: ClaimCell<Option<More>>,
    partial: ClaimCell<Option<Partial>>,
    /// The way to record a count, while no write of one is under way.
    mark: ClaimCell<Option<Mark>>,
    done: ClaimCell<Option<Arc<Handoff<Outcome>>>>,
    failure: ClaimCell<Option<ReferenceError>>,
    /// A chunk or a record failed: nothing more starts.
    failed: AtomicBool,
    /// Recording a count failed: the prefix is not recorded any further.
    broken: AtomicBool,
    ended: AtomicBool,
    /// A count is being made durable.
    marking: AtomicBool,
    /// The chunks this run goes up to: all of them, or `stop_after`.
    limit: AtomicUsize,
    /// The chunks before this one have started.
    next: AtomicUsize,
    /// The chunks before this one have been concluded.
    concluded: AtomicUsize,
    /// The chunks before this one are whole.
    prefix: AtomicUsize,
    /// The chunks before this one are recorded complete in the file.
    recorded: AtomicUsize,
    chunks: AtomicUsize,
}

impl Drop for Run {
    fn drop(&mut self) {
        if let Some(done) = self.done.get_mut().take() {
            done.send(Err(ReferenceError::Engine(
                "the run was dropped before it ended".to_owned(),
            )));
        }
    }
}

impl Run {
    fn end(&self, result: Outcome) {
        let done = self.done.try_claim().and_then(|mut done| done.take());
        if let Some(done) = done {
            done.send(result);
        }
    }

    fn fail(&self, error: ReferenceError) {
        if let Some(mut failure) = self.failure.try_claim() {
            failure.get_or_insert(error);
        }
        self.failed.store(true, Ordering::SeqCst);
    }

    /// The first chunks start: those after the chunks already complete, up to `batch` of them.
    fn opened(self: &Arc<Self>, partial: Partial, forcer: Arc<dyn Forcer>) {
        let chunks = partial.layout().chunks();
        let completed = partial.completed();
        let limit = self.stop_after.map_or(chunks, |stop| stop.min(chunks));
        self.chunks.store(chunks, Ordering::SeqCst);
        self.limit.store(limit, Ordering::SeqCst);
        for counter in [&self.next, &self.concluded, &self.prefix, &self.recorded] {
            counter.store(completed, Ordering::SeqCst);
        }
        let first = completed..limit.min(completed + self.batch);
        // Everything the first chunks and the later ones need is taken before any chunk starts, and the slots are
        // registered, in chunk order, before any can end.
        let layout = Arc::clone(partial.layout());
        let starts: Vec<_> = first
            .clone()
            .map(|chunk| (chunk, self.order.register(), partial.rows(), Arc::clone(&forcer)))
            .collect();
        self.next.store(first.end.max(completed), Ordering::SeqCst);
        if let Some(mut mark) = self.mark.try_claim() {
            *mark = Some(partial.mark());
        }
        if first.end < limit
            && let Some(mut more) = self.more.try_claim()
        {
            *more = Some(More {
                forcer,
                rows: partial.rows(),
                layout: Arc::clone(&layout),
            });
        }
        if let Some(mut held) = self.partial.try_claim() {
            *held = Some(partial);
        }
        if starts.is_empty() {
            return self.settle();
        }
        for (chunk, slot, rows, forcer) in starts {
            self.start(chunk, slot, &layout, rows, forcer);
        }
    }

    /// Forces chunk `chunk`.
    fn start(
        self: &Arc<Self>,
        chunk: usize,
        slot: Slot<Ended>,
        layout: &Arc<Reference>,
        rows: RowFile,
        forcer: Arc<dyn Forcer>,
    ) {
        let unit = match self.host.admit() {
            Ok(unit) => unit,
            Err(error) => {
                return slot.complete((
                    chunk,
                    Err(ReferenceError::Engine(format!(
                        "the host lake would not take work: {error}"
                    ))),
                ));
            }
        };
        let (prefix, forced) = (layout.prefix(chunk).to_vec(), layout.forced(chunk).to_vec());
        let run = Arc::new(ChunkRun {
            host: Arc::clone(&self.host),
            rows: Some(rows),
            chunk,
            forced: forced.clone(),
            written: (0..forced.len()).map(|_| AtomicBool::new(false)).collect(),
            fault: OnceLock::new(),
            outcome: OnceLock::new(),
            slot: Some(slot),
            _run: Arc::clone(self),
            _unit: unit,
        });
        let sink = ReferenceSink {
            run: Arc::clone(&run),
            row_bytes: 2 * layout.vocabulary,
        };
        let started = forcer.force(prefix, forced, self.rows, Box::new(sink));
        drop(forcer);
        if let Err(why) = started {
            run.fail(format!("chunk {chunk} did not start: {why}"));
        }
    }

    /// A chunk is concluded, after every chunk before it: it moves the prefix when it is whole and nothing failed
    /// before it, and the window moves on by one chunk.
    fn conclude(self: &Arc<Self>, (chunk, verdict): Ended) {
        match verdict {
            Ok(()) if !self.failed.load(Ordering::SeqCst) => self.prefix.store(chunk + 1, Ordering::SeqCst),
            Ok(()) => {}
            Err(error) => self.fail(error),
        }
        let started = self.start_next();
        self.concluded.fetch_add(1, Ordering::SeqCst);
        self.record();
        if !started {
            self.settle();
        }
    }

    /// Starts the chunk the window now has room for, if the run is to go on. Returns whether one started.
    fn start_next(self: &Arc<Self>) -> bool {
        let Some(mut more) = self.more.try_claim() else {
            return false;
        };
        let next = self.next.load(Ordering::SeqCst);
        if self.failed.load(Ordering::SeqCst) || next >= self.limit.load(Ordering::SeqCst) {
            drop(more.take());
            return false;
        }
        let Some(held) = more.as_ref() else {
            return false;
        };
        let (forcer, rows, layout) = (Arc::clone(&held.forcer), held.rows.clone(), Arc::clone(&held.layout));
        let slot = self.order.register();
        self.next.store(next + 1, Ordering::SeqCst);
        if next + 1 >= self.limit.load(Ordering::SeqCst) {
            drop(more.take());
        }
        drop(more);
        self.start(next, slot, &layout, rows, forcer);
        true
    }

    /// Records the whole prefix: its rows are made durable, then the count is. One write at a time; each ends by
    /// looking again, so a prefix that moved meanwhile is recorded next.
    fn record(self: &Arc<Self>) {
        loop {
            if self.broken.load(Ordering::SeqCst) || self.marking.swap(true, Ordering::SeqCst) {
                return;
            }
            let target = self.prefix.load(Ordering::SeqCst);
            if target > self.recorded.load(Ordering::SeqCst) {
                let Some(mark) = self.mark.try_claim().and_then(|mut mark| mark.take()) else {
                    self.marking.store(false, Ordering::SeqCst);
                    return;
                };
                let run = Arc::clone(self);
                let spare = mark.clone();
                mark.complete(
                    target,
                    Box::new(move |recorded| {
                        match recorded {
                            Ok(()) => run.recorded.store(target, Ordering::SeqCst),
                            Err(error) => {
                                run.broken.store(true, Ordering::SeqCst);
                                run.fail(error);
                            }
                        }
                        if let Some(mut mark) = run.mark.try_claim() {
                            *mark = Some(spare);
                        }
                        run.marking.store(false, Ordering::SeqCst);
                        run.record();
                        run.settle();
                    }),
                );
                return;
            }
            self.marking.store(false, Ordering::SeqCst);
            // A prefix published between the look and the release is not recorded by anyone else.
            if self.prefix.load(Ordering::SeqCst) <= self.recorded.load(Ordering::SeqCst) {
                return;
            }
        }
    }

    /// Reads the state `settled` decides on. The order is the contract; every load is SeqCst.
    /// - `concluded` before `next`: both only grow and `concluded <= next`, so their equality is one instant with
    ///   no chunk in flight.
    /// - `broken` before `marking`: a count write sets `broken` and calls `fail` while `marking` is held, and
    ///   releases it after, so a `marking` read of false after `broken` was seen true follows that `fail`.
    /// - `failed` after all of them: every chunk `fail` precedes its `concluded` increment, so a guard that passed
    ///   on equal counters and a released `marking` has the `failed` read after it see every `fail`. A `failed`
    ///   read before the guard can miss one that lands while the guard is still closed.
    fn read_state(&self) -> State {
        let concluded = self.concluded.load(Ordering::SeqCst);
        let next = self.next.load(Ordering::SeqCst);
        let recorded = self.recorded.load(Ordering::SeqCst);
        let prefix = self.prefix.load(Ordering::SeqCst);
        let broken = self.broken.load(Ordering::SeqCst);
        let marking = self.marking.load(Ordering::SeqCst);
        State {
            next,
            concluded,
            marking,
            recorded,
            prefix,
            broken,
            failed: self.failed.load(Ordering::SeqCst),
            limit: self.limit.load(Ordering::SeqCst),
        }
    }

    /// Ends the run when everything is settled: every chunk that started has been concluded, nothing more starts, and
    /// the count is recorded as far as it can be.
    fn settle(self: &Arc<Self>) {
        let state = self.read_state();
        let completed = match settled(state) {
            Settled::Pending => return,
            Settled::Failed => None,
            Settled::Done { completed } => Some(completed),
        };
        if self.ended.swap(true, Ordering::SeqCst) {
            return;
        }
        drop(self.more.try_claim().and_then(|mut more| more.take()));
        drop(self.mark.try_claim().and_then(|mut mark| mark.take()));
        let partial = self.partial.try_claim().and_then(|mut partial| partial.take());
        let Some(completed) = completed else {
            drop(partial);
            let error = self.failure.try_claim().and_then(|mut failure| failure.take());
            return self.end(Err(
                error.unwrap_or_else(|| ReferenceError::Engine("the run failed".to_owned()))
            ));
        };
        let Some(mut partial) = partial else {
            return self.end(Err(ReferenceError::Engine("the partial file is gone".to_owned())));
        };
        partial.set_completed(completed);
        if completed == self.chunks.load(Ordering::SeqCst) {
            let run = Arc::clone(self);
            partial.finish(Box::new(move |sealed| {
                run.end(sealed.map(|()| Written {
                    completed,
                    finished: true,
                }));
            }));
        } else {
            drop(partial);
            self.end(Ok(Written {
                completed,
                finished: false,
            }));
        }
    }
}

/// What `settle` reads, in the order `Run::read_state` reads it.
#[derive(Clone, Copy, Debug)]
struct State {
    next: usize,
    concluded: usize,
    marking: bool,
    recorded: usize,
    prefix: usize,
    broken: bool,
    failed: bool,
    limit: usize,
}

/// What a settle decides.
#[derive(Debug, PartialEq, Eq)]
enum Settled {
    /// Something is still running or to be recorded: a later settle decides.
    Pending,
    /// The run ends with the recorded failure; the durable count covers the chunks before it.
    Failed,
    /// The run ends without a failure, with `completed` chunks durable.
    Done { completed: usize },
}

/// Decides from one reading. The guard is: every started chunk concluded, no count write in flight, and the count
/// caught up with the prefix (or given up on). Only past the guard do `failed` and `limit` decide, so the reading
/// must have taken them after the rest (see `Run::read_state`).
fn settled(state: State) -> Settled {
    let State {
        next,
        concluded,
        marking,
        recorded,
        prefix,
        broken,
        failed,
        limit,
    } = state;
    if concluded != next || marking || (recorded != prefix && !broken) {
        return Settled::Pending;
    }
    if failed {
        return Settled::Failed;
    }
    if next >= limit {
        return Settled::Done { completed: recorded };
    }
    Settled::Pending
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
    slot: Option<Slot<Ended>>,
    /// Keeps the run alive while a chunk is in flight.
    _run: Arc<Run>,
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
        if let Some(slot) = self.slot.take() {
            let verdict = self.verdict();
            slot.complete((self.chunk, verdict));
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

#[cfg(test)]
mod tests {
    use super::{Settled, State, settled};

    /// Three chunks all started and concluded, the count recorded over the prefix of `prefix` chunks.
    fn quiet(failed: bool, prefix: usize) -> State {
        State {
            next: 3,
            concluded: 3,
            marking: false,
            recorded: prefix,
            prefix,
            broken: false,
            failed,
            limit: 3,
        }
    }

    #[test]
    fn a_failure_seen_after_the_guard_passes_ends_the_run_with_the_error() {
        // The count-write continuation settles while chunk 1 is still open; chunks 1 and 2 then conclude and the
        // failure lands. The reading after the guard carries it, whatever the continuation read earlier.
        assert_eq!(settled(quiet(true, 1)), Settled::Failed);
        assert_eq!(settled(quiet(true, 0)), Settled::Failed);
    }

    #[test]
    fn a_run_without_failure_is_done_with_the_recorded_count() {
        assert_eq!(settled(quiet(false, 3)), Settled::Done { completed: 3 });
        let capped = State {
            limit: 2,
            next: 2,
            concluded: 2,
            ..quiet(false, 2)
        };
        assert_eq!(settled(capped), Settled::Done { completed: 2 });
    }

    #[test]
    fn a_failed_count_write_ends_the_run_with_its_error_though_the_count_lags() {
        let broken = State {
            broken: true,
            recorded: 1,
            prefix: 2,
            ..quiet(true, 2)
        };
        assert_eq!(settled(broken), Settled::Failed);
    }

    #[test]
    fn a_run_waits_while_a_chunk_is_open_a_write_is_in_flight_or_the_count_lags() {
        let open = State {
            concluded: 2,
            ..quiet(true, 1)
        };
        let writing = State {
            marking: true,
            ..quiet(true, 1)
        };
        let lagging = State {
            recorded: 0,
            ..quiet(true, 1)
        };
        for state in [open, writing, lagging] {
            assert_eq!(settled(state), Settled::Pending);
        }
    }

    #[test]
    fn a_run_without_failure_waits_while_more_chunks_are_to_start() {
        let more = State {
            limit: 4,
            ..quiet(false, 3)
        };
        assert_eq!(settled(more), Settled::Pending);
    }
}
