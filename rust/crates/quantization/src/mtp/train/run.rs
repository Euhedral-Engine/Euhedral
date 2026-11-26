//! The run as frames: each window's hidden rows are read, staged and run on the lane, and the frame that retires the
//! work trains the next. Nothing waits: a read, a window, an update or a save each hands the run, with the lane, to the
//! call that ends it, and that call goes on.
//!
//! Per sequence, its windows are trained in order; a trained window may owe an update. At a sequence's end an
//! evaluation over the held-out sequences runs if one is due; its verdict saves the best parameters when they improved
//! and the state to resume from. The run ends after the last epoch's evaluation, or after the state save that makes
//! `stop_after_saves`.
//!
//! The run and its lane move together from call to call, and every way out ends in [`end`], which answers the thread
//! that started the run with the lane, so that the thread closes it. A call that cannot start (a refused window, a
//! file that will not open) hands the run and the lane back through [`Claim`] to the same `end`.

use std::collections::VecDeque;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::mem::replace;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use euhedral_graph::Lake;
use euhedral_inference_engine::Handoff;
use euhedral_io::{AsyncIo, OwnedFile};

use super::data::Data;
use super::error::RunError;
use super::lane::{Done, Job, Lane, Outcome};
use super::options::{Host, Log, Options};
use super::progress::{Evaluation, Identity, Next, Progress, Trained, Update};
use super::settings::{Settings, Window, windows};
use super::stage::stage;
use super::state::Placed;
use super::wait::Reply;
use super::window::Mode;
use crate::checkpoint::io::{Next as ReadNext, read_exact};
use crate::resume::{HEADER, Header, ResumeFile, Then};
use crate::step::idle;

/// What a run hands back to the thread that started it.
pub struct Ended<L> {
    /// The lane, free: the caller closes it.
    pub lane: L,
    /// How the run ended.
    pub outcome: Result<Summary, RunError>,
}

impl<L> Debug for Ended<L> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("Ended")
            .field("outcome", &self.outcome)
            .finish_non_exhaustive()
    }
}

/// How a run that did not fail ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    /// The state saves it made.
    pub saves: u32,
    /// It was stopped at a saved state, not finished.
    pub stopped: bool,
    /// The best held-out accepted drafts per lineage.
    pub best: f64,
}

/// How the run begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// From the checkpoint: the held-out sequences are evaluated before any training.
    Initial,
    /// From a saved state, which already had that evaluation.
    Continue,
    /// The saved state is of a run that ended: there is nothing to do.
    Complete,
}

/// A run ready to go: what the loader hands to [`start`].
pub struct Ready<L> {
    pub lane: L,
    pub data: Data,
    /// The width of the hidden state.
    pub width: u32,
    pub hidden: Arc<OwnedFile>,
    /// A token's row in the draft head, -1 for a token it does not hold.
    pub row_of_token: Vec<i32>,
    pub identity: Identity,
    pub progress: Progress,
    pub start: Start,
    /// Where the state's and the best parameters' pieces lie on the lane.
    pub state_pieces: Vec<Placed>,
    pub best_pieces: Vec<Placed>,
}

impl<L> Debug for Ready<L> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Ready").finish_non_exhaustive()
    }
}

/// What the windows being run are for.
enum Phase {
    Training,
    Evaluating(Box<Evaluating>),
}

/// An evaluation under way.
struct Evaluating {
    tally: Evaluation,
    /// The held-out sequences still to run.
    queue: VecDeque<usize>,
    /// The one before any training.
    initial: bool,
    /// The one after the last epoch.
    last: bool,
}

/// What the run owns besides the lane.
struct Run<L: Lane> {
    io: Arc<AsyncIo>,
    lake: Arc<Lake>,
    data: Data,
    settings: Settings,
    width: u32,
    hidden: Arc<OwnedFile>,
    row_of_token: Vec<i32>,
    identity: Identity,
    progress: Progress,
    out: PathBuf,
    state_pieces: Vec<Placed>,
    best_pieces: Vec<Placed>,
    windows: VecDeque<Window>,
    sequence: usize,
    phase: Phase,
    saves: u32,
    stop_after_saves: u32,
    fail_in_save: u32,
    log: Log,
    reply: Arc<Reply<Ended<L>>>,
}

type Step<L> = Box<Run<L>>;

/// The run and its lane while a call that may not start holds the continuation that would carry them on. Whichever
/// of the continuation and the caller's failure path comes first takes them; the other finds nothing and does
/// nothing, so the run ends once, with its lane.
struct Claim<L: Lane>(Arc<Handoff<(Step<L>, L)>>);

impl<L: Lane> Clone for Claim<L> {
    fn clone(&self) -> Claim<L> {
        Claim(Arc::clone(&self.0))
    }
}

impl<L: Lane> Claim<L> {
    fn new(run: Step<L>, lane: L) -> Claim<L> {
        let cell = Handoff::new();
        cell.send((run, lane));
        Claim(cell)
    }

    /// The continuation of a call: `result` goes to `then`, a failure ends the run.
    fn resume<T, E: Into<RunError>>(&self, result: Result<T, E>, then: impl FnOnce(Step<L>, L, T)) {
        let Some((run, lane)) = self.0.take() else {
            return;
        };
        let beat = run.reply.beat();
        match result.map_err(Into::into).and_then(|value| beat.map(|()| value)) {
            Ok(value) => then(run, lane, value),
            Err(error) => fail(run, lane, error),
        }
    }

    /// The caller's side of a call: one that could not start ends the run.
    fn started<E: Into<RunError>>(&self, started: Result<(), E>) {
        if let Err(error) = started
            && let Some((run, lane)) = self.0.take()
        {
            fail(run, lane, error);
        }
    }

    /// The continuation of a resume file's call.
    fn then<T: Send + 'static>(&self, then: impl FnOnce(Step<L>, L, T) + Send + 'static) -> Then<T> {
        let claim = self.clone();
        Box::new(move |result| {
            claim.resume(result, then);
            idle()
        })
    }
}

/// Starts the run: returns once its first call is under way; the answer reaches `reply`.
pub fn start<L: Lane>(ready: Ready<L>, options: &Options, host: &Host<'_>, log: Log, reply: Arc<Reply<Ended<L>>>) {
    let Ready {
        lane,
        data,
        width,
        hidden,
        row_of_token,
        identity,
        progress,
        start,
        state_pieces,
        best_pieces,
    } = ready;
    let run = Box::new(Run {
        io: Arc::clone(host.io),
        lake: Arc::clone(host.lake),
        data,
        settings: options.settings,
        width,
        hidden,
        row_of_token,
        identity,
        progress,
        out: options.out.clone(),
        state_pieces,
        best_pieces,
        windows: VecDeque::new(),
        sequence: 0,
        phase: Phase::Training,
        saves: 0,
        stop_after_saves: options.stop_after_saves,
        fail_in_save: options.fail_in_save,
        log,
        reply,
    });
    match start {
        Start::Initial => {
            let held_out = run.data.held_out().iter().copied().collect();
            evaluate(run, lane, held_out, true, false);
        }
        Start::Continue => train(run, lane),
        Start::Complete => end(run, lane, Ok(false)),
    }
}

fn say<L: Lane>(run: &Run<L>, line: &str) {
    (run.log)(line);
}

/// The one way out: ends the run, answering the thread that started it with the lane. `stopped` says it ended at a
/// saved state rather than at the end.
fn end<L: Lane>(run: Step<L>, lane: L, outcome: Result<bool, RunError>) {
    let summary = outcome.map(|stopped| Summary {
        saves: run.saves,
        stopped,
        best: run.progress.best(),
    });
    run.reply.send(Ok(Ended { lane, outcome: summary }));
}

fn fail<L: Lane>(run: Step<L>, lane: L, error: impl Into<RunError>) {
    end(run, lane, Err(error.into()));
}

/// The `done` of a lane call: its result goes to `then`, a failure ends the run.
fn lane_done<L: Lane, T: Send + 'static>(
    run: Step<L>,
    then: impl FnOnce(Step<L>, L, T) + Send + 'static,
) -> Done<L, T> {
    Box::new(move |finished| {
        let beat = run.reply.beat();
        match finished.result.and_then(|value| beat.map(|()| value)) {
            Ok(value) => then(run, finished.lane, value),
            Err(error) => fail(run, finished.lane, error),
        }
    })
}

/// The next sequence to train on, or the end of the epochs.
fn train<L: Lane>(mut run: Step<L>, lane: L) {
    run.phase = Phase::Training;
    match run.progress.advance() {
        Next::Sequence(index) => begin(run, lane, index),
        Next::Final => {
            let held_out = run.data.held_out().iter().copied().collect();
            evaluate(run, lane, held_out, false, true);
        }
    }
}

/// Starts sequence `index`: its windows, in order.
fn begin<L: Lane>(mut run: Step<L>, lane: L, index: usize) {
    let Some(entry) = run.data.entries().get(index) else {
        return fail(
            run,
            lane,
            RunError::Broken(format!("the corpus has no sequence {index}")),
        );
    };
    let (length, prompt) = (entry.length, entry.prompt);
    let (Ok(length), Ok(prompt)) = (u32::try_from(length), u32::try_from(prompt)) else {
        return fail(run, lane, RunError::Input(format!("sequence {index} is too long")));
    };
    run.sequence = index;
    run.windows = windows(length, prompt, &run.settings).into();
    advance(run, lane);
}

/// The next window of the sequence, or the sequence's end.
fn advance<L: Lane>(mut run: Step<L>, lane: L) {
    match run.windows.pop_front() {
        Some(window) => read(run, lane, window),
        None => match run.phase {
            Phase::Training => sequence_end(run, lane),
            Phase::Evaluating(_) => next_held_out(run, lane),
        },
    }
}

/// Reads the window's hidden rows, then runs it.
fn read<L: Lane>(run: Step<L>, lane: L, window: Window) {
    let range = run.data.hidden_range(run.sequence, window.a, window.count, run.width);
    let (offset, len) = match range {
        Ok(range) => range,
        Err(error) => return fail(run, lane, error),
    };
    let (io, file) = (Arc::clone(&run.io), Arc::clone(&run.hidden));
    let context = format!("the hidden rows of sequence {}", run.sequence);
    let claim = Claim::new(run, lane);
    let held = claim.clone();
    let next = ReadNext::new(move |bytes: Result<Vec<u8>, _>| {
        held.resume(bytes.map_err(RunError::from), |run, lane, bytes| {
            launch(run, lane, window, bytes);
        });
    });
    claim.started(read_exact(&io, file, offset, len, context, next).map_err(RunError::from));
}

/// Runs `window` on the lane over its hidden rows.
fn launch<L: Lane>(run: Step<L>, lane: L, window: Window, hidden: Vec<u8>) {
    let evaluating = matches!(run.phase, Phase::Evaluating(_));
    let staged = match run.data.sequence(run.sequence) {
        Ok(sequence) => stage(&sequence, &window, run.settings.depths, &run.row_of_token, evaluating),
        Err(error) => return fail(run, lane, error),
    };
    let job = Job {
        window,
        mode: if evaluating { Mode::Evaluate } else { Mode::Train },
        hidden,
        ids: staged.ids,
        labels: staged.labels,
    };
    let checked = staged.checked;
    let done = lane_done(run, move |run, lane, outcome: Outcome| {
        windowed(run, lane, window, &checked, &outcome);
    });
    lane.window(job, done);
}

/// A window ran: tally it, and owe an update if its lineages are enough.
fn windowed<L: Lane>(mut run: Step<L>, lane: L, window: Window, checked: &[i32], outcome: &Outcome) {
    if let Phase::Evaluating(evaluating) = &mut run.phase {
        let sampled = run.data.sampled(run.sequence);
        evaluating
            .tally
            .add(sampled, window.count as usize, checked, &outcome.predicted);
        return advance(run, lane);
    }
    let trained = Trained {
        labelled: window.labelled,
        count: window.count as usize,
        checked,
        predicted: &outcome.predicted,
        losses: &outcome.losses,
    };
    match run.progress.trained(&trained) {
        None => advance(run, lane),
        Some(update) => owed(run, lane, update),
    }
}

/// Applies the update, then goes on.
fn owed<L: Lane>(mut run: Step<L>, lane: L, update: Update) {
    if let Some(line) = run.progress.report(&update) {
        say(&run, &line);
    }
    let done = lane_done(run, |run, lane, ()| advance(run, lane));
    lane.update(update, done);
}

/// A sequence's windows are done: evaluate if one is due.
fn sequence_end<L: Lane>(mut run: Step<L>, lane: L) {
    if run.progress.take_evaluation_due() {
        let held_out = run.data.held_out().iter().copied().collect();
        evaluate(run, lane, held_out, false, false);
    } else {
        train(run, lane);
    }
}

/// Starts an evaluation over the `held_out` sequences.
fn evaluate<L: Lane>(mut run: Step<L>, lane: L, held_out: VecDeque<usize>, initial: bool, last: bool) {
    run.phase = Phase::Evaluating(Box::new(Evaluating {
        tally: Evaluation::new(run.settings.depths),
        queue: held_out,
        initial,
        last,
    }));
    next_held_out(run, lane);
}

/// The next held-out sequence, or the evaluation's verdict.
fn next_held_out<L: Lane>(mut run: Step<L>, lane: L) {
    let Phase::Evaluating(evaluating) = &mut run.phase else {
        return fail(
            run,
            lane,
            RunError::Broken("an evaluation step outside an evaluation".to_owned()),
        );
    };
    match evaluating.queue.pop_front() {
        Some(index) => begin(run, lane, index),
        None => verdict(run, lane),
    }
}

/// The evaluation is complete: log it, save the best parameters if they improved, save the state, and go on.
fn verdict<L: Lane>(mut run: Step<L>, lane: L) {
    let Phase::Evaluating(evaluating) = replace(&mut run.phase, Phase::Training) else {
        return fail(
            run,
            lane,
            RunError::Broken("a verdict outside an evaluation".to_owned()),
        );
    };
    let verdict = run
        .progress
        .evaluated(&evaluating.tally, evaluating.initial, evaluating.last);
    say(&run, &verdict.line);
    let after = move |run: Step<L>, lane: L| {
        if verdict.save_state {
            save_state(run, lane, verdict.finished);
        } else {
            carry_on(run, lane, verdict.finished);
        }
    };
    if verdict.improved {
        save_best(run, lane, Box::new(after));
    } else {
        after(run, lane);
    }
}

/// After the saves: the end, or more training.
fn carry_on<L: Lane>(run: Step<L>, lane: L, finished: bool) {
    if finished {
        say(
            &run,
            &format!("best held-out accepted drafts {:.4}", run.progress.best()),
        );
        end(run, lane, Ok(false));
    } else {
        train(run, lane);
    }
}

type After<L> = Box<dyn FnOnce(Step<L>, L) + Send>;

/// Saves the best parameters to `best.bin`, then `after`.
fn save_best<L: Lane>(run: Step<L>, lane: L, after: After<L>) {
    let header = run.progress.best_header(&run.identity);
    let (target, pieces) = (run.out.join("best.bin"), run.best_pieces.clone());
    save(run, lane, target, header, pieces, None, after);
}

/// Saves the state to `resume.bin`, then goes on (or stops, at the requested number of saves).
fn save_state<L: Lane>(run: Step<L>, lane: L, finished: bool) {
    let header = run.progress.header(&run.identity);
    let (target, pieces) = (run.out.join("resume.bin"), run.state_pieces.clone());
    // The injected failure of this save, if it is the one asked for: it fails with half of the pieces left.
    let fail_at = (run.fail_in_save == run.saves + 1).then_some(pieces.len() / 2);
    let after: After<L> = Box::new(move |mut run, lane| {
        run.saves += 1;
        if run.stop_after_saves > 0 && run.saves >= run.stop_after_saves {
            say(&run, &format!("stopped after {} saved state(s)", run.saves));
            end(run, lane, Ok(true));
        } else {
            carry_on(run, lane, finished);
        }
    });
    save(run, lane, target, header, pieces, fail_at, after);
}

/// A file being saved: its pieces are written to a sibling of `target`, which is renamed over `target` once whole.
struct Saving<L: Lane> {
    target: PathBuf,
    header: Header,
    rest: VecDeque<Placed>,
    /// The file's length when whole.
    len: u64,
    /// The save fails when this many pieces are left to write (an injected failure).
    fail_at: Option<usize>,
    after: After<L>,
}

/// The name a file is written under until it is whole.
fn unfinished(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_owned();
    name.push(".part");
    PathBuf::from(name)
}

/// Writes `pieces` of the lane's memory to a sibling of `target` and renames it over `target` when it is whole and
/// durable, so that a stop at any point leaves `target` as it was or as the new file, never between. Then `after`.
fn save<L: Lane>(
    run: Step<L>,
    lane: L,
    target: PathBuf,
    header: Header,
    pieces: Vec<Placed>,
    fail_at: Option<usize>,
    after: After<L>,
) {
    let len = pieces
        .last()
        .map_or(HEADER, |last| last.piece.offset + last.piece.bytes as u64);
    let saving = Saving {
        target,
        header,
        rest: pieces.into(),
        len,
        fail_at,
        after,
    };
    let (io, lake, part) = (Arc::clone(&run.io), Arc::clone(&run.lake), unfinished(&saving.target));
    let claim = Claim::new(run, lane);
    let opened = claim.then(move |run, lane, file: ResumeFile| write_piece(run, lane, file, saving));
    claim.started(ResumeFile::open(&io, &lake, &part, opened));
}

/// Reads the next piece from the device and writes it; after the last, seals the file under its own name.
fn write_piece<L: Lane>(run: Step<L>, lane: L, file: ResumeFile, mut saving: Saving<L>) {
    if saving.fail_at == Some(saving.rest.len()) {
        return fail(
            run,
            lane,
            RunError::Broken(format!(
                "a failure injected into the save of {}",
                saving.target.display()
            )),
        );
    }
    let Some(placed) = saving.rest.pop_front() else {
        let claim = Claim::new(run, lane);
        let sealed = claim.then(move |run, lane, ()| (saving.after)(run, lane));
        return claim.started(file.finish(&saving.header, saving.len, &saving.target, sealed));
    };
    let done = lane_done(run, move |run, lane, bytes: Vec<u8>| {
        let claim = Claim::new(run, lane);
        let written =
            claim.then(move |run, lane, (file, _end): (ResumeFile, u64)| write_piece(run, lane, file, saving));
        claim.started(file.write_through(placed.piece.offset, bytes, written));
    });
    lane.read(placed.address, placed.piece.bytes, done);
}

#[cfg(test)]
mod tests;
