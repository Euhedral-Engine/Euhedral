//! The run as frames: each window's hidden rows are read, staged and run on the lane, and the frame that retires the
//! work trains the next. Nothing waits: a read, a window, an update or a save each hands the run, with the lane, to the
//! call that ends it, and that call goes on.
//!
//! Per sequence, its windows are trained in order; a trained window may owe an update. At a sequence's end an
//! evaluation over the held-out sequences runs if one is due; its verdict saves the best parameters when they improved
//! and the state to resume from. The run ends after the last epoch's evaluation, or after the state save that makes
//! `stop_after_saves`.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_graph::Lake;
use euhedral_io::{AsyncIo, OwnedFile};

use super::data::Data;
use super::error::RunError;
use super::lane::{CHUNK_BYTES, Finished, Job, Outcome, TrainLane};
use super::load::{Keep, Loaded, Start};
use super::options::{Host, Log, Options};
use super::progress::{Evaluation, Next, Progress, Trained, Update};
use super::settings::{Settings, Window, windows};
use super::stage::stage;
use super::state::{File, Part, Piece, pieces};
use super::wait::Reply;
use super::window::Mode;
use crate::checkpoint::io::read_exact;
use crate::resume::{Header, ResumeError, ResumeFile, Then};
use crate::step::idle;

/// What a run hands back to the thread that started it.
pub struct Ended {
    /// The lane, free: the caller closes it.
    pub lane: TrainLane,
    /// What the device needs to stay open until then.
    pub keep: Keep,
    /// How the run ended.
    pub outcome: Result<Summary, RunError>,
}

impl std::fmt::Debug for Ended {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

/// A piece of a file with the device address it is read from or written to.
#[derive(Clone, Copy, Debug)]
struct Placed {
    piece: Piece,
    address: u64,
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
struct Run {
    io: Arc<AsyncIo>,
    lake: Arc<Lake>,
    data: Data,
    settings: Settings,
    width: u32,
    hidden: Arc<OwnedFile>,
    row_of_token: Vec<i32>,
    identity: String,
    progress: Progress,
    state: ResumeFile,
    best_file: Option<ResumeFile>,
    out: PathBuf,
    state_pieces: Vec<Placed>,
    best_pieces: Vec<Placed>,
    windows: VecDeque<Window>,
    sequence: usize,
    phase: Phase,
    saves: u32,
    stop_after_saves: u32,
    log: Log,
    keep: Option<Keep>,
    reply: Arc<Reply<Ended>>,
}

type Step = Box<Run>;

/// Where the device keeps a part of a parameter.
fn address(lane: &TrainLane, piece: &Piece) -> u64 {
    let parameter = &lane.drafter().parameters()[piece.parameter];
    let base = match piece.part {
        Part::Values => parameter.values,
        Part::First => parameter.first,
        Part::Second => parameter.second,
        Part::Gradient => parameter.gradient,
    };
    base + piece.inside
}

fn place(lane: &TrainLane, file: File) -> Vec<Placed> {
    let counts: Vec<u64> = lane
        .drafter()
        .parameters()
        .iter()
        .map(|parameter| parameter.count())
        .collect();
    pieces(file, &counts, CHUNK_BYTES)
        .into_iter()
        .map(|piece| Placed {
            piece,
            address: address(lane, &piece),
        })
        .collect()
}

/// Starts the run: returns once its first call is under way; the answer reaches `reply`.
pub fn start(loaded: Loaded, options: &Options, host: &Host<'_>, log: Log, reply: Arc<Reply<Ended>>) {
    let Loaded {
        lane,
        data,
        width,
        hidden,
        row_of_token,
        identity,
        progress,
        state,
        start,
        keep,
    } = loaded;
    let (state_pieces, best_pieces) = (place(&lane, File::State), place(&lane, File::Best));
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
        state,
        best_file: None,
        out: options.out.clone(),
        state_pieces,
        best_pieces,
        windows: VecDeque::new(),
        sequence: 0,
        phase: Phase::Training,
        saves: 0,
        stop_after_saves: options.stop_after_saves,
        log,
        keep: Some(keep),
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

fn say(run: &Run, line: &str) {
    (run.log)(line);
}

/// Ends the run, answering the thread that started it. `stopped` says it ended at a saved state rather than at the
/// end.
fn end(mut run: Step, lane: TrainLane, outcome: Result<bool, RunError>) {
    let summary = outcome.map(|stopped| Summary {
        saves: run.saves,
        stopped,
        best: run.progress.best(),
    });
    if let Some(keep) = run.keep.take() {
        run.reply.send(Ok(Ended {
            lane,
            keep,
            outcome: summary,
        }));
    }
}

fn fail(run: Step, lane: TrainLane, error: impl Into<RunError>) {
    end(run, lane, Err(error.into()));
}

/// The `done` of a lane call: its result goes to `then`, a failure ends the run.
fn lane_done<T: Send + 'static>(
    run: Step,
    then: impl FnOnce(Step, TrainLane, T) + Send + 'static,
) -> Box<dyn FnOnce(Finished<T>) + Send> {
    Box::new(move |finished| match finished.result {
        Ok(value) => then(run, finished.lane, value),
        Err(error) => fail(run, finished.lane, error),
    })
}

/// The continuation of a resume file's call: its result goes to `then`, a failure ends the run. A call that could not
/// start gives the run's error back to the caller, which answers with it (see [`started`]).
fn resumed<T: Send + 'static>(
    run: Step,
    lane: TrainLane,
    then: impl FnOnce(Step, TrainLane, T) + Send + 'static,
) -> Then<T> {
    Box::new(move |result| {
        match result {
            Ok(value) => then(run, lane, value),
            Err(error) => fail(run, lane, error),
        }
        idle()
    })
}

/// A call that could not start dropped its continuation, and with it the run: answer for it.
fn started(reply: &Arc<Reply<Ended>>, result: Result<(), ResumeError>) {
    if let Err(error) = result {
        reply.send(Err(error.into()));
    }
}

/// The next sequence to train on, or the end of the epochs.
fn train(mut run: Step, lane: TrainLane) {
    run.phase = Phase::Training;
    match run.progress.next() {
        Next::Sequence(index) => begin(run, lane, index),
        Next::Final => {
            let held_out = run.data.held_out().iter().copied().collect();
            evaluate(run, lane, held_out, false, true);
        }
    }
}

/// Starts sequence `index`: its windows, in order.
fn begin(mut run: Step, lane: TrainLane, index: usize) {
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
fn advance(mut run: Step, lane: TrainLane) {
    match run.windows.pop_front() {
        Some(window) => read(run, lane, window),
        None => match run.phase {
            Phase::Training => sequence_end(run, lane),
            Phase::Evaluating(_) => next_held_out(run, lane),
        },
    }
}

/// Reads the window's hidden rows, then runs it.
fn read(run: Step, lane: TrainLane, window: Window) {
    let range = run.data.hidden_range(run.sequence, window.a, window.count, run.width);
    let (offset, len) = match range {
        Ok(range) => range,
        Err(error) => return fail(run, lane, error),
    };
    let (io, file, reply) = (Arc::clone(&run.io), Arc::clone(&run.hidden), Arc::clone(&run.reply));
    let context = format!("the hidden rows of sequence {}", run.sequence);
    let next = Box::new(move |bytes: Result<Vec<u8>, _>| match bytes {
        Ok(bytes) => launch(run, lane, window, bytes),
        Err(error) => fail(run, lane, error),
    });
    if let Err(error) = read_exact(&io, file, offset, len, context, next) {
        reply.send(Err(error.into()));
    }
}

/// Runs `window` on the lane over its hidden rows.
fn launch(run: Step, lane: TrainLane, window: Window, hidden: Vec<u8>) {
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
    // A refused window's closure was not made yet: the lane comes back in the refusal.
    let (parked, pending) = (Arc::clone(&run.reply), run);
    let done = lane_done(pending, move |run, lane, outcome: Outcome| {
        windowed(run, lane, window, &checked, &outcome);
    });
    if let Err(refused) = lane.window(job, done) {
        parked.send(Err(refused.error));
        drop(refused.lane);
    }
}

/// A window ran: tally it, and owe an update if its lineages are enough.
fn windowed(mut run: Step, lane: TrainLane, window: Window, checked: &[i32], outcome: &Outcome) {
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
fn owed(mut run: Step, lane: TrainLane, update: Update) {
    if let Some(line) = run.progress.report(&update) {
        say(&run, &line);
    }
    lane.update(update, lane_done(run, |run, lane, ()| advance(run, lane)));
}

/// A sequence's windows are done: evaluate if one is due.
fn sequence_end(mut run: Step, lane: TrainLane) {
    if run.progress.take_evaluation_due() {
        let held_out = run.data.held_out().iter().copied().collect();
        evaluate(run, lane, held_out, false, false);
    } else {
        train(run, lane);
    }
}

/// Starts an evaluation over the `held_out` sequences.
fn evaluate(mut run: Step, lane: TrainLane, held_out: VecDeque<usize>, initial: bool, last: bool) {
    run.phase = Phase::Evaluating(Box::new(Evaluating {
        tally: Evaluation::new(run.settings.depths),
        queue: held_out,
        initial,
        last,
    }));
    next_held_out(run, lane);
}

/// The next held-out sequence, or the evaluation's verdict.
fn next_held_out(mut run: Step, lane: TrainLane) {
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
fn verdict(mut run: Step, lane: TrainLane) {
    let Phase::Evaluating(evaluating) = std::mem::replace(&mut run.phase, Phase::Training) else {
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
    let after = move |run: Step, lane: TrainLane| {
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
fn carry_on(run: Step, lane: TrainLane, finished: bool) {
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

type After = Box<dyn FnOnce(Step, TrainLane) + Send>;

/// Saves the best parameters to `best.bin`, then `after`.
fn save_best(run: Step, lane: TrainLane, after: After) {
    let header = run.progress.best_header(&run.identity);
    let pieces = run.best_pieces.clone();
    match run.best_file.clone() {
        Some(file) => save(run, lane, file, header, pieces, after),
        None => {
            let (io, lake, path, reply) = (
                Arc::clone(&run.io),
                Arc::clone(&run.lake),
                run.out.join("best.bin"),
                Arc::clone(&run.reply),
            );
            let opened = resumed(run, lane, move |mut run, lane, file: ResumeFile| {
                run.best_file = Some(file.clone());
                save(run, lane, file, header, pieces, after);
            });
            started(&reply, ResumeFile::open(&io, &lake, &path, opened));
        }
    }
}

/// Saves the state to `resume.bin`, then goes on (or stops, at the requested number of saves).
fn save_state(run: Step, lane: TrainLane, finished: bool) {
    let header = run.progress.header(&run.identity);
    let (file, pieces) = (run.state.clone(), run.state_pieces.clone());
    let after: After = Box::new(move |mut run, lane| {
        run.saves += 1;
        if run.stop_after_saves > 0 && run.saves >= run.stop_after_saves {
            say(&run, &format!("stopped after {} saved state(s)", run.saves));
            end(run, lane, Ok(true));
        } else {
            carry_on(run, lane, finished);
        }
    });
    save(run, lane, file, header, pieces, after);
}

/// Writes `pieces` to `file` between an invalid header and `header`, so that a stop at any point leaves the old file or
/// the new one.
fn save(run: Step, lane: TrainLane, file: ResumeFile, header: Header, pieces: Vec<Placed>, after: After) {
    let invalid = Progress::invalid_header(&run.identity);
    let (target, reply) = (file.clone(), Arc::clone(&run.reply));
    let marked = resumed(run, lane, move |run, lane, ()| {
        write_piece(run, lane, file, header, VecDeque::from(pieces), after);
    });
    started(&reply, target.set_header(&invalid, marked));
}

/// Reads the next piece from the device and writes it; at the last, writes the valid header.
fn write_piece(run: Step, lane: TrainLane, file: ResumeFile, header: Header, mut rest: VecDeque<Placed>, after: After) {
    let Some(placed) = rest.pop_front() else {
        let (target, reply) = (file, Arc::clone(&run.reply));
        let sealed = resumed(run, lane, move |run, lane, ()| after(run, lane));
        return started(&reply, target.set_header(&header, sealed));
    };
    let reply = Arc::clone(&run.reply);
    let done = lane_done(run, move |run, lane, bytes: Vec<u8>| {
        let (target, reply) = (file.clone(), Arc::clone(&run.reply));
        let written = resumed(run, lane, move |run, lane, _end: u64| {
            write_piece(run, lane, file, header, rest, after);
        });
        started(&reply, target.write(placed.piece.offset, bytes, written));
    });
    if let Err(refused) = lane.read(placed.address, placed.piece.bytes, done) {
        reply.send(Err(refused.error));
        drop(refused.lane);
    }
}
