//! The run's frames with a lane that has no device: the order of its calls, the verdict, the order of its saves and
//! the way each failure ends it, with the lane given back to the thread that started it.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test: a failed unwrap is the failure, and the test thread waits for the frames it started"
)]

use std::fs;
use std::mem::discriminant;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::LatticeSource;
use euhedral_graph::Lake;
use euhedral_inference_engine::lattice_config;
use euhedral_inference_settings::{ModelSettings, Settings as Product};
use euhedral_io::{AsyncIo, OwnedFile};

use super::*;
use crate::chain::{self, Scratch};
use crate::checkpoint::io::open_read;
use crate::corpus::Entry;
use crate::mtp::train::data::Files;
use crate::mtp::train::lane::Job;
use crate::mtp::train::settings::expected_steps;
use crate::mtp::train::state::{Part, Piece};
use crate::mtp::train::wait::{call, pair};

/// The hang deadline of a run: it fails if no call finishes over this long, which a run of fakes never does.
const HANG: Duration = Duration::from_secs(20);
const WIDTH: u32 = 4;
const SEQUENCES: u64 = 4;
/// Tokens of a sequence, two of them the prompt; it has `LENGTH - 1` hidden rows.
const LENGTH: u64 = 6;
const BEST: u64 = 0x9000;
const STATE: u64 = 0x1000;

fn settings() -> Settings {
    Settings {
        depths: 2,
        rows: 4,
        keys: 8,
        tokens: 2,
        epochs: 1,
        warmup: 1,
        holdout: 2,
        evaluate_every: 1,
        ..Settings::default()
    }
}

/// What the lane was asked, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Call {
    Window(Mode),
    Update,
    Read(u64),
}

/// What a fake lane records and where it fails; shared by all of a run's lane values.
#[derive(Default)]
struct Script {
    calls: Mutex<Vec<Call>>,
    closed: AtomicU32,
    /// The window, update or read of this number (from 0, of its kind) fails.
    fail_window: Option<usize>,
    fail_read: Option<usize>,
}

impl Script {
    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }

    /// Records `call` and says whether it is the one that fails.
    fn record(&self, call: Call, fail: Option<usize>) -> bool {
        let mut calls = self.calls.lock().unwrap();
        let same = |other: &&Call| discriminant(*other) == discriminant(&call);
        let number = calls.iter().filter(same).count();
        calls.push(call);
        fail == Some(number)
    }
}

/// A lane that answers at once: every window predicts the labels it was given, and a read gives bytes named by its
/// address.
struct Fake {
    script: Arc<Script>,
}

fn answer<T>(lane: Fake, result: Result<T, RunError>, done: Done<Fake, T>) {
    done(chain::Finished {
        lane,
        scratch: Scratch::new(()),
        result,
    });
}

impl Lane for Fake {
    fn window(self, job: Job, done: Done<Self, Outcome>) {
        let failed = self.script.record(Call::Window(job.mode), self.script.fail_window);
        let result = if failed {
            Err(RunError::Broken("the window failed".to_owned()))
        } else {
            Ok(Outcome {
                predicted: if job.window.labelled {
                    job.labels.clone()
                } else {
                    Vec::new()
                },
                losses: vec![0.5; job.ids.len() / job.window.count as usize],
            })
        };
        answer(self, result, done);
    }

    fn update(self, _update: Update, done: Done<Self, ()>) {
        self.script.record(Call::Update, None);
        answer(self, Ok(()), done);
    }

    fn read(self, address: u64, bytes: usize, done: Done<Self, Vec<u8>>) {
        let failed = self.script.record(Call::Read(address), self.script.fail_read);
        let result = if failed {
            Err(RunError::Broken("the read failed".to_owned()))
        } else {
            Ok(vec![(address >> 8) as u8; bytes])
        };
        answer(self, result, done);
    }

    fn close(self) -> Result<(), RunError> {
        self.script.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct World {
    lattice: Arc<ControlPlaneLattice>,
    io: Arc<AsyncIo>,
    lake: Arc<Lake>,
}

impl World {
    fn start() -> World {
        let lattice = ControlPlaneLattice::new(lattice_config(&Product::with_model(ModelSettings::default()))).unwrap();
        let io = AsyncIo::open().unwrap();
        lattice.add_upstream(Arc::clone(&io) as Arc<dyn LatticeSource>).unwrap();
        let lake = Lake::new(
            Arc::clone(&lattice),
            NonZeroUsize::new(2).unwrap(),
            Duration::from_secs(1),
        );
        World { lattice, io, lake }
    }

    fn stop(self) {
        self.lake.complete_gracefully();
        self.io.complete();
        self.lattice.close();
    }

    fn hidden(&self, path: &Path, rows: u64) -> Arc<OwnedFile> {
        fs::write(path, vec![1u8; (rows * 2 * u64::from(WIDTH)) as usize]).unwrap();
        call(|reply| {
            open_read(
                &self.io,
                path,
                Box::new(move |opened| reply.send(opened.map(Arc::new).map_err(RunError::from))),
            )
        })
        .unwrap()
    }
}

fn piece(offset: u64, bytes: usize) -> Piece {
    Piece {
        parameter: 0,
        part: Part::Values,
        inside: 0,
        offset,
        bytes,
    }
}

/// Three pieces of the state and two of the best parameters, at addresses that tell them apart.
fn pieces() -> (Vec<Placed>, Vec<Placed>) {
    let at = |base: u64, count: u64| {
        (0..count)
            .map(|index| Placed {
                piece: piece(HEADER + 16 * index, 16),
                address: base + 0x100 * index,
            })
            .collect()
    };
    (at(STATE, 3), at(BEST, 2))
}

fn ready(world: &World, out: &Path, script: &Arc<Script>, start: Start, rows: u64) -> Ready<Fake> {
    let entries = (0..SEQUENCES)
        .map(|index| Entry {
            id: format!("s{index}"),
            token: LENGTH * index,
            row: (LENGTH - 1) * index,
            length: LENGTH,
            prompt: 2,
            sampled: false,
        })
        .collect();
    let tokens: Vec<i32> = (0..(LENGTH * SEQUENCES) as i32).map(|token| token % 7).collect();
    let labels: Vec<i32> = tokens.iter().map(|token| (token + 1) % 7).collect();
    let data = Data::new(
        Files {
            entries,
            tokens,
            labels,
            hidden_rows: (LENGTH - 1) * SEQUENCES,
            vocabulary: 8,
        },
        &settings(),
    )
    .unwrap();
    let expected = expected_steps(data.lengths(), settings().tokens);
    let (state_pieces, best_pieces) = pieces();
    Ready {
        lane: Fake {
            script: Arc::clone(script),
        },
        hidden: world.hidden(&out.join("hidden.bin"), rows),
        width: WIDTH,
        row_of_token: (0..8).collect(),
        identity: Identity::new(
            "c".to_owned(),
            "m".to_owned(),
            "a".to_owned(),
            &settings(),
            SEQUENCES as usize,
        ),
        progress: Progress::new(settings(), data.training().to_vec(), expected),
        data,
        start,
        state_pieces,
        best_pieces,
    }
}

fn options(out: &Path, stop_after_saves: u32, fail_in_save: u32) -> Options {
    Options {
        corpus: out.to_path_buf(),
        model: out.to_path_buf(),
        artifact: out.to_path_buf(),
        out: out.to_path_buf(),
        settings: settings(),
        library: out.to_path_buf(),
        math: out.to_path_buf(),
        stop_after_saves,
        fresh: false,
        poison: None,
        fail_in_save,
    }
}

/// Runs the run to its end and gives what the thread that started it gets back.
fn run(world: &World, ready: Ready<Fake>, options: &Options) -> Ended<Fake> {
    let attach = |_: Arc<dyn LatticeSource>| Ok(());
    let host = Host {
        io: &world.io,
        lake: &world.lake,
        attach: &attach,
    };
    let (wait, reply) = pair();
    start(ready, options, &host, Arc::new(|_| {}), reply);
    wait.follow_within(HANG).expect("the run never ended")
}

fn modes(calls: &[Call]) -> Vec<Mode> {
    calls
        .iter()
        .filter_map(|call| match call {
            Call::Window(mode) => Some(*mode),
            _ => None,
        })
        .collect()
}

fn reads(calls: &[Call]) -> Vec<u64> {
    calls
        .iter()
        .filter_map(|call| match call {
            Call::Read(address) => Some(*address),
            _ => None,
        })
        .collect()
}

#[test]
fn a_run_from_the_checkpoint_evaluates_first_trains_and_ends_with_its_lane_and_both_files_sealed() {
    let (world, out) = (World::start(), tempfile::tempdir().unwrap());
    let script = Arc::new(Script::default());
    let ended = run(
        &world,
        ready(&world, out.path(), &script, Start::Initial, SEQUENCES * (LENGTH - 1)),
        &options(out.path(), 0, 0),
    );
    let summary = ended.outcome.unwrap();
    assert!(!summary.stopped);
    let calls = script.calls();
    let modes = modes(&calls);
    assert_eq!(
        modes.first(),
        Some(&Mode::Evaluate),
        "the evaluation before any training"
    );
    assert!(modes.contains(&Mode::Train));
    assert!(calls.contains(&Call::Update));
    // The first evaluation always improves: the best parameters are read, all of them, before the state is.
    let reads = reads(&calls);
    assert_eq!(&reads[..2], &[BEST, BEST + 0x100], "the best parameters first");
    assert_eq!(&reads[2..5], &[STATE, STATE + 0x100, STATE + 0x200], "then the state");
    assert_eq!(
        summary.saves as usize * 3,
        reads.iter().filter(|a| (STATE..BEST).contains(a)).count()
    );
    // The files are whole under their own names, with the pieces where they belong, and nothing is left unfinished.
    let state = fs::read(out.path().join("resume.bin")).unwrap();
    assert_eq!(state.len(), HEADER as usize + 48);
    assert_eq!(&state[HEADER as usize..][..16], &[(STATE >> 8) as u8; 16]);
    assert_eq!(
        &state[HEADER as usize + 32..][..16],
        &[((STATE + 0x200) >> 8) as u8; 16]
    );
    let best = fs::read(out.path().join("best.bin")).unwrap();
    assert_eq!(best.len(), HEADER as usize + 32);
    assert_eq!(&best[HEADER as usize + 16..][..16], &[((BEST + 0x100) >> 8) as u8; 16]);
    assert!(!out.path().join("best.bin.part").exists());
    assert!(!out.path().join("resume.bin.part").exists());
    // The lane comes back to the starter, which closes it once.
    ended.lane.close().unwrap();
    assert_eq!(script.closed.load(Ordering::SeqCst), 1);
    world.stop();
}

#[test]
fn a_run_from_a_saved_state_does_not_evaluate_before_training() {
    let (world, out) = (World::start(), tempfile::tempdir().unwrap());
    let script = Arc::new(Script::default());
    let ended = run(
        &world,
        ready(&world, out.path(), &script, Start::Continue, SEQUENCES * (LENGTH - 1)),
        &options(out.path(), 0, 0),
    );
    ended.outcome.unwrap();
    assert_eq!(modes(&script.calls()).first(), Some(&Mode::Train));
    world.stop();
}

#[test]
fn a_run_that_was_complete_makes_no_call_and_gives_the_lane_back() {
    let (world, out) = (World::start(), tempfile::tempdir().unwrap());
    let script = Arc::new(Script::default());
    let ended = run(
        &world,
        ready(&world, out.path(), &script, Start::Complete, SEQUENCES * (LENGTH - 1)),
        &options(out.path(), 0, 0),
    );
    assert!(!ended.outcome.unwrap().stopped);
    assert!(script.calls().is_empty());
    assert!(!out.path().join("resume.bin").exists());
    world.stop();
}

#[test]
fn a_run_asked_to_stop_after_a_save_ends_at_that_saved_state() {
    let (world, out) = (World::start(), tempfile::tempdir().unwrap());
    let script = Arc::new(Script::default());
    let ended = run(
        &world,
        ready(&world, out.path(), &script, Start::Initial, SEQUENCES * (LENGTH - 1)),
        &options(out.path(), 1, 0),
    );
    let summary = ended.outcome.unwrap();
    assert!(summary.stopped);
    assert_eq!(summary.saves, 1);
    assert!(out.path().join("resume.bin").exists());
    // The run ends with the state's last piece: nothing is run after the save it stopped at.
    let calls = script.calls();
    assert_eq!(calls.last(), Some(&Call::Read(STATE + 0x200)), "{calls:?}");
    assert!(
        calls.contains(&Call::Update),
        "the state saved is of a run that trained"
    );
    world.stop();
}

#[test]
fn a_window_the_lane_fails_ends_the_run_with_the_lane_and_the_error() {
    let (world, out) = (World::start(), tempfile::tempdir().unwrap());
    let script = Arc::new(Script {
        fail_window: Some(1),
        ..Script::default()
    });
    let ended = run(
        &world,
        ready(&world, out.path(), &script, Start::Initial, SEQUENCES * (LENGTH - 1)),
        &options(out.path(), 0, 0),
    );
    assert!(matches!(&ended.outcome, Err(RunError::Broken(message)) if message == "the window failed"));
    assert_eq!(modes(&script.calls()).len(), 2, "nothing runs after the failure");
    ended.lane.close().unwrap();
    assert_eq!(script.closed.load(Ordering::SeqCst), 1);
    world.stop();
}

#[test]
fn a_hidden_file_too_short_for_a_window_ends_the_run_with_the_lane() {
    let (world, out) = (World::start(), tempfile::tempdir().unwrap());
    let script = Arc::new(Script::default());
    let ended = run(
        &world,
        ready(&world, out.path(), &script, Start::Initial, 1),
        &options(out.path(), 0, 0),
    );
    assert!(ended.outcome.is_err());
    assert!(script.calls().is_empty(), "no window ran over rows that are not there");
    ended.lane.close().unwrap();
    world.stop();
}

#[test]
fn a_read_of_the_lane_that_fails_while_saving_ends_the_run_and_leaves_the_files_as_they_were() {
    let (world, out) = (World::start(), tempfile::tempdir().unwrap());
    fs::write(out.path().join("best.bin"), b"the earlier best parameters").unwrap();
    let script = Arc::new(Script {
        fail_read: Some(1),
        ..Script::default()
    });
    let ended = run(
        &world,
        ready(&world, out.path(), &script, Start::Initial, SEQUENCES * (LENGTH - 1)),
        &options(out.path(), 0, 0),
    );
    assert!(matches!(&ended.outcome, Err(RunError::Broken(message)) if message == "the read failed"));
    assert_eq!(
        fs::read(out.path().join("best.bin")).unwrap(),
        b"the earlier best parameters"
    );
    ended.lane.close().unwrap();
    world.stop();
}

#[test]
fn a_failure_inside_the_state_save_leaves_the_saved_state_and_the_best_parameters_whole() {
    let (world, out) = (World::start(), tempfile::tempdir().unwrap());
    fs::write(out.path().join("resume.bin"), b"the earlier state").unwrap();
    let script = Arc::new(Script::default());
    let ended = run(
        &world,
        ready(&world, out.path(), &script, Start::Initial, SEQUENCES * (LENGTH - 1)),
        &options(out.path(), 0, 1),
    );
    assert!(
        matches!(&ended.outcome, Err(RunError::Broken(message)) if message.contains("injected into the save of")),
        "{:?}",
        ended.outcome
    );
    assert_eq!(fs::read(out.path().join("resume.bin")).unwrap(), b"the earlier state");
    // The best parameters of the evaluation before it were sealed: they are whole.
    assert_eq!(
        fs::read(out.path().join("best.bin")).unwrap().len(),
        HEADER as usize + 32
    );
    ended.lane.close().unwrap();
    assert_eq!(script.closed.load(Ordering::SeqCst), 1);
    world.stop();
}

#[test]
fn a_save_that_cannot_open_its_file_ends_the_run_with_the_lane_and_writes_nothing() {
    let (world, out) = (World::start(), tempfile::tempdir().unwrap());
    let script = Arc::new(Script::default());
    let gone = out.path().join("not-there");
    let ended = run(
        &world,
        ready(&world, out.path(), &script, Start::Initial, SEQUENCES * (LENGTH - 1)),
        &options(&gone, 0, 0),
    );
    assert!(ended.outcome.is_err());
    assert!(
        reads(&script.calls()).is_empty(),
        "nothing was read for a file that could not be opened"
    );
    ended.lane.close().unwrap();
    assert_eq!(script.closed.load(Ordering::SeqCst), 1);
    world.stop();
}
