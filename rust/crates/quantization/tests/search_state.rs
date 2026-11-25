//! The search's resume file on a real lattice, without a device: its calls start work and hand the state to
//! continuations.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting call: a test thread waits on the hand-off for the frames it started, and reads its files"
)]

use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::DummyFrame;
use euhedral_core::generics::LatticeSource;
use euhedral_graph::Lake;
use euhedral_inference_engine::{Handoff, lattice_config};
use euhedral_inference_quantization::mixed::{
    Allocation, Measured, Progress, QuantOption, SearchError, SearchState, SearchThen,
};
use euhedral_inference_quantization::resume::{HEADER, Header, ResumeError, ResumeFile, Then as ResumeThen};
use euhedral_inference_settings::{ModelSettings, Settings};
use euhedral_io::AsyncIo;

/// The longest a test waits for a call that must finish.
const LONGEST: Duration = Duration::from_secs(60);

struct World {
    lattice: Arc<ControlPlaneLattice>,
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
}

impl World {
    fn start() -> World {
        let settings = Settings::with_model(ModelSettings::default());
        let lattice = ControlPlaneLattice::new(lattice_config(&settings)).expect("the lattice starts");
        let io = AsyncIo::open().expect("the asynchronous I/O opens");
        lattice
            .add_upstream(Arc::clone(&io) as Arc<dyn LatticeSource>)
            .expect("the lattice takes the I/O");
        let host = Lake::new(
            Arc::clone(&lattice),
            NonZeroUsize::new(2).unwrap(),
            Duration::from_secs(1),
        );
        World { lattice, io, host }
    }

    fn open(&self, path: &Path, fingerprint: &str, seed: i64, free: u64) -> Result<SearchState, SearchError> {
        call(|then| SearchState::open(&self.io, &self.host, path, fingerprint, seed, free, then))
    }

    fn read(&self, path: &Path) -> Result<Vec<Measured>, SearchError> {
        call(|then| SearchState::read(&self.io, &self.host, path, then))
    }

    fn finish(self) {
        self.host.complete_gracefully();
        self.io.complete();
        self.lattice.close();
    }
}

/// Starts a call and waits, outside the lattice, for the value its continuation is given.
fn call<T: Send + 'static>(start: impl FnOnce(SearchThen<T>) -> Result<(), SearchError>) -> Result<T, SearchError> {
    let handoff = Handoff::new();
    let sender = Arc::clone(&handoff);
    start(Box::new(move |result| {
        sender.send(result);
        Box::new(DummyFrame::INSTANCE)
    }))?;
    handoff.wait(LONGEST).expect("the continuation never ran")
}

fn add(state: SearchState, candidate: Measured) -> (SearchState, bool) {
    call(|then| state.add(candidate, then)).unwrap()
}

fn allocation(first: QuantOption, second: QuantOption) -> Allocation {
    Allocation::new(BTreeMap::from([
        ("text/layers/0/mlp/down".to_string(), first),
        ("text/layers/1/mlp/down".to_string(), second),
    ]))
}

fn measured(allocation: Allocation, kl: f64, round: u32) -> Measured {
    Measured {
        allocation,
        kl,
        kl_se: 0.001,
        bytes: 1234,
        prefill: 1.5,
        decode: 2.5,
        fits: true,
        round,
        origin: "test".to_string(),
    }
}

fn nvfp4() -> Allocation {
    allocation(QuantOption::Nvfp4, QuantOption::Nvfp4)
}

fn mixed() -> Allocation {
    allocation(QuantOption::Nvfp4, QuantOption::Trellis3_0)
}

/// What is recorded survives a stop, with the progress and the first run's seed, and a candidate offered again after
/// the restart is refused without a second record.
#[test]
fn what_is_recorded_survives_a_stop_and_comes_back_without_duplicates() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("search.resume");
    let world = World::start();
    let state = world.open(&path, "fingerprint", 17, 1 << 30).unwrap();
    let (state, added) = add(state, measured(nvfp4(), 0.1, 0));
    assert!(added);
    let (state, added) = add(state, measured(mixed(), 0.2, 1));
    assert!(added);
    let progress = Progress {
        round: 1,
        explore: 0.4,
        stale: 2,
        held_out: 0.07,
    };
    let state = call(|then| state.set_progress(progress, then)).unwrap();
    // The same allocation offered again in the same run.
    let (state, added) = add(state, measured(nvfp4(), 0.5, 1));
    assert!(!added, "no duplicate within a run");
    assert_eq!(state.measured().len(), 2);
    drop(state);

    let state = world.open(&path, "fingerprint", 99, 5).unwrap();
    assert_eq!(state.measured().len(), 2);
    assert_eq!(state.seed(), 17, "the seed of the first run is kept");
    assert_eq!(state.free_bytes(), 1 << 30);
    assert_eq!(state.progress(), progress);
    assert!(state.contains(&mixed()));
    let size = fs::metadata(&path).unwrap().len();
    let (state, added) = add(state, measured(mixed(), 0.9, 1));
    assert!(!added, "no duplicate");
    assert_eq!(
        fs::metadata(&path).unwrap().len(),
        size,
        "a refused candidate writes nothing"
    );
    let (state, added) = add(
        state,
        measured(allocation(QuantOption::Trellis3_0, QuantOption::Nvfp4), 0.3, 1),
    );
    assert!(added);
    assert_eq!(state.measured().len(), 3);
    let first = &state.measured()[0];
    assert_eq!((first.kl, first.bytes, first.origin.as_str()), (0.1, 1234, "test"));
    drop(state);

    assert_eq!(world.open(&path, "fingerprint", 0, 0).unwrap().measured().len(), 3);
    world.finish();
}

/// Runs a call of the resume file itself, to act on a state file as a stop would leave it.
fn file_call<T: Send + 'static>(
    start: impl FnOnce(ResumeThen<T>) -> Result<(), ResumeError>,
) -> Result<T, ResumeError> {
    let handoff = Handoff::new();
    let sender = Arc::clone(&handoff);
    start(Box::new(move |result| {
        sender.send(result);
        Box::new(DummyFrame::INSTANCE)
    }))?;
    handoff.wait(LONGEST).expect("the continuation never ran")
}

fn open_file(world: &World, path: &Path) -> ResumeFile {
    file_call(|then| ResumeFile::open(&world.io, &world.host, path, then)).unwrap()
}

/// Bytes of a record that never reached the header are not kept, and the next record is written over them.
#[test]
fn a_record_that_was_being_written_when_the_run_stopped_is_not_kept() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("search.resume");
    let world = World::start();
    let (state, _) = add(
        world.open(&path, "fingerprint", 1, 1).unwrap(),
        measured(nvfp4(), 0.1, 0),
    );
    drop(state);
    let file = open_file(&world, &path);
    let end = fs::metadata(&path).unwrap().len();
    file_call(|then| file.write(end, vec![9, 0, 0, 0, b'{', b'"', b'a'], then)).unwrap();
    drop(file);

    let state = world.open(&path, "fingerprint", 1, 1).unwrap();
    assert_eq!(state.measured().len(), 1);
    let (state, added) = add(state, measured(mixed(), 0.2, 0));
    assert!(added);
    drop(state);
    assert_eq!(world.open(&path, "fingerprint", 1, 1).unwrap().measured().len(), 2);
    world.finish();
}

#[test]
fn a_state_of_another_search_is_refused_and_left_alone() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("search.resume");
    let world = World::start();
    let (state, _) = add(world.open(&path, "one", 1, 1).unwrap(), measured(nvfp4(), 0.1, 0));
    drop(state);
    let before = fs::read(&path).unwrap();
    let failure = world.open(&path, "two", 1, 1).unwrap_err();
    assert!(matches!(failure, SearchError::DifferentSearch { .. }), "{failure}");
    assert!(failure.to_string().contains("different search"), "{failure}");
    assert_eq!(fs::read(&path).unwrap(), before);
    world.finish();
}

/// Another search reads what one measured whatever its fingerprint, without taking the file over; a file that is
/// another run's resume file is refused.
#[test]
fn another_search_reads_what_one_measured_whatever_its_fingerprint() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("other.resume");
    let world = World::start();
    let state = world.open(&path, "a search at 32K", 3, 1 << 30).unwrap();
    let (state, _) = add(state, measured(nvfp4(), 0.1, 0));
    let wide = allocation(QuantOption::Trellis2_5, QuantOption::Trellis4_0);
    let (state, _) = add(state, measured(wide.clone(), 0.2, 4));
    drop(state);

    let read = world.read(&path).unwrap();
    assert_eq!(read.len(), 2);
    assert_eq!((read[1].kl, read[1].round), (0.2, 4));
    assert_eq!(read[1].allocation, wide);
    assert_eq!(
        world
            .open(&path, "a search at 32K", 3, 1 << 30)
            .unwrap()
            .measured()
            .len(),
        2
    );

    let foreign = directory.path().join("foreign.resume");
    let file = open_file(&world, &foreign);
    let mut header = Header::new();
    header.set("kind", "calibrate");
    file_call(|then| file.set_header(&header, then)).unwrap();
    drop(file);
    let failure = world.read(&foreign).unwrap_err();
    assert!(matches!(failure, SearchError::NotSearchState { .. }), "{failure}");
    world.finish();
}

/// Every double comes back as the bits that were written, including the values JSON has no number for.
#[test]
fn doubles_come_back_bit_for_bit() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("search.resume");
    let world = World::start();
    let values = [
        0.1,
        1e-300,
        -0.0,
        f64::MAX,
        f64::MIN_POSITIVE,
        1.0 / 3.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let mut state = world.open(&path, "fingerprint", -5, 7).unwrap();
    for (round, value) in values.iter().enumerate() {
        let mut candidate = measured(Allocation::default(), *value, round as u32);
        candidate.allocation = Allocation::new(BTreeMap::from([(format!("object {round}"), QuantOption::Nvfp4)]));
        candidate.kl_se = *value;
        candidate.prefill = *value;
        candidate.decode = *value;
        state = add(state, candidate).0;
    }
    let progress = Progress {
        round: 3,
        explore: 1e-7,
        stale: 0,
        held_out: f64::NAN,
    };
    let state = call(|then| state.set_progress(progress, then)).unwrap();
    drop(state);

    let state = world.open(&path, "fingerprint", 0, 0).unwrap();
    assert_eq!(state.seed(), -5);
    for (each, value) in state.measured().iter().zip(values) {
        for read in [each.kl, each.kl_se, each.prefill, each.decode] {
            assert_eq!(read.to_bits(), value.to_bits(), "{value}");
        }
    }
    let read = state.progress();
    assert_eq!((read.round, read.explore.to_bits()), (3, 1e-7f64.to_bits()));
    assert!(read.held_out.is_nan());
    world.finish();
}

/// A new state has no progress yet: the errors are NaN and the counts zero.
#[test]
fn a_new_state_has_no_progress() {
    let directory = tempfile::tempdir().unwrap();
    let world = World::start();
    let state = world.open(&directory.path().join("search.resume"), "f", 1, 1).unwrap();
    let progress = state.progress();
    assert_eq!((progress.round, progress.stale), (0, 0));
    assert!(progress.explore.is_nan() && progress.held_out.is_nan());
    assert!(state.measured().is_empty());
    drop(state);
    let state = world.open(&directory.path().join("search.resume"), "f", 2, 2).unwrap();
    let progress = state.progress();
    assert_eq!((progress.round, progress.stale), (0, 0));
    assert!(progress.explore.is_nan() && progress.held_out.is_nan());
    assert_eq!(state.seed(), 1);
    world.finish();
}

/// A header whose records end where the file's records do not is refused, not read as far as it goes.
#[test]
fn a_header_that_covers_more_than_the_records_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("search.resume");
    let world = World::start();
    let (state, _) = add(world.open(&path, "f", 1, 1).unwrap(), measured(nvfp4(), 0.1, 0));
    drop(state);
    let file = open_file(&world, &path);
    let mut header = file_call(|then| file.header(then)).unwrap();
    let end: u64 = header.get("end").unwrap().parse().unwrap();
    header.set("end", (end + 1).to_string());
    file_call(|then| file.set_header(&header, then)).unwrap();
    drop(file);
    let failure = world.open(&path, "f", 1, 1).unwrap_err();
    assert!(matches!(failure, SearchError::Damaged(_)), "{failure}");

    let file = open_file(&world, &path);
    header.set("end", (end - 2).to_string());
    file_call(|then| file.set_header(&header, then)).unwrap();
    drop(file);
    assert!(matches!(
        world.open(&path, "f", 1, 1).unwrap_err(),
        SearchError::Damaged(_)
    ));
    world.finish();
}

/// The header is a fixed number of bytes however many candidates there are, so the records never outgrow it.
#[test]
fn the_header_does_not_grow_with_the_candidates() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("search.resume");
    let world = World::start();
    let mut state = world.open(&path, "f", 1, 1).unwrap();
    for round in 0..300u32 {
        let choices = BTreeMap::from([(format!("text/layers/{round}/mlp/down"), QuantOption::Trellis3_5)]);
        state = add(state, measured(Allocation::new(choices), 0.5, round)).0;
    }
    drop(state);
    let file = open_file(&world, &path);
    let header = file_call(|then| file.header(then)).unwrap();
    assert!(header.text_len() < 300);
    assert_eq!(header.get("count"), Some("300"));
    assert!(fs::metadata(&path).unwrap().len() > HEADER + 300 * 50);
    drop(file);
    assert_eq!(world.open(&path, "f", 1, 1).unwrap().measured().len(), 300);
    world.finish();
}

/// The import of a path that does not exist is an error and creates nothing; a file that is only readable imports.
#[test]
fn reading_a_missing_path_creates_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("mistyped.resume");
    let world = World::start();
    let failure = world.read(&missing).unwrap_err();
    match failure {
        SearchError::Resume(ResumeError::Io(error)) => assert_eq!(error.kind(), ErrorKind::NotFound),
        other => panic!("{other}"),
    }
    assert!(!missing.exists());

    let path = directory.path().join("search.resume");
    let (state, _) = add(world.open(&path, "f", 1, 1).unwrap(), measured(nvfp4(), 0.1, 0));
    drop(state);
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&path, permissions).unwrap();
    assert_eq!(world.read(&path).unwrap().len(), 1);
    world.finish();
}

/// A path mistyped onto a small file that is not a state is refused and the file is untouched; an empty file is a new
/// state.
#[test]
fn a_short_file_that_is_not_a_state_is_not_overwritten() {
    let directory = tempfile::tempdir().unwrap();
    let small = directory.path().join("notes.txt");
    fs::write(&small, b"keep me").unwrap();
    let empty = directory.path().join("empty.resume");
    fs::write(&empty, b"").unwrap();
    let world = World::start();
    let failure = world.open(&small, "f", 1, 1).unwrap_err();
    assert!(matches!(failure, SearchError::NotSearchState { .. }), "{failure}");
    assert_eq!(fs::read(&small).unwrap(), b"keep me");
    let state = world.open(&empty, "f", 1, 1).unwrap();
    assert!(state.measured().is_empty());
    world.finish();
}

/// The names in a record on disk are fixed: a record written as literal text reads back, so a renamed key fails.
#[test]
fn a_record_in_the_documented_text_reads() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("search.resume");
    let world = World::start();
    drop(world.open(&path, "f", 1, 1).unwrap());
    let text = br#"{"objects":{"text/layers/0/mlp/down":"trellis-3.5"},"kl":0.25,"klSe":0.5,"bytes":99,"prefill":1.5,"decode":2.5,"fits":false,"round":7,"origin":"explore:move"}"#;
    let mut record = (text.len() as u32).to_le_bytes().to_vec();
    record.extend_from_slice(text);
    let file = open_file(&world, &path);
    let end = file_call(|then| file.write(HEADER, record, then)).unwrap();
    let mut header = file_call(|then| file.header(then)).unwrap();
    header.set("count", "1");
    header.set("end", end.to_string());
    file_call(|then| file.set_header(&header, then)).unwrap();
    drop(file);

    let read = world.read(&path).unwrap();
    let expected = Measured {
        allocation: Allocation::new(BTreeMap::from([(
            "text/layers/0/mlp/down".to_string(),
            QuantOption::Trellis3_5,
        )])),
        kl: 0.25,
        kl_se: 0.5,
        bytes: 99,
        prefill: 1.5,
        decode: 2.5,
        fits: false,
        round: 7,
        origin: "explore:move".to_string(),
    };
    assert_eq!(read, vec![expected]);
    world.finish();
}

/// What a stop during the first header write can leave, a file of zeros no longer than a header, is a new state; the
/// same zeros under bytes past the header are some other file and are kept.
#[test]
fn a_torn_first_header_is_a_new_state_and_a_big_file_of_zeros_is_not() {
    let directory = tempfile::tempdir().unwrap();
    let torn = directory.path().join("torn.resume");
    fs::write(&torn, vec![0u8; 1000]).unwrap();
    let full = directory.path().join("full.resume");
    fs::write(&full, vec![0u8; HEADER as usize]).unwrap();
    let mut data = vec![0u8; HEADER as usize + 10];
    data[HEADER as usize + 3] = 7;
    let big = directory.path().join("big.resume");
    fs::write(&big, &data).unwrap();
    let marked = directory.path().join("marked.resume");
    fs::write(&marked, b"\0\0not zero").unwrap();
    let world = World::start();
    for path in [&torn, &full] {
        assert!(world.open(path, "f", 1, 1).unwrap().measured().is_empty());
    }
    for path in [&big, &marked] {
        let failure = world.open(path, "f", 1, 1).unwrap_err();
        assert!(matches!(failure, SearchError::NotSearchState { .. }), "{failure}");
    }
    assert_eq!(fs::read(&big).unwrap(), data);
    assert_eq!(fs::read(&marked).unwrap(), b"\0\0not zero");
    world.finish();
}
