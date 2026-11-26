//! The calibration's files: a file saved whole, and the resume states written and read in chunks, over a fake of the
//! resume file for the order of the calls and over a real lattice for the files.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "no-wait, waiting call: a test thread waits on the hand-off for the frames it started; the fake records in a mutex"
)]

use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::DummyFrame;
use euhedral_core::generics::LatticeSource;
use euhedral_graph::Lake;
use euhedral_inference_engine::lattice_config;
use euhedral_inference_quantization::calibrate::CalibrateError;
use euhedral_inference_quantization::calibrate::files::{Region, StatesFile, read_states, save, write_checkpoint};
use euhedral_inference_quantization::calibrate::plan::resume_at;
use euhedral_inference_quantization::resume::{HEADER, Header, ResumeError, Then};
use euhedral_inference_settings::{ModelSettings, Settings};
use euhedral_io::AsyncIo;

const HANG: Duration = Duration::from_secs(10);

/// What the fake resume file was asked.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Call {
    Header(Option<usize>),
    Write(u64, usize),
    Read(u64, usize),
}

/// A resume file in memory: it records the calls, answers at once and fails the write number `fail_write` on request.
#[derive(Clone)]
struct Fake {
    calls: Arc<Mutex<Vec<Call>>>,
    bytes: Arc<Mutex<Vec<u8>>>,
    fail_write: Option<usize>,
}

impl Fake {
    fn new(bytes: Vec<u8>, fail_write: Option<usize>) -> Fake {
        Fake {
            calls: Arc::new(Mutex::new(Vec::new())),
            bytes: Arc::new(Mutex::new(bytes)),
            fail_write,
        }
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }
}

impl StatesFile for Fake {
    fn set_header(&self, header: &Header, then: Then<()>) -> Result<(), ResumeError> {
        let layers = header.get("layers").map(|layers| layers.parse().unwrap());
        self.calls.lock().unwrap().push(Call::Header(layers));
        drop(then(Ok(())));
        Ok(())
    }

    fn write(&self, offset: u64, bytes: Vec<u8>, then: Then<u64>) -> Result<(), ResumeError> {
        let mut calls = self.calls.lock().unwrap();
        calls.push(Call::Write(offset, bytes.len()));
        let writes = calls.iter().filter(|call| matches!(call, Call::Write(..))).count();
        drop(calls);
        if self.fail_write == Some(writes) {
            drop(then(Err(ResumeError::Busy)));
            return Ok(());
        }
        let mut held = self.bytes.lock().unwrap();
        let at = (offset - HEADER) as usize;
        held[at..at + bytes.len()].copy_from_slice(&bytes);
        drop(held);
        drop(then(Ok(offset + bytes.len() as u64)));
        Ok(())
    }

    fn read(&self, offset: u64, len: usize, then: Then<Vec<u8>>) -> Result<(), ResumeError> {
        self.calls.lock().unwrap().push(Call::Read(offset, len));
        let at = (offset - HEADER) as usize;
        let held = self.bytes.lock().unwrap();
        let end = (at + len).min(held.len());
        let bytes = held[at.min(end)..end].to_vec();
        drop(held);
        drop(then(Ok(bytes)));
        Ok(())
    }
}

/// The result of a call that ends through a `Then<()>`.
fn outcome(start: impl FnOnce(Box<dyn FnOnce(Result<(), CalibrateError>) + Send>)) -> Result<(), CalibrateError> {
    let (sender, receiver) = channel();
    start(Box::new(move |result| sender.send(result).unwrap()));
    receiver.recv_timeout(HANG).expect("the call never ended")
}

fn region(buffer: &mut [u8]) -> Region {
    // SAFETY: the buffer outlives the call, which the test waits for, and nothing else touches it meanwhile.
    unsafe { Region::new(buffer.as_mut_ptr(), buffer.len()) }
}

#[test]
fn a_checkpoint_names_no_layers_first_then_writes_every_chunk_then_names_the_layers_done() {
    let fake = Fake::new(vec![0; 10], None);
    let mut states: Vec<u8> = (1..=10).collect();
    let region = region(&mut states);
    outcome(|then| write_checkpoint(&fake, "run", region, 4, 9, then)).unwrap();
    assert_eq!(
        fake.calls(),
        vec![
            Call::Header(Some(0)),
            Call::Write(HEADER, 4),
            Call::Write(HEADER + 4, 4),
            Call::Write(HEADER + 8, 2),
            Call::Header(Some(9)),
        ]
    );
    assert_eq!(*fake.bytes.lock().unwrap(), (1..=10).collect::<Vec<u8>>());
}

#[test]
fn a_checkpoint_that_fails_in_a_chunk_leaves_the_header_naming_no_layers() {
    let fake = Fake::new(vec![0; 10], Some(2));
    let mut states = vec![5u8; 10];
    let region = region(&mut states);
    let error = outcome(|then| write_checkpoint(&fake, "run", region, 4, 9, then)).unwrap_err();
    assert!(error.to_string().contains("resume states"), "{error}");
    let calls = fake.calls();
    assert_eq!(calls.first(), Some(&Call::Header(Some(0))));
    assert!(
        !calls.contains(&Call::Header(Some(9))),
        "the header never named the layers: {calls:?}"
    );
    assert_eq!(calls.len(), 3, "no chunk is written after the failure");
}

#[test]
fn a_header_that_names_no_layers_resumes_from_the_start_and_the_valid_one_from_its_layers() {
    let mut invalid = Header::new();
    invalid.set("key", "run");
    invalid.set("layers", "0");
    assert_eq!(resume_at(&invalid, "run", 64), 0);
    let mut valid = Header::new();
    valid.set("key", "run");
    valid.set("layers", "9");
    assert_eq!(resume_at(&valid, "run", 64), 9);
}

#[test]
fn the_states_are_read_in_chunks_into_the_region() {
    let fake = Fake::new((1..=10).collect(), None);
    let mut states = vec![0u8; 10];
    let into = region(&mut states);
    outcome(|then| read_states(&fake, into, 4, then)).unwrap();
    assert_eq!(states, (1..=10).collect::<Vec<u8>>());
    assert_eq!(
        fake.calls(),
        vec![
            Call::Read(HEADER, 4),
            Call::Read(HEADER + 4, 4),
            Call::Read(HEADER + 8, 2)
        ]
    );
}

#[test]
fn states_the_file_holds_too_few_bytes_of_fail_the_read() {
    let fake = Fake::new((1..=6).collect(), None);
    let mut states = vec![0u8; 10];
    let into = region(&mut states);
    let error = outcome(|then| read_states(&fake, into, 4, then)).unwrap_err();
    assert!(error.to_string().contains("hold"), "{error}");
}

// ---- save, on a real lattice ----

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

    fn save(&self, path: &Path, bytes: Vec<u8>) -> Result<(), CalibrateError> {
        let (sender, receiver) = channel();
        save(
            &self.io,
            &self.host,
            path.to_path_buf(),
            bytes,
            Box::new(move |result| {
                sender.send(result).unwrap();
                Box::new(DummyFrame::INSTANCE)
            }),
        );
        receiver.recv_timeout(HANG).expect("the save never ended")
    }

    fn finish(self) {
        self.host.complete_gracefully();
        self.io.complete();
        self.lattice.close();
    }
}

#[test]
fn a_save_writes_the_file_whole_and_leaves_no_temporary_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("layer.npz");
    let world = World::start();
    world.save(&path, vec![7; 5000]).unwrap();
    assert_eq!(fs::read(&path).unwrap(), vec![7; 5000]);
    assert_eq!(
        fs::read_dir(directory.path()).unwrap().count(),
        1,
        "only the sealed file"
    );
    world.finish();
}

#[test]
fn a_temporary_file_a_stop_left_is_not_the_file_and_a_new_save_replaces_it_whole() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("layer.npz");
    let torn = directory.path().join("layer.npz.tmp");
    fs::write(&torn, vec![9; 20_000]).unwrap();
    assert!(!path.exists(), "a torn temporary file is not the file");
    let world = World::start();
    world.save(&path, vec![3; 100]).unwrap();
    assert_eq!(
        fs::read(&path).unwrap(),
        vec![3; 100],
        "no byte of the torn file is in the sealed one"
    );
    assert!(!torn.exists());
    world.finish();
}

#[test]
fn a_save_that_cannot_open_fails_and_leaves_no_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("missing").join("layer.npz");
    let world = World::start();
    let error = world.save(&path, vec![1; 10]).unwrap_err();
    assert!(error.to_string().contains("open"), "{error}");
    assert!(!path.exists());
    world.finish();
}
