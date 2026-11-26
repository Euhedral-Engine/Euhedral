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
use std::ops::Range;
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
use euhedral_inference_quantization::calibrate::files::{
    Region, StatesFile, read_states, resume, save, write_checkpoint,
};
use euhedral_inference_quantization::calibrate::manifest::{Ledger, Manifest, inspect};
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
    header: Arc<Mutex<Header>>,
    bytes: Arc<Mutex<Vec<u8>>>,
    fail_write: Option<usize>,
}

impl Fake {
    fn new(bytes: Vec<u8>, fail_write: Option<usize>) -> Fake {
        Fake {
            calls: Arc::new(Mutex::new(Vec::new())),
            header: Arc::new(Mutex::new(Header::new())),
            bytes: Arc::new(Mutex::new(bytes)),
            fail_write,
        }
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }
}

impl StatesFile for Fake {
    fn header(&self, then: Then<Header>) -> Result<(), ResumeError> {
        let header = self.header.lock().unwrap().clone();
        drop(then(Ok(header)));
        Ok(())
    }

    fn set_header(&self, header: &Header, then: Then<()>) -> Result<(), ResumeError> {
        let layers = header.get("layers").map(|layers| layers.parse().unwrap());
        self.calls.lock().unwrap().push(Call::Header(layers));
        *self.header.lock().unwrap() = header.clone();
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

/// Where a run starts, from `fake` and `manifest`, over `states`.
fn resumed(fake: &Fake, manifest: Manifest, states: &mut [u8]) -> Result<usize, CalibrateError> {
    let into = region(states);
    let (sender, receiver) = channel();
    resume(
        fake,
        manifest,
        64,
        into,
        4,
        Box::new(move |at| sender.send(at).unwrap()),
    );
    receiver.recv_timeout(HANG).expect("the resume never ended")
}

const FIELDS: [&str; 1] = ["the key"];

fn touch(directory: &Path, names: &[String]) {
    for name in names {
        fs::write(directory.join(name), name.as_bytes()).unwrap();
    }
}

/// The files of layers `layers`, as a run names them.
fn layer_files(layers: Range<usize>) -> Vec<String> {
    layers
        .map(|layer| format!("text.layers.{layer}.mlp.down.npz"))
        .collect()
}

/// A directory of run `key` after 9 layers, with its checkpoint after 8 sealed and listed, and the resume file's
/// states after those 8 layers in a fake.
fn after_nine_layers(key: &str) -> (tempfile::TempDir, Fake) {
    let directory = tempfile::tempdir().unwrap();
    let mut manifest = Manifest::new(key, 64).after(9);
    manifest.files = layer_files(0..9);
    manifest.files.push("hidden.bin".to_owned());
    touch(directory.path(), &manifest.files);
    fs::write(directory.path().join("calibration.json"), manifest.to_bytes().unwrap()).unwrap();
    let fake = Fake::new(vec![0; 10], None);
    let mut states: Vec<u8> = (1..=10).collect();
    let from = region(&mut states);
    outcome(|then| write_checkpoint(&fake, key, from, 4, 8, then)).unwrap();
    (directory, fake)
}

#[test]
fn the_same_runs_directory_resumes_at_its_checkpoint_and_reads_the_states() {
    let (directory, fake) = after_nine_layers("A");
    let manifest = inspect(directory.path(), "A", 64, &FIELDS, false)
        .unwrap()
        .apply()
        .unwrap();
    let mut states = vec![0u8; 10];
    assert_eq!(resumed(&fake, manifest, &mut states).unwrap(), 8);
    assert_eq!(states, (1..=10).collect::<Vec<u8>>());
}

/// Run A reaches layer 9 with its checkpoint after 8; `--fresh` with key B removes A's files, and B rounds 4 layers
/// with no checkpoint; `--fresh` back to A must start at layer 0, though the resume file left beside B's manifest
/// (here: A's states after 8 layers, the worst it can hold) names A, and must never count layer 8 or 9.
#[test]
fn fresh_back_to_a_run_whose_states_survived_starts_at_layer_0_and_counts_no_later_layer() {
    let (directory, fake) = after_nine_layers("A");
    inspect(directory.path(), "B", 64, &FIELDS, true)
        .unwrap()
        .apply()
        .unwrap();
    assert!(
        !directory.path().join("hidden.bin").exists(),
        "--fresh removed the listed resume file"
    );
    let mut b = Manifest::new("B", 64).after(4);
    b.files = layer_files(0..4);
    touch(directory.path(), &b.files);
    touch(directory.path(), &["hidden.bin".to_owned()]);
    fs::write(directory.path().join("calibration.json"), b.to_bytes().unwrap()).unwrap();
    let manifest = inspect(directory.path(), "A", 64, &FIELDS, true)
        .unwrap()
        .apply()
        .unwrap();
    assert_eq!(manifest, Manifest::new("A", 64));
    let mut states = vec![0u8; 10];
    assert_eq!(resumed(&fake, manifest.clone(), &mut states).unwrap(), 0);
    assert_eq!(states, vec![0u8; 10], "the states are left for the tokens");
    let ledger = Arc::new(Ledger::new(manifest));
    for layer in [8, 9] {
        let (sender, receiver) = channel();
        ledger.record(
            layer,
            Ok(()),
            |_bytes, _saved| panic!("a layer past the first uncounted one was saved"),
            Box::new(move |result| sender.send(result).unwrap()),
        );
        assert!(
            receiver.recv_timeout(HANG).unwrap().is_err(),
            "layer {layer} was counted"
        );
    }
    assert_eq!(ledger.manifest().unwrap().layers_done, 0);
}

#[test]
fn a_checkpoint_torn_before_its_header_named_the_layers_resumes_from_the_start() {
    let fake = Fake::new(vec![0; 10], Some(2));
    let mut written = vec![5u8; 10];
    let from = region(&mut written);
    outcome(|then| write_checkpoint(&fake, "run", from, 4, 8, then)).unwrap_err();
    let mut manifest = Manifest::new("run", 64).after(9);
    manifest.files.push("hidden.bin".to_owned());
    let mut states = vec![0u8; 10];
    assert_eq!(resumed(&fake, manifest, &mut states).unwrap(), 0);
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

/// A trellis run lists its heads' state files as `heads/<name>`: the one directory inside the output a listed name may
/// be in.
#[test]
fn a_listed_name_may_be_inside_the_heads_directory_and_nowhere_deeper_or_outside() {
    let listing = |names: &[&str]| {
        let mut manifest = Manifest::new("A", 64);
        manifest.files = names.iter().map(|name| name.to_string()).collect();
        Manifest::parse(&manifest.to_bytes().unwrap())
    };
    assert!(listing(&["heads/text__output_head.npz", "layer.safetensors"]).is_ok());
    for refused in [
        "heads/deeper/x.npz",
        "other/x.npz",
        "heads/",
        "heads/..",
        "heads//x",
        "../heads/x.npz",
    ] {
        assert!(listing(&[refused]).is_err(), "{refused} was accepted");
    }
}

/// `--fresh` removes the files of the heads directory the manifest lists and no other file in it; the directory is
/// made again by `apply` for the run that follows.
#[test]
fn fresh_removes_only_the_listed_files_of_the_heads_directory_and_apply_makes_it() {
    let directory = tempfile::tempdir().unwrap();
    let heads = directory.path().join("heads");
    fs::create_dir(&heads).unwrap();
    fs::write(heads.join("listed.npz"), b"listed").unwrap();
    fs::write(heads.join("note.txt"), b"mine").unwrap();
    let mut manifest = Manifest::new("A", 64);
    manifest.files = vec!["heads/listed.npz".to_owned()];
    fs::write(directory.path().join("calibration.json"), manifest.to_bytes().unwrap()).unwrap();
    let kept = inspect(directory.path(), "A", 64, &FIELDS, true).unwrap();
    assert!(heads.join("listed.npz").exists(), "inspect changes nothing");
    kept.creating("heads").apply().unwrap();
    assert!(!heads.join("listed.npz").exists());
    assert!(heads.join("note.txt").exists());

    let fresh = tempfile::tempdir().unwrap();
    inspect(fresh.path(), "A", 64, &FIELDS, false)
        .unwrap()
        .creating("heads")
        .apply()
        .unwrap();
    assert!(fresh.path().join("heads").is_dir());
}

/// A resumed directory whose listed heads file is gone is refused, naming it.
#[test]
fn a_listed_heads_file_that_is_gone_refuses_the_resume() {
    let directory = tempfile::tempdir().unwrap();
    let mut manifest = Manifest::new("A", 64);
    manifest.files = vec!["heads/gone.npz".to_owned()];
    fs::write(directory.path().join("calibration.json"), manifest.to_bytes().unwrap()).unwrap();
    let why = inspect(directory.path(), "A", 64, &FIELDS, false)
        .unwrap_err()
        .to_string();
    assert!(why.contains("heads/gone.npz"), "{why}");
}

/// A trellis directory whose manifest is complete (the MTP block and the heads sealed, the heads in their directory) is
/// finished: the run does nothing, and applying it, as the command does not, would change none of its files.
#[test]
fn a_complete_trellis_directory_is_finished_and_its_files_stay() {
    let directory = tempfile::tempdir().unwrap();
    fs::create_dir(directory.path().join("heads")).unwrap();
    fs::write(directory.path().join("heads/text__output_head.npz"), b"head").unwrap();
    fs::write(directory.path().join("mtp.safetensors"), b"mtp").unwrap();
    let mut manifest = Manifest::new("A", 2);
    manifest.layers_done = 2;
    manifest.mtp_done = true;
    manifest.heads_done = true;
    manifest.files = vec!["mtp.safetensors".to_owned(), "heads/text__output_head.npz".to_owned()];
    fs::write(directory.path().join("calibration.json"), manifest.to_bytes().unwrap()).unwrap();
    let before = fs::read(directory.path().join("calibration.json")).unwrap();
    let prepared = inspect(directory.path(), "A", 2, &FIELDS, false).unwrap();
    assert!(prepared.finished());
    prepared.creating("heads").apply().unwrap();
    assert_eq!(fs::read(directory.path().join("calibration.json")).unwrap(), before);
    assert!(directory.path().join("heads/text__output_head.npz").exists());
}
