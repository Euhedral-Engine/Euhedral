//! The resume file on a real lattice, without a device: its calls start work and hand their results to continuations.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting call: a test thread waits on the hand-off for the frames it started, and reads its files"
)]

use std::fs;
use std::io;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{DummyFrame, Frame};
use euhedral_core::generics::LatticeSource;
use euhedral_graph::Lake;
use euhedral_inference_engine::{Handoff, lattice_config};
use euhedral_inference_quantization::resume::{HEADER, Header, ResumeError, ResumeFile, Then};
use euhedral_inference_settings::{ModelSettings, Settings};
use euhedral_io::AsyncIo;

/// The wait for a call whose hang is the likely failure: long enough for a loaded machine, short enough to fail fast.
const HANG: Duration = Duration::from_secs(10);

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

    fn open(&self, path: &Path) -> ResumeFile {
        call(|then| ResumeFile::open(&self.io, &self.host, path, then)).expect("the file opens")
    }

    fn finish(self) {
        self.host.complete_gracefully();
        self.io.complete();
        self.lattice.close();
    }
}

/// Starts a call and waits, outside the lattice, for the value its continuation is given.
fn call<T: Send + 'static>(start: impl FnOnce(Then<T>) -> Result<(), ResumeError>) -> Result<T, ResumeError> {
    let handoff = Handoff::new();
    let sender = Arc::clone(&handoff);
    start(Box::new(move |result| {
        sender.send(result);
        Box::new(DummyFrame::INSTANCE)
    }))?;
    handoff.wait(HANG).expect("the continuation never ran")
}

fn step_header(step: u64) -> Header {
    let mut header = Header::new();
    header.set("step", step.to_string());
    header
}

fn longs(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// A section written at the end of a run is what a run that opens the file again reads, and the file stays one file of
/// the size of the last write.
#[test]
fn overwriting_in_place_keeps_one_file_and_the_last_state() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("resume.bin");
    let world = World::start();
    for values in [[1i64, 2, 3], [7, 8, 9]] {
        let file = world.open(&path);
        call(|then| file.set_header(&Header::new(), then)).unwrap();
        let end = call(|then| file.write(HEADER, longs(&values), then)).unwrap();
        assert_eq!(end, HEADER + 24);
        call(|then| file.set_header(&step_header(values[0] as u64), then)).unwrap();
    }
    let file = world.open(&path);
    let header = call(|then| file.header(then)).unwrap();
    assert_eq!(header.get("step"), Some("7"));
    let section = call(|then| file.read(HEADER, 24, then)).unwrap();
    assert_eq!(section, longs(&[7, 8, 9]));
    assert_eq!(fs::metadata(&path).unwrap().len(), HEADER + 24);
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    drop(file);
    world.finish();
}

#[test]
fn a_new_file_has_an_empty_header() {
    let directory = tempfile::tempdir().unwrap();
    let world = World::start();
    let file = world.open(&directory.path().join("resume.bin"));
    assert!(call(|then| file.header(then)).unwrap().is_empty());
    drop(file);
    world.finish();
}

/// A header that does not fit is refused before anything is written, so the state on disk is still whole: the file's
/// bytes are what they were.
#[test]
fn a_header_too_large_is_refused_and_the_file_is_unchanged() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("resume.bin");
    let world = World::start();
    let file = world.open(&path);
    call(|then| file.set_header(&step_header(3), then)).unwrap();
    call(|then| file.write(HEADER, longs(&[5, 6]), then)).unwrap();
    let before = fs::read(&path).unwrap();

    let mut oversize = step_header(4);
    oversize.set("ids", (0..1200).map(|id| id.to_string()).collect::<Vec<_>>().join(","));
    assert!(oversize.text_len() > HEADER as usize);
    assert!(matches!(oversize.check(), Err(ResumeError::HeaderTooLarge { .. })));
    let refused = call(|then| file.set_header(&oversize, then));
    assert!(
        matches!(refused, Err(ResumeError::HeaderTooLarge { .. })),
        "{refused:?}"
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    drop(file);
    world.finish();
}

/// A header whose text takes all 4096 bytes has no NUL to end it, and reads back whole.
#[test]
fn a_header_of_exactly_the_limit_is_written_and_read_whole() {
    let directory = tempfile::tempdir().unwrap();
    let world = World::start();
    let file = world.open(&directory.path().join("resume.bin"));
    let mut header = step_header(1);
    let filler = HEADER as usize - header.text_len() - "pad=\n".len();
    header.set("pad", "x".repeat(filler));
    assert_eq!(header.text_len(), HEADER as usize);
    call(|then| file.set_header(&header, then)).unwrap();
    assert_eq!(call(|then| file.header(then)).unwrap(), header);
    header.set("pad", "x".repeat(filler + 1));
    assert!(matches!(header.check(), Err(ResumeError::HeaderTooLarge { bytes }) if bytes == HEADER as usize + 1));
    drop(file);
    world.finish();
}

/// Sections larger than one I/O piece, and empty ones, cross the file intact.
#[test]
fn sections_of_any_size_round_trip() {
    let directory = tempfile::tempdir().unwrap();
    let world = World::start();
    let file = world.open(&directory.path().join("resume.bin"));
    let big: Vec<u8> = (0..5 * 1024 * 1024 + 123u32)
        .map(|index| (index.wrapping_mul(2654435761) >> 24) as u8)
        .collect();
    let end = call(|then| file.write(HEADER, big.clone(), then)).unwrap();
    assert_eq!(end, HEADER + big.len() as u64);
    assert_eq!(call(|then| file.write(end, Vec::new(), then)).unwrap(), end);
    assert!(call(|then| file.read(end, 0, then)).unwrap().is_empty());
    assert_eq!(call(|then| file.read(HEADER, big.len(), then)).unwrap(), big);
    drop(file);
    world.finish();
}

/// Reading past the end names the position where the file ends.
#[test]
fn reading_past_the_end_names_where_the_file_ends() {
    let directory = tempfile::tempdir().unwrap();
    let world = World::start();
    let file = world.open(&directory.path().join("resume.bin"));
    call(|then| file.write(0, vec![1; 100], then)).unwrap();
    let short = call(|then| file.read(60, 100, then));
    assert!(matches!(short, Err(ResumeError::ShortRead { at: 100 })), "{short:?}");
    drop(file);
    world.finish();
}

/// A file cut back to a length has that length, and the sections before it are untouched.
#[test]
fn truncating_cuts_the_file_back_and_keeps_what_precedes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("resume.bin");
    let world = World::start();
    let file = world.open(&path);
    call(|then| file.write(0, (0..200u8).collect(), then)).unwrap();
    call(|then| file.sync(then)).unwrap();
    call(|then| file.truncate(120, then)).unwrap();
    assert_eq!(fs::read(&path).unwrap(), (0..120u8).collect::<Vec<_>>());
    drop(file);
    world.finish();
}

/// A finished run removes its file; while another handle is open the removal is refused and the file stays. The call
/// that was just finished holds no handle, so the last handle always removes, however soon it asks.
#[test]
fn removing_needs_the_last_handle_and_deletes_the_file() {
    let directory = tempfile::tempdir().unwrap();
    let world = World::start();
    for round in 0..50 {
        let path = directory.path().join(format!("resume-{round}.bin"));
        let file = world.open(&path);
        let other = file.clone();
        call(|then| file.set_header(&step_header(1), then)).unwrap();
        let busy = call(|then| file.remove(then));
        assert!(matches!(busy, Err(ResumeError::Busy)), "{busy:?}");
        assert!(path.exists());
        call(|then| other.remove(then)).unwrap();
        assert!(!path.exists());
    }
    world.finish();
}

/// The continuation of a call may remove the file at once: the call holds no handle while it runs. The handle the call
/// was started from is dropped before the continuation goes on (it waits for a flag, and reports a flag that never
/// comes), so a refusal of the removal can only come from a handle the call itself kept.
#[test]
fn a_continuation_may_remove_the_file_it_was_started_from() {
    let directory = tempfile::tempdir().unwrap();
    let world = World::start();
    for round in 0..50 {
        let path = directory.path().join(format!("resume-{round}.bin"));
        let file = world.open(&path);
        let report: Arc<Handoff<Result<(), ResumeError>>> = Handoff::new();
        let gate = Arc::new(AtomicBool::new(false));
        let (handle, tell, open) = (file.clone(), Arc::clone(&report), Arc::clone(&gate));
        let continuation = move |synced: Result<(), ResumeError>| {
            let again = Arc::clone(&tell);
            let mut result = synced;
            let deadline = Instant::now() + Duration::from_secs(10);
            while result.is_ok() && !open.load(Ordering::SeqCst) {
                if Instant::now() > deadline {
                    result = Err(ResumeError::Io(io::ErrorKind::TimedOut.into()));
                }
                thread::yield_now();
            }
            match result {
                Ok(()) => {
                    let removal = handle.remove(Box::new(move |removed| {
                        again.send(removed);
                        Box::new(DummyFrame::INSTANCE)
                    }));
                    if let Err(error) = removal {
                        tell.send(Err(error));
                    }
                }
                Err(error) => tell.send(Err(error)),
            }
            Box::new(DummyFrame::INSTANCE) as Box<dyn Frame>
        };
        file.set_header(&step_header(2), Box::new(continuation)).unwrap();
        drop(file);
        gate.store(true, Ordering::SeqCst);
        let removed = report.wait(HANG).expect("the removal was never reported");
        assert!(removed.is_ok(), "{removed:?}");
        assert!(!path.exists());
    }
    world.finish();
}

/// A step that is refused after the call started reaches the continuation as an error: the I/O engine is closed, so
/// the write of the header is refused. The call itself started (the barrier before the header is a host call), so
/// the refusal is told to the continuation, not returned.
#[test]
fn a_refused_later_step_is_told_to_the_continuation() {
    let directory = tempfile::tempdir().unwrap();
    let world = World::start();
    let file = world.open(&directory.path().join("resume.bin"));
    world.io.complete();
    let report: Arc<Handoff<Result<(), ResumeError>>> = Handoff::new();
    let tell = Arc::clone(&report);
    let started = file.set_header(
        &step_header(1),
        Box::new(move |result| {
            tell.send(result);
            Box::new(DummyFrame::INSTANCE)
        }),
    );
    assert!(started.is_ok(), "{started:?}");
    let result = report.wait(HANG).expect("the refusal was never reported");
    assert!(matches!(result, Err(ResumeError::Io(_))), "{result:?}");
    drop(file);
    world.finish();
}

/// A read at or past the end names where the file ends or where the read began.
#[test]
fn a_read_at_or_past_the_end_names_the_position() {
    let directory = tempfile::tempdir().unwrap();
    let world = World::start();
    let file = world.open(&directory.path().join("resume.bin"));
    call(|then| file.write(0, vec![1; 100], then)).unwrap();
    for at in [100u64, 150] {
        let result = call(|then| file.read(at, 8, then));
        assert!(
            matches!(result, Err(ResumeError::ShortRead { at: found }) if found == at),
            "{at}: {result:?}"
        );
    }
    drop(file);
    world.finish();
}

/// A lake that no longer takes work refuses the call at once, so a caller learns of it instead of waiting.
#[test]
fn a_closed_host_lake_refuses_to_open_at_once() {
    let directory = tempfile::tempdir().unwrap();
    let world = World::start();
    world.host.complete_gracefully();
    let refused = ResumeFile::open(
        &world.io,
        &world.host,
        &directory.path().join("resume.bin"),
        Box::new(|_| Box::new(DummyFrame::INSTANCE)),
    );
    assert!(matches!(refused, Err(ResumeError::LakeClosed)), "{refused:?}");
    world.finish();
}

/// The forms a properties file has: comments, escapes, separators, continuation lines and raw UTF-8.
#[test]
fn the_reader_takes_the_forms_a_properties_file_has() {
    let text = "# written by an earlier run\n\
        ! another comment\n\
        \n\
        digest=aa\\=bb\\:cc\n\
        name=na\u{ef}ve \\\\ path\n\
        spaced   =   value with spaces  \n\
        colon:colon value\n\
        bare\n\
        joined=one \\\n    \ttwo\n\
        escapes=tab\\there\\nnew\\u00e9\\uD83D\\uDE00\n\
        \\ lead\\ key=\\ lead\n";
    let mut padded = text.as_bytes().to_vec();
    padded.resize(HEADER as usize, 0);
    let header = Header::decode(&padded).unwrap();
    let read = |key: &str| header.get(key).map(str::to_string);
    assert_eq!(read("digest").as_deref(), Some("aa=bb:cc"));
    assert_eq!(read("name").as_deref(), Some("na\u{ef}ve \\ path"));
    assert_eq!(read("spaced").as_deref(), Some("value with spaces  "));
    assert_eq!(read("colon").as_deref(), Some("colon value"));
    assert_eq!(read("bare").as_deref(), Some(""));
    assert_eq!(read("joined").as_deref(), Some("one two"));
    assert_eq!(read("escapes").as_deref(), Some("tab\there\nnew\u{e9}\u{1f600}"));
    assert_eq!(read(" lead key").as_deref(), Some(" lead"));
    assert_eq!(header.iter().count(), 8);
}

/// Whatever a header holds is read back as it was, and the written text never holds a NUL before its padding.
#[test]
fn awkward_text_round_trips_and_never_writes_a_nul() {
    let mut header = Header::new();
    for (key, value) in [
        ("plain", "value"),
        (" leading space", " leading space"),
        ("trailing ", "trailing "),
        ("sep=:#!", "sep=:#!"),
        ("tab\tnew\nline\rform\u{c}", "tab\tnew\nline\rform\u{c}"),
        ("nul", "before\0after"),
        ("unicode", "caf\u{e9} \u{1f600} \u{4e2d}"),
        ("slashes", "\\\\ \\u0041 \\"),
        ("empty", ""),
    ] {
        header.set(key, value);
    }
    let bytes = header.encode().unwrap();
    assert_eq!(bytes.len(), HEADER as usize);
    let text_end = header.text_len();
    assert!(bytes[..text_end].iter().all(|&byte| byte != 0));
    assert!(bytes[text_end..].iter().all(|&byte| byte == 0));
    assert_eq!(Header::decode(&bytes).unwrap(), header);
}

#[test]
fn a_malformed_unicode_escape_is_an_error_naming_it() {
    let bytes = b"key=\\u12G4\n";
    assert!(matches!(Header::decode(bytes), Err(ResumeError::HeaderText { .. })));
    assert!(matches!(
        Header::decode(b"key=\\u12"),
        Err(ResumeError::HeaderText { .. })
    ));
}

/// A call with nothing to wait for still ends on the host lake: its continuation does not run before the call returns.
#[test]
fn a_settled_call_ends_on_the_host_lake_not_inline() {
    let directory = tempfile::tempdir().unwrap();
    let world = World::start();
    let file = world.open(&directory.path().join("resume.bin"));
    let caller = thread::current().id();
    let inline = Arc::new(AtomicBool::new(false));
    let (flag, seen) = (Arc::clone(&inline), Arc::clone(&inline));
    let ran_on_caller = call(|then| {
        let started = file.settled(Box::new(move |result: Result<(), ResumeError>| {
            flag.store(thread::current().id() == caller, Ordering::SeqCst);
            then(result)
        }));
        // Returning here, before the continuation had to run, must not have run it on this thread.
        assert!(!seen.load(Ordering::SeqCst));
        started
    });
    ran_on_caller.unwrap();
    assert!(
        !inline.load(Ordering::SeqCst),
        "the continuation ran on the caller's thread"
    );
    drop(file);
    world.finish();
}

/// A file opened for reading only reads, reports its length, and cannot be written; a missing one is not created.
#[test]
fn a_read_only_open_reads_and_cannot_write() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("resume.bin");
    let world = World::start();
    let file = world.open(&path);
    call(|then| file.write(0, longs(&[4, 5, 6]), then)).unwrap();
    drop(file);

    let file = call(|then| ResumeFile::open_read_only(&world.io, &world.host, &path, then)).unwrap();
    assert_eq!(call(|then| file.len(then)).unwrap(), 24);
    assert_eq!(call(|then| file.read(8, 8, then)).unwrap(), longs(&[5]));
    let refused = call(|then| file.write(0, longs(&[9]), then));
    assert!(matches!(refused, Err(ResumeError::Io(_))), "{refused:?}");
    drop(file);
    assert_eq!(fs::read(&path).unwrap(), longs(&[4, 5, 6]));

    let missing = directory.path().join("missing.bin");
    let failure = call(|then| ResumeFile::open_read_only(&world.io, &world.host, &missing, then));
    assert!(matches!(failure, Err(ResumeError::Io(error)) if error.kind() == io::ErrorKind::NotFound));
    assert!(!missing.exists());
    world.finish();
}

#[test]
fn the_length_follows_the_writes() {
    let directory = tempfile::tempdir().unwrap();
    let world = World::start();
    let file = world.open(&directory.path().join("resume.bin"));
    assert_eq!(call(|then| file.len(then)).unwrap(), 0);
    call(|then| file.write(HEADER, longs(&[1, 2]), then)).unwrap();
    assert_eq!(call(|then| file.len(then)).unwrap(), HEADER + 16);
    drop(file);
    world.finish();
}
