//! The disk reader over a fake port: its reads end when a test says, its gates are the graph's own.

use std::collections::VecDeque;
use std::env::{temp_dir, var_os};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::ptr::copy_nonoverlapping;
use std::slice::from_raw_parts;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use euhedral_core::frames::FrameError;
use euhedral_core::generics::LatticeSource;
use euhedral_graph::testing::{FakeDevice, GateEvent, StageHarness};
use euhedral_graph::{GateRef, Stage, StageCx, StageError, Submitted};

use super::*;

const BLOCK: u64 = HOST_ALIGNMENT;
const FILE_BYTES: usize = 64 * 1024;
const SLOT_BYTES: u64 = 3 * BLOCK;
const BOUND: Duration = Duration::from_secs(10);

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn inline() -> Spawn {
    Box::new(|work| {
        work();
        true
    })
}

fn contents() -> Vec<u8> {
    (0..FILE_BYTES).map(|index| (index % 251) as u8).collect()
}

fn geometry(slots: usize) -> WindowGeometry {
    WindowGeometry {
        slots,
        slot_bytes: SLOT_BYTES,
    }
}

fn use_of(name: &str, file_offset: u64, bytes: u64, slot: usize) -> DiskUse {
    DiskUse {
        name: name.to_string(),
        file_offset,
        bytes,
        slot,
    }
}

/// Reads that wait for the test to end them, over a file of known bytes.
struct FakePort {
    file: Vec<u8>,
    file_len: u64,
    direct: bool,
    window: Arc<DiskWindow>,
    pending: Mutex<VecDeque<(ReadSpec, ReadDone)>>,
    started: Mutex<Vec<ReadSpec>>,
}

impl FakePort {
    fn new(window: Arc<DiskWindow>) -> Arc<FakePort> {
        Arc::new(FakePort {
            file: contents(),
            file_len: FILE_BYTES as u64,
            direct: true,
            window,
            pending: Mutex::new(VecDeque::new()),
            started: Mutex::new(Vec::new()),
        })
    }

    fn short(window: Arc<DiskWindow>, file_len: u64) -> Arc<FakePort> {
        Arc::new(FakePort {
            file: contents(),
            file_len,
            direct: true,
            window,
            pending: Mutex::new(VecDeque::new()),
            started: Mutex::new(Vec::new()),
        })
    }

    fn pending(&self) -> Vec<ReadSpec> {
        lock(&self.pending).iter().map(|(spec, _)| *spec).collect()
    }

    fn started(&self) -> Vec<ReadSpec> {
        lock(&self.started).clone()
    }

    /// Ends the oldest read of `slot` with its bytes in the window.
    fn finish(&self, slot: usize) {
        let (spec, done) = self.take(slot);
        let end = (spec.offset as usize + spec.len).min(self.file.len());
        let bytes = &self.file[spec.offset as usize..end];
        // SAFETY: the window is a live heap region of at least `slot_bytes` per slot, and the reader reads one use of a
        // slot at a time.
        unsafe {
            copy_nonoverlapping(bytes.as_ptr(), self.window.slot_ptr(spec.slot).cast_mut(), bytes.len());
        }
        done(Ok(bytes.len()));
    }

    /// Ends the oldest read of `slot` with `result`.
    fn finish_with(&self, slot: usize, result: io::Result<usize>) {
        let (_, done) = self.take(slot);
        done(result);
    }

    fn take(&self, slot: usize) -> (ReadSpec, ReadDone) {
        let mut pending = lock(&self.pending);
        let index = pending
            .iter()
            .position(|(spec, _)| spec.slot == slot)
            .expect("a read of the slot");
        pending.remove(index).expect("present")
    }
}

impl ReadPort for FakePort {
    fn start(&self, spec: ReadSpec, done: ReadDone) -> io::Result<()> {
        lock(&self.started).push(spec);
        lock(&self.pending).push_back((spec, done));
        Ok(())
    }

    fn has_direct(&self) -> bool {
        self.direct
    }

    fn file_len(&self) -> u64 {
        self.file_len
    }
}

/// The transfer stage of a test: claims `(slot, ticket)` in `pass_gate`.
struct Copy(DiskReader);

impl Stage<(usize, u64)> for Copy {
    fn run(&self, _: &(usize, u64), _: &StageCx<'_>) -> Result<Submitted, StageError> {
        Ok(Submitted::done())
    }

    fn pass_gate(&self, quantum: &(usize, u64), gate: GateRef<'_>) -> bool {
        self.0.claim(quantum.0, quantum.1, || gate.hold()) == Claimed::Ready
    }
}

struct Rig {
    reader: DiskReader,
    port: Arc<FakePort>,
    window: Arc<DiskWindow>,
    harness: StageHarness,
    stage: Copy,
    failures: Arc<Mutex<Vec<String>>>,
}

impl Rig {
    fn new(slots: usize) -> Rig {
        let window = Arc::new(DiskWindow::on_heap(geometry(slots)));
        Rig::over(FakePort::new(Arc::clone(&window)), window, slots)
    }

    fn over(port: Arc<FakePort>, window: Arc<DiskWindow>, slots: usize) -> Rig {
        let reader = DiskReader::new(Arc::clone(&port) as Arc<dyn ReadPort>, geometry(slots), inline());
        let (lanes, _) = FakeDevice::pool(&["lane"]);
        Rig {
            stage: Copy(reader.clone()),
            reader,
            port,
            window,
            harness: StageHarness::new(lanes),
            failures: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn fail_handle(&self) -> FailHandle {
        let failures = Arc::clone(&self.failures);
        Arc::new(move |why| lock(&failures).push(why))
    }

    fn enqueue(&self, ticket: u64, uses: Vec<DiskUse>) {
        self.reader.enqueue(ticket, uses, self.fail_handle()).expect("accepted");
    }

    /// The claim of the copy: `true` when the stage may be published at once.
    fn claim(&self, slot: usize, ticket: u64) -> bool {
        let passed = self.harness.pass_gate(&self.stage, &(slot, ticket));
        assert_eq!(passed.open, !passed.held, "a claim either opens or holds");
        passed.open
    }

    fn events(&self) -> Vec<GateEvent> {
        self.harness.gate_events()
    }

    fn payload(&self, slot: usize, lead: usize, len: usize) -> Vec<u8> {
        // SAFETY: a live window; the slot's read ended before the test looks.
        unsafe { from_raw_parts(self.window.slot_ptr(slot).add(lead), len).to_vec() }
    }
}

#[test]
fn a_use_reads_the_aligned_blocks_around_its_payload() {
    let unaligned = use_of("w", 5000, 100, 0);
    assert_eq!(unaligned.aligned_offset(), BLOCK);
    assert_eq!(unaligned.lead(), 5000 - BLOCK);
    assert_eq!(unaligned.read_bytes(), BLOCK);
    let straddling = use_of("w", 4090, 20, 0);
    assert_eq!((straddling.aligned_offset(), straddling.lead()), (0, 4090));
    assert_eq!(straddling.read_bytes(), 2 * BLOCK);
    let aligned = use_of("w", 2 * BLOCK, BLOCK, 0);
    assert_eq!((aligned.lead(), aligned.read_bytes()), (0, BLOCK));
}

#[test]
fn uses_fill_slots_in_order_and_wait_for_the_slot_to_be_freed() {
    let rig = Rig::new(2);
    rig.enqueue(
        1,
        vec![
            use_of("a", 0, 100, 0),
            use_of("b", BLOCK, 100, 1),
            use_of("c", 2 * BLOCK, 100, 0),
        ],
    );
    // The first two start; the third waits for slot 0.
    assert_eq!(
        rig.port.pending().iter().map(|spec| spec.slot).collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(rig.reader.reads_in_flight(), 2);

    // The copy that asks before its read arrived is held; the read's arrival releases it.
    assert!(!rig.claim(0, 1));
    assert!(rig.events().is_empty());
    rig.port.finish(0);
    assert_eq!(rig.events(), [GateEvent::Released]);
    assert_eq!(rig.payload(0, 0, 100), contents()[..100]);
    // The slot is the copy's until freed: the third use does not start.
    assert_eq!(rig.port.started().len(), 2);

    // A copy that asks after its read arrived owns the slot at once and takes no gate.
    rig.port.finish(1);
    assert!(rig.claim(1, 1));
    assert_eq!(rig.events().len(), 1);

    rig.reader.slot_freed(0);
    assert_eq!(rig.port.started().len(), 3);
    assert_eq!(rig.port.started()[2].offset, 2 * BLOCK);
    rig.port.finish(0);
    assert!(rig.claim(0, 1));
    assert_eq!(rig.payload(0, 0, 100), contents()[2 * BLOCK as usize..][..100]);
    assert_eq!(rig.reader.reads_in_flight(), 0);
}

#[test]
fn an_aligned_superset_covers_unaligned_payloads() {
    let rig = Rig::new(2);
    rig.enqueue(1, vec![use_of("a", 5000, 100, 0), use_of("b", 4090, 20, 1)]);
    assert_eq!(
        rig.port.pending(),
        [
            ReadSpec {
                slot: 0,
                offset: BLOCK,
                len: BLOCK as usize,
                direct: true
            },
            ReadSpec {
                slot: 1,
                offset: 0,
                len: 2 * BLOCK as usize,
                direct: true
            },
        ]
    );
    rig.port.finish(0);
    rig.port.finish(1);
    assert!(rig.claim(0, 1));
    assert!(rig.claim(1, 1));
    assert_eq!(rig.payload(0, 904, 100), contents()[5000..5100]);
    assert_eq!(rig.payload(1, 4090, 20), contents()[4090..4110]);
}

#[test]
fn a_range_that_would_pass_the_end_of_the_file_is_read_buffered_and_exact() {
    let window = Arc::new(DiskWindow::on_heap(geometry(1)));
    let port = FakePort::short(Arc::clone(&window), 5100);
    let rig = Rig::over(port, window, 1);
    // The payload ends at 5100; the aligned read would end at 8192.
    rig.enqueue(1, vec![use_of("tail", 5000, 100, 0)]);
    assert_eq!(
        rig.port.pending(),
        [ReadSpec {
            slot: 0,
            offset: BLOCK,
            len: 1004,
            direct: false
        }]
    );
    rig.port.finish(0);
    assert!(rig.claim(0, 1));
    assert!(rig.failures_empty());
}

impl Rig {
    fn failures_empty(&self) -> bool {
        lock(&self.failures).is_empty()
    }
}

#[test]
fn successive_quanta_share_the_window_in_admission_order() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0)]);
    rig.enqueue(2, vec![use_of("b", BLOCK, 100, 0)]);
    assert_eq!(rig.port.started().len(), 1);
    rig.port.finish(0);
    assert!(rig.claim(0, 1));
    assert_eq!(rig.port.started().len(), 1);
    rig.reader.slot_freed(0);
    assert_eq!(rig.port.started().len(), 2);
    assert_eq!(rig.port.started()[1].offset, BLOCK);
    // The second quantum's copy is held until its bytes are in.
    assert!(!rig.claim(0, 2));
    rig.port.finish(0);
    assert_eq!(rig.events(), [GateEvent::Released]);
    assert_eq!(rig.payload(0, 0, 100), contents()[BLOCK as usize..][..100]);
}

#[test]
fn abandon_drops_the_failed_quantums_uses() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0), use_of("b", BLOCK, 100, 0)]);
    rig.enqueue(2, vec![use_of("c", 2 * BLOCK, 100, 0)]);
    rig.reader.abandon(1);
    // The read in flight cannot be stopped; it is dropped when it arrives, and the next quantum's read follows it.
    assert_eq!(rig.port.started().len(), 1);
    rig.port.finish(0);
    assert_eq!(rig.port.started().len(), 2);
    assert_eq!(rig.port.started()[1].offset, 2 * BLOCK);
    rig.port.finish(0);
    assert!(rig.claim(0, 2));
    assert_eq!(rig.payload(0, 0, 100), contents()[2 * BLOCK as usize..][..100]);
    rig.reader.slot_freed(0);
    assert_eq!(rig.port.started().len(), 2);
    assert_eq!(rig.reader.reads_in_flight(), 0);
}

#[test]
fn a_read_nobody_claimed_of_an_abandoned_quantum_is_dropped() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0)]);
    rig.enqueue(2, vec![use_of("b", BLOCK, 100, 0)]);
    rig.port.finish(0);
    // The bytes are in and unclaimed: abandoning the quantum frees the slot for the next one.
    rig.reader.abandon(1);
    assert_eq!(rig.port.started().len(), 2);
    rig.port.finish(0);
    assert!(rig.claim(0, 2));
}

#[test]
fn a_quantum_abandoned_before_its_registration_reads_nothing() {
    let rig = Rig::new(2);
    rig.reader.abandon(5);
    rig.enqueue(5, vec![use_of("a", 0, 100, 0)]);
    assert!(rig.port.started().is_empty());
    assert_eq!(rig.reader.reads_in_flight(), 0);
    // The one after it reads.
    rig.enqueue(6, vec![use_of("b", 0, 100, 0)]);
    assert_eq!(rig.port.started().len(), 1);
}

#[test]
fn a_failed_read_fails_its_quantum_and_still_releases_the_gate() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("weights", 0, 100, 0)]);
    assert!(!rig.claim(0, 1));
    rig.port.finish_with(0, Err(io::Error::other("no disk")));
    let failures = lock(&rig.failures).clone();
    assert_eq!(failures.len(), 1);
    assert!(
        failures[0].contains("weights") && failures[0].contains("no disk"),
        "{failures:?}"
    );
    assert_eq!(rig.events(), [GateEvent::Released]);
}

#[test]
fn a_short_read_is_a_failed_read() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("weights", 0, 100, 0)]);
    rig.port.finish_with(0, Ok(40));
    assert_eq!(lock(&rig.failures).len(), 1);
    // The copy still gets the slot: its quantum cannot commit.
    assert!(rig.claim(0, 1));
}

#[test]
fn a_direct_read_the_file_system_refuses_is_repeated_through_the_page_cache() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 100, 100, 0), use_of("b", BLOCK, 100, 0)]);
    assert!(rig.port.pending()[0].direct);
    rig.port.finish_with(0, Err(io::Error::from_raw_os_error(EINVAL)));
    // The same use again, now through the page cache, and nothing failed.
    let pending = rig.port.pending();
    assert_eq!(pending.len(), 1);
    assert!(!pending[0].direct);
    assert_eq!((pending[0].offset, pending[0].len), (0, 200));
    assert!(rig.failures_empty());
    rig.port.finish(0);
    assert!(rig.claim(0, 1));
    assert_eq!(rig.payload(0, 100, 100), contents()[100..200]);
    // Later reads stay buffered.
    rig.reader.slot_freed(0);
    assert!(!rig.port.pending()[0].direct);
}

#[test]
fn a_held_gate_of_an_abandoned_quantum_is_released_within_a_bound() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0)]);
    assert!(!rig.claim(0, 1));
    assert!(rig.events().is_empty());
    // The quantum ends while its copy waits for a read that never matters: the gate must go, without any arrival.
    rig.reader.abandon(1);
    assert_eq!(rig.events(), [GateEvent::Released]);
    // The read in flight ends later and is dropped.
    rig.port.finish(0);
    assert_eq!(rig.events().len(), 1);
    assert_eq!(rig.reader.reads_in_flight(), 0);
}

#[test]
fn a_claim_that_finds_the_owner_busy_is_decided_by_the_owner() {
    let rig = Rig::new(2);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0), use_of("b", BLOCK, 100, 1)]);
    // The claim is held by this thread, as when a worker is in a turn.
    let held = Arc::new(Mutex::new(Vec::new()));
    {
        let (rig_ref, held) = (&rig, Arc::clone(&held));
        rig.reader.mailbox.with(move |_| {
            // The read of slot 0 has not arrived: held.
            lock(&held).push(rig_ref.claim(0, 1));
        });
    }
    assert_eq!(*lock(&held), [false]);
    // The gate is the owner's now and the arrival releases it.
    assert!(rig.events().is_empty());
    rig.port.finish(0);
    assert_eq!(rig.events(), [GateEvent::Released]);

    // A read already in: the owner decides the claim is ready and releases the gate it was handed.
    rig.port.finish(1);
    {
        let rig_ref = &rig;
        rig.reader.mailbox.with(move |_| {
            assert!(!rig_ref.claim(1, 1));
        });
    }
    assert_eq!(rig.events(), [GateEvent::Released, GateEvent::Released]);
}

#[test]
fn a_claim_on_a_closed_reader_fails_its_gate() {
    let rig = Rig::new(1);
    let closed = rig.reader.close();
    assert!(closed.is_set());
    assert!(!rig.claim(0, 1));
    assert!(matches!(
        rig.events().as_slice(),
        [GateEvent::Failed(StageError::Fatal(_))]
    ));
}

#[test]
fn close_completes_only_when_reads_in_flight_end() {
    let rig = Rig::new(2);
    rig.enqueue(
        1,
        vec![
            use_of("a", 0, 100, 0),
            use_of("b", BLOCK, 100, 1),
            use_of("c", 2 * BLOCK, 100, 0),
        ],
    );
    assert!(!rig.claim(1, 1));
    let closed = rig.reader.close();
    // Two reads are out: not closed. The gate that waited is failed at once, and the unread use is dropped.
    assert!(!closed.is_set());
    assert!(matches!(
        rig.events().as_slice(),
        [GateEvent::Failed(StageError::Fatal(_))]
    ));
    rig.port.finish(0);
    assert!(!closed.is_set());
    assert_eq!(rig.reader.reads_in_flight(), 1);
    rig.port.finish(1);
    assert!(closed.is_set());
    assert_eq!(rig.reader.reads_in_flight(), 0);
    // Nothing was started after the close.
    assert_eq!(rig.port.started().len(), 2);
    rig.reader.slot_freed(0);
    assert_eq!(rig.port.started().len(), 2);
}

#[test]
fn close_with_nothing_in_flight_completes_at_once() {
    let rig = Rig::new(1);
    assert!(rig.reader.close().is_set());
}

#[test]
fn a_use_the_window_cannot_hold_is_refused() {
    let rig = Rig::new(2);
    let fail = rig.fail_handle();
    let refused = |uses| rig.reader.enqueue(1, uses, Arc::clone(&fail));
    assert!(matches!(
        refused(vec![use_of("x", 0, 10, 2)]),
        Err(DiskError::NoSuchSlot { .. })
    ));
    assert!(matches!(
        refused(vec![use_of("x", 0, 0, 0)]),
        Err(DiskError::Empty { .. })
    ));
    assert!(matches!(
        refused(vec![use_of("x", 0, SLOT_BYTES + 1, 0)]),
        Err(DiskError::TooLarge { .. })
    ));
    assert!(rig.port.started().is_empty());
}

#[test]
fn a_window_checks_its_memory() {
    let geometry = geometry(2);
    let aligned = BLOCK as usize * 4;
    assert_eq!(DiskWindow::check(aligned, 6 * BLOCK as usize, geometry), Ok(()));
    assert!(matches!(
        DiskWindow::check(aligned, 5 * BLOCK as usize, geometry),
        Err(DiskError::Memory { .. })
    ));
    assert_eq!(
        DiskWindow::check(aligned + 8, 6 * BLOCK as usize, geometry),
        Err(DiskError::Unaligned)
    );
    let odd = WindowGeometry {
        slots: 1,
        slot_bytes: 1000,
    };
    assert_eq!(DiskWindow::check(aligned, 4096, odd), Err(DiskError::SlotSize(1000)));
}

// The real file and ring.

/// A file under `EUHEDRAL_SCRATCH` (a directory of a file system that may take direct reads), else under the system's
/// temporary directory (which may refuse them: the reads are then buffered).
fn file_in_scratch(bytes: &[u8]) -> Option<(tempfile::TempDir, PathBuf)> {
    let base = var_os("EUHEDRAL_SCRATCH").map_or_else(temp_dir, PathBuf::from);
    let dir = tempfile::Builder::new().prefix("disk-reader").tempdir_in(base).ok()?;
    let path = dir.path().join("artifact.bin");
    File::create(&path).and_then(|mut file| file.write_all(bytes)).ok()?;
    Some((dir, path))
}

/// Reaps and runs the continuations of `io` until `ended` says the reads are over.
fn drive(io: &AsyncIo, ended: impl Fn() -> bool) {
    let deadline = Instant::now() + BOUND;
    while !ended() {
        assert!(Instant::now() < deadline, "the reads did not end in time");
        io.reap();
        let mut frames = Vec::new();
        io.pull(&mut |frame| frames.push(frame), &mut |_| false, 8);
        for mut frame in frames {
            match frame.execute() {
                Ok(()) | Err(FrameError::Cancelled) => frame.do_finally(),
                Err(error) => frame.do_finally_with_error(error),
            }
        }
    }
}

fn read_through_the_ring(file: ArtifactFile) -> (Vec<u8>, Vec<u8>, bool) {
    let io = Arc::new(AsyncIo::open().expect("the ring opens"));
    let file = Arc::new(file);
    let direct = file.has_direct();
    let window = DiskWindow::on_heap(geometry(2));
    let port = Arc::new(FilePort::new(Arc::clone(&io), file, window));
    let reader = DiskReader::new(Arc::clone(&port) as Arc<dyn ReadPort>, geometry(2), inline());
    let failures = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&failures);
    let fail: FailHandle = Arc::new(move |_| {
        counted.fetch_add(1, Ordering::SeqCst);
    });
    // One payload in the middle of the file, one at its very end (a direct read of the whole block would pass it).
    let end = FILE_BYTES as u64;
    reader
        .enqueue(
            1,
            vec![use_of("mid", 5000, 9000, 0), use_of("tail", end - 100, 100, 1)],
            fail,
        )
        .expect("accepted");
    drive(&io, || reader.reads_in_flight() == 0);
    assert_eq!(failures.load(Ordering::SeqCst), 0);
    let window = port.window();
    // SAFETY: both reads ended; the window is live.
    let read =
        |slot: usize, lead: usize, len: usize| unsafe { from_raw_parts(window.slot_ptr(slot).add(lead), len).to_vec() };
    (
        read(0, 5000 - BLOCK as usize, 9000),
        read(1, (end - 100 - (end - 100) / BLOCK * BLOCK) as usize, 100),
        direct,
    )
}

#[test]
fn the_ring_reads_the_artifact_into_the_window_through_the_direct_handle_when_there_is_one() {
    let Some((_dir, path)) = file_in_scratch(&contents()) else {
        return;
    };
    let (middle, tail, _direct) = read_through_the_ring(ArtifactFile::open(&path).expect("opens"));
    assert_eq!(middle, contents()[5000..14000]);
    assert_eq!(tail, contents()[FILE_BYTES - 100..]);
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn a_direct_open_that_fails_leaves_every_read_buffered() {
    let Some((_dir, path)) = file_in_scratch(&contents()) else {
        return;
    };
    let file = ArtifactFile::open_with(&path, |_| Err(io::Error::from_raw_os_error(EINVAL)))
        .expect("the buffered open does not depend on it");
    assert!(!file.has_direct());
    assert_eq!(file.len(), FILE_BYTES as u64);
    let (middle, tail, direct) = read_through_the_ring(file);
    assert!(!direct);
    assert_eq!(middle, contents()[5000..14000]);
    assert_eq!(tail, contents()[FILE_BYTES - 100..]);
}

#[test]
fn a_missing_artifact_is_an_error() {
    let missing = Path::new("/nonexistent-disk-reader-test/artifact.bin");
    assert!(ArtifactFile::open(missing).is_err());
}
