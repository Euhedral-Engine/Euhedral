//! The disk reader over a fake port: its reads end when a test says, its gates are the graph's own.

use std::collections::VecDeque;
use std::env::{current_exe, var_os};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::ptr::copy_nonoverlapping;
use std::slice::from_raw_parts;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::LatticeSource;
use euhedral_graph::testing::{FakeDevice, GateEvent, StageHarness};
use euhedral_graph::{GateRef, Stage, StageCx, StageError, Submitted};
use euhedral_inference_settings::{ModelSettings, Settings};

use super::*;
use crate::runtime::lattice_config;

const BLOCK: u64 = HOST_ALIGNMENT;
const FILE_BYTES: usize = 64 * 1024;
const SLOT_BYTES: u64 = 3 * BLOCK;
/// The error a direct read gets for a request the file system cannot align.
const ALIGNMENT_REFUSAL: i32 = if cfg!(windows) { 87 } else { 22 };
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
    harness: Arc<StageHarness>,
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
            harness: Arc::new(StageHarness::new(lanes)),
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

    rig.reader.slot_freed(0, 1);
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

/// `count` uses at distinct offsets of the file, slots `first + index` round robin over `slots`.
fn uses_of(quantum: &str, count: usize, first: usize, offset: u64, slots: usize) -> Vec<DiskUse> {
    (0..count)
        .map(|index| {
            let at = (offset + 12_288 * index as u64 + 512 * (index as u64 % 3)) % (FILE_BYTES as u64 - 2 * SLOT_BYTES);
            use_of(
                &format!("{quantum}/{index}"),
                at,
                9_000 + 100 * index as u64,
                (first + index) % slots,
            )
        })
        .collect()
}

impl Rig {
    /// The device's side of one use: the claim (the read finished before it on even uses, after it on odd ones), the
    /// bytes checked, the copy's end.
    fn consume(&self, ticket: u64, use_: &DiskUse, index: usize) {
        let pending = |rig: &Rig| rig.port.pending().iter().any(|spec| spec.slot == use_.slot);
        if index.is_multiple_of(2) && pending(self) {
            self.port.finish(use_.slot);
        }
        if !self.claim(use_.slot, ticket) {
            assert!(pending(self), "{} was never read", use_.name);
            self.port.finish(use_.slot);
        }
        let from = use_.file_offset as usize;
        assert_eq!(
            self.payload(use_.slot, use_.lead() as usize, use_.bytes as usize),
            contents()[from..from + use_.bytes as usize],
            "{}",
            use_.name
        );
        self.reader.slot_freed(use_.slot, ticket);
    }
}

#[test]
fn successive_quanta_share_the_window_in_admission_order() {
    let rig = Rig::new(8);
    // The draft after a decode: its uses start on slots the decode still has queued.
    let first = uses_of("q1", 12, 0, 0, 8);
    let second = uses_of("q2", 9, 3, 393_216, 8);
    rig.enqueue(1, first.clone());
    rig.enqueue(2, second.clone());
    for (index, use_) in first.iter().enumerate() {
        rig.consume(1, use_, index);
    }
    for (index, use_) in second.iter().enumerate() {
        rig.consume(2, use_, index);
    }
    assert_eq!(rig.reader.reads_in_flight(), 0);
    assert_eq!(rig.port.started().len(), 21);
    assert!(rig.failures_empty());
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
    rig.reader.slot_freed(0, 2);
    assert_eq!(rig.port.started().len(), 2);
    assert_eq!(rig.reader.reads_in_flight(), 0);
}

#[test]
fn a_kill_flag_set_abandons_the_quanta_watching_it_and_no_others() {
    let rig = Rig::new(1);
    let (flag, other) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    rig.enqueue(1, vec![use_of("a", 0, 100, 0)]);
    rig.reader.watch(1, Arc::clone(&flag));
    rig.enqueue(2, vec![use_of("b", BLOCK, 100, 0)]);
    rig.reader.watch(2, Arc::clone(&other));
    // The read of quantum 1 is in flight; the flag is set and the reader told: quantum 2 reads once it arrives.
    flag.store(true, Ordering::Release);
    rig.reader.killed(&flag);
    rig.port.finish(0);
    assert_eq!(
        rig.port.started().len(),
        2,
        "the slot went on to the quantum that is not killed"
    );
    assert_eq!(rig.port.started()[1].offset, BLOCK);
    rig.port.finish(0);
    assert!(rig.claim(0, 2));
    // Telling again changes nothing.
    rig.reader.killed(&flag);
    assert_eq!(rig.port.started().len(), 2);
}

#[test]
fn a_quantum_watched_with_its_flag_already_set_is_abandoned_at_once() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0), use_of("b", BLOCK, 100, 0)]);
    rig.enqueue(2, vec![use_of("c", 2 * BLOCK, 100, 0)]);
    rig.reader.watch(1, Arc::new(AtomicBool::new(true)));
    // The read in flight is dropped when it arrives, and the quantum's second use is never read.
    rig.port.finish(0);
    assert_eq!(rig.port.started().len(), 2);
    assert_eq!(rig.port.started()[1].offset, 2 * BLOCK);
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
fn a_failed_read_fails_its_quantum_and_the_gate_of_its_waiting_copy() {
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
    assert!(matches!(
        rig.events().as_slice(),
        [GateEvent::Failed(StageError::Fatal(why))] if why.contains("no disk")
    ));
}

#[test]
fn a_short_read_is_a_failed_read_and_its_copy_is_failed_not_handed_the_slot() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("weights", 0, 100, 0)]);
    rig.port.finish_with(0, Ok(40));
    assert_eq!(lock(&rig.failures).len(), 1);
    // No copy was waiting: the failure stays with the slot, and the copy that comes is failed with it.
    assert!(!rig.claim(0, 1));
    assert!(
        matches!(rig.events().as_slice(), [GateEvent::Failed(StageError::Fatal(why))] if why.contains("40 of 100")),
        "{:?}",
        rig.events()
    );
}

#[test]
fn a_failed_slot_goes_back_once_the_failed_copy_frees_it() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0)]);
    rig.enqueue(2, vec![use_of("b", 0, 100, 0)]);
    rig.port.finish_with(0, Err(io::Error::other("no disk")));
    assert!(!rig.claim(0, 1));
    // The copy owns the slot until its quantum ends; the next quantum reads only after.
    assert!(rig.port.pending().is_empty());
    rig.reader.slot_freed(0, 1);
    assert_eq!(rig.port.pending().len(), 1);
}

#[test]
fn a_claim_between_the_owners_release_and_the_failure_is_never_ready() {
    let rig = Rig::new(1);
    let (reader, harness) = (rig.reader.clone(), Arc::clone(&rig.harness));
    let seen: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));
    let record = Arc::clone(&seen);
    // The fail handle runs where a racing claim would: after the owner's claim is released, before the failure is told.
    let fail: FailHandle = Arc::new(move |_why| {
        let passed = harness.pass_gate(&Copy(reader.clone()), &(0, 1));
        lock(&record).push(passed.open);
    });
    rig.reader
        .enqueue(1, vec![use_of("weights", 0, 100, 0)], fail)
        .expect("accepted");
    rig.port.finish_with(0, Err(io::Error::other("no disk")));
    assert_eq!(*lock(&seen), [false], "the copy of a failed read was handed the slot");
    assert!(matches!(
        rig.events().as_slice(),
        [GateEvent::Failed(StageError::Fatal(_))]
    ));
}

#[test]
fn a_claim_the_owner_decides_later_on_a_failed_read_fails_its_gate() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("weights", 0, 100, 0)]);
    rig.port.finish_with(0, Err(io::Error::other("no disk")));
    // The claim finds the owner busy and leaves its gate to it.
    {
        let rig_ref = &rig;
        rig.reader.mailbox.with(move |_| {
            assert!(!rig_ref.claim(0, 1));
        });
    }
    assert!(matches!(
        rig.events().as_slice(),
        [GateEvent::Failed(StageError::Fatal(why))] if why.contains("no disk")
    ));
}

#[test]
fn a_direct_read_the_file_system_refuses_is_repeated_through_the_page_cache() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 100, 100, 0), use_of("b", BLOCK, 100, 0)]);
    assert!(rig.port.pending()[0].direct);
    rig.port
        .finish_with(0, Err(io::Error::from_raw_os_error(ALIGNMENT_REFUSAL)));
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
    rig.reader.slot_freed(0, 1);
    assert!(!rig.port.pending()[0].direct);
}

#[test]
fn a_held_gate_of_an_abandoned_quantum_is_released_within_a_bound() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0)]);
    assert!(!rig.claim(0, 1));
    assert!(rig.events().is_empty());
    // The quantum ends while its copy waits for a read that never matters: the gate must go, without any arrival. It
    // fails: a release always means the copy owns the slot.
    rig.reader.abandon(1);
    assert!(matches!(
        rig.events().as_slice(),
        [GateEvent::Failed(StageError::Fatal(_))]
    ));
    // The read in flight ends later and is dropped.
    rig.port.finish(0);
    assert_eq!(rig.events().len(), 1);
    assert_eq!(rig.reader.reads_in_flight(), 0);
}

#[test]
fn a_claim_after_its_quantum_was_abandoned_fails_at_once_and_leaves_the_slot_alone() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0)]);
    rig.enqueue(2, vec![use_of("b", BLOCK, 100, 0)]);
    rig.reader.abandon(1);
    // The stage of the ended quantum was published between its end and its gate pass: its claim comes late.
    assert!(!rig.claim(0, 1));
    assert!(matches!(
        rig.events().as_slice(),
        [GateEvent::Failed(StageError::Fatal(_))]
    ));
    // The read in flight is dropped on arrival and quantum 2 reads and claims the slot.
    rig.port.finish(0);
    assert_eq!(rig.port.started().len(), 2);
    rig.port.finish(0);
    assert!(rig.claim(0, 2), "the next quantum's read was lost to the stale claim");
    assert_eq!(rig.payload(0, 0, 100), contents()[BLOCK as usize..][..100]);
}

#[test]
fn a_claim_after_the_abandoned_read_arrived_leaves_the_next_quantums_bytes() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0)]);
    rig.enqueue(2, vec![use_of("b", BLOCK, 100, 0)]);
    rig.port.finish(0);
    rig.reader.abandon(1);
    rig.port.finish(0);
    // Quantum 2's bytes are in; the late claim of quantum 1 must not drop them.
    assert!(!rig.claim(0, 1));
    assert!(rig.claim(0, 2));
    assert_eq!(rig.payload(0, 0, 100), contents()[BLOCK as usize..][..100]);
}

#[test]
fn a_stale_free_does_not_clear_another_quantums_filled_slot() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0)]);
    rig.enqueue(2, vec![use_of("b", BLOCK, 100, 0)]);
    assert!(!rig.claim(0, 1));
    rig.reader.abandon(1);
    rig.port.finish(0);
    rig.port.finish(0);
    // Quantum 2's read is in and unclaimed; a free naming quantum 1 changes nothing.
    rig.reader.slot_freed(0, 1);
    assert!(rig.claim(0, 2), "the stale free dropped the bytes");
    assert_eq!(rig.payload(0, 0, 100), contents()[BLOCK as usize..][..100]);
}

#[test]
fn a_stale_free_does_not_free_another_quantums_claimed_slot() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0)]);
    rig.enqueue(2, vec![use_of("b", BLOCK, 100, 0)]);
    rig.enqueue(3, vec![use_of("c", 2 * BLOCK, 100, 0)]);
    assert!(!rig.claim(0, 1));
    rig.reader.abandon(1);
    rig.port.finish(0);
    rig.port.finish(0);
    assert!(rig.claim(0, 2));
    // The copy of quantum 2 owns the slot: a free naming quantum 1 must not start the read of quantum 3.
    rig.reader.slot_freed(0, 1);
    assert_eq!(rig.port.started().len(), 2);
    assert_eq!(rig.payload(0, 0, 100), contents()[BLOCK as usize..][..100]);
    rig.reader.slot_freed(0, 2);
    assert_eq!(rig.port.started().len(), 3);
}

#[test]
fn a_stale_read_is_dropped_and_the_next_quantum_gets_its_own() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0)]);
    rig.port.finish(0);
    rig.reader.abandon(1);
    rig.enqueue(2, vec![use_of("b", BLOCK, 100, 0)]);
    // The slot held the stopped quantum's read: it is dropped and the copy waits for its own.
    assert!(!rig.claim(0, 2));
    assert!(rig.events().is_empty());
    assert_eq!(rig.port.started().len(), 2);
    rig.port.finish(0);
    assert_eq!(rig.events(), [GateEvent::Released]);
    assert_eq!(rig.payload(0, 0, 100), contents()[BLOCK as usize..][..100]);
}

#[test]
fn a_stopped_quantums_read_arriving_while_another_copy_waits_is_dropped() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0)]);
    // The next quantum's copy asks before its quantum is registered, while the stopped quantum's read is out.
    assert!(!rig.claim(0, 2));
    rig.reader.abandon(1);
    rig.enqueue(2, vec![use_of("b", BLOCK, 100, 0)]);
    rig.port.finish(0);
    assert!(
        rig.events().is_empty(),
        "the stopped quantum's read released the waiting copy"
    );
    assert_eq!(rig.port.started().len(), 2);
    rig.port.finish(0);
    assert_eq!(rig.events(), [GateEvent::Released]);
    assert_eq!(rig.payload(0, 0, 100), contents()[BLOCK as usize..][..100]);
}

#[test]
fn every_close_gets_its_hand_off() {
    let rig = Rig::new(1);
    rig.enqueue(1, vec![use_of("a", 0, 100, 0)]);
    let (first, second) = (rig.reader.close(), rig.reader.close());
    assert!(!first.is_set() && !second.is_set());
    rig.port.finish(0);
    assert!(first.is_set() && second.is_set());
}

#[cfg(not(windows))]
#[test]
fn only_einval_is_an_alignment_refusal_on_unix() {
    assert!(refuses_alignment(22));
    assert!(!refuses_alignment(87));
    assert!(!refuses_alignment(5));
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
    rig.reader.slot_freed(0, 1);
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

/// A file under `EUHEDRAL_SCRATCH` (a directory of a file system that may take direct reads), else in the directory of
/// the test binary, inside the target directory (which may refuse them: the reads are then buffered).
fn file_in_scratch(bytes: &[u8]) -> Option<(tempfile::TempDir, PathBuf)> {
    let base = match var_os("EUHEDRAL_SCRATCH") {
        Some(base) => PathBuf::from(base),
        None => current_exe().ok()?.parent()?.to_path_buf(),
    };
    let dir = tempfile::Builder::new().prefix("disk-reader").tempdir_in(base).ok()?;
    let path = dir.path().join("artifact.bin");
    File::create(&path).and_then(|mut file| file.write_all(bytes)).ok()?;
    Some((dir, path))
}

/// A file in the default temporary directory: for the opens that do not need a file system that takes `O_DIRECT`.
fn file_in_temp() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("artifact.bin");
    fs::write(&path, contents()).expect("the artifact");
    (dir, path)
}

fn read_through_the_ring(file: ArtifactFile) -> (Vec<u8>, Vec<u8>, bool) {
    let settings = Settings::with_model(ModelSettings::default());
    let lattice = ControlPlaneLattice::new(lattice_config(&settings)).expect("the lattice starts");
    let io = AsyncIo::open().expect("the ring opens");
    lattice
        .add_upstream(Arc::clone(&io) as Arc<dyn LatticeSource>)
        .expect("the lattice takes the ring");
    let file = Arc::new(file);
    let direct = file.has_direct();
    let window = DiskWindow::on_heap(geometry(2));
    let port = Arc::new(FilePort::new(
        Arc::clone(&io),
        file,
        window,
        DiskReadPiece::from_setting(Auto::Auto).unwrap(),
    ));
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
    // The lattice reaps the ring and runs the continuations; the close hand-off is sent when the reads have ended.
    reader.close().wait(BOUND).expect("the reads ended in time");
    lattice.close();
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
    let file = ArtifactFile::open_checked(
        &path,
        |_| Err(io::Error::from_raw_os_error(ALIGNMENT_REFUSAL)),
        |_| Ok(()),
    )
    .expect("the buffered open does not depend on it");
    assert!(!file.has_direct());
    assert_eq!(file.len(), FILE_BYTES as u64);
    let (middle, tail, direct) = read_through_the_ring(file);
    assert!(!direct);
    assert_eq!(middle, contents()[5000..14000]);
    assert_eq!(tail, contents()[FILE_BYTES - 100..]);
}

#[test]
fn an_open_whose_buffered_handle_is_refused_by_the_check_fails() {
    let (_dir, path) = file_in_temp();
    let error = ArtifactFile::open_admitted(&path, |_| Err(io::Error::other("not the held file"))).unwrap_err();
    assert_eq!(error.to_string(), "not the held file");
}

#[test]
fn an_open_whose_direct_handle_is_refused_by_the_check_fails_and_is_not_read_buffered() {
    let (_dir, path) = file_in_temp();
    let shown = AtomicUsize::new(0);
    let error = ArtifactFile::open_checked(
        &path,
        |named| File::open(named),
        |_| match shown.fetch_add(1, Ordering::SeqCst) {
            0 => Ok(()),
            _ => Err(io::Error::other("not the held file")),
        },
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "not the held file");
    assert_eq!(shown.load(Ordering::SeqCst), 2, "both handles are shown to the check");
}

#[test]
fn a_missing_artifact_is_an_error() {
    let missing = Path::new("/nonexistent-disk-reader-test/artifact.bin");
    assert!(ArtifactFile::open(missing).is_err());
}

#[test]
fn an_unset_read_piece_is_the_most_the_ring_allows() {
    let piece = DiskReadPiece::from_setting(Auto::Auto).expect("auto is valid");
    assert_eq!(piece.bytes(), MAX_PIECE);
}

#[test]
fn a_stated_read_piece_of_whole_blocks_is_kept() {
    let piece = DiskReadPiece::from_setting(Auto::Value(ByteSize::mib(8))).expect("8 MiB is whole blocks");
    assert_eq!(piece.bytes(), 8 << 20);
    let limit = DiskReadPiece::from_setting(Auto::Value(ByteSize(MAX_PIECE as u64))).expect("the limit");
    assert_eq!(limit.bytes(), MAX_PIECE);
}

#[test]
fn a_read_piece_that_is_zero_unaligned_or_over_the_limit_is_refused_naming_the_value() {
    for bytes in [0, 1, BLOCK + 1, MAX_PIECE as u64 + BLOCK] {
        let refusal = DiskReadPiece::from_setting(Auto::Value(ByteSize(bytes))).expect_err("refused");
        assert!(refusal.contains(&format!("is {bytes} bytes")), "{refusal}");
        assert!(refusal.contains(&MAX_PIECE.to_string()), "{refusal}");
    }
}

#[test]
fn a_read_piece_is_the_option_every_read_of_the_port_carries() {
    let piece = DiskReadPiece::from_setting(Auto::Value(ByteSize(BLOCK))).expect("a block");
    assert_eq!(piece.options().piece, Some(BLOCK as usize));
}
