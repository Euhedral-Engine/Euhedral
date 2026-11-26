//! The copies of a held weight into its staging slot, ahead of the stage that reads it: from the pinned host arena, or
//! from the disk window the artifact was read into.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use euhedral_graph::{Completions, DeviceQueue, GateRef, Marker, Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{DevicePtr, HostWeightsArena, KernelOps, PinnedBuffer};

use super::{Launch, Quantum, fatal, run_stage};
use crate::frames::{Kind, step};
use crate::handoff::Handoff;
use crate::residency::disk::{Claimed, DiskError, DiskReader, DiskUse, FailHandle};
use crate::residency::offload::WindowGeometry;

/// Copies one weight from the pinned host arena into a staging slot.
///
/// The stage reads nothing of the quantum: it is ordered by the plan's edges alone, behind the stage that last read
/// the slot and ahead of the one that reads the weight. It never joins a captured graph, as a staged plan's views are
/// not captured.
pub struct WeightTransfer {
    name: String,
    source: Option<Arc<HostWeightsArena>>,
    offset: u64,
    bytes: u64,
    destination: DevicePtr,
}

impl WeightTransfer {
    /// The copy of `bytes` bytes at `offset` of `source` to `destination`. A transfer of a plan over described weights
    /// has no `source` and fails when it runs.
    pub fn new(
        name: String,
        source: Option<Arc<HostWeightsArena>>,
        offset: u64,
        bytes: u64,
        destination: DevicePtr,
    ) -> Self {
        Self {
            name,
            source,
            offset,
            bytes,
            destination,
        }
    }

    /// The bytes the stage copies.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The slot the stage fills.
    pub fn destination(&self) -> DevicePtr {
        self.destination
    }
}

impl fmt::Debug for WeightTransfer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WeightTransfer")
            .field("name", &self.name)
            .field("offset", &self.offset)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl Launch for WeightTransfer {
    fn launch(&self, _quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let source = self
            .source
            .as_ref()
            .ok_or_else(|| fatal(format!("{} has no host arena to be staged from", self.name)))?;
        let offset = usize::try_from(self.offset).map_err(fatal)?;
        let bytes = usize::try_from(self.bytes).map_err(fatal)?;
        // SAFETY: the stage holds the arena, which is only read once the model is loaded, so its bytes in the range
        // stay unmodified; the plan orders the copy behind the last read of the slot and ahead of the next, so no
        // other copy writes the slot meanwhile.
        unsafe { kernels.copy_host_weights_to_device(self.destination, source, offset, bytes) }.map_err(fatal)
    }
}

super::eager!(WeightTransfer);

/// What a ledger tells the disk reader.
pub(crate) trait Release {
    /// The copy of `ticket` out of window slot `slot` ended (or never began): the slot may be read into again.
    fn slot_freed(&self, slot: usize, ticket: u64);
    /// `ticket` reads no more.
    fn abandon(&self, ticket: u64);
}

impl Release for DiskReader {
    fn slot_freed(&self, slot: usize, ticket: u64) {
        DiskReader::slot_freed(self, slot, ticket);
    }

    fn abandon(&self, ticket: u64) {
        DiskReader::abandon(self, ticket);
    }
}

/// The use has not reached its stage's gate.
const FRESH: u8 = 0;
/// The stage asked for the use's read: the slot may be, or become, the copy's.
const ATTEMPTED: u8 = 1;
/// The copy is queued: its completion frees the slot.
const SUBMITTED: u8 = 2;
/// The slot was handed back.
const FREED: u8 = 3;

/// What one quantum owes the disk reader: for each of its disk uses, whether its window slot may be the quantum's, so
/// that every slot a read handed to the quantum goes back exactly once, by the completion of the copy that read it or,
/// for a copy that never ran, when the quantum ends.
#[derive(Debug)]
pub struct DiskLedger {
    ticket: u64,
    /// The window slot of each use, in use order.
    windows: Box<[usize]>,
    states: Box<[AtomicU8]>,
    /// Why a read of the quantum failed, set before the copy that would have read the bytes is released.
    failure: OnceLock<String>,
    /// Set once the quantum's stages ended without all of their copies.
    ended: AtomicBool,
}

impl DiskLedger {
    fn new(ticket: u64, windows: Box<[usize]>) -> DiskLedger {
        let states = windows.iter().map(|_| AtomicU8::new(FRESH)).collect();
        DiskLedger {
            ticket,
            windows,
            states,
            failure: OnceLock::new(),
            ended: AtomicBool::new(false),
        }
    }

    /// The quantum's number at the reader.
    pub fn ticket(&self) -> u64 {
        self.ticket
    }

    /// Why a read of the quantum failed, if one did.
    pub fn failure(&self) -> Option<&str> {
        self.failure.get().map(String::as_str)
    }

    fn fail(&self, why: String) {
        let _ = self.failure.set(why);
    }

    fn attempt(&self, use_index: usize) {
        self.states[use_index].store(ATTEMPTED, Ordering::SeqCst);
    }

    fn submitted(&self, use_index: usize) {
        self.states[use_index].store(SUBMITTED, Ordering::SeqCst);
    }

    /// The copy of `use_index` ended: its slot goes back.
    fn freed(&self, use_index: usize, release: &dyn Release) {
        if self.states[use_index]
            .compare_exchange(SUBMITTED, FREED, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            release.slot_freed(self.windows[use_index], self.ticket);
            // A use behind it that never ran may have waited for this slot.
            if self.ended.load(Ordering::SeqCst) {
                self.sweep(release);
            }
        }
    }

    /// The quantum's stages ended: what it never copied is not read. Once.
    pub(crate) fn end(&self, release: &dyn Release) {
        if !self.ended.swap(true, Ordering::SeqCst) {
            release.abandon(self.ticket);
            self.sweep(release);
        }
    }

    /// Hands back the slot of each use whose stage asked for it and never queued the copy, as soon as no earlier copy
    /// of the same slot is still to end: the reader frees a slot only for the ticket that holds it, so a free sent ahead
    /// of an earlier copy of the same ticket would free the slot under that copy.
    fn sweep(&self, release: &dyn Release) {
        for use_index in 0..self.states.len() {
            if self.states[use_index].load(Ordering::SeqCst) == ATTEMPTED
                && !self.behind_a_copy(use_index)
                && self.states[use_index]
                    .compare_exchange(ATTEMPTED, FREED, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            {
                release.slot_freed(self.windows[use_index], self.ticket);
            }
        }
    }

    fn behind_a_copy(&self, use_index: usize) -> bool {
        (0..use_index).any(|earlier| {
            self.windows[earlier] == self.windows[use_index] && self.states[earlier].load(Ordering::SeqCst) == SUBMITTED
        })
    }
}

/// The disk side of a run: the reader and the window it reads into, the order quanta register their reads in, and where
/// a copy's completion is told.
pub struct DiskTier {
    reader: DiskReader,
    /// The pinned memory the copies read; `None` for a tier that only registers (host tests).
    window: Option<Arc<PinnedBuffer>>,
    geometry: WindowGeometry,
    completions: Arc<Completions>,
    /// The next ticket: taken as a quantum registers, so tickets grow with the order quanta start in.
    next: AtomicU64,
}

impl fmt::Debug for DiskTier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiskTier")
            .field("geometry", &self.geometry)
            .finish_non_exhaustive()
    }
}

impl DiskTier {
    /// The tier over `reader`, whose reads land in `window`; a copy's completion is told through `completions`.
    pub fn new(
        reader: DiskReader,
        window: Arc<PinnedBuffer>,
        geometry: WindowGeometry,
        completions: Arc<Completions>,
    ) -> Arc<DiskTier> {
        Arc::new(DiskTier {
            reader,
            window: Some(window),
            geometry,
            completions,
            next: AtomicU64::new(0),
        })
    }

    /// A tier with no window behind it: plans and quanta register their reads, and no copy can run.
    pub fn detached(reader: DiskReader, geometry: WindowGeometry, completions: Arc<Completions>) -> Arc<DiskTier> {
        Arc::new(DiskTier {
            reader,
            window: None,
            geometry,
            completions,
            next: AtomicU64::new(0),
        })
    }

    /// The window's slots.
    pub fn geometry(&self) -> WindowGeometry {
        self.geometry
    }

    /// The reader, as what a ledger tells.
    pub(crate) fn reader(&self) -> &DiskReader {
        &self.reader
    }

    /// Registers the reads of a quantum that is about to start, with the next ticket. Called from the one place that
    /// starts quanta, one quantum at a time, so tickets and the reader's queues follow the order quanta start in.
    /// `kill` is the flag that stops the quantum: setting it ends the ticket ([`DiskTier::killed`]).
    pub fn register(&self, uses: &[DiskUse], kill: &Arc<AtomicBool>) -> Result<Arc<DiskLedger>, DiskError> {
        let ticket = self.next.fetch_add(1, Ordering::SeqCst);
        let windows = uses.iter().map(|use_| use_.slot).collect();
        let ledger = Arc::new(DiskLedger::new(ticket, windows));
        let told = Arc::clone(&ledger);
        let fail: FailHandle = Arc::new(move |why| told.fail(why));
        self.reader.enqueue(ticket, uses.to_vec(), fail)?;
        self.reader.watch(ticket, Arc::clone(kill));
        Ok(ledger)
    }

    /// `kill` was set: the quanta it stops read no more.
    pub fn killed(&self, kill: &Arc<AtomicBool>) {
        self.reader.killed(kill);
    }

    /// Stops reading: the hand-off is sent once no read is in flight. Free the window only after every quantum retired
    /// as well.
    pub fn close(&self) -> Arc<Handoff<()>> {
        self.reader.close()
    }

    /// `slot` of the ledger goes back once the copy that `marker` ends is done.
    fn free_after(self: &Arc<Self>, marker: Marker, window_slot: usize, ledger: &Arc<DiskLedger>, use_index: usize) {
        let (tier, ledger) = (Arc::clone(self), Arc::clone(ledger));
        self.completions.after_marker(
            marker,
            step(Kind::DiskCopy, window_slot, move || {
                ledger.freed(use_index, &tier.reader);
            }),
        );
    }
}

/// Copies one weight out of the disk window into a staging slot, once the read that fills the window slot is in.
///
/// Its gate is taken when its last edge arrives ([`Stage::pass_gate`]): the reader releases it when the read has
/// arrived, so the copy is queued only when the bytes are there and nothing on the device waits for the host. The copy's
/// marker completion gives the window slot back; a copy that never ran gives it back when the quantum ends.
pub struct DiskTransfer {
    name: String,
    tier: Arc<DiskTier>,
    /// This use's place among the view's disk uses.
    use_index: usize,
    window_slot: usize,
    /// Where the payload starts in the window: its slot's start and the lead of the aligned read.
    window_offset: u64,
    bytes: u64,
    destination: DevicePtr,
}

impl DiskTransfer {
    /// The copy of `use_`'s payload from its window slot to `destination`, the `use_index`th disk use of its view.
    pub fn new(tier: Arc<DiskTier>, use_: &DiskUse, use_index: usize, destination: DevicePtr) -> Self {
        Self {
            name: use_.name.clone(),
            window_offset: use_.slot as u64 * tier.geometry.slot_bytes + use_.lead(),
            window_slot: use_.slot,
            tier,
            use_index,
            bytes: use_.bytes,
            destination,
        }
    }

    /// The bytes the stage copies.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The window slot the stage reads.
    pub fn window_slot(&self) -> usize {
        self.window_slot
    }
}

impl fmt::Debug for DiskTransfer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiskTransfer")
            .field("name", &self.name)
            .field("use", &self.use_index)
            .field("window_slot", &self.window_slot)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl Launch for DiskTransfer {
    fn launch(&self, _quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let window = self
            .tier
            .window
            .as_ref()
            .ok_or_else(|| fatal(format!("{} has no disk window to be copied from", self.name)))?;
        let offset = usize::try_from(self.window_offset).map_err(fatal)?;
        let bytes = usize::try_from(self.bytes).map_err(fatal)?;
        // SAFETY: the window holds the bytes (the stage's gate was released after the read arrived) and the reader does
        // not read into the slot again until the copy's marker completion frees it; the plan orders the copy behind the
        // last read of the staging slot and ahead of the next, so no other copy writes it meanwhile.
        unsafe { kernels.copy_upload_to_device(self.destination, window, offset, bytes) }.map_err(fatal)
    }
}

impl DiskTransfer {
    /// The copy's claim on its window slot: `true` when the bytes are in and the stage may be published; otherwise the
    /// gate is the reader's (released when they arrive, failed when they never will) or already failed. A quantum with
    /// no ledger had no reads registered: its gate fails.
    fn claim(&self, ledger: Option<&DiskLedger>, gate: GateRef<'_>) -> bool {
        let Some(ledger) = ledger else {
            gate.hold().fail(fatal(format!(
                "{} was started with no disk reads registered",
                self.name
            )));
            return false;
        };
        ledger.attempt(self.use_index);
        match self.tier.reader.claim(self.window_slot, ledger.ticket, || gate.hold()) {
            Claimed::Ready => true,
            Claimed::Held => false,
        }
    }

    /// The ledger the copy may run under: refused when none was registered, or when a read of the quantum failed, as
    /// the copy would move bytes that are not the weights.
    fn admit<'a>(&self, ledger: Option<&'a Arc<DiskLedger>>) -> Result<&'a Arc<DiskLedger>, StageError> {
        let ledger = ledger
            .ok_or_else(|| fatal(format!("{} was started with no disk reads registered", self.name)))?;
        if let Some(why) = ledger.failure() {
            return Err(fatal(why));
        }
        Ok(ledger)
    }

    /// The quantum's stages ended: what it never copied is not read, and the slots it holds go back.
    fn ended(&self, ledger: Option<&DiskLedger>) {
        if let Some(ledger) = ledger {
            ledger.end(&self.tier.reader);
        }
    }

    /// The copy was queued, ending at `marker` (`None`: it ended already): its window slot goes back when it has.
    fn queued(&self, ledger: &Arc<DiskLedger>, marker: Option<Marker>) {
        ledger.submitted(self.use_index);
        match marker {
            Some(marker) => self.tier.free_after(marker, self.window_slot, ledger, self.use_index),
            None => ledger.freed(self.use_index, &self.tier.reader),
        }
    }
}

impl Stage<Quantum> for DiskTransfer {
    fn pass_gate(&self, quantum: &Quantum, gate: GateRef<'_>) -> bool {
        self.claim(quantum.disk.as_deref(), gate)
    }

    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let ledger = self.admit(quantum.disk.as_ref())?;
        let submitted = run_stage(self, quantum, cx)?;
        self.queued(ledger, submitted.marker);
        Ok(submitted)
    }

    fn abandoned(&self, quantum: &Quantum, home: &(dyn DeviceQueue + 'static)) {
        quantum
            .unmarked
            .await_on(home, |lane| quantum.lanes.get(lane).map(|queue| queue.id()));
        self.ended(quantum.disk.as_deref());
    }
}

#[cfg(test)]
#[allow(
    clippy::disallowed_types,
    reason = "the fake port and the recording release are shared between the test and the closures the reader is given"
)]
mod tests {
    use std::io;
    use std::sync::Mutex;

    use euhedral_graph::testing::{FakeDevice, GateEvent, StageHarness};

    use super::*;
    use crate::residency::disk::{ReadDone, ReadPort, ReadSpec};
    use crate::residency::offload::WindowGeometry;

    /// What a ledger told the reader, in order.
    #[derive(Default)]
    struct Told(Mutex<Vec<String>>);

    impl Release for Told {
        fn slot_freed(&self, slot: usize, ticket: u64) {
            self.0.lock().unwrap().push(format!("free {slot} for {ticket}"));
        }

        fn abandon(&self, ticket: u64) {
            self.0.lock().unwrap().push(format!("abandon {ticket}"));
        }
    }

    impl Told {
        fn told(&self) -> Vec<String> {
            self.0.lock().unwrap().clone()
        }
    }

    #[test]
    fn a_copy_that_ended_hands_its_slot_back_once() {
        let (ledger, told) = (DiskLedger::new(7, vec![2].into()), Told::default());
        ledger.attempt(0);
        ledger.submitted(0);
        ledger.freed(0, &told);
        ledger.freed(0, &told);
        assert_eq!(told.told(), ["free 2 for 7"]);
    }

    #[test]
    fn the_end_abandons_once_and_frees_only_the_slots_it_asked_for() {
        let (ledger, told) = (DiskLedger::new(3, vec![0, 1, 2].into()), Told::default());
        ledger.attempt(0);
        ledger.submitted(0);
        ledger.attempt(1);
        // The third use never reached its gate: the reader has not handed it a slot.
        ledger.end(&told);
        ledger.end(&told);
        assert_eq!(told.told(), ["abandon 3", "free 1 for 3"]);
        ledger.freed(0, &told);
        assert_eq!(told.told(), ["abandon 3", "free 1 for 3", "free 0 for 3"]);
    }

    #[test]
    fn a_slot_never_copied_is_freed_behind_the_copy_that_still_holds_it() {
        // Two uses of one window slot: the second asked for it and never copied, the first is still being copied.
        let (ledger, told) = (DiskLedger::new(5, vec![4, 4].into()), Told::default());
        ledger.attempt(0);
        ledger.submitted(0);
        ledger.attempt(1);
        ledger.end(&told);
        assert_eq!(told.told(), ["abandon 5"], "nothing frees the slot under the copy");
        ledger.freed(0, &told);
        assert_eq!(told.told(), ["abandon 5", "free 4 for 5", "free 4 for 5"]);
    }

    #[test]
    fn a_failed_read_is_kept_for_the_copy_that_would_have_used_it() {
        let ledger = DiskLedger::new(1, vec![0].into());
        assert_eq!(ledger.failure(), None);
        ledger.fail("short read".to_string());
        ledger.fail("second".to_string());
        assert_eq!(ledger.failure(), Some("short read"));
    }

    /// A port that records the reads it is asked for and ends none.
    #[derive(Default)]
    struct Recording(Mutex<Vec<u64>>);

    impl ReadPort for Arc<Recording> {
        fn start(&self, spec: ReadSpec, _done: ReadDone) -> io::Result<()> {
            self.0.lock().unwrap().push(spec.offset);
            Ok(())
        }
        fn has_direct(&self) -> bool {
            false
        }
        fn file_len(&self) -> u64 {
            u64::MAX
        }
    }

    fn use_at(slot: usize, file_offset: u64) -> DiskUse {
        DiskUse {
            name: format!("w@{file_offset}"),
            file_offset,
            bytes: 100,
            slot,
        }
    }

    #[test]
    fn quanta_register_their_reads_in_the_order_they_start() {
        let geometry = WindowGeometry {
            slots: 2,
            slot_bytes: 4096,
        };
        let kill = Arc::new(AtomicBool::new(false));
        for (first, second) in [(0x1000, 0x2000), (0x2000, 0x1000)] {
            let port = Arc::new(Recording::default());
            let reader = DiskReader::new(
                Arc::new(Arc::clone(&port)),
                geometry,
                Box::new(|work| {
                    work();
                    true
                }),
            );
            let tier = DiskTier::detached(reader, geometry, Completions::without_lanes());
            // Two quanta of different sequences read the same window slot: the one that started first is read first,
            // and the other waits behind its copy.
            let a = tier.register(&[use_at(0, first)], &kill).unwrap();
            let b = tier.register(&[use_at(0, second)], &kill).unwrap();
            assert_eq!(
                (a.ticket(), b.ticket()),
                (0, 1),
                "tickets follow the order of registration"
            );
            assert_eq!(
                *port.0.lock().unwrap(),
                [first],
                "the second quantum's read waits for the slot"
            );
        }
    }

    /// A port whose reads end when the test says.
    #[derive(Default)]
    struct Reads {
        pending: Mutex<Vec<(usize, ReadDone)>>,
        started: Mutex<usize>,
    }

    impl ReadPort for Arc<Reads> {
        fn start(&self, spec: ReadSpec, done: ReadDone) -> io::Result<()> {
            *self.started.lock().unwrap() += 1;
            self.pending.lock().unwrap().push((spec.slot, done));
            Ok(())
        }
        fn has_direct(&self) -> bool {
            false
        }
        fn file_len(&self) -> u64 {
            u64::MAX
        }
    }

    impl Reads {
        fn started(&self) -> usize {
            *self.started.lock().unwrap()
        }

        /// Ends the oldest read of `slot` as moving `moved` bytes, or failing.
        fn end(&self, slot: usize, result: io::Result<usize>) {
            let done = {
                let mut pending = self.pending.lock().unwrap();
                let index = pending.iter().position(|(held, _)| *held == slot).expect("a read of the slot");
                pending.remove(index).1
            };
            done(result);
        }
    }

    /// A tier over `slots` window slots, a quantum's copy of its first use, and the harness that takes its gates.
    struct Copying {
        reads: Arc<Reads>,
        tier: Arc<DiskTier>,
        harness: StageHarness,
        kill: Arc<AtomicBool>,
    }

    /// The copy as a stage of the harness, over the ledger a quantum would carry.
    struct Probe<'a> {
        copy: &'a DiskTransfer,
        ledger: Option<&'a Arc<DiskLedger>>,
    }

    impl Stage<()> for Probe<'_> {
        fn run(&self, _: &(), _: &StageCx<'_>) -> Result<Submitted, StageError> {
            Ok(Submitted::done())
        }

        fn pass_gate(&self, _: &(), gate: GateRef<'_>) -> bool {
            self.copy.claim(self.ledger.map(|ledger| &**ledger), gate)
        }
    }

    impl Copying {
        fn new() -> Copying {
            let geometry = WindowGeometry {
                slots: 1,
                slot_bytes: 4096,
            };
            let reads = Arc::new(Reads::default());
            let reader = DiskReader::new(
                Arc::new(Arc::clone(&reads)),
                geometry,
                Box::new(|work| {
                    work();
                    true
                }),
            );
            let (lanes, _) = FakeDevice::pool(&["lane"]);
            Copying {
                reads,
                tier: DiskTier::detached(reader, geometry, Completions::without_lanes()),
                harness: StageHarness::new(lanes),
                kill: Arc::new(AtomicBool::new(false)),
            }
        }

        fn register(&self) -> Arc<DiskLedger> {
            self.tier.register(&[use_at(0, 0)], &self.kill).unwrap()
        }

        fn copy(&self) -> DiskTransfer {
            DiskTransfer::new(Arc::clone(&self.tier), &use_at(0, 0), 0, DevicePtr::NULL)
        }

        /// The claim of the copy: `true` when its stage may be published at once.
        fn claim(&self, copy: &DiskTransfer, ledger: Option<&Arc<DiskLedger>>) -> bool {
            let passed = self.harness.pass_gate(&Probe { copy, ledger }, &());
            assert!(!(passed.open && passed.held), "a claim that opens takes no gate");
            passed.open
        }
    }

    #[test]
    fn a_copy_is_published_when_its_bytes_are_in_and_waits_for_them_otherwise() {
        let rig = Copying::new();
        let (ledger, copy) = (rig.register(), rig.copy());
        assert!(!rig.claim(&copy, Some(&ledger)), "the read is still out");
        assert!(rig.harness.gate_events().is_empty());
        rig.reads.end(0, Ok(100));
        assert_eq!(rig.harness.gate_events(), [GateEvent::Released]);
        // A second quantum's read arrives before its copy asks: the claim is ready at once.
        rig.tier.reader().slot_freed(0, ledger.ticket());
        let next = rig.register();
        rig.reads.end(0, Ok(100));
        assert!(rig.claim(&copy, Some(&next)));
    }

    #[test]
    fn a_copy_of_a_quantum_with_no_registered_reads_is_failed_at_its_gate_and_refused_at_its_run() {
        let rig = Copying::new();
        let copy = rig.copy();
        assert!(!rig.claim(&copy, None));
        assert!(matches!(
            rig.harness.gate_events().as_slice(),
            [GateEvent::Failed(StageError::Fatal(why))] if why.contains("no disk reads")
        ));
        assert!(matches!(copy.admit(None), Err(StageError::Fatal(why)) if why.contains("no disk reads")));
    }

    #[test]
    fn a_copy_of_a_failed_read_is_failed_at_its_gate_whether_it_waited_or_came_late() {
        for waited in [true, false] {
            let rig = Copying::new();
            let (ledger, copy) = (rig.register(), rig.copy());
            if waited {
                assert!(!rig.claim(&copy, Some(&ledger)));
            }
            rig.reads.end(0, Err(io::Error::other("no disk")));
            if !waited {
                assert!(!rig.claim(&copy, Some(&ledger)), "a failed read is never ready");
            }
            assert!(
                matches!(
                    rig.harness.gate_events().as_slice(),
                    [GateEvent::Failed(StageError::Fatal(why))] if why.contains("no disk")
                ),
                "waited: {waited}: {:?}",
                rig.harness.gate_events()
            );
        }
    }

    #[test]
    fn a_copy_does_not_run_once_a_read_of_its_quantum_failed() {
        let rig = Copying::new();
        let (ledger, copy) = (rig.register(), rig.copy());
        assert!(copy.admit(Some(&ledger)).is_ok());
        ledger.fail("the artifact ended".to_string());
        assert!(matches!(copy.admit(Some(&ledger)), Err(StageError::Fatal(why)) if why == "the artifact ended"));
    }

    #[test]
    fn a_queued_copy_that_ended_hands_its_window_slot_to_the_next_read() {
        let rig = Copying::new();
        let (first, copy) = (rig.register(), rig.copy());
        let _second = rig.register();
        rig.reads.end(0, Ok(100));
        assert_eq!(rig.reads.started(), 1, "the slot is the first copy's until it ends");
        assert!(rig.claim(&copy, Some(&first)));
        copy.queued(&first, None);
        assert_eq!(rig.reads.started(), 2);
    }

    #[test]
    fn the_end_of_a_quantum_fails_the_copy_that_waits_and_frees_the_slot_for_the_next() {
        let rig = Copying::new();
        let (first, copy) = (rig.register(), rig.copy());
        let _second = rig.register();
        assert!(!rig.claim(&copy, Some(&first)));
        copy.ended(Some(&first));
        assert!(matches!(
            rig.harness.gate_events().as_slice(),
            [GateEvent::Failed(StageError::Fatal(_))]
        ));
        // The read in flight is discarded when it arrives, and the next quantum's read follows it.
        rig.reads.end(0, Ok(100));
        assert_eq!(rig.reads.started(), 2);
        copy.ended(None);
    }
}
