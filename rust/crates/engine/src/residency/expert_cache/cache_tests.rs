//! The cache's contracts through its interface, on a fake device: the reads are held by a fake port and the copies and
//! their retirements by a fake lane, and each test moves them by hand, so an order, a refusal or a leak is forced and
//! nothing is waited for.

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixture: the artifact is written with blocking calls, and the fakes' queues are Mutexes"
)]

use std::collections::VecDeque;
use std::fs::write;
use std::io;
use std::mem::forget;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError, Weak};
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Failure, Frame};
use euhedral_core::generics::{LatticeReceiver, LatticeSource, LatticeTerminal};
use euhedral_graph::{LaneId, Marker, QueueId};
use euhedral_inference_edrl::{
    ComponentGroup, ExpertBank, ExpertProjection, ExpertRecord, TensorDataType, WeightFormat, WeightLayout,
};
use euhedral_inference_settings::{ModelSettings, Settings as RuntimeSettings};
use euhedral_io::{Continuation, IoDoneOwned};
use euhedral_memory::{LocationKind, LocationSpec, OwnerSpec, Planner, Priority, Scope, SlotArena, TierPool};

use super::host::{CopyJob, CopyLane, DeviceSlots, Frames, SubmitFailure};
use super::owner::{Reply, Request};
use super::{CacheConfig, CacheError, ExpertCache, ExpertLease, Listener, Outcome};
use crate::handoff::Handoff;
use crate::residency::experts::{
    ExpertError, ExpertId, ExpertLayout, ExpertStore, FileRecordSource, HostArena, PartPort, RamTier, WritePart,
};
use crate::runtime::lattice_config;

const RECORD: u64 = 4096;
const EXPERTS: u32 = 6;
/// A hang deadline: the bound of a wait whose failure names what never ended.
const HANG: Duration = Duration::from_secs(60);

/// Takes a fake's lock. A fake is never entered while it is held (a frame run under its lock would re-enter it), so a
/// held lock fails the test at once, naming itself, instead of hanging it.
fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.try_lock() {
        Ok(guard) => guard,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => {
            panic!("a fake was entered while its own lock was held: a guard outlived its use")
        }
    }
}

fn id(expert: u32) -> ExpertId {
    ExpertId { bank: 0, expert }
}

// ---------------------------------------------------------------- the fake port

struct Held {
    offset: u64,
    memory: WritePart,
    then: Continuation<IoDoneOwned<WritePart>>,
}

struct FakePort {
    bytes: Vec<u8>,
    held: Mutex<VecDeque<Held>>,
}

impl FakePort {
    fn held(&self) -> usize {
        locked(&self.held).len()
    }

    /// Ends the oldest held read, failing it when `fail`.
    fn release(&self, fail: bool) {
        let Some(Held {
            offset,
            mut memory,
            then,
        }) = locked(&self.held).pop_front()
        else {
            return;
        };
        let len = memory.len();
        let result = if fail {
            Err(io::Error::from_raw_os_error(5))
        } else {
            memory
                .bytes_mut()
                .copy_from_slice(&self.bytes[offset as usize..][..len]);
            Ok(len)
        };
        let filled = result.as_ref().map_or(0, |&moved| moved);
        let mut frame = then(IoDoneOwned {
            buf: Some(memory),
            result,
            filled,
        });
        frame.execute().expect("the frame runs");
    }

    fn release_all(&self) {
        while self.held() > 0 {
            self.release(false);
        }
    }
}

impl PartPort for FakePort {
    fn start(
        &self,
        offset: u64,
        _direct: bool,
        memory: WritePart,
        then: Continuation<IoDoneOwned<WritePart>>,
    ) -> io::Result<()> {
        locked(&self.held).push_back(Held { offset, memory, then });
        Ok(())
    }

    fn has_direct(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------- the fake lane

/// Holds what the cache publishes: frames for the lake, frames that wait for a copy's retirement and frames that wait
/// for room, and answers the submissions as a test says.
struct FakeLane {
    queue: QueueId,
    state: Mutex<LaneState>,
    /// Lake units admitted and not yet dropped.
    units: Arc<AtomicUsize>,
}

/// What one copy was asked to do: its destination, the bytes it would read, and what it waits behind.
struct Submission {
    device: u64,
    bytes: Vec<u8>,
    waits: Vec<Marker>,
}

#[derive(Default)]
struct LaneState {
    offered: VecDeque<Box<dyn Frame>>,
    after: VecDeque<(Marker, Box<dyn Frame>)>,
    room: VecDeque<Box<dyn Frame>>,
    submitted: Vec<Submission>,
    busy: usize,
    refuse: bool,
    refuse_admit: bool,
    /// How many of the next submissions queue the copy and then fail with no marker.
    uncertain: usize,
    /// How many of the next marks find the lane full.
    mark_busy: usize,
    /// Whether the lane refuses every mark.
    mark_refuse: bool,
    /// The values of the markers the lane has made.
    issued: u64,
    /// Whether the lake's hand-off of a frame, or the admission of a unit, panics.
    panic_offer: bool,
    panic_admit: bool,
}

struct Unit(Arc<AtomicUsize>);

impl Drop for Unit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl FakeLane {
    fn submitted(&self) -> usize {
        locked(&self.state).submitted.len()
    }

    /// Takes one entry out, with the lock released before the caller runs it (a frame calls back into the lane).
    fn next<T>(&self, take: impl FnOnce(&mut LaneState) -> Option<T>) -> Option<T> {
        take(&mut locked(&self.state))
    }

    /// Runs what the cache offered to the lake, including what that offers in turn.
    fn run_offered(&self) {
        while let Some(mut frame) = self.next(|state| state.offered.pop_front()) {
            frame.execute().expect("the frame runs");
        }
    }

    /// Retires every submitted copy.
    fn retire_all(&self) {
        while let Some((_, mut frame)) = self.next(|state| state.after.pop_front()) {
            frame.execute().expect("the frame runs");
        }
    }

    fn run_room(&self) {
        while let Some(mut frame) = self.next(|state| state.room.pop_front()) {
            frame.execute().expect("the frame runs");
        }
    }
}

impl Frames for FakeLane {
    fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        let mut state = locked(&self.state);
        if state.panic_offer {
            drop(state);
            drop(frame);
            panic!("the lake's hand-off of a frame panicked");
        }
        state.offered.push_back(frame);
        Ok(())
    }

    fn after_marker(&self, marker: Marker, frame: Box<dyn Frame>) {
        locked(&self.state).after.push_back((marker, frame));
    }

    fn admit(&self) -> Result<Box<dyn Send>, String> {
        if locked(&self.state).panic_admit {
            panic!("the lake's admission of a unit panicked");
        }
        if locked(&self.state).refuse_admit {
            return Err("the lake is closed".to_owned());
        }
        self.units.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Unit(Arc::clone(&self.units))))
    }
}

impl CopyLane for FakeLane {
    fn submit(&self, job: &CopyJob) -> Result<Marker, SubmitFailure> {
        let mut state = locked(&self.state);
        if state.busy > 0 {
            state.busy -= 1;
            return Err(SubmitFailure::Busy);
        }
        if state.refuse {
            return Err(SubmitFailure::Refused("the lane is shut".to_owned()));
        }
        state.submitted.push(Submission {
            device: job.device,
            bytes: job.staging.bytes()[..job.bytes].to_vec(),
            waits: job.waits.clone(),
        });
        state.issued += 1;
        if state.uncertain > 0 {
            state.uncertain -= 1;
            return Err(SubmitFailure::Uncertain(
                "the lane failed after the copy was queued".to_owned(),
            ));
        }
        Ok(Marker::new(self.queue, LaneId(0), state.issued))
    }

    fn mark(&self) -> Result<Marker, SubmitFailure> {
        let mut state = locked(&self.state);
        if state.mark_busy > 0 {
            state.mark_busy -= 1;
            return Err(SubmitFailure::Busy);
        }
        if state.mark_refuse {
            return Err(SubmitFailure::Refused("the lane takes no marker".to_owned()));
        }
        state.issued += 1;
        Ok(Marker::new(self.queue, LaneId(0), state.issued))
    }

    fn when_room(&self, frame: Box<dyn Frame>) {
        locked(&self.state).room.push_back(frame);
    }
}

// ---------------------------------------------------------------- the fake worker

/// The downstream of the owner's source: it keeps the frames the source pushes, and the test runs them as a worker would.
#[derive(Default)]
struct FakeReceiver {
    pushed: Mutex<VecDeque<Box<dyn Frame>>>,
}

impl FakeReceiver {
    fn waiting(&self) -> usize {
        locked(&self.pushed).len()
    }

    fn take(&self) -> Option<Box<dyn Frame>> {
        locked(&self.pushed).pop_front()
    }
}

impl LatticeTerminal for FakeReceiver {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}

impl LatticeReceiver for FakeReceiver {
    fn push(&self, frame: Box<dyn Frame>) {
        locked(&self.pushed).push_back(frame);
    }

    fn on_complete(&self) {}

    fn on_error(&self, _error: Failure) {}
}

// ---------------------------------------------------------------- the rig

struct Settings {
    device_slots: usize,
    staging_slots: u32,
    max_reads: usize,
    /// Slots in the device pool, when not the cache's.
    pool_slots: Option<u32>,
    /// The device slot size, when not the record's.
    device_slot_bytes: Option<u64>,
    /// Slots in the staging free list, when not the arena's.
    free_list: Option<u32>,
    /// Slots of a host tier between the store and the cache, when it has one.
    tier_slots: Option<usize>,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            device_slots: 2,
            staging_slots: 2,
            max_reads: 2,
            pool_slots: None,
            device_slot_bytes: None,
            free_list: None,
            tier_slots: None,
        }
    }
}

struct Rig {
    _dir: tempfile::TempDir,
    cache: ExpertCache,
    port: Arc<FakePort>,
    lane: Arc<FakeLane>,
    /// The downstream of the owner's source, when the rig is its worker.
    worker: Arc<FakeReceiver>,
    source: Arc<dyn LatticeSource>,
    artifact: Vec<u8>,
}

/// Hands the owner's source to the fake worker, as a lattice's attach does.
fn to_the_fake_worker(worker: Arc<FakeReceiver>) -> impl FnOnce(Arc<dyn LatticeSource>) -> Result<(), String> {
    move |source| {
        source.add_downstream(worker as Arc<dyn LatticeReceiver>);
        Ok(())
    }
}

/// What a request was told, once.
#[derive(Clone, Default)]
struct Told(Arc<Mutex<Option<Outcome>>>);

impl Told {
    fn take(&self) -> Option<Outcome> {
        locked(&self.0).take()
    }

    fn is_told(&self) -> bool {
        locked(&self.0).is_some()
    }

    fn lease(&self) -> ExpertLease {
        match self.take() {
            Some(Outcome::Leased(lease)) => lease,
            Some(other) => panic!("expected a lease, was told {other:?}"),
            None => panic!("the request was not told"),
        }
    }

    fn failure(&self) -> CacheError {
        match self.take() {
            Some(Outcome::Failed(error)) => error,
            Some(other) => panic!("expected a failure, was told {other:?}"),
            None => panic!("the request was not told"),
        }
    }

    fn cancelled(&self) -> bool {
        matches!(self.take(), Some(Outcome::Cancelled))
    }
}

fn bank(records: Vec<ExpertRecord>) -> ExpertBank {
    let projection = ExpertProjection {
        name: "gate".to_owned(),
        shape: vec![1],
        dtype: TensorDataType::Bf16,
        format: WeightFormat::Nvfp4,
        layout: WeightLayout::RowSplitK128V1,
        record_offset: 0,
        bytes: 1,
    };
    ExpertBank::new("experts", ComponentGroup::RoutedExpert, 0, vec![projection], records).expect("a bank")
}

fn pool(planner: &Arc<Planner>, slots: u32, slot_bytes: u64) -> Arc<TierPool> {
    let location = planner
        .location(LocationKind::Device { ordinal: 0 })
        .expect("the device location");
    let owner = planner.open_owner(OwnerSpec::new("cache", Priority(10), Scope::Epoch(1)));
    TierPool::new(planner, owner, location, 0x1000_0000, slot_bytes, slots).expect("a pool")
}

fn planner() -> Arc<Planner> {
    Planner::new([
        LocationSpec {
            kind: LocationKind::Device { ordinal: 0 },
            capacity: 1 << 30,
            reserve: 0,
        },
        LocationSpec {
            kind: LocationKind::HostPinned,
            capacity: 1 << 30,
            reserve: 0,
        },
    ])
}

fn free_list(planner: &Arc<Planner>, slots: u32, slot_bytes: u64) -> Arc<SlotArena> {
    let location = planner.location(LocationKind::HostPinned).expect("the host location");
    let owner = planner.open_owner(OwnerSpec::new("staging", Priority(10), Scope::Epoch(1)));
    let claim = planner
        .claim(owner, location, u64::from(slots) * slot_bytes, "staging slots")
        .expect("the claim fits");
    SlotArena::new(0, slot_bytes, slots, claim)
}

impl Rig {
    fn new(settings: Settings) -> Rig {
        Rig::try_new(settings).unwrap_or_else(|error| panic!("{error}"))
    }

    fn try_new(settings: Settings) -> Result<Rig, CacheError> {
        let worker = Arc::new(FakeReceiver::default());
        let attach = to_the_fake_worker(Arc::clone(&worker));
        Rig::build(settings, worker, attach)
    }

    /// A rig whose owner is attached by `attach`; `worker` is the fake worker when `attach` hands it the source.
    fn build(
        settings: Settings,
        worker: Arc<FakeReceiver>,
        attach: impl FnOnce(Arc<dyn LatticeSource>) -> Result<(), String>,
    ) -> Result<Rig, CacheError> {
        let artifact: Vec<u8> = (0..(EXPERTS as usize + 1) * RECORD as usize)
            .map(|position| (position * 7) as u8)
            .collect();
        let records = (0..u64::from(EXPERTS))
            .map(|index| {
                let offset = RECORD * (index + 1);
                ExpertRecord {
                    offset,
                    bytes: RECORD,
                    crc32: crc32fast::hash(&artifact[offset as usize..][..RECORD as usize]),
                }
            })
            .collect();
        let layout = Arc::new(ExpertLayout::new(&[bank(records)], artifact.len() as u64).expect("a layout"));
        let dir = tempfile::tempdir().expect("a directory");
        write(dir.path().join("artifact.bin"), &artifact).expect("the artifact is written");
        let port = Arc::new(FakePort {
            bytes: artifact.clone(),
            held: Mutex::new(VecDeque::new()),
        });
        let source = FileRecordSource::new(Arc::clone(&layout), Arc::clone(&port) as Arc<dyn PartPort>, 1, true)
            .expect("a source");
        let staging = HostArena::new(settings.staging_slots as usize, layout.slot_bytes()).expect("an arena");
        let (memory, tier) = match settings.tier_slots {
            Some(slots) => {
                let (memory, shards) = RamTier::new(Arc::clone(&layout), slots, 1)
                    .expect("a tier")
                    .into_parts();
                (Some(memory), shards)
            }
            None => (None, Vec::new()),
        };
        let store = Arc::new(ExpertStore::new(source, memory, Arc::clone(&staging)));
        let planner = planner();
        let device = DeviceSlots::lease(
            &pool(
                &planner,
                settings.pool_slots.unwrap_or(settings.device_slots as u32),
                settings.device_slot_bytes.unwrap_or(layout.max_record_bytes()),
            ),
            planner.open_owner(OwnerSpec::new("leaser", Priority(10), Scope::Epoch(1))),
            settings.device_slots,
        )?;
        let lane = Arc::new(FakeLane {
            queue: QueueId::next(),
            state: Mutex::new(LaneState::default()),
            units: Arc::new(AtomicUsize::new(0)),
        });
        let config = CacheConfig {
            store,
            tier,
            device,
            staging_slots: free_list(
                &planner,
                settings.free_list.unwrap_or(settings.staging_slots),
                layout.slot_bytes() as u64,
            ),
            max_reads: settings.max_reads,
            frames: Arc::clone(&lane) as Arc<dyn Frames>,
            copier: Arc::clone(&lane) as Arc<dyn CopyLane>,
        };
        let attached = Arc::new(Mutex::new(None));
        let keep = Arc::clone(&attached);
        let cache = ExpertCache::new(config, move |source| {
            *locked(&keep) = Some(Arc::clone(&source));
            attach(source)
        })?;
        let source = locked(&attached).take().expect("the cache attached its source");
        Ok(Rig {
            _dir: dir,
            cache,
            port,
            lane,
            worker,
            source,
            artifact,
        })
    }

    /// A worker's visits: the source is requested, and the frames it pushes run, until it pushes none. Nothing runs
    /// unless a worker visits, so a record posted is not applied before this.
    fn poll(&self) {
        for _ in 0..1000 {
            self.source.request(1);
            let Some(mut frame) = self.worker.take() else {
                return;
            };
            frame.execute().expect("the frame runs");
            // The executor sends a key home after the frame ran.
            frame.recycle();
        }
        panic!("the owner's turns never ran out: a turn keeps posting a record for the next");
    }

    /// Posts a request for `expert` and returns what it will be told, without a worker visiting.
    fn ask(&self, expert: u32, listener: Listener) -> Arc<AtomicBool> {
        let cancel = Arc::new(AtomicBool::new(false));
        self.cache
            .fetch(id(expert), false, Arc::clone(&cancel), listener)
            .expect("the request is accepted");
        cancel
    }

    /// Posts a request for `expert` and has a worker visit.
    fn fetch(&self, expert: u32) -> (Told, Arc<AtomicBool>) {
        let told = Told::default();
        let sink = told.clone();
        let cancel = self.ask(expert, Box::new(move |outcome| *locked(&sink.0) = Some(outcome)));
        self.poll();
        (told, cancel)
    }

    /// Moves every read, submission and retirement along until nothing is held.
    fn settle(&self) {
        for _ in 0..1000 {
            self.poll();
            let reads = self.port.held();
            let offered = self.lane.next(|state| Some(!state.offered.is_empty())).unwrap_or(false);
            let after = self.lane.next(|state| Some(!state.after.is_empty())).unwrap_or(false);
            if reads == 0 && !offered && !after {
                return;
            }
            self.port.release_all();
            self.lane.run_offered();
            self.lane.retire_all();
        }
        panic!("the reads, submissions and retirements never ran out: a load keeps starting another");
    }

    /// Fetches `expert` and moves it to a lease.
    fn lease(&self, expert: u32) -> ExpertLease {
        let (told, _) = self.fetch(expert);
        self.settle();
        told.lease()
    }

    fn record(&self, expert: u32) -> &[u8] {
        &self.artifact[(RECORD as usize) * (expert as usize + 1)..][..RECORD as usize]
    }
}

// ---------------------------------------------------------------- the contracts

#[test]
fn a_resident_expert_is_leased_at_once_and_a_miss_after_its_load() {
    let rig = Rig::new(Settings::default());
    let (told, _) = rig.fetch(0);
    assert!(!told.is_told(), "a miss is not leased before its copy is submitted");
    assert_eq!(rig.port.held(), 1, "the miss started a read");
    rig.settle();
    let lease = told.lease();
    assert_eq!(lease.id(), id(0));
    assert_eq!(lease.bytes(), RECORD);
    assert_eq!(rig.lane.submitted(), 1);
    drop(lease);
    let (hit, _) = rig.fetch(0);
    assert!(hit.is_told(), "a resident expert is leased by the request itself");
    drop(hit.lease());
    let stats = rig.cache.stats();
    assert_eq!((stats.misses, stats.hits, stats.transfers), (1, 1, 1));
    assert_eq!(stats.transferred_bytes, RECORD);
    assert_eq!(
        rig.lane.units.load(Ordering::SeqCst),
        0,
        "the load's lake unit ended with its copy"
    );
}

#[test]
fn a_request_for_an_expert_being_loaded_joins_its_load() {
    let rig = Rig::new(Settings::default());
    let (first, _) = rig.fetch(1);
    let (second, _) = rig.fetch(1);
    assert!(!second.is_told());
    assert_eq!(rig.port.held(), 1, "one read serves both");
    rig.port.release_all();
    rig.poll();
    rig.lane.run_offered();
    rig.poll();
    let (a, b) = (first.lease(), second.lease());
    assert_eq!(a.device_address(), b.device_address());
    assert!(
        b.ready().is_some(),
        "the joined lease waits for the copy that is still writing the slot"
    );
    rig.settle();
    let stats = rig.cache.stats();
    assert_eq!((stats.misses, stats.coalesced), (1, 1));
}

#[test]
fn the_copy_is_made_to_the_slots_address_from_the_experts_record() {
    let rig = Rig::new(Settings::default());
    let lease = rig.lease(3);
    let state = locked(&rig.lane.state);
    let copy = &state.submitted[0];
    assert_eq!(copy.device, lease.device_address());
    assert_eq!(copy.bytes, rig.record(3), "the copy reads the expert's own record");
}

#[test]
fn a_request_with_every_slot_leased_is_asked_again_after_a_release_and_not_before() {
    let rig = Rig::new(Settings {
        device_slots: 1,
        ..Settings::default()
    });
    let held = rig.lease(0);
    let (waiting, _) = rig.fetch(1);
    assert!(!waiting.is_told());
    assert_eq!(rig.port.held(), 0, "no read starts while the only slot is leased");
    let stats = rig.cache.stats();
    assert_eq!((stats.waits[0], stats.asked_again), (1, 0));
    // Other requests and their ends change other causes, not the slots: the waiter is not asked.
    let (other, _) = rig.fetch(0);
    drop(other.lease());
    assert_eq!(rig.cache.stats().asked_again, 0);
    drop(held);
    assert_eq!(
        rig.cache.stats().asked_again,
        0,
        "a release only posts: nothing runs until the owner's turn"
    );
    assert_eq!(rig.port.held(), 0);
    rig.poll();
    assert_eq!(rig.cache.stats().asked_again, 1, "the release asked it once");
    assert_eq!(rig.port.held(), 1, "and it began its load");
    rig.settle();
    drop(waiting.lease());
    let stats = rig.cache.stats();
    assert_eq!((stats.waits[0], stats.evictions), (1, 1));
}

#[test]
fn eviction_never_takes_a_leased_slot() {
    let rig = Rig::new(Settings::default());
    let (first, second) = (rig.lease(0), rig.lease(1));
    let (third, _) = rig.fetch(2);
    assert!(!third.is_told(), "both slots are leased: the request waits");
    assert_eq!(rig.cache.stats().evictions, 0);
    let (kept, freed) = (second.device_address(), first.device_address());
    drop(first);
    rig.settle();
    let third = third.lease();
    assert_eq!(third.device_address(), freed, "the released slot was the one taken");
    assert_ne!(third.device_address(), kept);
    assert_eq!(second.id(), id(1));
    assert_eq!(rig.cache.stats().evictions, 1);
}

#[test]
fn a_read_that_fails_fails_the_request_with_the_experts_name_and_frees_what_it_held() {
    let rig = Rig::new(Settings {
        device_slots: 1,
        staging_slots: 1,
        max_reads: 1,
        ..Settings::default()
    });
    let (told, _) = rig.fetch(4);
    rig.port.release(true);
    rig.poll();
    match told.failure() {
        CacheError::Transfer { bank, expert, .. } => assert_eq!((bank.as_str(), expert), ("experts", 4)),
        other => panic!("expected a transfer failure, got {other:?}"),
    }
    assert_eq!(rig.cache.stats().failed_transfers, 1);
    assert_eq!(rig.lane.units.load(Ordering::SeqCst), 0);
    drop(rig.lease(2));
}

#[test]
fn a_copy_the_lane_refuses_fails_the_request_with_the_experts_name_and_gives_its_staging_slot_back() {
    let rig = Rig::new(Settings {
        staging_slots: 1,
        ..Settings::default()
    });
    locked(&rig.lane.state).refuse = true;
    let (told, _) = rig.fetch(5);
    rig.settle();
    match told.failure() {
        CacheError::Transfer { bank, expert, why } => {
            assert_eq!((bank.as_str(), expert), ("experts", 5));
            assert!(why.contains("shut"), "the lane's reason is kept: {why}");
        }
        other => panic!("expected a transfer failure, got {other:?}"),
    }
    locked(&rig.lane.state).refuse = false;
    drop(rig.lease(5));
}

#[test]
fn a_lane_with_no_room_gets_the_copy_again_when_it_has_room() {
    let rig = Rig::new(Settings::default());
    locked(&rig.lane.state).busy = 1;
    let (told, _) = rig.fetch(0);
    rig.port.release_all();
    rig.poll();
    rig.lane.run_offered();
    assert_eq!(rig.lane.submitted(), 0);
    assert!(!told.is_told());
    rig.lane.run_room();
    rig.settle();
    drop(told.lease());
    assert_eq!(rig.lane.submitted(), 1);
}

#[test]
fn a_request_cancelled_while_waiting_is_ended_and_leaves_nothing_held() {
    let rig = Rig::new(Settings {
        device_slots: 1,
        ..Settings::default()
    });
    let held = rig.lease(0);
    let (waiting, cancel) = rig.fetch(1);
    rig.cache.cancel(&cancel);
    rig.poll();
    assert!(waiting.cancelled());
    drop(held);
    rig.poll();
    assert_eq!(rig.port.held(), 0, "the cancelled request never began its load");
    assert_eq!(rig.cache.stats().asked_again, 0);
}

#[test]
fn a_request_cancelled_before_its_copy_is_submitted_gets_no_lease_and_its_slot_is_free() {
    let rig = Rig::new(Settings {
        device_slots: 1,
        ..Settings::default()
    });
    let (told, cancel) = rig.fetch(0);
    cancel.store(true, Ordering::Release);
    rig.settle();
    assert!(told.cancelled());
    assert_eq!(rig.cache.stats().cancelled_misses, 1);
    drop(rig.lease(1));
}

#[test]
fn a_request_cancelled_between_its_read_and_its_copy_gets_no_lease_and_its_slot_is_free() {
    let rig = Rig::new(Settings {
        device_slots: 1,
        ..Settings::default()
    });
    let (told, cancel) = rig.fetch(0);
    rig.port.release_all();
    rig.poll();
    cancel.store(true, Ordering::Release);
    rig.lane.run_offered();
    rig.poll();
    assert!(
        told.cancelled(),
        "a request cancelled after its read ended is not leased"
    );
    rig.lane.retire_all();
    rig.poll();
    assert_eq!(rig.cache.stats().cancelled_misses, 1);
    drop(rig.lease(1));
}

#[test]
fn a_request_cancelled_before_its_read_ends_has_no_copy_made() {
    let rig = Rig::new(Settings {
        device_slots: 1,
        ..Settings::default()
    });
    let (told, cancel) = rig.fetch(0);
    cancel.store(true, Ordering::Release);
    rig.port.release_all();
    rig.poll();
    assert!(told.cancelled(), "the read ended for a request that was cancelled");
    assert_eq!(rig.lane.submitted(), 0, "no copy was made for it");
    drop(rig.lease(1));
}

#[test]
fn reads_in_flight_are_limited_and_the_next_miss_is_asked_again_when_one_ends() {
    let rig = Rig::new(Settings {
        device_slots: 3,
        staging_slots: 3,
        max_reads: 1,
        ..Settings::default()
    });
    let (first, _) = rig.fetch(0);
    let (second, _) = rig.fetch(1);
    assert_eq!(rig.port.held(), 1, "one read at a time");
    assert_eq!(rig.cache.stats().waits[3], 1);
    rig.port.release(false);
    rig.poll();
    assert_eq!(rig.cache.stats().asked_again, 1);
    assert_eq!(rig.port.held(), 1, "the second began when the first ended");
    rig.settle();
    drop((first.lease(), second.lease()));
}

#[test]
fn staging_buffers_in_use_hold_the_next_miss_until_a_copy_retires() {
    let rig = Rig::new(Settings {
        device_slots: 3,
        staging_slots: 1,
        max_reads: 2,
        ..Settings::default()
    });
    let (first, _) = rig.fetch(0);
    let (second, _) = rig.fetch(1);
    assert_eq!(rig.port.held(), 1);
    assert_eq!(rig.cache.stats().waits[2], 1);
    rig.port.release_all();
    rig.poll();
    rig.lane.run_offered();
    rig.poll();
    assert_eq!(rig.port.held(), 0, "the staging buffer is held until the copy retires");
    rig.lane.retire_all();
    rig.poll();
    assert_eq!(rig.port.held(), 1);
    rig.settle();
    drop((first.lease(), second.lease()));
}

#[test]
fn closing_waits_for_the_load_in_flight_and_its_unit_ends_with_the_copy() {
    let rig = Rig::new(Settings::default());
    let (told, _) = rig.fetch(0);
    rig.port.release_all();
    rig.poll();
    rig.lane.run_offered();
    rig.poll();
    let closed = rig.cache.close();
    rig.poll();
    assert!(!closed.is_set(), "the copy has not retired");
    assert_eq!(rig.lane.units.load(Ordering::SeqCst), 1);
    rig.lane.retire_all();
    rig.poll();
    assert!(
        closed.wait(HANG).is_ok(),
        "the close was never told after the copy retired"
    );
    assert_eq!(rig.lane.units.load(Ordering::SeqCst), 0);
    drop(told.lease());
}

#[test]
fn a_lease_that_is_never_dropped_does_not_hang_the_close_and_is_counted() {
    let rig = Rig::new(Settings::default());
    forget(rig.lease(0));
    let closed = rig.cache.close();
    rig.poll();
    assert!(
        closed.wait(HANG).is_ok(),
        "the close hung on a lease that was never dropped"
    );
    assert_eq!(rig.cache.stats().forced_leases, 1);
}

#[test]
fn a_dropped_lease_frees_its_slot_for_the_next_request() {
    let rig = Rig::new(Settings {
        device_slots: 1,
        ..Settings::default()
    });
    drop(rig.lease(0));
    let (next, _) = rig.fetch(1);
    assert!(rig.port.held() == 1, "the dropped lease's slot was not given back");
    rig.settle();
    drop(next.lease());
}

#[test]
fn a_lease_dropped_after_the_close_is_harmless() {
    let rig = Rig::new(Settings::default());
    let lease = rig.lease(0);
    let closed = rig.cache.close();
    rig.poll();
    assert!(closed.wait(HANG).is_ok(), "the close was never told");
    drop(lease);
    let (late, _) = rig.fetch(1);
    assert_eq!(
        late.failure(),
        CacheError::Closed,
        "a request after the close is refused with the error that names it"
    );
}

#[test]
fn the_cache_refuses_what_it_cannot_serve() {
    let error = Rig::try_new(Settings {
        device_slots: 0,
        ..Settings::default()
    })
    .err()
    .expect("no device slot");
    assert!(matches!(error, CacheError::Slots { slots: 0, .. }), "{error}");
    let error = Rig::try_new(Settings {
        max_reads: 0,
        ..Settings::default()
    })
    .err()
    .expect("no read");
    assert_eq!(error, CacheError::Reads { reads: 0 });
    let rig = Rig::new(Settings::default());
    let error = rig
        .cache
        .fetch(
            ExpertId {
                bank: 0,
                expert: EXPERTS,
            },
            false,
            Arc::new(AtomicBool::new(false)),
            Box::new(|_| {}),
        )
        .expect_err("no such expert");
    assert!(matches!(error, ExpertError::NoSuchExpert { .. }), "{error}");
}

#[test]
fn a_lease_fence_is_waited_for_by_the_next_copy_into_its_slot() {
    let rig = Rig::new(Settings {
        device_slots: 1,
        ..Settings::default()
    });
    let fence = Marker::new(QueueId::next(), LaneId(7), 42);
    let mut lease = rig.lease(0);
    lease.fence_after(fence);
    drop(lease);
    drop(rig.lease(1));
    let state = locked(&rig.lane.state);
    assert!(state.submitted[0].waits.is_empty());
    assert_eq!(
        state.submitted[1].waits,
        vec![fence],
        "the copy into the slot waits for the work that read it"
    );
}

#[test]
fn a_lake_that_admits_no_unit_fails_the_request_with_the_experts_name_and_frees_the_slot() {
    let rig = Rig::new(Settings {
        device_slots: 1,
        ..Settings::default()
    });
    locked(&rig.lane.state).refuse_admit = true;
    let (told, _) = rig.fetch(2);
    match told.failure() {
        CacheError::Transfer { bank, expert, why } => {
            assert_eq!((bank.as_str(), expert), ("experts", 2));
            assert!(why.contains("closed"), "{why}");
        }
        other => panic!("expected a transfer failure, got {other:?}"),
    }
    locked(&rig.lane.state).refuse_admit = false;
    drop(rig.lease(2));
}

#[test]
fn a_posted_record_is_applied_by_the_next_visit_of_a_worker_and_not_by_the_poster() {
    let rig = Rig::new(Settings::default());
    rig.source.request(1);
    assert_eq!(rig.worker.waiting(), 0, "no record waits: no turn frame is pushed");
    rig.ask(0, Box::new(|_| {}));
    assert_eq!(
        rig.port.held(),
        0,
        "a request posted is not applied on the posting thread"
    );
    assert_eq!(
        rig.worker.waiting(),
        0,
        "posting pushes nothing: only a worker's request does"
    );
    rig.source.request(1);
    assert_eq!(
        rig.worker.waiting(),
        1,
        "the request found a record and pushed the turn frame"
    );
    rig.source.request(1);
    assert_eq!(rig.worker.waiting(), 1, "one turn frame is out at a time");
    rig.poll();
    assert_eq!(rig.port.held(), 1, "the turn applied the request and began its load");
}

#[test]
fn a_copy_that_may_be_queued_holds_its_staging_slot_until_a_later_marker_retires() {
    let rig = Rig::new(Settings {
        staging_slots: 1,
        ..Settings::default()
    });
    locked(&rig.lane.state).uncertain = 1;
    let (first, _) = rig.fetch(0);
    let (second, _) = rig.fetch(1);
    rig.port.release_all();
    rig.poll();
    rig.lane.run_offered();
    rig.poll();
    match first.failure() {
        CacheError::Transfer { expert, .. } => assert_eq!(expert, 0),
        other => panic!("expected a transfer failure, got {other:?}"),
    }
    assert!(!second.is_told());
    assert_eq!(
        rig.port.held(),
        0,
        "the staging slot is not reused while the copy may still read it"
    );
    rig.settle();
    drop(second.lease());
    assert_eq!(rig.cache.stats().failed_transfers, 1);
}

#[test]
fn a_marker_the_lane_has_no_room_for_is_taken_again_when_it_has_room() {
    let rig = Rig::new(Settings {
        staging_slots: 1,
        ..Settings::default()
    });
    {
        let mut state = locked(&rig.lane.state);
        state.uncertain = 1;
        state.mark_busy = 1;
    }
    let (first, _) = rig.fetch(0);
    let (second, _) = rig.fetch(1);
    rig.port.release_all();
    rig.poll();
    rig.lane.run_offered();
    rig.poll();
    first.failure();
    assert_eq!(locked(&rig.lane.state).room.len(), 1, "the mark waits for room");
    assert!(!second.is_told());
    rig.lane.run_room();
    rig.settle();
    drop(second.lease());
}

#[test]
fn when_every_staging_slot_is_lost_the_waiting_requests_are_told_so() {
    let rig = Rig::new(Settings {
        staging_slots: 1,
        ..Settings::default()
    });
    {
        let mut state = locked(&rig.lane.state);
        state.uncertain = 1;
        state.mark_refuse = true;
    }
    let (first, _) = rig.fetch(0);
    let (second, _) = rig.fetch(1);
    rig.port.release_all();
    rig.poll();
    rig.lane.run_offered();
    rig.poll();
    first.failure();
    assert_eq!(
        second.failure(),
        CacheError::StagingLost { lost: 1 },
        "the request that waited for the lost slot was never told"
    );
}

#[test]
fn when_every_staging_slot_is_lost_a_miss_is_refused_and_a_resident_expert_is_still_served() {
    let rig = Rig::new(Settings {
        staging_slots: 1,
        ..Settings::default()
    });
    drop(rig.lease(3));
    {
        let mut state = locked(&rig.lane.state);
        state.uncertain = 1;
        state.mark_refuse = true;
    }
    let (first, _) = rig.fetch(0);
    rig.port.release_all();
    rig.poll();
    rig.lane.run_offered();
    rig.poll();
    first.failure();
    let (miss, _) = rig.fetch(1);
    assert_eq!(miss.failure(), CacheError::StagingLost { lost: 1 });
    let (hit, _) = rig.fetch(3);
    assert_eq!(hit.lease().id(), id(3));
}

#[test]
fn a_panic_applying_a_record_poisons_the_owner_and_every_request_is_told_the_error() {
    let rig = Rig::new(Settings {
        device_slots: 1,
        ..Settings::default()
    });
    let (loading, _) = rig.fetch(0);
    let (waiting, _) = rig.fetch(1);
    locked(&rig.lane.state).panic_offer = true;
    rig.port.release_all();
    rig.poll();
    assert!(matches!(loading.failure(), CacheError::Poisoned { .. }));
    assert!(matches!(waiting.failure(), CacheError::Poisoned { .. }));
    let (late, _) = rig.fetch(2);
    match late.failure() {
        CacheError::Poisoned { why } => assert!(why.contains("hand-off"), "the panic's text is kept: {why}"),
        other => panic!("expected a poisoned owner, got {other:?}"),
    }
    locked(&rig.lane.state).panic_offer = false;
    let closed = rig.cache.close();
    rig.poll();
    assert!(
        closed.wait(HANG).is_ok(),
        "the close never ended after a panic poisoned the owner"
    );
    assert_eq!(rig.lane.units.load(Ordering::SeqCst), 0, "the load's unit never ended");
}

#[test]
fn a_request_in_the_record_that_panicked_is_told_it_was_abandoned() {
    let rig = Rig::new(Settings::default());
    locked(&rig.lane.state).panic_admit = true;
    let (told, _) = rig.fetch(0);
    assert!(matches!(told.failure(), CacheError::Abandoned { .. }));
    let (late, _) = rig.fetch(1);
    assert!(matches!(late.failure(), CacheError::Poisoned { .. }));
}

#[test]
fn a_listener_that_panics_does_not_stop_the_load_from_retiring_or_the_close_or_the_listeners_after_it() {
    let rig = Rig::new(Settings::default());
    rig.ask(0, Box::new(|_| panic!("the listener of the first request panicked")));
    let (second, _) = rig.fetch(1);
    rig.settle();
    drop(second.lease());
    let closed = rig.cache.close();
    rig.poll();
    assert!(
        closed.wait(HANG).is_ok(),
        "the close never ended after a listener panicked while a copy was submitted"
    );
    assert_eq!(rig.lane.units.load(Ordering::SeqCst), 0, "the load's unit never ended");
    let (late, _) = rig.fetch(2);
    assert_eq!(
        late.failure(),
        CacheError::Closed,
        "a listener's panic poisoned the owner"
    );
}

#[test]
fn a_cache_whose_source_the_lattice_refuses_is_not_built() {
    let error = Rig::build(Settings::default(), Arc::default(), |_| {
        Err("the lattice is closed".to_owned())
    })
    .err()
    .expect("a cache that nothing would serve is refused");
    assert_eq!(
        error,
        CacheError::Attach {
            why: "the lattice is closed".to_owned()
        }
    );
}

#[test]
fn a_lake_whose_hand_off_panics_as_a_load_begins_does_not_leave_the_load_to_hang_the_close() {
    let rig = Rig::new(Settings {
        device_slots: 1,
        tier_slots: Some(3),
        ..Settings::default()
    });
    // Two experts through the one device slot leave the first in the host tier, whose load is then one frame.
    drop(rig.lease(0));
    drop(rig.lease(1));
    locked(&rig.lane.state).panic_offer = true;
    let (told, _) = rig.fetch(0);
    assert!(
        matches!(told.failure(), CacheError::Poisoned { .. }),
        "the request of the load whose frame was lost was not told the poison"
    );
    assert_eq!(
        rig.lane.units.load(Ordering::SeqCst),
        0,
        "the load whose read was never handed off still holds its unit"
    );
    locked(&rig.lane.state).panic_offer = false;
    let closed = rig.cache.close();
    rig.poll();
    assert!(
        closed.wait(HANG).is_ok(),
        "the close never ended: a load whose read was never handed off was never ended"
    );
}

#[test]
fn a_record_posted_while_a_turn_frame_is_out_is_applied_by_the_next_visit_that_finds_it() {
    let rig = Rig::new(Settings {
        device_slots: 1,
        ..Settings::default()
    });
    // The listener drops its lease: a record posted by a listener, after the turn's drain.
    rig.ask(0, Box::new(drop));
    rig.poll();
    rig.port.release_all();
    rig.poll();
    rig.lane.run_offered();
    rig.lane.retire_all();
    rig.source.request(1);
    let mut turn = rig
        .worker
        .take()
        .expect("the submission is a record waiting: the turn frame is pushed");
    turn.execute().expect("the turn runs");
    rig.source.request(1);
    assert_eq!(
        rig.worker.waiting(),
        0,
        "a second turn frame was pushed while one is out"
    );
    turn.recycle();
    rig.source.request(1);
    assert_eq!(
        rig.worker.waiting(),
        1,
        "the release the listener posted during the turn was stranded: the frame that came home pushed nothing"
    );
    let (second, _) = rig.fetch(1);
    assert_eq!(
        rig.port.held(),
        1,
        "the release was not applied: the only slot is still pinned"
    );
    assert!(!second.is_told());
}

#[test]
fn a_record_posted_from_outside_the_lattice_is_applied_by_a_worker_of_a_real_lattice() {
    let runtime = RuntimeSettings::with_model(ModelSettings::default());
    let lattice = ControlPlaneLattice::new(lattice_config(&runtime)).expect("the lattice starts");
    let rig = Rig::build(Settings::default(), Arc::default(), |source| {
        lattice
            .add_upstream(source)
            .map(drop)
            .map_err(|error| error.to_string())
    })
    .expect("the cache is built");
    let told = Handoff::<Outcome>::new();
    let sink = Arc::clone(&told);
    // A request already cancelled needs no lane: it is ended by the turn and told.
    let cancel = Arc::new(AtomicBool::new(true));
    rig.cache
        .fetch(id(0), false, cancel, Box::new(move |outcome| sink.send(outcome)))
        .expect("the request is accepted");
    assert!(
        matches!(told.wait(HANG), Ok(Outcome::Cancelled)),
        "no worker of the lattice applied the request posted from outside it"
    );
    let closed = rig.cache.close();
    assert!(
        closed.wait(HANG).is_ok(),
        "no worker of the lattice applied the close posted from outside it"
    );
    lattice.close();
}

#[test]
fn a_closed_cache_that_is_done_refuses_a_request_on_the_calling_thread() {
    let rig = Rig::new(Settings::default());
    let closed = rig.cache.close();
    rig.poll();
    assert!(closed.wait(HANG).is_ok(), "the close never ended");
    assert!(rig.source.is_complete(), "a closed cache's source was never completed");
    let told = Told::default();
    let sink = told.clone();
    rig.ask(0, Box::new(move |outcome| *locked(&sink.0) = Some(outcome)));
    assert_eq!(
        told.failure(),
        CacheError::Closed,
        "a request after the source completed was left for a worker that will never come"
    );
    assert!(
        rig.cache.close().wait(HANG).is_ok(),
        "a second close of a done cache never ended"
    );
}

#[test]
fn a_request_posted_after_the_source_completed_is_told_closed_once_by_its_poster() {
    let rig = Rig::new(Settings::default());
    let closed = rig.cache.close();
    rig.poll();
    assert!(closed.wait(HANG).is_ok(), "the close never ended");
    // The interleaving: `fetch` passed its check while the source was live, and the source completed before the post.
    let told = Told::default();
    let sink = told.clone();
    let told_twice = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&told_twice);
    rig.cache.send(Request {
        id: id(0),
        key: 0,
        bytes: RECORD,
        scan: false,
        cancel: Arc::new(AtomicBool::new(false)),
        reply: Reply::new(
            Box::new(move |outcome| {
                counted.fetch_add(1, Ordering::SeqCst);
                *locked(&sink.0) = Some(outcome);
            }),
            Weak::new(),
        ),
    });
    assert_eq!(
        told.failure(),
        CacheError::Closed,
        "a request posted after the source completed was left silent"
    );
    rig.poll();
    assert_eq!(told_twice.load(Ordering::SeqCst), 1, "the request was told twice");
}

#[test]
fn a_retirement_frame_the_lake_drops_without_running_still_ends_the_load_and_the_close() {
    let rig = Rig::new(Settings::default());
    let (told, _) = rig.fetch(0);
    rig.port.release_all();
    rig.poll();
    rig.lane.run_offered();
    rig.poll();
    drop(told.lease());
    locked(&rig.lane.state).after.clear();
    let closed = rig.cache.close();
    rig.poll();
    assert!(
        closed.wait(HANG).is_ok(),
        "the close never ended: the retirement frame was dropped and posted nothing"
    );
    assert_eq!(rig.lane.units.load(Ordering::SeqCst), 0, "the load's unit never ended");
}

#[test]
fn a_quarantine_frame_the_lake_drops_without_running_loses_the_slot() {
    let rig = Rig::new(Settings {
        staging_slots: 1,
        ..Settings::default()
    });
    locked(&rig.lane.state).uncertain = 1;
    let (first, _) = rig.fetch(0);
    let (second, _) = rig.fetch(1);
    rig.port.release_all();
    rig.poll();
    rig.lane.run_offered();
    locked(&rig.lane.state).after.clear();
    rig.poll();
    first.failure();
    assert_eq!(
        second.failure(),
        CacheError::StagingLost { lost: 1 },
        "the request waiting for the staging slot was never told: the dropped frame posted nothing"
    );
}

#[test]
fn the_pool_must_hold_the_slots_and_the_slots_the_record() {
    let error = Rig::try_new(Settings {
        device_slots: 3,
        pool_slots: Some(2),
        ..Settings::default()
    })
    .err()
    .expect("a pool too small");
    assert_eq!(error, CacheError::PoolTooSmall { wanted: 3, free: 2 });
    let error = Rig::try_new(Settings {
        device_slot_bytes: Some(RECORD - 1),
        ..Settings::default()
    })
    .err()
    .expect("a slot too small");
    assert!(
        matches!(error, CacheError::SlotTooSmall { what: "device", .. }),
        "{error}"
    );
    let error = Rig::try_new(Settings {
        free_list: Some(3),
        ..Settings::default()
    })
    .err()
    .expect("a free list longer than the arena");
    assert_eq!(error, CacheError::StagingSlots { free_list: 3, arena: 2 });
}
