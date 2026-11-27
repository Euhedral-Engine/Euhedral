//! The cache's one owner: its device slots' bookkeeping and the host tier's, and the loads in flight.
//!
//! Requests, lease releases, read completions, copy submissions and copy retirements are records posted to it; the one
//! worker that holds its claim applies them in order, with no lock and no wait. A request that cannot be served (every
//! slot is loading or leased, every staging buffer or read is in use) stays here with the cause, and is asked again
//! when that cause has changed, once per change.
//!
//! A load is: the owner reserves a device slot, a staging slot and a read, and begins the store's load; the store's
//! frames read the record; its end posts [`Message::Read`]; the owner offers the copy's frame, which a worker submits on
//! the copy lane and answers with [`Message::Submitted`] (the lease, which carries the copy's marker, is told then);
//! the marker's retirement runs a frame that posts [`Message::Retired`], which frees the staging slot and ends the load.
//! The load holds one unit of the lake from its start to its retirement, so closing waits for it.

use std::collections::{HashMap, VecDeque};
use std::fmt::{self, Display, Formatter};
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use euhedral_core::frames::Frame;
use euhedral_data_structures::queues::{Consumer, Single};
use euhedral_graph::Marker;
use euhedral_memory::{SlotArena, SlotLease};

use super::CacheError;
use super::host::{CopyJob, CopyLane, DeviceSlots, Frames, SubmitFailure};
use super::lease::ExpertLease;
use super::shard::{Claimed, Reserved, Shard};
use super::stats::{CacheStats, Cause};
use crate::frames::{Kind, step};
use crate::generation::{Address, Turn};
use crate::handoff::Handoff;
use crate::residency::experts::{
    ExpertError, ExpertId, ExpertLayout, ExpertStore, HostArena, LoadEnd, RamTierShard, ReadOutcome, Started,
    StoreLoad, TierDirective, TierSettle, shard_of,
};

/// What a request came to.
#[derive(Debug)]
pub enum Outcome {
    /// The expert is leased.
    Leased(ExpertLease),
    /// The expert could not be had; the error names it.
    Failed(CacheError),
    /// The request was cancelled, or the cache closed, before a lease was made.
    Cancelled,
}

/// Told once with a request's outcome, on a worker, once the owner's claim is free. It must not wait.
pub type Listener = Box<dyn FnOnce(Outcome) + Send>;

/// One request for an expert.
pub(crate) struct Request {
    pub id: ExpertId,
    pub key: usize,
    pub bytes: u64,
    /// Whether the load is a prefill scan's, which the host tier may refuse to keep.
    pub scan: bool,
    pub cancel: Arc<AtomicBool>,
    pub listener: Listener,
}

/// What is posted to the owner.
pub(crate) enum Message {
    Request(Request),
    /// A lease was dropped.
    Release {
        slot: u32,
        generation: u64,
        fence: Option<Marker>,
    },
    /// A store load ended.
    Read(LoadEnd),
    /// The copy of the load of `id` was submitted, or why it was not.
    Submitted {
        id: ExpertId,
        result: Result<Marker, String>,
    },
    /// The copy of the load of `id` retired.
    Retired {
        id: ExpertId,
    },
    /// Some request was cancelled: those waiting here end.
    Sweep,
}

/// An outcome to tell, or the end of a shutdown.
pub(crate) enum Finished {
    Told(Listener, Outcome),
    Idle(Arc<Handoff<()>>),
}

/// The load of one expert, from the reservation of its slots to the retirement of its copy.
struct Load {
    request: Option<Request>,
    slot: u32,
    generation: u64,
    bytes: u64,
    fences: Vec<Marker>,
    staging: SlotLease,
    cancel: Arc<AtomicBool>,
    /// The lake unit the load holds until its copy retires.
    _unit: Box<dyn Send>,
}

/// A request that could not start, and what it waits for.
struct Waiting {
    request: Request,
    cause: Cause,
    epoch: u64,
}

/// What an owner is built from.
pub(super) struct Parts {
    pub layout: Arc<ExpertLayout>,
    pub store: Arc<ExpertStore>,
    pub tier: Vec<RamTierShard>,
    pub device: DeviceSlots,
    pub staging: Arc<HostArena>,
    pub staging_slots: Arc<SlotArena>,
    pub max_reads: usize,
    pub frames: Arc<dyn Frames>,
    pub copier: Arc<dyn CopyLane>,
    pub stats: Arc<CacheStats>,
}

pub(crate) struct Owner {
    inbox: Consumer<Message, Single>,
    address: Arc<Address<Owner>>,
    layout: Arc<ExpertLayout>,
    store: Arc<ExpertStore>,
    tier: Vec<RamTierShard>,
    device: DeviceSlots,
    shard: Shard,
    staging: Arc<HostArena>,
    staging_slots: Arc<SlotArena>,
    max_reads: usize,
    reads: usize,
    frames: Arc<dyn Frames>,
    copier: Arc<dyn CopyLane>,
    stats: Arc<CacheStats>,
    loads: HashMap<ExpertId, Load>,
    blocked: VecDeque<Waiting>,
    /// How many times each cause (by [`Cause`] order) has changed.
    epochs: [u64; 4],
    told: Vec<Finished>,
    closed: bool,
    closing: Vec<Arc<Handoff<()>>>,
}

impl Owner {
    pub(super) fn new(
        inbox: Consumer<Message, Single>,
        address: Arc<Address<Owner>>,
        parts: Parts,
    ) -> Result<Owner, CacheError> {
        let Parts {
            layout,
            store,
            tier,
            device,
            staging,
            staging_slots,
            max_reads,
            frames,
            copier,
            stats,
        } = parts;
        let record = layout.max_record_bytes();
        if device.slot_bytes() < record {
            return Err(CacheError::SlotTooSmall {
                what: "device",
                slot_bytes: device.slot_bytes(),
                record,
            });
        }
        if (staging.slot_bytes() as u64) < record {
            return Err(CacheError::SlotTooSmall {
                what: "staging",
                slot_bytes: staging.slot_bytes() as u64,
                record,
            });
        }
        if staging_slots.capacity() == 0 || staging_slots.capacity() as usize > staging.slots() {
            return Err(CacheError::StagingSlots {
                free_list: staging_slots.capacity() as usize,
                arena: staging.slots(),
            });
        }
        if max_reads == 0 {
            return Err(CacheError::Reads { reads: max_reads });
        }
        let shard = Shard::new(layout.keys(), device.len(), Arc::clone(&stats))?;
        Ok(Owner {
            inbox,
            address,
            layout,
            store,
            tier,
            device,
            shard,
            staging,
            staging_slots,
            max_reads,
            reads: 0,
            frames,
            copier,
            stats,
            loads: HashMap::new(),
            blocked: VecDeque::new(),
            epochs: [0; 4],
            told: Vec::new(),
            closed: false,
            closing: Vec::new(),
        })
    }

    fn handle(&mut self, message: Message) {
        match message {
            Message::Request(request) => self.ask(request, false),
            Message::Release {
                slot,
                generation,
                fence,
            } => {
                if self.shard.release(slot, generation, fence) {
                    self.changed(Cause::Slots);
                }
            }
            Message::Read(end) => self.read_ended(end),
            Message::Submitted { id, result } => self.submitted(id, result),
            Message::Retired { id } => self.retired(id),
            Message::Sweep => self.sweep(),
        }
    }

    fn changed(&mut self, cause: Cause) {
        self.epochs[cause as usize] += 1;
    }

    /// Why a miss could not start now, if it could not.
    fn blocked_by(&self) -> Option<Cause> {
        if self.staging_slots.leased() >= self.staging_slots.capacity() {
            Some(Cause::Staging)
        } else if self.reads >= self.max_reads {
            Some(Cause::Reads)
        } else {
            None
        }
    }

    /// Serves `request`: a lease now, a load started, or a place in the waiting list. `again` says it was asked before.
    fn ask(&mut self, request: Request, again: bool) {
        if self.closed || request.cancel.load(Ordering::Acquire) {
            self.told.push(Finished::Told(request.listener, Outcome::Cancelled));
            return;
        }
        let claimed = match self.shard.claim(request.key, request.bytes, self.blocked_by()) {
            Ok(claimed) => claimed,
            Err(error) => {
                self.told.push(Finished::Told(request.listener, Outcome::Failed(error)));
                return;
            }
        };
        match claimed {
            Claimed::Resident { slot, generation } => {
                let lease = self.lease(request.id, slot, generation, request.bytes, None);
                self.told.push(Finished::Told(request.listener, Outcome::Leased(lease)));
            }
            Claimed::Joined {
                slot,
                generation,
                ready,
            } => {
                let lease = self.lease(request.id, slot, generation, request.bytes, Some(ready));
                self.told.push(Finished::Told(request.listener, Outcome::Leased(lease)));
            }
            Claimed::Miss(reserved) => self.begin(request, reserved),
            Claimed::Wait(cause) => {
                if !again {
                    self.stats.waited(cause);
                }
                let epoch = self.epochs[cause as usize];
                self.blocked.push_back(Waiting { request, cause, epoch });
            }
        }
    }

    fn lease(&self, id: ExpertId, slot: u32, generation: u64, bytes: u64, ready: Option<Marker>) -> ExpertLease {
        ExpertLease::new(
            Arc::clone(&self.address),
            id,
            slot,
            generation,
            self.device.address(slot as usize),
            bytes,
            ready,
        )
    }

    /// Asks again each waiting request whose cause has changed since it waited, in the order they came.
    fn ask_changed(&mut self) {
        for waiting in mem::take(&mut self.blocked) {
            if self.epochs[waiting.cause as usize] == waiting.epoch {
                self.blocked.push_back(waiting);
            } else {
                self.stats.asked_again();
                self.ask(waiting.request, true);
            }
        }
    }

    /// Ends the waiting requests that were cancelled.
    fn sweep(&mut self) {
        for waiting in mem::take(&mut self.blocked) {
            if waiting.request.cancel.load(Ordering::Acquire) {
                self.told
                    .push(Finished::Told(waiting.request.listener, Outcome::Cancelled));
            } else {
                self.blocked.push_back(waiting);
            }
        }
    }

    fn failure(&self, id: ExpertId, why: impl Display) -> CacheError {
        CacheError::Transfer {
            bank: self.layout.bank_name(id.bank).to_owned(),
            expert: id.expert,
            why: why.to_string(),
        }
    }

    /// Starts the load of `request`'s expert into the slot just reserved.
    fn begin(&mut self, request: Request, reserved: Reserved) {
        let Reserved {
            slot,
            generation,
            fences,
        } = reserved;
        let id = request.id;
        let Some(staging) = self.staging_slots.lease() else {
            // `blocked_by` saw a free staging slot and only this owner takes one: not reached.
            let error = self.failure(id, "no staging slot was free after one was counted free");
            return self.give_up(request, slot, generation, None, Outcome::Failed(error));
        };
        let unit = match self.frames.admit() {
            Ok(unit) => unit,
            Err(why) => {
                let error = self.failure(id, why);
                return self.give_up(request, slot, generation, None, Outcome::Failed(error));
            }
        };
        let directive = match self.plan(id, request.key, request.scan) {
            Ok(directive) => directive,
            Err(error) => {
                return self.give_up(
                    request,
                    slot,
                    generation,
                    None,
                    Outcome::Failed(self.failure(id, error)),
                );
            }
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let address = Arc::clone(&self.address);
        let load = StoreLoad {
            id,
            directive,
            staging_slot: staging.index() as usize,
            cancel: Arc::clone(&cancel),
            done: Box::new(move |end| address.post(Message::Read(end))),
        };
        let started = match self.store.begin(load) {
            Ok(started) => started,
            Err(error) => {
                let failure = self.failure(id, error);
                return self.give_up(request, slot, generation, Some(directive), Outcome::Failed(failure));
            }
        };
        let bytes = request.bytes;
        self.loads.insert(
            id,
            Load {
                request: Some(request),
                slot,
                generation,
                bytes,
                fences,
                staging,
                cancel,
                _unit: unit,
            },
        );
        self.reads += 1;
        if let Started::Frame(frame) = started
            && self.frames.offer(frame).is_err()
        {
            // The frame carried the load's end and is gone with its lease: the load can only be ended here.
            self.reads -= 1;
            let error = self.failure(id, "the lake took no frame");
            self.end_load(id, directive.settle_unstarted(), Outcome::Failed(error));
        }
    }

    fn plan(&mut self, id: ExpertId, key: usize, scan: bool) -> Result<TierDirective, ExpertError> {
        if self.tier.is_empty() {
            return Ok(TierDirective::Bypass);
        }
        let index = shard_of(key, self.tier.len());
        self.tier[index].plan(id, scan)
    }

    /// Ends a load that never became one: the reservation goes back, and the tier's pin with it.
    fn give_up(
        &mut self,
        request: Request,
        slot: u32,
        generation: u64,
        planned: Option<TierDirective>,
        outcome: Outcome,
    ) {
        if let Some(settle) = planned.and_then(|directive| directive.settle_unstarted()) {
            self.settle(request.key, settle);
        }
        if let Err(error) = self.shard.abandon(slot, generation) {
            log::error!("an expert load that never began could not be given back: {error}");
        }
        self.stats.failed_transfer();
        self.changed(Cause::Slots);
        self.changed(Cause::Copying);
        self.told.push(Finished::Told(request.listener, outcome));
    }

    fn settle(&mut self, key: usize, settle: TierSettle) {
        if self.tier.is_empty() {
            return;
        }
        let index = shard_of(key, self.tier.len());
        if let Err(error) = self.tier[index].apply(settle) {
            log::error!("the host tier refused a settle: {error}");
        }
    }

    /// The store's load of `end.id` ended.
    fn read_ended(&mut self, end: LoadEnd) {
        self.reads = self.reads.saturating_sub(1);
        self.changed(Cause::Reads);
        let id = end.id;
        let key = self.layout.key(id).unwrap_or(0);
        if let Some(settle) = end.settle {
            self.settle(key, settle);
        }
        let Some(load) = self.loads.get(&id) else {
            log::error!("a read ended for an expert with no load: {id:?}");
            return;
        };
        let cancelled = self.closed
            || load.cancel.load(Ordering::Acquire)
            || load
                .request
                .as_ref()
                .is_some_and(|request| request.cancel.load(Ordering::Acquire));
        match end.outcome {
            ReadOutcome::Complete if !cancelled => self.submit(id),
            ReadOutcome::Complete | ReadOutcome::Cancelled { .. } => self.end_load(id, None, Outcome::Cancelled),
            ReadOutcome::Failed(error) => {
                let failure = self.failure(id, error);
                self.end_load(id, None, Outcome::Failed(failure));
            }
            ReadOutcome::Lost => {
                let failure = self.failure(id, "the read's memory was lost");
                self.end_load(id, None, Outcome::Failed(failure));
            }
        }
    }

    /// Offers the frame that submits the copy of the record in the load's staging slot.
    fn submit(&mut self, id: ExpertId) {
        let Some(load) = self.loads.get(&id) else {
            return;
        };
        let (slot, bytes) = (load.slot, load.bytes);
        let staging_slot = load.staging.index() as usize;
        let staging = match self.staging.read(staging_slot) {
            Ok(staging) => staging,
            Err(error) => {
                let failure = self.failure(id, error);
                return self.end_load(id, None, Outcome::Failed(failure));
            }
        };
        let job = CopyJob {
            device: self.device.address(slot as usize),
            bytes: bytes as usize,
            staging,
            waits: load.fences.clone(),
        };
        let frame = copy_frame(
            Arc::clone(&self.copier),
            Arc::clone(&self.frames),
            Arc::clone(&self.address),
            id,
            slot as usize,
            job,
        );
        if self.frames.offer(frame).is_err() {
            let failure = self.failure(id, "the lake took no frame");
            self.end_load(id, None, Outcome::Failed(failure));
        }
    }

    fn submitted(&mut self, id: ExpertId, result: Result<Marker, String>) {
        let Some(load) = self.loads.get_mut(&id) else {
            log::error!("a copy was submitted for an expert with no load: {id:?}");
            return;
        };
        let (slot, generation, bytes) = (load.slot, load.generation, load.bytes);
        match result {
            Err(why) => {
                let failure = self.failure(id, why);
                self.end_load(id, None, Outcome::Failed(failure));
            }
            Ok(marker) => {
                if let Err(error) = self.shard.submitted(slot, generation, marker) {
                    log::error!("{error}");
                    return;
                }
                self.changed(Cause::Copying);
                let lease = self.lease(id, slot, generation, bytes, Some(marker));
                let Some(request) = self.loads.get_mut(&id).and_then(|load| load.request.take()) else {
                    return;
                };
                let outcome = if self.closed || request.cancel.load(Ordering::Acquire) {
                    // The lease is dropped here: it posts its own release, and the slot is unpinned in this turn.
                    drop(lease);
                    self.stats.cancelled_miss();
                    Outcome::Cancelled
                } else {
                    Outcome::Leased(lease)
                };
                self.told.push(Finished::Told(request.listener, outcome));
            }
        }
    }

    fn retired(&mut self, id: ExpertId) {
        let Some(load) = self.loads.remove(&id) else {
            log::error!("a copy retired for an expert with no load: {id:?}");
            return;
        };
        match self.shard.retired(load.slot, load.generation) {
            Ok(true) => self.changed(Cause::Slots),
            Ok(false) => {}
            Err(error) => log::error!("{error}"),
        }
        // The staging slot goes back to the free list as the load drops, and the lake unit with it.
        drop(load);
        self.changed(Cause::Staging);
    }

    /// Ends the load of `id` that has not submitted its copy, as `outcome`, telling its request if it still waits.
    fn end_load(&mut self, id: ExpertId, settle: Option<TierSettle>, outcome: Outcome) {
        let Some(mut load) = self.loads.remove(&id) else {
            return;
        };
        if let Some(settle) = settle
            && let Ok(key) = self.layout.key(id)
        {
            self.settle(key, settle);
        }
        if let Err(error) = self.shard.abandon(load.slot, load.generation) {
            log::error!("{error}");
        }
        match &outcome {
            Outcome::Cancelled => self.stats.cancelled_miss(),
            _ => self.stats.failed_transfer(),
        }
        if let Some(request) = load.request.take() {
            self.told.push(Finished::Told(request.listener, outcome));
        }
        drop(load);
        self.changed(Cause::Slots);
        self.changed(Cause::Copying);
        self.changed(Cause::Staging);
    }

    /// Tells the handoffs of a close once nothing is in flight, and forces the leases still out shut.
    fn signal_closed(&mut self) {
        if !self.closed || !self.loads.is_empty() {
            return;
        }
        if let Err(error) = self.shard.close() {
            log::error!("{error}");
        }
        for closed in mem::take(&mut self.closing) {
            self.told.push(Finished::Idle(closed));
        }
    }
}

/// The frame that submits `job` on the copy lane and tells the owner. A lane with no room hands the job to a frame
/// that runs once there is; nothing waits.
fn copy_frame(
    copier: Arc<dyn CopyLane>,
    frames: Arc<dyn Frames>,
    owner: Arc<Address<Owner>>,
    id: ExpertId,
    chain: usize,
    job: CopyJob,
) -> Box<dyn Frame> {
    step(Kind::ExpertSubmit, chain, move || match copier.submit(&job) {
        Ok(marker) => {
            // The owner hears of the submission before the retirement can be: the message is in its inbox before the
            // frame that posts the retirement is registered.
            owner.post(Message::Submitted { id, result: Ok(marker) });
            let retired = Arc::clone(&owner);
            frames.after_marker(
                marker,
                step(Kind::ExpertRetire, 0, move || {
                    // The staging bytes were read by the copy that has now retired.
                    drop(job);
                    retired.post(Message::Retired { id });
                }),
            );
        }
        Err(SubmitFailure::Busy) => {
            let again = copy_frame(Arc::clone(&copier), frames, owner, id, chain, job);
            copier.when_room(again);
        }
        Err(SubmitFailure::Refused(why)) => owner.post(Message::Submitted { id, result: Err(why) }),
        Err(SubmitFailure::Uncertain(why)) => {
            // The copy may still read the staging slot: it is never reused.
            mem::forget(job);
            owner.post(Message::Submitted { id, result: Err(why) });
        }
    })
}

impl Turn for Owner {
    type Message = Message;
    type Finished = Finished;

    fn turn(&mut self) {
        while let Some(message) = self.inbox.poll() {
            self.handle(message);
        }
        self.ask_changed();
        self.signal_closed();
    }

    fn take_finished(&mut self) -> Vec<Finished> {
        mem::take(&mut self.told)
    }

    fn abandon(&mut self, why: &str) -> Vec<Finished> {
        self.closed = true;
        for load in self.loads.values() {
            load.cancel.store(true, Ordering::Release);
        }
        for waiting in mem::take(&mut self.blocked) {
            let error = CacheError::Abandoned { why: why.to_owned() };
            self.told
                .push(Finished::Told(waiting.request.listener, Outcome::Failed(error)));
        }
        self.signal_closed();
        mem::take(&mut self.told)
    }

    fn deliver(finished: Finished) {
        match finished {
            Finished::Told(listener, outcome) => listener(outcome),
            Finished::Idle(closed) => closed.send(()),
        }
    }

    fn shut(&mut self, idle: Arc<Handoff<()>>) {
        self.closed = true;
        self.closing.push(idle);
        for load in self.loads.values() {
            load.cancel.store(true, Ordering::Release);
        }
        for waiting in mem::take(&mut self.blocked) {
            self.told
                .push(Finished::Told(waiting.request.listener, Outcome::Cancelled));
        }
        self.signal_closed();
    }

    fn into_barrier(finished: Finished) -> Result<Arc<Handoff<()>>, Finished> {
        match finished {
            Finished::Idle(closed) => Ok(closed),
            told => Err(told),
        }
    }

    fn idle(&self) -> bool {
        self.loads.is_empty() && self.blocked.is_empty()
    }
}

impl fmt::Debug for Owner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Owner")
            .field("loads", &self.loads.len())
            .field("blocked", &self.blocked.len())
            .finish_non_exhaustive()
    }
}
