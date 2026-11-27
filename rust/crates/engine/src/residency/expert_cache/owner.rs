//! The cache's one owner: its device slots' bookkeeping and the host tier's, and the loads in flight.
//!
//! Requests, lease releases, read completions, copy submissions and copy retirements are records posted to it (see
//! `source`); the one worker that runs its turn applies them in order, with no lock and no wait. A request that cannot
//! be served (every slot is loading or leased, every staging buffer or read is in use) stays here with the cause, and is
//! asked again when that cause has changed, once per change.
//!
//! A record whose application panics poisons the owner: the loads and waiting requests are ended with the error that
//! names it, new requests are refused with it, and the records that give things back (reads, copies, leases) are still
//! applied, so that every load ends and a close completes.
//!
//! A load is: the owner reserves a device slot, a staging slot and a read, and begins the store's load; the store's
//! frames read the record; its end posts [`Message::Read`]; the owner offers the copy's frame, which a worker submits on
//! the copy lane and answers with [`Message::Submitted`] (the lease, which carries the copy's marker, is told then);
//! the marker's retirement runs a frame that posts [`Message::Retired`], which frees the staging slot and ends the load.
//! The load holds one unit of the lake from its start to its retirement, so closing waits for it.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::fmt::{self, Display, Formatter};
use std::mem::{self, forget};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::KeySlot;
use euhedral_data_structures::queues::{Consumer, Single};
use euhedral_graph::Marker;
use euhedral_memory::{SlotArena, SlotLease};

use super::CacheError;
use super::host::{CopyJob, CopyLane, DeviceSlots, Frames, SubmitFailure};
use super::lease::{ExpertLease, Home};
use super::shard::{Claimed, Reserved, Shard};
use super::source::Inbox;
use super::stats::{CacheStats, Cause};
use crate::frames::{Kind, id_hash, step};
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
    /// The expert could not be had; the error names why: the expert's own failure, a closed cache, a poisoned owner.
    Failed(CacheError),
    /// The request's own cancel flag was set before a lease was made.
    Cancelled,
}

/// Told once with a request's outcome, on a worker, once the owner's claim is free. It must not wait.
pub type Listener = Box<dyn FnOnce(Outcome) + Send>;

/// A request's listener, which is told exactly once. A reply dropped untold (the record that held it panicked) posts
/// itself to the owner, whose next turn tells the listener that the request was abandoned: on a worker, with the claim
/// free, and not in the drop (which may run inside the claim or while a panic unwinds). Only when the owner is gone, and
/// nothing could tell it, is the listener told in the drop.
///
/// The listener sits in a [`KeySlot`], shared with whoever posted the request: whoever takes it first (one swap) tells it,
/// so the owner and a poster that finds the source complete after posting never both tell it, and neither can miss it.
pub(crate) struct Reply {
    listener: Arc<KeySlot<Listener>>,
    inbox: Weak<Inbox>,
}

impl Reply {
    pub(super) fn new(listener: Listener, inbox: Weak<Inbox>) -> Reply {
        Reply {
            listener: Arc::new(KeySlot::with_key(Box::new(listener))),
            inbox,
        }
    }

    /// The listener's slot, for a poster that must tell it itself.
    pub(super) fn slot(&self) -> Arc<KeySlot<Listener>> {
        Arc::clone(&self.listener)
    }

    /// Tells the listener in `slot` that the cache is closed, if nobody has told it.
    pub(super) fn refuse(slot: &KeySlot<Listener>) {
        if let Some(listener) = slot.take() {
            listener(Outcome::Failed(CacheError::Closed));
        }
    }

    fn tell(self, outcome: Outcome) {
        if let Some(listener) = self.listener.take() {
            listener(outcome);
        }
    }

    fn abandoned() -> Outcome {
        let why = "the request was dropped before it was answered".to_owned();
        Outcome::Failed(CacheError::Abandoned { why })
    }
}

impl Drop for Reply {
    fn drop(&mut self) {
        let Some(listener) = self.listener.take() else {
            return;
        };
        let reply = Reply::new(listener, Weak::new());
        match self.inbox.upgrade() {
            Some(inbox) => inbox.post(Message::Orphaned(reply)),
            None => {
                // A drop may run while a panic unwinds: a listener that panics must not escape it.
                if catch_unwind(AssertUnwindSafe(|| reply.tell(Reply::abandoned()))).is_err() {
                    log::error!("a listener of the expert cache panicked while told its request was dropped");
                }
            }
        }
    }
}

/// One request for an expert.
pub(crate) struct Request {
    pub id: ExpertId,
    pub key: usize,
    pub bytes: u64,
    /// Whether the load is a prefill scan's, which the host tier may refuse to keep.
    pub scan: bool,
    pub cancel: Arc<AtomicBool>,
    pub reply: Reply,
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
        result: Result<Marker, Unsubmitted>,
    },
    /// The copy of the load of `id` retired.
    Retired {
        id: ExpertId,
    },
    /// The copy that may have read staging slot `staging` has retired: the slot is free to be written again.
    Unquarantined {
        staging: u32,
    },
    /// No marker will say when the copy that may have read staging slot `staging` ends: the slot is never used again.
    Lost {
        staging: u32,
    },
    /// A request's reply was dropped untold: its listener is told the request was abandoned.
    Orphaned(Reply),
    /// Some request was cancelled: those waiting here end.
    Sweep,
    /// The cache is closing: `done` is sent when nothing is in flight.
    Close(Arc<Handoff<()>>),
}

/// Why a copy was not submitted, and whether the device may still read its staging slot.
pub(crate) struct Unsubmitted {
    why: String,
    /// The copy may be queued: the staging slot waits for a later marker (see [`Message::Unquarantined`]).
    quarantined: bool,
}

impl Unsubmitted {
    pub(super) fn refused(why: String) -> Unsubmitted {
        Unsubmitted {
            why,
            quarantined: false,
        }
    }

    pub(super) fn uncertain(why: String) -> Unsubmitted {
        Unsubmitted { why, quarantined: true }
    }
}

/// An outcome to tell, or the end of a shutdown.
pub(crate) enum Finished {
    Told(Reply, Outcome),
    Idle(Arc<Handoff<()>>),
}

impl Finished {
    /// Tells the outcome, or the end of the shutdown.
    pub(super) fn deliver(self) {
        match self {
            Finished::Told(reply, outcome) => reply.tell(outcome),
            Finished::Idle(closed) => closed.send(()),
        }
    }
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
    /// The frame that submits the copy was offered: from then on a record of the copy's retirement is to come.
    copy_offered: bool,
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
    records: Consumer<Message, Single>,
    /// Where the leases, the store's ends and the copy frames post.
    inbox: Arc<Inbox>,
    layout: Arc<ExpertLayout>,
    store: Arc<ExpertStore>,
    tier: Vec<RamTierShard>,
    /// Shared with the leases, so the pool's slots stay leased while any lease is out.
    device: Arc<DeviceSlots>,
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
    /// Staging slots a copy may still read, by slot index, with the lease that keeps them from the free list.
    quarantined: HashMap<u32, SlotLease>,
    /// Staging slots that are never used again.
    lost: u32,
    /// How many times each cause (by [`Cause`] order) has changed.
    epochs: [u64; 4],
    told: Vec<Finished>,
    closed: bool,
    /// Why the owner can no longer serve, once a record panicked.
    poisoned: Option<String>,
    closing: Vec<Arc<Handoff<()>>>,
}

impl Owner {
    pub(super) fn new(
        records: Consumer<Message, Single>,
        inbox: Arc<Inbox>,
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
            records,
            inbox,
            layout,
            store,
            tier,
            device: Arc::new(device),
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
            quarantined: HashMap::new(),
            lost: 0,
            epochs: [0; 4],
            told: Vec::new(),
            closed: false,
            poisoned: None,
            closing: Vec::new(),
        })
    }

    /// Applies every record posted, in order, then asks again the requests whose cause changed, and returns what is to be
    /// told. A record that panics poisons the owner and the records after it are still applied.
    pub(super) fn turn(&mut self) -> Vec<Finished> {
        while let Some(message) = self.records.poll() {
            let read = match &message {
                Message::Read(end) => Some(end.id),
                _ => None,
            };
            if self.guarded(|owner| owner.handle(message))
                && let Some(id) = read
            {
                self.end_stranded(id);
            }
        }
        self.guarded(Owner::ask_changed);
        self.guarded(Owner::signal_closed);
        mem::take(&mut self.told)
    }

    /// Runs `step`; a panic in it poisons the owner instead of leaving it. Whether it panicked.
    fn guarded(&mut self, step: impl FnOnce(&mut Owner)) -> bool {
        let Err(panic) = catch_unwind(AssertUnwindSafe(|| step(self))) else {
            return false;
        };
        self.poison(&panic_text(&*panic));
        true
    }

    /// Ends the load of `id` whose read ended in a record that panicked, if no copy was offered for it: no record of it
    /// is to come, so it would hold the close for ever.
    fn end_stranded(&mut self, id: ExpertId) {
        if self.loads.get(&id).is_some_and(|load| !load.copy_offered) {
            self.guarded(|owner| owner.end_load(id, None, Outcome::Cancelled));
        }
    }

    /// Ends what is waiting or loading with the error that names the panic, and refuses requests from now on. What is
    /// in flight is still given back by the records that end it, so a close completes.
    fn poison(&mut self, why: &str) {
        log::error!("the expert cache owner panicked and is poisoned: {why}");
        if self.poisoned.is_some() {
            return;
        }
        self.poisoned = Some(why.to_owned());
        for load in self.loads.values_mut() {
            load.cancel.store(true, Ordering::Release);
            if let Some(request) = load.request.take() {
                let error = CacheError::Poisoned { why: why.to_owned() };
                self.told.push(Finished::Told(request.reply, Outcome::Failed(error)));
            }
        }
        for waiting in mem::take(&mut self.blocked) {
            let error = CacheError::Poisoned { why: why.to_owned() };
            self.told
                .push(Finished::Told(waiting.request.reply, Outcome::Failed(error)));
        }
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
            Message::Unquarantined { staging } => self.unquarantined(staging),
            Message::Lost { staging } => self.lost_slot(staging),
            Message::Orphaned(reply) => self.told.push(Finished::Told(reply, Reply::abandoned())),
            Message::Sweep => self.sweep(),
            Message::Close(done) => self.shut(done),
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
        if request.cancel.load(Ordering::Acquire) {
            self.told.push(Finished::Told(request.reply, Outcome::Cancelled));
            return;
        }
        if let Some(refusal) = self.refusal() {
            self.told.push(Finished::Told(request.reply, Outcome::Failed(refusal)));
            return;
        }
        let claimed = match self.shard.claim(request.key, request.bytes, self.blocked_by()) {
            Ok(claimed) => claimed,
            Err(error) => {
                self.told.push(Finished::Told(request.reply, Outcome::Failed(error)));
                return;
            }
        };
        match claimed {
            Claimed::Resident { slot, generation } => {
                let lease = self.lease(request.id, slot, generation, request.bytes, None);
                self.told.push(Finished::Told(request.reply, Outcome::Leased(lease)));
            }
            Claimed::Joined {
                slot,
                generation,
                ready,
            } => {
                let lease = self.lease(request.id, slot, generation, request.bytes, Some(ready));
                self.told.push(Finished::Told(request.reply, Outcome::Leased(lease)));
            }
            Claimed::Miss(reserved) => self.begin(request, reserved),
            Claimed::Wait(Cause::Staging) if self.staging_gone() => {
                let error = CacheError::StagingLost { lost: self.lost };
                self.told.push(Finished::Told(request.reply, Outcome::Failed(error)));
            }
            Claimed::Wait(cause) => {
                if !again {
                    self.stats.waited(cause);
                }
                let epoch = self.epochs[cause as usize];
                self.blocked.push_back(Waiting { request, cause, epoch });
            }
        }
    }

    /// Why no request is served now, if so: the owner is poisoned, or closed.
    fn refusal(&self) -> Option<CacheError> {
        if let Some(why) = &self.poisoned {
            Some(CacheError::Poisoned { why: why.clone() })
        } else if self.closed {
            Some(CacheError::Closed)
        } else {
            None
        }
    }

    /// Whether every staging slot is lost, so that no load can start again.
    fn staging_gone(&self) -> bool {
        self.lost >= self.staging_slots.capacity()
    }

    fn lease(&self, id: ExpertId, slot: u32, generation: u64, bytes: u64, ready: Option<Marker>) -> ExpertLease {
        ExpertLease::new(
            Home {
                inbox: Arc::clone(&self.inbox),
                _slots: Arc::clone(&self.device),
            },
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
        // One pass over the requests waiting now; what is not asked stays in the list, so a panic poisons it with them.
        for _ in 0..self.blocked.len() {
            let Some(waiting) = self.blocked.pop_front() else {
                break;
            };
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
                    .push(Finished::Told(waiting.request.reply, Outcome::Cancelled));
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
        let inbox = Arc::clone(&self.inbox);
        let load = StoreLoad {
            id,
            directive,
            staging_slot: staging.index() as usize,
            cancel: Arc::clone(&cancel),
            done: Box::new(move |end| inbox.post(Message::Read(end))),
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
                copy_offered: false,
                _unit: unit,
            },
        );
        self.reads += 1;
        if let Started::Frame(frame) = started {
            // A hand-off that panics drops the frame, which carried the load's end: as one that refuses, the load can only
            // be ended here. The panic is not swallowed: it poisons the owner.
            let handed = catch_unwind(AssertUnwindSafe(|| self.frames.offer(frame)));
            let why = match handed {
                Ok(Ok(())) => return,
                Ok(Err(_)) => None,
                Err(panic) => Some(panic_text(&*panic)),
            };
            self.reads -= 1;
            self.changed(Cause::Reads);
            let outcome = match why {
                Some(why) => {
                    self.poison(&why);
                    self.ended_by(id)
                }
                None => Outcome::Failed(self.failure(id, "the lake took no frame")),
            };
            self.end_load(id, directive.settle_unstarted(), outcome);
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
        // The staging slot and the read the request held are free again with the caller's locals.
        self.changed(Cause::Staging);
        self.told.push(Finished::Told(request.reply, outcome));
    }

    /// Applies `settle` to the tier shard of `id`'s key; an id the layout does not hold is logged and settles nothing.
    fn settle_of(&mut self, id: ExpertId, settle: TierSettle) {
        match self.layout.key(id) {
            Ok(key) => self.settle(key, settle),
            Err(error) => log::error!("a tier settle for an expert the layout does not hold was dropped: {error}"),
        }
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
        if let Some(settle) = end.settle {
            self.settle_of(id, settle);
        }
        let Some(load) = self.loads.get(&id) else {
            log::error!("a read ended for an expert with no load: {id:?}");
            return;
        };
        let cancelled = self.closed
            || self.poisoned.is_some()
            || load.cancel.load(Ordering::Acquire)
            || load
                .request
                .as_ref()
                .is_some_and(|request| request.cancel.load(Ordering::Acquire));
        match end.outcome {
            ReadOutcome::Complete if !cancelled => self.submit(id),
            ReadOutcome::Complete | ReadOutcome::Cancelled { .. } => {
                let outcome = self.ended_by(id);
                self.end_load(id, None, outcome);
            }
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
            Arc::clone(&self.inbox),
            Copy {
                id,
                chain: slot as usize,
                staging: staging_slot as u32,
            },
            job,
        );
        if self.frames.offer(frame).is_err() {
            let failure = self.failure(id, "the lake took no frame");
            self.end_load(id, None, Outcome::Failed(failure));
        } else if let Some(load) = self.loads.get_mut(&id) {
            load.copy_offered = true;
        }
    }

    /// What a request is told when its load ends before the copy was made and no failure ended it: its own cancel flag,
    /// else the close or the poison that ended it.
    fn ended_by(&self, id: ExpertId) -> Outcome {
        let asked = self
            .loads
            .get(&id)
            .and_then(|load| load.request.as_ref())
            .is_some_and(|request| request.cancel.load(Ordering::Acquire));
        match self.refusal() {
            Some(refusal) if !asked => Outcome::Failed(refusal),
            _ => Outcome::Cancelled,
        }
    }

    fn submitted(&mut self, id: ExpertId, result: Result<Marker, Unsubmitted>) {
        let Some(load) = self.loads.get_mut(&id) else {
            log::error!("a copy was submitted for an expert with no load: {id:?}");
            return;
        };
        let (slot, generation, bytes) = (load.slot, load.generation, load.bytes);
        match result {
            Err(Unsubmitted { why, quarantined }) => {
                let failure = self.failure(id, why);
                self.finish_load(id, None, Outcome::Failed(failure), quarantined);
            }
            Ok(marker) => {
                if let Err(error) = self.shard.submitted(slot, generation, marker) {
                    // The copy is queued and will retire: the load stays for that, and its request is told.
                    log::error!("{error}");
                    let failure = self.failure(id, error);
                    if let Some(request) = self.loads.get_mut(&id).and_then(|load| load.request.take()) {
                        self.told.push(Finished::Told(request.reply, Outcome::Failed(failure)));
                    }
                    return;
                }
                self.changed(Cause::Copying);
                let lease = self.lease(id, slot, generation, bytes, Some(marker));
                let Some(request) = self.loads.get_mut(&id).and_then(|load| load.request.take()) else {
                    return;
                };
                let outcome = if request.cancel.load(Ordering::Acquire) {
                    // The lease is dropped here: it posts its own release, and the slot is unpinned in a later turn.
                    drop(lease);
                    self.stats.cancelled_miss();
                    Outcome::Cancelled
                } else if let Some(refusal) = self.refusal() {
                    drop(lease);
                    self.stats.cancelled_miss();
                    Outcome::Failed(refusal)
                } else {
                    Outcome::Leased(lease)
                };
                self.told.push(Finished::Told(request.reply, outcome));
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
        self.finish_load(id, settle, outcome, false);
    }

    /// As [`end_load`](Owner::end_load); with `quarantined` the staging slot is held out of the free list, as the device
    /// may still read it, until [`Message::Unquarantined`] or [`Message::Lost`] says what became of the copy.
    fn finish_load(&mut self, id: ExpertId, settle: Option<TierSettle>, outcome: Outcome, quarantined: bool) {
        let Some(mut load) = self.loads.remove(&id) else {
            return;
        };
        if let Some(settle) = settle {
            self.settle_of(id, settle);
        }
        if let Err(error) = self.shard.abandon(load.slot, load.generation) {
            log::error!("{error}");
        }
        match &outcome {
            Outcome::Failed(CacheError::Transfer { .. } | CacheError::Abandoned { .. }) => self.stats.failed_transfer(),
            _ => self.stats.cancelled_miss(),
        }
        if let Some(request) = load.request.take() {
            self.told.push(Finished::Told(request.reply, outcome));
        }
        if quarantined {
            self.quarantined.insert(load.staging.index(), load.staging);
        } else {
            drop(load);
        }
        self.changed(Cause::Slots);
        self.changed(Cause::Copying);
        self.changed(Cause::Staging);
    }

    /// A copy that may have read staging slot `staging` has retired: the slot is free again.
    fn unquarantined(&mut self, staging: u32) {
        if self.quarantined.remove(&staging).is_none() {
            log::error!("staging slot {staging} was released from a quarantine it was not in");
            return;
        }
        self.changed(Cause::Staging);
    }

    /// Staging slot `staging` can never be said free: its lease is forgotten, so it never returns to the free list. When
    /// every slot is gone, the requests that wait for one are told so: nothing would ever end their wait.
    fn lost_slot(&mut self, staging: u32) {
        let Some(lease) = self.quarantined.remove(&staging) else {
            log::error!("staging slot {staging} was lost from a quarantine it was not in");
            return;
        };
        forget(lease);
        self.lost += 1;
        if !self.staging_gone() {
            return;
        }
        let error = CacheError::StagingLost { lost: self.lost };
        for waiting in mem::take(&mut self.blocked) {
            if waiting.cause == Cause::Staging {
                self.told
                    .push(Finished::Told(waiting.request.reply, Outcome::Failed(error.clone())));
            } else {
                self.blocked.push_back(waiting);
            }
        }
    }

    /// Closes: requests waiting and loads in flight are ended with [`CacheError::Closed`]; `done` is sent when nothing
    /// is in flight.
    fn shut(&mut self, done: Arc<Handoff<()>>) {
        self.closed = true;
        self.closing.push(done);
        for load in self.loads.values_mut() {
            load.cancel.store(true, Ordering::Release);
        }
        for waiting in mem::take(&mut self.blocked) {
            self.told.push(Finished::Told(
                waiting.request.reply,
                Outcome::Failed(CacheError::Closed),
            ));
        }
    }

    /// Whether a close was applied and nothing is in flight: the owner has nothing more to do.
    pub(super) fn is_shut(&self) -> bool {
        self.closed && self.loads.is_empty()
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

/// The text of a caught panic's payload.
fn panic_text(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "a panic with no text".to_owned()
    }
}

/// What a copy frame knows of its load.
#[derive(Clone, Copy)]
struct Copy {
    id: ExpertId,
    /// The copy lane's chain: the load's device slot.
    chain: usize,
    /// The load's staging slot.
    staging: u32,
}

/// The frame that submits `job` on the copy lane and tells the owner. A lane with no room hands the job to a frame
/// that runs once there is; nothing waits. It only posts: every outcome is applied by the owner's turn.
fn copy_frame(
    copier: Arc<dyn CopyLane>,
    frames: Arc<dyn Frames>,
    owner: Arc<Inbox>,
    copy: Copy,
    job: CopyJob,
) -> Box<dyn Frame> {
    let Copy { id, chain, .. } = copy;
    step(Kind::ExpertSubmit, chain, move || match copier.submit(&job) {
        Ok(marker) => {
            // The owner hears of the submission before the retirement can be: the record is in its inbox before the
            // frame that posts the retirement is registered.
            owner.post(Message::Submitted { id, result: Ok(marker) });
            // The staging bytes were read by the copy once it has retired.
            frames.after_marker(
                marker,
                Posts::frame(
                    chain,
                    owner,
                    Some(job),
                    Message::Retired { id },
                    Message::Retired { id },
                ),
            );
        }
        Err(SubmitFailure::Busy) => {
            let again = copy_frame(Arc::clone(&copier), frames, owner, copy, job);
            copier.when_room(again);
        }
        Err(SubmitFailure::Refused(why)) => owner.post(Message::Submitted {
            id,
            result: Err(Unsubmitted::refused(why)),
        }),
        Err(SubmitFailure::Uncertain(why)) => {
            // The copy may be queued and has no marker: the owner keeps its staging slot out of the free list, and the
            // next marker of the lane gives it back.
            owner.post(Message::Submitted {
                id,
                result: Err(Unsubmitted::uncertain(why)),
            });
            Quarantine {
                copier,
                frames,
                owner,
                copy,
                job,
            }
            .run();
        }
    })
}

/// The frame that runs behind a copy's marker: it lets go of the copy's job and posts what the retirement means. A frame
/// the lake drops without having run (it was discarded as the lattice closed) posts `unrun` instead, so that the load
/// still ends and a close completes: for a retirement the same record (the lake is going, and nothing will read the
/// slot), for a quarantine the slot's loss (nothing said the copy ran, so the slot is never written again).
struct Posts {
    header: FrameHeader,
    owner: Arc<Inbox>,
    job: Option<CopyJob>,
    run: Option<Message>,
    unrun: Option<Message>,
}

impl Posts {
    fn frame(chain: usize, owner: Arc<Inbox>, job: Option<CopyJob>, run: Message, unrun: Message) -> Box<dyn Frame> {
        Box::new(Posts {
            header: FrameHeader::new(id_hash(Kind::ExpertRetire, chain)),
            owner,
            job,
            run: Some(run),
            unrun: Some(unrun),
        })
    }
}

impl Frame for Posts {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.unrun = None;
        self.job = None;
        if let Some(message) = self.run.take() {
            self.owner.post(message);
        }
        Ok(())
    }
}

impl Drop for Posts {
    fn drop(&mut self) {
        if let Some(message) = self.unrun.take() {
            self.job = None;
            self.owner.post(message);
        }
    }
}

/// A copy that may be queued with no marker of its own, waiting for one behind it.
///
/// The lane runs its work in order, so the marker of work submitted after the copy retires only after the copy has run
/// (or been dropped by the lane): once it has, the device no longer reads the staging slot, and the job (the slot's read
/// lease) can go. The marker is taken with no work behind it, as the lane's own mark. When the lane cannot mark either,
/// nothing will say when the copy ran: the slot is lost, and the owner is told.
struct Quarantine {
    copier: Arc<dyn CopyLane>,
    frames: Arc<dyn Frames>,
    owner: Arc<Inbox>,
    copy: Copy,
    job: CopyJob,
}

impl Quarantine {
    /// Takes the marker and registers the release of the slot behind it; with the lane full, tries again when it has
    /// room.
    fn run(self) {
        let Quarantine {
            copier,
            frames,
            owner,
            copy,
            job,
        } = self;
        match copier.mark() {
            Ok(marker) => {
                let Copy { chain, staging, .. } = copy;
                frames.after_marker(
                    marker,
                    Posts::frame(
                        chain,
                        owner,
                        Some(job),
                        Message::Unquarantined { staging },
                        Message::Lost { staging },
                    ),
                );
            }
            Err(SubmitFailure::Busy) => {
                let again = Arc::clone(&copier);
                let frame = step(Kind::ExpertSubmit, copy.chain, move || {
                    Quarantine {
                        copier,
                        frames,
                        owner,
                        copy,
                        job,
                    }
                    .run();
                });
                again.when_room(frame);
            }
            Err(SubmitFailure::Refused(why) | SubmitFailure::Uncertain(why)) => {
                log::error!("a staging slot is lost: no marker can say when its copy ran: {why}");
                // The owner forgets the slot's own lease, so nothing writes the slot again: the job's read lease can go.
                owner.post(Message::Lost { staging: copy.staging });
                drop(job);
            }
        }
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
