//! The bookkeeping of the device slots: which expert each holds, which may be replaced, and which are pinned.
//!
//! Plain state with one owner: only the cache owner's turns call it. An expert is used through a lease, which pins its
//! slot; a slot that is loading or pinned is never evicted or refilled, so the address a lease carries is valid until
//! the lease is let go. Everything else is replaced by S3-FIFO: a new record enters a small queue (a tenth of the slots),
//! one asked for again there moves to the main queue, and the main queue gives a record asked for again another round
//! (a count of at most three). A record evicted unused from the small queue is remembered for as many such evictions as
//! there are slots, and enters the main queue if it returns.
//!
//! A slot goes `Empty` -> `Loading` (reserved, its copy being made) -> `Resident` (evictable, in a queue) or `InUse`
//! (pinned). A slot whose copy is not retired is never evictable, even once every lease on it was let go: the copy may
//! still be writing it. The device work that read a slot's previous expert is a fence the next copy waits behind.

use std::mem;
use std::sync::Arc;

use euhedral_graph::Marker;

use super::CacheError;
use super::stats::{CacheStats, Cause};

const NONE: u32 = u32::MAX;
const SMALL: usize = 0;
const MAIN: usize = 1;

/// The most times a record is given another round in the main queue.
const MAX_FREQUENCY: u8 = 3;

/// What a slot is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Empty,
    Loading,
    Resident,
    InUse,
}

/// What a request for an expert came to.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Claimed {
    /// The expert is resident: a lease at once, its slot pinned.
    Resident { slot: u32, generation: u64 },
    /// The expert is loading and its copy was submitted: a lease that waits for `ready`.
    Joined { slot: u32, generation: u64, ready: Marker },
    /// A slot was reserved for the expert: the caller starts the load. The copy waits for `fences`.
    Miss(Reserved),
    /// Nothing changed; ask again when `Cause` has changed.
    Wait(Cause),
}

/// A reserved slot.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Reserved {
    pub slot: u32,
    pub generation: u64,
    pub fences: Vec<Marker>,
}

pub(super) struct Shard {
    slots: usize,
    stats: Arc<CacheStats>,
    directory: Vec<u32>,
    state: Vec<State>,
    slot_key: Vec<u32>,
    bytes: Vec<u64>,
    pins: Vec<u32>,
    generation: Vec<u64>,
    /// The device work that read the slot's expert: the next copy into the slot waits behind it.
    fence: Vec<Vec<Marker>>,
    /// The fences the slot's load waits for, kept until it ends so that a failed load gives them back.
    waits: Vec<Vec<Marker>>,
    /// The marker of the slot's copy once submitted, until it retires.
    ready: Vec<Option<Marker>>,
    prev: Vec<u32>,
    next: Vec<u32>,
    queue: Vec<usize>,
    freq: Vec<u8>,
    /// When each key was last evicted unused from the small queue, by `ghost_clock`; 0 for never.
    ghost_stamp: Vec<u64>,
    ghost_clock: u64,
    small_cap: usize,
    free_head: u32,
    head: [u32; 2],
    tail: [u32; 2],
    used: [usize; 2],
    loading: usize,
    pinned: usize,
    leases_open: u64,
    closed: bool,
}

/// Adds `fence` to `fences`: one marker per lane, the later of two on the same lane.
fn merge(fences: &mut Vec<Marker>, fence: Marker) {
    match fences
        .iter_mut()
        .find(|held| held.queue == fence.queue && held.lane == fence.lane)
    {
        Some(held) => held.value = held.value.max(fence.value),
        None => fences.push(fence),
    }
}

impl Shard {
    /// A shard of `slots` slots for `keys` experts. Refuses neither being zero, and more slots than a slot index holds.
    pub(super) fn new(keys: usize, slots: usize, stats: Arc<CacheStats>) -> Result<Shard, CacheError> {
        if keys == 0 || keys >= NONE as usize {
            return Err(CacheError::Keys {
                keys,
                limit: NONE as usize - 1,
            });
        }
        if slots == 0 || slots >= NONE as usize {
            return Err(CacheError::Slots {
                slots,
                limit: NONE as usize - 1,
            });
        }
        let next = (0..slots)
            .map(|slot| if slot + 1 < slots { slot as u32 + 1 } else { NONE })
            .collect();
        Ok(Shard {
            slots,
            stats,
            directory: vec![NONE; keys],
            state: vec![State::Empty; slots],
            slot_key: vec![NONE; slots],
            bytes: vec![0; slots],
            pins: vec![0; slots],
            generation: vec![0; slots],
            fence: vec![Vec::new(); slots],
            waits: vec![Vec::new(); slots],
            ready: vec![None; slots],
            prev: vec![NONE; slots],
            next,
            queue: vec![SMALL; slots],
            freq: vec![0; slots],
            ghost_stamp: vec![0; keys],
            ghost_clock: 0,
            small_cap: (slots / 10).max(1),
            free_head: 0,
            head: [NONE; 2],
            tail: [NONE; 2],
            used: [0; 2],
            loading: 0,
            pinned: 0,
            leases_open: 0,
            closed: false,
        })
    }

    pub(super) fn slots(&self) -> usize {
        self.slots
    }

    pub(super) fn keys(&self) -> usize {
        self.directory.len()
    }

    /// Asks for the expert of `key`, whose record is `bytes` long. A resident one is a lease at once. An absent one is a
    /// miss when `blocked` is `None` and a slot is free or evictable; otherwise the answer is to wait, and nothing has
    /// changed.
    pub(super) fn claim(&mut self, key: usize, bytes: u64, blocked: Option<Cause>) -> Result<Claimed, CacheError> {
        if self.closed {
            return Err(CacheError::Closed);
        }
        let Some(&mapped) = self.directory.get(key) else {
            return Err(CacheError::Key {
                key,
                keys: self.directory.len(),
            });
        };
        if mapped != NONE {
            let slot = mapped as usize;
            if self.state[slot] == State::Loading {
                // Another request's load: leased once its copy has a marker to wait for.
                let Some(ready) = self.ready[slot] else {
                    return Ok(Claimed::Wait(Cause::Copying));
                };
                self.pins[slot] += 1;
                self.leases_open += 1;
                self.asked(slot);
                self.stats.coalesce();
                return Ok(Claimed::Joined {
                    slot: mapped,
                    generation: self.generation[slot],
                    ready,
                });
            }
            self.stats.hit();
            self.asked(slot);
            self.pin(slot);
            return Ok(Claimed::Resident {
                slot: mapped,
                generation: self.generation[slot],
            });
        }
        if let Some(cause) = blocked {
            return Ok(Claimed::Wait(cause));
        }
        let victim = match self.free_head {
            NONE => self.s3_victim(),
            free => Some(free),
        };
        let Some(victim) = victim else {
            return Ok(Claimed::Wait(Cause::Slots));
        };
        let reserved = self.reserve(victim as usize, key, bytes);
        self.stats.miss();
        Ok(Claimed::Miss(reserved))
    }

    /// The copy of the load of `slot` was submitted and will record `marker`: claims of the expert may join it.
    /// Counts the load's own lease, which the caller makes.
    pub(super) fn submitted(&mut self, slot: u32, generation: u64, marker: Marker) -> Result<(), CacheError> {
        let index = self.loading_slot(slot, generation, "submitted")?;
        if self.ready[index].is_some() {
            return Err(CacheError::BadLoad {
                slot,
                what: "submitted twice",
            });
        }
        self.ready[index] = Some(marker);
        self.leases_open += 1;
        Ok(())
    }

    /// The copy of `slot` retired: the expert is resident once nobody holds it.
    pub(super) fn retired(&mut self, slot: u32, generation: u64) -> Result<(), CacheError> {
        let index = self.loading_slot(slot, generation, "retired")?;
        if self.ready[index].is_none() {
            return Err(CacheError::BadLoad {
                slot,
                what: "retired before it was submitted",
            });
        }
        self.loading -= 1;
        self.ready[index] = None;
        self.waits[index].clear();
        if self.pins[index] > 0 {
            self.state[index] = State::InUse;
        } else {
            self.make_evictable(index);
        }
        self.stats.transferred(self.bytes[index]);
        Ok(())
    }

    /// Ends the load of `slot` that never submitted a copy (it was refused, its read failed, or it was cancelled): the
    /// slot is empty again, and the fences its copy would have waited for still order the next one.
    pub(super) fn abandon(&mut self, slot: u32, generation: u64) -> Result<(), CacheError> {
        let index = self.loading_slot(slot, generation, "abandoned")?;
        if self.ready[index].is_some() {
            return Err(CacheError::BadLoad {
                slot,
                what: "abandoned after its copy was submitted",
            });
        }
        self.loading -= 1;
        self.directory[self.slot_key[index] as usize] = NONE;
        self.state[index] = State::Empty;
        self.pins[index] = 0;
        for fence in mem::take(&mut self.waits[index]) {
            merge(&mut self.fence[index], fence);
        }
        self.slot_key[index] = NONE;
        self.used[self.queue[index]] -= 1;
        self.next[index] = self.free_head;
        self.free_head = slot;
        self.pinned -= 1;
        Ok(())
    }

    /// A lease on `slot` was let go; `fence` is the device work that read it, if any. A lease of an earlier life of the
    /// slot (the shard closed with it open) is ignored.
    pub(super) fn release(&mut self, slot: u32, generation: u64, fence: Option<Marker>) {
        let index = slot as usize;
        if self.closed || self.generation.get(index) != Some(&generation) || self.pins[index] == 0 {
            return;
        }
        self.leases_open -= 1;
        if let Some(fence) = fence {
            merge(&mut self.fence[index], fence);
        }
        self.pins[index] -= 1;
        // A slot whose copy has not retired keeps its place: the leases may all be let go (the kernels that read it
        // wait behind the copy's marker), but the copy is still writing.
        if self.pins[index] == 0 && self.state[index] != State::Loading {
            self.make_evictable(index);
        }
    }

    /// Ends the shard: every lease still open is forced shut, and the fences pending are dropped. Refuses while a load
    /// is in flight. Returns the leases that were open.
    pub(super) fn close(&mut self) -> Result<u64, CacheError> {
        if self.closed {
            return Ok(0);
        }
        if self.loading > 0 {
            return Err(CacheError::LoadsInFlight { loads: self.loading });
        }
        self.closed = true;
        let forced = mem::take(&mut self.leases_open);
        self.stats.forced(forced);
        for slot in 0..self.slots {
            // A new generation makes every lease still out stale, and the release of one a no-op.
            self.generation[slot] += 1;
            self.fence[slot].clear();
        }
        Ok(forced)
    }

    pub(super) fn is_closed(&self) -> bool {
        self.closed
    }

    pub(super) fn loading(&self) -> usize {
        self.loading
    }

    pub(super) fn leases_open(&self) -> u64 {
        self.leases_open
    }

    /// Slots holding a loaded expert that no lease pins: what a miss may evict.
    pub(super) fn evictable(&self) -> usize {
        self.used_in_queues()
    }

    /// Whether the expert of `key` has a slot, resident, leased or loading.
    pub(super) fn holds(&self, key: usize) -> bool {
        self.directory.get(key).is_some_and(|&slot| slot != NONE)
    }

    /// The slot of the expert of `key`, when it is loaded and not loading.
    pub(super) fn resident_slot(&self, key: usize) -> Option<u32> {
        let slot = *self.directory.get(key)?;
        (slot != NONE && self.state[slot as usize] != State::Loading).then_some(slot)
    }

    /// The fences the load of `slot` waits behind: the device work that read its previous expert.
    #[cfg(test)]
    pub(super) fn fence_of(&self, slot: u32) -> &[Marker] {
        &self.fence[slot as usize]
    }

    fn loading_slot(&self, slot: u32, generation: u64, what: &'static str) -> Result<usize, CacheError> {
        let index = slot as usize;
        if index >= self.slots || self.generation[index] != generation || self.state[index] != State::Loading {
            return Err(CacheError::BadLoad { slot, what });
        }
        Ok(index)
    }

    fn asked(&mut self, slot: usize) {
        if self.freq[slot] < MAX_FREQUENCY {
            self.freq[slot] += 1;
        }
    }

    /// Pins a resident or leased slot for one more lease.
    fn pin(&mut self, slot: usize) {
        if self.state[slot] == State::Resident {
            self.dequeue(slot);
            self.state[slot] = State::InUse;
            self.pinned += 1;
            self.stats.slots_in_use(self.pinned as u64);
        }
        self.pins[slot] += 1;
        self.leases_open += 1;
    }

    /// The slot an expert about to load takes: the small queue's oldest while that queue holds its share (moving it to
    /// the main queue instead when it was asked for again), else the main queue's oldest that was not asked for since
    /// its last round (each other goes round once more with its count lowered). `None` when no slot is evictable.
    fn s3_victim(&mut self) -> Option<u32> {
        loop {
            let small = self.head[SMALL];
            let main = self.head[MAIN];
            if small != NONE && (self.used[SMALL] >= self.small_cap || main == NONE) {
                let slot = small as usize;
                if self.freq[slot] > 0 {
                    self.dequeue(slot);
                    self.used[SMALL] -= 1;
                    self.queue[slot] = MAIN;
                    self.used[MAIN] += 1;
                    self.freq[slot] = 0;
                    self.enqueue(slot);
                    continue;
                }
                self.ghost_clock += 1;
                self.ghost_stamp[self.slot_key[slot] as usize] = self.ghost_clock;
                return Some(small);
            }
            if main == NONE {
                return None;
            }
            let slot = main as usize;
            if self.freq[slot] > 0 {
                self.freq[slot] -= 1;
                self.dequeue(slot);
                self.enqueue(slot);
                continue;
            }
            return Some(main);
        }
    }

    /// Takes the free or evicted slot `victim` for the expert of `key`.
    fn reserve(&mut self, victim: usize, key: usize, bytes: u64) -> Reserved {
        if self.state[victim] == State::Resident {
            self.dequeue(victim);
            self.directory[self.slot_key[victim] as usize] = NONE;
            self.used[self.queue[victim]] -= 1;
            self.stats.eviction();
        } else {
            self.free_head = self.next[victim];
        }
        let stamp = self.ghost_stamp[key];
        let remembered = stamp != 0 && self.ghost_clock - stamp < self.slots as u64;
        self.queue[victim] = if remembered { MAIN } else { SMALL };
        self.freq[victim] = 0;
        self.used[self.queue[victim]] += 1;
        self.generation[victim] += 1;
        self.slot_key[victim] = key as u32;
        self.bytes[victim] = bytes;
        self.directory[key] = victim as u32;
        self.state[victim] = State::Loading;
        self.pins[victim] = 1;
        self.ready[victim] = None;
        self.waits[victim] = mem::take(&mut self.fence[victim]);
        self.loading += 1;
        self.pinned += 1;
        self.stats.slots_in_use(self.pinned as u64);
        Reserved {
            slot: victim as u32,
            generation: self.generation[victim],
            fences: self.waits[victim].clone(),
        }
    }

    /// A slot nobody holds becomes an evictable resident: it goes to the tail of its queue.
    fn make_evictable(&mut self, slot: usize) {
        self.state[slot] = State::Resident;
        self.enqueue(slot);
        self.pinned -= 1;
    }

    fn enqueue(&mut self, slot: usize) {
        let list = self.queue[slot];
        self.prev[slot] = self.tail[list];
        self.next[slot] = NONE;
        if self.tail[list] == NONE {
            self.head[list] = slot as u32;
        } else {
            self.next[self.tail[list] as usize] = slot as u32;
        }
        self.tail[list] = slot as u32;
    }

    fn dequeue(&mut self, slot: usize) {
        let list = self.queue[slot];
        let (before, after) = (self.prev[slot], self.next[slot]);
        if before == NONE {
            self.head[list] = after;
        } else {
            self.next[before as usize] = after;
        }
        if after == NONE {
            self.tail[list] = before;
        } else {
            self.prev[after as usize] = before;
        }
        self.prev[slot] = NONE;
        self.next[slot] = NONE;
    }

    fn used_in_queues(&self) -> usize {
        let mut count = 0;
        for head in self.head {
            let mut slot = head;
            while slot != NONE {
                count += 1;
                slot = self.next[slot as usize];
            }
        }
        count
    }

    /// Verifies the bookkeeping, which holds between any two calls: the directory and the slots agree, and every slot is
    /// in exactly the list its state says. Names the first inconsistency.
    pub(super) fn check_invariants(&self) -> Result<(), String> {
        let mut free = 0;
        let mut slot = self.free_head;
        while slot != NONE {
            free += 1;
            if free > self.slots {
                return Err("the free list loops".into());
            }
            if self.state[slot as usize] != State::Empty {
                return Err(format!("slot {slot} is free but not empty"));
            }
            slot = self.next[slot as usize];
        }
        let mut evictable = 0;
        for list in 0..2 {
            let mut before = NONE;
            let mut slot = self.head[list];
            while slot != NONE {
                evictable += 1;
                if evictable > self.slots {
                    return Err("a queue loops".into());
                }
                let index = slot as usize;
                if self.state[index] != State::Resident || self.pins[index] != 0 {
                    return Err(format!("slot {slot} is evictable but pinned or not resident"));
                }
                if self.queue[index] != list {
                    return Err(format!("slot {slot} is on another list"));
                }
                if self.prev[index] != before {
                    return Err(format!("slot {slot} has a broken back link"));
                }
                before = slot;
                slot = self.next[index];
            }
            if before != self.tail[list] {
                return Err("a queue tail is wrong".into());
            }
        }
        let mut counted = [0usize; 2];
        let (mut empty, mut resident, mut in_use, mut loading) = (0, 0, 0, 0);
        for slot in 0..self.slots {
            match self.state[slot] {
                State::Empty => empty += 1,
                State::Resident => resident += 1,
                State::InUse => {
                    in_use += 1;
                    if self.pins[slot] == 0 {
                        return Err(format!("slot {slot} is in use but unpinned"));
                    }
                }
                State::Loading => {
                    loading += 1;
                    if self.pins[slot] == 0 && self.ready[slot].is_none() {
                        return Err(format!("slot {slot} is loading with no claim on it"));
                    }
                }
            }
            if self.state[slot] != State::Empty {
                counted[self.queue[slot]] += 1;
                let key = self.slot_key[slot] as usize;
                if self.directory[key] != slot as u32 {
                    return Err(format!("slot {slot} is not in the directory"));
                }
            }
        }
        if counted != self.used {
            return Err("the per-queue slot counts are wrong".into());
        }
        if empty != free {
            return Err(format!("the free list holds {free} of {empty} empty slots"));
        }
        if resident != evictable {
            return Err(format!("the queues hold {evictable} of {resident} residents"));
        }
        if loading != self.loading {
            return Err("the loading count is wrong".into());
        }
        if in_use + loading != self.pinned {
            return Err("the pinned count is wrong".into());
        }
        Ok(())
    }
}
