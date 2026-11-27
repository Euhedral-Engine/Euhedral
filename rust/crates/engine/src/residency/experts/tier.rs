//! The host tier's bookkeeping: which record each slot of host memory holds, and what a load of a record should do
//! with the tier.
//!
//! A [`RamTierShard`] owns the slots of the records that hash to it and nothing else. It has one owner (the owner of
//! the cache shard with the same index), and everything it does runs there: the directory, the slot states, the
//! request counts and the pins are plain arrays. A load asks [`plan`](RamTierShard::plan) what to do:
//!
//! - [`TierDirective::Hit`]: copy the ready record out of its slot (pins the slot until the load ends);
//! - [`TierDirective::Fill`]: read the artifact into a slot reserved for the record, which is not ready until the load
//!   ends;
//! - [`TierDirective::Bypass`]: read the artifact into staging alone, because no slot may be taken.
//!
//! The load itself runs elsewhere and touches only slot memory. It ends by handing the owner one [`TierSettle`], which
//! the owner passes to [`apply`](RamTierShard::apply): that is the only way a load changes the bookkeeping.
//!
//! ## Slots
//!
//! A slot is free, filling, ready, or lost (a read's memory was leaked by the ring: the slot is out of use for good).
//! A slot that is filling or pinned is skipped by the victim search, so it is never taken while a load uses it.
//!
//! ## Replacement
//!
//! The tier sees only the device cache's misses, from which recency is already filtered, so it replaces by frequency:
//! the victim is the least requested of [`SAMPLE`] ready, unpinned slots drawn at random, the least recently planned
//! among equals (when the sample finds none, the first such slot). The counts are halved every [`AGE_PER_SLOT`]
//! requests per slot, so a record that was hot long ago does not hold its slot forever.
//!
//! ## Admission
//!
//! A prefill's record that finds no free slot replaces the victim only if it was asked for more often than the victim
//! was; a decode step's records always enter. A prefill visits nearly every expert of a layer once per chunk, so
//! admitting every record would replace each just before the next chunk needs it, and the tier would serve almost
//! nothing; with admission the records already in the tier stay, and a share of every chunk is served from memory.
//! A record that loses is read into staging (a bypass).

use std::fmt::{self, Formatter};
use std::sync::Arc;

use super::arena::HostArena;
use super::layout::{ExpertError, ExpertId, ExpertLayout, shard_of};

/// Requests per slot between halvings of the request counts.
pub const AGE_PER_SLOT: usize = 10;
/// Slots sampled for a victim.
pub const SAMPLE: usize = 32;

const NONE: u32 = u32::MAX;

/// What a load does with the tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierDirective {
    /// Copy the record out of `slot` (pinned until [`TierSettle::Used`]).
    Hit { slot: usize },
    /// Read the artifact into `slot`, and copy each part on to staging (the slot is filling until a settle).
    Fill { slot: usize },
    /// Read the artifact into staging alone.
    Bypass,
}

/// What a load that planned a directive tells the owner when it ends; the owner applies it to the shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierSettle {
    /// A hit is done with the slot.
    Used(usize),
    /// A fill read the record into the slot: it is ready for the next load.
    Filled(usize),
    /// A fill gave up: the slot holds nothing usable and is free again.
    Abandoned(usize),
    /// A fill's memory was leaked: the slot is out of use for good.
    Lost(usize),
}

/// Draws the slots a victim search looks at. A test supplies a fixed sequence.
pub trait VictimSampler: Send {
    /// A number below `bound` (which is at least 1).
    fn below(&mut self, bound: usize) -> usize;
}

/// The default sampler: SplitMix64.
#[derive(Debug)]
pub struct SplitMix(u64);

impl SplitMix {
    pub fn new(seed: u64) -> SplitMix {
        SplitMix(seed)
    }
}

impl VictimSampler for SplitMix {
    fn below(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // The multiply-high maps the draw onto 0..bound without a division.
        ((u128::from(z) * bound as u128) >> 64) as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Free,
    Filling,
    Ready,
    Lost,
}

/// What a shard counted, per bank, and what is in it now.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TierStats {
    pub hits: Vec<u64>,
    pub misses: Vec<u64>,
    pub evictions: Vec<u64>,
    pub bypasses: Vec<u64>,
    /// Ready records per bank.
    pub ready: Vec<u64>,
    /// Prefetched records a later request found.
    pub prefetches_used: u64,
}

/// One shard of the host tier. See the module documentation.
pub struct RamTierShard {
    layout: Arc<ExpertLayout>,
    index: usize,
    shards: usize,
    first_slot: usize,
    slots: usize,
    resident: bool,
    sampler: Box<dyn VictimSampler>,
    /// Each slot's latest plan, by `clock`.
    last_use: Vec<u64>,
    clock: u64,
    /// Requests of each record since the counts were last halved (-1: assigned a slot by share and not asked for);
    /// empty for a resident shard.
    requests: Vec<i32>,
    since_aging: usize,
    /// The local slot of each key, or `NONE`.
    directory: Vec<u32>,
    slot_key: Vec<u32>,
    state: Vec<State>,
    pins: Vec<u32>,
    prefetched: Vec<bool>,
    free: Vec<u32>,
    stats: TierStats,
}

impl RamTierShard {
    fn new(layout: Arc<ExpertLayout>, index: usize, shards: usize, first_slot: usize, slots: usize) -> RamTierShard {
        let keys = layout.keys();
        let owned = (0..keys).filter(|&key| shard_of(key, shards) == index).count();
        let resident = slots >= owned;
        let banks = layout.banks();
        RamTierShard {
            index,
            shards,
            first_slot,
            slots,
            resident,
            sampler: Box::new(SplitMix::new(0x5eed)),
            last_use: vec![0; slots],
            clock: 0,
            requests: if resident { Vec::new() } else { vec![0; keys] },
            since_aging: 0,
            directory: vec![NONE; keys],
            slot_key: vec![NONE; slots],
            state: vec![State::Free; slots],
            pins: vec![0; slots],
            prefetched: vec![false; slots],
            free: (0..slots as u32).rev().collect(),
            stats: TierStats {
                hits: vec![0; banks],
                misses: vec![0; banks],
                evictions: vec![0; banks],
                bypasses: vec![0; banks],
                ready: vec![0; banks],
                prefetches_used: 0,
            },
            layout,
        }
    }

    /// Replaces the sampler (tests fix the victims it draws).
    pub fn set_sampler(&mut self, sampler: Box<dyn VictimSampler>) {
        self.sampler = sampler;
    }

    /// Whether every record of this shard has a slot: nothing is ever evicted or filled.
    pub fn holds_all(&self) -> bool {
        self.resident
    }

    pub fn slot_count(&self) -> usize {
        self.slots
    }

    pub fn stats(&self) -> &TierStats {
        &self.stats
    }

    // ---------------------------------------------------------------- planning a load

    /// Decides what the load of `id` does with the tier, for a load a prefill chunk asked for. A hit or a fill pins the
    /// slot until the load's settle is applied.
    pub fn plan_scan(&mut self, id: ExpertId) -> Result<TierDirective, ExpertError> {
        self.plan(id, true)
    }

    /// As [`plan_scan`](Self::plan_scan), for a decode step's load, which is not subject to admission.
    pub fn plan_decode(&mut self, id: ExpertId) -> Result<TierDirective, ExpertError> {
        self.plan(id, false)
    }

    /// Decides what the load of `id` does with the tier; only a `scan`'s records are subject to admission.
    pub fn plan(&mut self, id: ExpertId, scan: bool) -> Result<TierDirective, ExpertError> {
        let key = self.layout.key(id)?;
        let asked = if self.resident { 0 } else { self.request(key) };
        self.clock += 1;
        let bank = id.bank as usize;
        let slot = self.directory[key];
        if slot != NONE {
            let slot = slot as usize;
            if self.state[slot] != State::Ready {
                self.stats.bypasses[bank] += 1;
                return Ok(TierDirective::Bypass);
            }
            self.pins[slot] += 1;
            self.last_use[slot] = self.clock;
            if self.prefetched[slot] {
                self.prefetched[slot] = false;
                self.stats.prefetches_used += 1;
            }
            self.stats.hits[bank] += 1;
            return Ok(TierDirective::Hit {
                slot: self.first_slot + slot,
            });
        }
        let slot = if self.resident { None } else { self.take(asked, scan) };
        let Some(slot) = slot else {
            self.stats.bypasses[bank] += 1;
            return Ok(TierDirective::Bypass);
        };
        self.stats.misses[bank] += 1;
        self.reserve(slot, key);
        Ok(TierDirective::Fill {
            slot: self.first_slot + slot,
        })
    }

    /// Plans a prefetch of `id`: a fill of a slot ahead of any request, which counts no request and faces no
    /// admission. `Some(Fill)` (the slot is filling until a settle) or `None` when the tier holds or is filling the
    /// record, or no slot can be taken.
    pub fn plan_prefetch(&mut self, id: ExpertId) -> Result<Option<TierDirective>, ExpertError> {
        let key = self.layout.key(id)?;
        if self.resident || self.directory[key] != NONE {
            return Ok(None);
        }
        self.clock += 1;
        let Some(slot) = self.take(0, false) else {
            return Ok(None);
        };
        self.reserve(slot, key);
        self.prefetched[slot] = true;
        Ok(Some(TierDirective::Fill {
            slot: self.first_slot + slot,
        }))
    }

    /// Whether the record's slot is being filled.
    pub fn is_filling(&self, id: ExpertId) -> bool {
        self.slot_of_id(id)
            .is_some_and(|slot| self.state[slot] == State::Filling)
    }

    /// Whether the record has a ready slot.
    pub fn is_ready(&self, id: ExpertId) -> bool {
        self.slot_of_id(id).is_some_and(|slot| self.state[slot] == State::Ready)
    }

    /// The global slot of a record that has one (ready or filling).
    pub fn slot_of(&self, id: ExpertId) -> Option<usize> {
        self.slot_of_id(id).map(|slot| self.first_slot + slot)
    }

    fn slot_of_id(&self, id: ExpertId) -> Option<usize> {
        let slot = self.directory[self.layout.key(id).ok()?];
        (slot != NONE).then_some(slot as usize)
    }

    /// Applies what a load told when it ended: the owner's one record of it. Refuses a settle that does not match the
    /// slot (a hit's settle of a slot that is not pinned, a fill's of one that is not filling).
    pub fn apply(&mut self, settle: TierSettle) -> Result<(), ExpertError> {
        let (global, what) = match settle {
            TierSettle::Used(slot) => (slot, "used"),
            TierSettle::Filled(slot) => (slot, "filled"),
            TierSettle::Abandoned(slot) => (slot, "abandoned"),
            TierSettle::Lost(slot) => (slot, "lost"),
        };
        let bad = ExpertError::BadSettle { slot: global, what };
        let slot = global
            .checked_sub(self.first_slot)
            .filter(|&slot| slot < self.slots)
            .ok_or_else(|| bad.clone())?;
        let state = self.state[slot];
        match settle {
            TierSettle::Used(_) if state == State::Ready && self.pins[slot] > 0 => self.pins[slot] -= 1,
            TierSettle::Filled(_) if state == State::Filling => {
                self.state[slot] = State::Ready;
                self.pins[slot] = 0;
                let bank = self.bank_of(slot);
                self.stats.ready[bank] += 1;
            }
            TierSettle::Abandoned(_) if state == State::Filling => {
                self.prefetched[slot] = false;
                self.directory[self.slot_key[slot] as usize] = NONE;
                self.state[slot] = State::Free;
                self.pins[slot] = 0;
                self.free.push(slot as u32);
            }
            TierSettle::Lost(_) if state == State::Filling => {
                self.prefetched[slot] = false;
                self.directory[self.slot_key[slot] as usize] = NONE;
                self.state[slot] = State::Lost;
                self.pins[slot] = 0;
            }
            _ => return Err(bad),
        }
        Ok(())
    }

    // ---------------------------------------------------------------- slots

    fn bank_of(&self, slot: usize) -> usize {
        self.layout.id_of(self.slot_key[slot] as usize).bank as usize
    }

    fn reserve(&mut self, slot: usize, key: usize) {
        self.state[slot] = State::Filling;
        self.pins[slot] = 1;
        self.slot_key[slot] = key as u32;
        self.directory[key] = slot as u32;
        self.last_use[slot] = self.clock;
    }

    /// Counts a request of `key` and returns how often it was asked for before it.
    fn request(&mut self, key: usize) -> i32 {
        let before = self.requests[key];
        self.requests[key] = before.saturating_add(1);
        self.since_aging += 1;
        if self.since_aging >= AGE_PER_SLOT * self.slots {
            self.since_aging = 0;
            for count in &mut self.requests {
                *count >>= 1;
            }
        }
        before
    }

    /// A free slot, or the slot of the least requested victim, or `None` when every slot is filling or pinned, or (with
    /// admission, `admit`) when the victim was asked for at least as often as the record was before this request.
    fn take(&mut self, asked: i32, admit: bool) -> Option<usize> {
        if let Some(slot) = self.free.pop() {
            return Some(slot as usize);
        }
        let victim = self.least_requested()?;
        if admit && self.requests[self.slot_key[victim] as usize] >= asked {
            return None;
        }
        self.evict(victim);
        Some(victim)
    }

    fn evictable(&self, slot: usize) -> bool {
        self.state[slot] == State::Ready && self.pins[slot] == 0
    }

    /// The least requested of [`SAMPLE`] ready, unpinned slots drawn at random, the least recently planned among equals;
    /// when the sample finds none, the first such slot.
    fn least_requested(&mut self) -> Option<usize> {
        let mut victim: Option<usize> = None;
        for _ in 0..SAMPLE {
            let slot = self.sampler.below(self.slots);
            if self.evictable(slot) && victim.is_none_or(|best| self.before(slot, best)) {
                victim = Some(slot);
            }
        }
        victim.or_else(|| (0..self.slots).find(|&slot| self.evictable(slot)))
    }

    fn before(&self, slot: usize, other: usize) -> bool {
        let a = self.requests[self.slot_key[slot] as usize];
        let b = self.requests[self.slot_key[other] as usize];
        a < b || (a == b && self.last_use[slot] < self.last_use[other])
    }

    fn evict(&mut self, slot: usize) {
        let bank = self.bank_of(slot);
        self.prefetched[slot] = false;
        self.directory[self.slot_key[slot] as usize] = NONE;
        self.state[slot] = State::Free;
        self.stats.evictions[bank] += 1;
        self.stats.ready[bank] -= 1;
    }

    // ---------------------------------------------------------------- assigned tiers

    /// Gives every record of this shard its slot, ready, in key order, for a tier that holds them all. Before any load;
    /// the preload reads the records into the slots. Refuses a shard with fewer slots than records.
    pub fn assign_resident(&mut self) -> Result<(), ExpertError> {
        if !self.resident {
            return Err(ExpertError::BadSettle {
                slot: self.first_slot,
                what: "assign_resident of a shard with fewer slots than records",
            });
        }
        let mut slot = 0;
        for key in 0..self.layout.keys() {
            if shard_of(key, self.shards) != self.index {
                continue;
            }
            self.directory[key] = slot as u32;
            self.slot_key[slot] = key as u32;
            self.state[slot] = State::Ready;
            self.stats.ready[self.layout.id_of(key).bank as usize] += 1;
            slot += 1;
        }
        self.free = (slot as u32..self.slots as u32).rev().collect();
        Ok(())
    }

    /// Gives each bank its share of this shard's slots (in proportion to its experts), filled by its lowest experts in
    /// consecutive slots, ready. Before any load, for a bounded tier; the startup fill reads the records into the slots.
    /// They hold their slots only until a request needs one: each counts as asked for less than never (-1), so the
    /// first request of any record replaces it as it would take a free slot, and the tier adapts as an empty one would.
    pub fn assign_share(&mut self) {
        if self.resident {
            return;
        }
        let banks = self.layout.banks();
        let mut owned = vec![0usize; banks];
        let mut owned_total = 0usize;
        for key in (0..self.layout.keys()).filter(|&key| shard_of(key, self.shards) == self.index) {
            owned[self.layout.id_of(key).bank as usize] += 1;
            owned_total += 1;
        }
        let mut taken = vec![0usize; banks];
        let mut slot = 0;
        for key in 0..self.layout.keys() {
            if owned_total == 0 {
                break;
            }
            if shard_of(key, self.shards) != self.index {
                continue;
            }
            let bank = self.layout.id_of(key).bank as usize;
            if taken[bank] >= self.slots * owned[bank] / owned_total {
                continue;
            }
            taken[bank] += 1;
            self.directory[key] = slot as u32;
            self.slot_key[slot] = key as u32;
            self.state[slot] = State::Ready;
            self.pins[slot] = 0;
            self.requests[key] = -1;
            self.stats.ready[bank] += 1;
            slot += 1;
        }
        self.free = (slot as u32..self.slots as u32).rev().collect();
    }

    /// The records this shard holds ready, with their global slots: what a preload reads in.
    pub fn assigned(&self) -> Vec<(ExpertId, usize)> {
        (0..self.slots)
            .filter(|&slot| self.state[slot] == State::Ready)
            .map(|slot| (self.layout.id_of(self.slot_key[slot] as usize), self.first_slot + slot))
            .collect()
    }

    // ---------------------------------------------------------------- inspection

    /// Checks the bookkeeping against itself: every directory entry names a slot that names it back, and the counts
    /// agree. Panics with the broken fact.
    pub fn check_invariants(&self) {
        let mut mapped = 0;
        for (key, &slot) in self.directory.iter().enumerate() {
            if slot == NONE {
                continue;
            }
            mapped += 1;
            let slot = slot as usize;
            assert_eq!(self.slot_key[slot] as usize, key, "slot {slot} does not hold key {key}");
            assert!(
                matches!(self.state[slot], State::Filling | State::Ready),
                "slot {slot} is mapped but {:?}",
                self.state[slot]
            );
        }
        let count = |wanted: State| self.state.iter().filter(|&&state| state == wanted).count();
        let (ready, filling, lost) = (count(State::Ready), count(State::Filling), count(State::Lost));
        assert_eq!(mapped, ready + filling, "the directory holds {mapped}");
        assert_eq!(
            self.free.len(),
            self.slots - ready - filling - lost,
            "free slots {}",
            self.free.len()
        );
        assert!(
            self.free.iter().all(|&slot| self.state[slot as usize] == State::Free),
            "a slot on the free list is not free"
        );
        assert_eq!(
            self.stats.ready.iter().sum::<u64>(),
            ready as u64,
            "ready records are miscounted"
        );
    }

    /// Slots filling or pinned. Zero when no load is in flight.
    pub fn pinned_slots(&self) -> usize {
        (0..self.slots)
            .filter(|&slot| self.state[slot] == State::Filling || self.pins[slot] > 0)
            .count()
    }
}

/// The host tier: slots of ordinary memory for records, and one shard of bookkeeping per owner. `into_parts` hands the
/// shards out to their owners; the memory is shared, reached through leases.
pub struct RamTier {
    arena: Arc<HostArena>,
    shards: Vec<RamTierShard>,
    resident: bool,
}

impl RamTier {
    /// A tier of about `slot_count` slots over the records of `layout`, split among `shards` shards. A tier with a slot
    /// for every record is resident (every shard is assigned its records, ready, for the preload to read in). Refuses no
    /// shard, and fewer slots than shards (each needs one).
    pub fn new(layout: Arc<ExpertLayout>, slot_count: usize, shards: usize) -> Result<RamTier, ExpertError> {
        if shards == 0 || slot_count < shards {
            return Err(ExpertError::TierTooSmall {
                slots: slot_count,
                shards,
            });
        }
        let resident = slot_count >= layout.keys();
        let mut owned = vec![0usize; shards];
        for key in 0..layout.keys() {
            owned[shard_of(key, shards)] += 1;
        }
        let sizes: Vec<usize> = (0..shards)
            .map(|shard| {
                if resident {
                    owned[shard]
                } else {
                    owned[shard].min(slot_count / shards + usize::from(shard < slot_count % shards))
                }
            })
            .collect();
        let total: usize = sizes.iter().sum();
        let arena = HostArena::new(total.max(1), layout.slot_bytes())?;
        let mut first = 0;
        let mut list = Vec::with_capacity(shards);
        for (index, &size) in sizes.iter().enumerate() {
            let mut shard = RamTierShard::new(Arc::clone(&layout), index, shards, first, size);
            // A shard whose slots hold all its records is resident whatever the whole tier is.
            if shard.holds_all() {
                shard.assign_resident()?;
            }
            list.push(shard);
            first += size;
        }
        Ok(RamTier {
            arena,
            shards: list,
            resident,
        })
    }

    /// Whether every record has a slot.
    pub fn is_resident(&self) -> bool {
        self.resident
    }

    /// The slots' memory.
    pub fn arena(&self) -> &Arc<HostArena> {
        &self.arena
    }

    /// The memory and the shards, which go to their owners (shard `i` to the owner of cache shard `i`).
    pub fn into_parts(self) -> (Arc<HostArena>, Vec<RamTierShard>) {
        (self.arena, self.shards)
    }
}

impl fmt::Debug for RamTierShard {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RamTierShard")
            .field("index", &self.index)
            .field("slots", &self.slots)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for RamTier {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RamTier")
            .field("resident", &self.resident)
            .finish_non_exhaustive()
    }
}
