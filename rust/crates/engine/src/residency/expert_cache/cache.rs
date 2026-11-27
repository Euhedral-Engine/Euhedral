//! The cache's public face: built from the plan's memory and the store, asked for experts, closed.

use std::fmt::{self, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use euhedral_data_structures::queues::MpscQueue;
use euhedral_memory::SlotArena;

use super::CacheError;
use super::host::{CopyLane, DeviceSlots, Frames};
use super::owner::{Listener, Message, Owner, Parts, Request};
use super::stats::{CacheSnapshot, CacheStats};
use crate::generation::{Address, Mailbox, Spawn};
use crate::handoff::Handoff;
use crate::residency::experts::{ExpertError, ExpertId, ExpertStore, RamTierShard};

/// Slots in a chunk of the owner's inbox.
const INBOX_CHUNK: usize = 64;

/// What a cache is built from.
pub struct CacheConfig {
    /// The store loads come from. Its layout fixes the experts the cache holds; its staging memory is the loads' target.
    pub store: Arc<ExpertStore>,
    /// The host tier's shards, as the store's tier was split; empty when the store has no tier.
    pub tier: Vec<RamTierShard>,
    /// The device slots, leased from the plan's pool.
    pub device: DeviceSlots,
    /// The free list over the staging memory's slots: at most the staging arena's slots, and each load takes one.
    pub staging_slots: Arc<SlotArena>,
    /// Store reads in flight at once, at least one.
    pub max_reads: usize,
    pub frames: Arc<dyn Frames>,
    pub copier: Arc<dyn CopyLane>,
}

impl fmt::Debug for CacheConfig {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CacheConfig")
            .field("device_slots", &self.device.len())
            .field("max_reads", &self.max_reads)
            .finish_non_exhaustive()
    }
}

/// The device cache of routed experts.
pub struct ExpertCache {
    mailbox: Arc<Mailbox<Owner>>,
    address: Arc<Address<Owner>>,
    store: Arc<ExpertStore>,
    stats: Arc<CacheStats>,
}

impl fmt::Debug for ExpertCache {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("ExpertCache").finish_non_exhaustive()
    }
}

impl ExpertCache {
    /// A cache over `config`; its hand-offs to a worker go through `spawn`. Refuses a device slot smaller than the
    /// largest record, a staging free list larger than the staging memory, and no device slot or no read.
    pub fn new(config: CacheConfig, spawn: Spawn) -> Result<ExpertCache, CacheError> {
        let CacheConfig {
            store,
            tier,
            device,
            staging_slots,
            max_reads,
            frames,
            copier,
        } = config;
        let stats = Arc::new(CacheStats::default());
        let queue = MpscQueue::<Message>::new(INBOX_CHUNK);
        let (producer, inbox) = (queue.producer(), queue.consumer());
        let address = Address::new();
        let parts = Parts {
            layout: Arc::clone(store.layout()),
            store: Arc::clone(&store),
            tier,
            device,
            staging: Arc::clone(store.staging()),
            staging_slots,
            max_reads,
            frames,
            copier,
            stats: Arc::clone(&stats),
        };
        let owner = Owner::new(inbox, Arc::clone(&address), parts)?;
        let mailbox = Arc::new(Mailbox::new(owner, producer, spawn));
        address.set(&mailbox);
        Ok(ExpertCache {
            mailbox,
            address,
            store,
            stats,
        })
    }

    /// Asks for `id`'s expert: `listener` is told, on a worker, with a lease (at once when the expert is resident), the
    /// error that names the expert, or that the request was cancelled. `cancel` ends the request when set (and the
    /// cache told by [`cancel`](ExpertCache::cancel)); `scan` marks a prefill scan, which the host tier may refuse to
    /// keep. Refuses an id the layout does not hold, with the listener not called.
    pub fn fetch(
        &self,
        id: ExpertId,
        scan: bool,
        cancel: Arc<AtomicBool>,
        listener: Listener,
    ) -> Result<(), ExpertError> {
        let layout = self.store.layout();
        let key = layout.key(id)?;
        let bytes = layout.record(id)?.bytes;
        self.address.post(Message::Request(Request {
            id,
            key,
            bytes,
            scan,
            cancel,
            listener,
        }));
        Ok(())
    }

    /// Sets `cancel` and has the owner end the requests that wait on it.
    pub fn cancel(&self, cancel: &AtomicBool) {
        cancel.store(true, Ordering::Release);
        self.address.post(Message::Sweep);
    }

    /// Closes the cache: waiting requests are cancelled, loads in flight are cancelled and awaited, and the handoff
    /// is sent when none is left. `None` when the caller holds the owner's claim (a listener).
    pub fn close(&self) -> Option<Arc<Handoff<()>>> {
        self.mailbox.stopping()
    }

    /// The counters, read now.
    pub fn stats(&self) -> CacheSnapshot {
        self.stats.snapshot()
    }
}
