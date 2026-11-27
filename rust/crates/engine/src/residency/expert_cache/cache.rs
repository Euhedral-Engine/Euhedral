//! The cache's public face: built from the plan's memory and the store, asked for experts, closed. Every call posts a
//! record to the owner and returns; the owner is a source of the lattice (see `source`) and a worker applies the records.

use std::fmt::{self, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::LatticeSource;
use euhedral_data_structures::queues::MpscQueue;
use euhedral_memory::SlotArena;

use super::CacheError;
use super::host::{CopyLane, DeviceSlots, Frames};
use super::owner::{Listener, Message, Outcome, Owner, Parts, Request};
use super::source::{Inbox, OwnerSource};
use super::stats::{CacheSnapshot, CacheStats};
use crate::handoff::Handoff;
use crate::residency::experts::{ExpertError, ExpertId, ExpertStore, RamTierShard};

/// Slots in a chunk of the owner's record queue.
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
    source: Arc<OwnerSource>,
    store: Arc<ExpertStore>,
    stats: Arc<CacheStats>,
}

impl fmt::Debug for ExpertCache {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("ExpertCache").finish_non_exhaustive()
    }
}

impl ExpertCache {
    /// A cache over `config`, attached to `lattice`: a worker of it applies the records posted, so there is no cache
    /// that nothing serves. Refuses what [`new`](ExpertCache::new) refuses, and a lattice that is closed.
    pub fn on_lattice(config: CacheConfig, lattice: &ControlPlaneLattice) -> Result<ExpertCache, CacheError> {
        ExpertCache::new(config, |source| {
            lattice
                .add_upstream(source)
                .map(drop)
                .map_err(|error| error.to_string())
        })
    }

    /// A cache over `config`, whose owner `attach` hands to the lattice as one of its sources (`add_upstream`): the owner
    /// does nothing until a worker visits it, so a cache that `attach` refuses is not built, and its error is returned.
    /// Refuses a device slot smaller than the largest record, a staging free list larger than the staging memory, and no
    /// device slot or no read.
    pub fn new(
        config: CacheConfig,
        attach: impl FnOnce(Arc<dyn LatticeSource>) -> Result<(), String>,
    ) -> Result<ExpertCache, CacheError> {
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
        let (producer, records) = (queue.producer(), queue.consumer());
        let inbox = Arc::new(Inbox::new(producer));
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
        let owner = Owner::new(records, Arc::clone(&inbox), parts)?;
        let source = OwnerSource::new(inbox, owner);
        attach(Arc::clone(&source) as Arc<dyn LatticeSource>).map_err(|why| CacheError::Attach { why })?;
        Ok(ExpertCache { source, store, stats })
    }

    /// Asks for `id`'s expert: `listener` is told, on a worker, with a lease (at once when the expert is resident), the
    /// error that names the expert, or that the request was cancelled. `cancel` ends the request when set (and the
    /// cache told by [`cancel`](ExpertCache::cancel)); `scan` marks a prefill scan, which the host tier may refuse to
    /// keep. Refuses an id the layout does not hold, with the listener not called. A cache that is closed and done (its
    /// source has completed, so no worker would apply the request) tells the listener [`CacheError::Closed`] at once,
    /// on the calling thread.
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
        if self.source.is_complete() {
            listener(Outcome::Failed(CacheError::Closed));
            return Ok(());
        }
        self.source.post(Message::Request(Request {
            id,
            key,
            bytes,
            scan,
            cancel,
            reply: self.source.reply(listener),
        }));
        Ok(())
    }

    /// Sets `cancel` and has the owner end the requests that wait on it.
    pub fn cancel(&self, cancel: &AtomicBool) {
        cancel.store(true, Ordering::Release);
        self.source.post(Message::Sweep);
    }

    /// Closes the cache: requests waiting and loads in flight are ended with [`CacheError::Closed`], the loads are awaited,
    /// and the handoff is sent when none is left. Requests after it are refused with the same error.
    pub fn close(&self) -> Arc<Handoff<()>> {
        let done = Handoff::new();
        if self.source.is_complete() {
            done.send(());
            return done;
        }
        self.source.post(Message::Close(Arc::clone(&done)));
        done
    }

    /// The counters, read now.
    pub fn stats(&self) -> CacheSnapshot {
        self.stats.snapshot()
    }
}
