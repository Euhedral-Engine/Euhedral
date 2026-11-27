//! What the cache needs of the engine around it, as two small interfaces: the frames' home (the lake and the device
//! completions) and the copy lane. The engine's implementations are in `lanes`; tests replace both.

use std::fmt::{self, Formatter};

use euhedral_core::frames::Frame;
use euhedral_graph::Marker;
use euhedral_memory::{Claim, OwnerId, TierPool};
use std::sync::Arc;

use super::CacheError;
use crate::residency::experts::SlotRead;

/// Where the cache's frames go, and what holds the lake open while a load is out.
pub trait Frames: Send + Sync {
    /// Publishes `frame` to the lake's sinks; hands it back when the lake takes none (no unit admitted, or completed).
    fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>>;

    /// Publishes `frame` once the work `marker` names has completed on its lane. Never waits.
    fn after_marker(&self, marker: Marker, frame: Box<dyn Frame>);

    /// Admits one unit to the lake: the lake does not close while the guard is held. Refuses once the lake is closed.
    fn admit(&self) -> Result<Box<dyn Send>, String>;
}

/// One copy of a record from a staging slot to a device slot.
pub struct CopyJob {
    /// The device address of the slot the record goes to.
    pub device: u64,
    /// The record's length.
    pub bytes: usize,
    /// The staging slot's read lease: the bytes stay as they are, and the slot unwritten, while it is held. It is held
    /// until the copy retires.
    pub staging: SlotRead,
    /// The device work the copy waits behind: the kernels that read the slot's previous expert.
    pub waits: Vec<Marker>,
}

impl fmt::Debug for CopyJob {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CopyJob")
            .field("device", &self.device)
            .field("bytes", &self.bytes)
            .field("waits", &self.waits)
            .finish_non_exhaustive()
    }
}

/// Why a copy was not submitted.
#[derive(Debug)]
pub enum SubmitFailure {
    /// The lane has no room: nothing was queued, and the job is tried again by [`CopyLane::when_room`].
    Busy,
    /// Nothing was queued and the lane cannot take it.
    Refused(String),
    /// The copy may be queued without a marker: the lane failed after the work was taken. The job's staging bytes stay
    /// as they are until a later [`CopyLane::mark`] retires.
    Uncertain(String),
}

/// The lane the copies run on.
pub trait CopyLane: Send + Sync {
    /// Queues `job` behind its waits and returns the marker its retirement is told by. The caller keeps the job (and so
    /// its staging lease) until that retirement. Never waits.
    fn submit(&self, job: &CopyJob) -> Result<Marker, SubmitFailure>;

    /// A marker behind everything submitted on the lane so far, with no work of its own: its retirement says that every
    /// copy submitted before it, a copy that failed with [`SubmitFailure::Uncertain`] included, has run. `Busy` when the
    /// lane has no room for a marker.
    fn mark(&self) -> Result<Marker, SubmitFailure>;

    /// Runs `frame` once the lane has room again (a copy was refused as `Busy`). Never waits.
    fn when_room(&self, frame: Box<dyn Frame>);
}

/// The device slots: the whole of a tier pool's slots, leased once. The pool is the memory planner's, so the budget is
/// the plan's; the cache keeps no allocator of its own.
pub struct DeviceSlots {
    pool: Arc<TierPool>,
    held: Vec<(u32, Claim)>,
}

impl DeviceSlots {
    /// Leases `count` slots of `pool` for `owner`. Refuses no slot, and more than the pool has free, naming both.
    pub fn lease(pool: &Arc<TierPool>, owner: OwnerId, count: usize) -> Result<DeviceSlots, CacheError> {
        if count == 0 {
            return Err(CacheError::Slots {
                slots: 0,
                limit: pool.free() as usize,
            });
        }
        if count > pool.free() as usize {
            return Err(CacheError::PoolTooSmall {
                wanted: count,
                free: pool.free() as usize,
            });
        }
        let mut held = Vec::with_capacity(count);
        while held.len() < count {
            match pool.lease(owner, false) {
                Some(slot) => held.push(slot),
                None => {
                    let free = held.len();
                    for (index, claim) in held {
                        pool.give_back(index, claim);
                    }
                    return Err(CacheError::PoolTooSmall { wanted: count, free });
                }
            }
        }
        Ok(DeviceSlots {
            pool: Arc::clone(pool),
            held,
        })
    }

    pub fn len(&self) -> usize {
        self.held.len()
    }

    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }

    pub fn slot_bytes(&self) -> u64 {
        self.pool.slot_bytes()
    }

    /// The device address of cache slot `slot`.
    pub fn address(&self, slot: usize) -> u64 {
        self.pool.address(self.held[slot].0)
    }
}

impl Drop for DeviceSlots {
    fn drop(&mut self) {
        for (index, claim) in self.held.drain(..) {
            self.pool.give_back(index, claim);
        }
    }
}

impl fmt::Debug for DeviceSlots {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceSlots")
            .field("slots", &self.held.len())
            .field("slot_bytes", &self.pool.slot_bytes())
            .finish()
    }
}
