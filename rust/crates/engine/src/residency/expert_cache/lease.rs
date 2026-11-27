//! The right to use one expert's slot: while a lease is held the slot is neither evicted nor refilled.

use std::fmt::{self, Formatter};
use std::sync::Arc;

use euhedral_graph::Marker;

use super::owner::{Message, Owner};
use crate::generation::Address;
use crate::residency::experts::ExpertId;

/// A leased expert. Dropping it tells the cache's owner, which unpins the slot: nothing waits, and a lease dropped on
/// any thread is let go in the owner's own turn.
pub struct ExpertLease {
    id: ExpertId,
    slot: u32,
    generation: u64,
    address: u64,
    bytes: u64,
    ready: Option<Marker>,
    fence: Option<Marker>,
    owner: Arc<Address<Owner>>,
}

impl ExpertLease {
    pub(super) fn new(
        owner: Arc<Address<Owner>>,
        id: ExpertId,
        slot: u32,
        generation: u64,
        address: u64,
        bytes: u64,
        ready: Option<Marker>,
    ) -> ExpertLease {
        ExpertLease {
            id,
            slot,
            generation,
            address,
            bytes,
            ready,
            fence: None,
            owner,
        }
    }

    pub fn id(&self) -> ExpertId {
        self.id
    }

    /// The device address of the record, valid until the lease is dropped.
    pub fn device_address(&self) -> u64 {
        self.address
    }

    /// The record's length.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The marker of the copy that is still writing the slot, when it is: device work that reads the slot waits for it
    /// on its own lane. `None` when the record was in place.
    pub fn ready(&self) -> Option<Marker> {
        self.ready
    }

    /// Records that device work ending at `marker` reads the slot: the next copy into it waits behind `marker`. Call it
    /// once, with the last work submitted that reads the slot, before dropping the lease.
    pub fn fence_after(&mut self, marker: Marker) {
        self.fence = Some(marker);
    }
}

impl Drop for ExpertLease {
    fn drop(&mut self) {
        self.owner.post(Message::Release {
            slot: self.slot,
            generation: self.generation,
            fence: self.fence,
        });
    }
}

impl fmt::Debug for ExpertLease {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExpertLease")
            .field("id", &self.id)
            .field("slot", &self.slot)
            .field("address", &self.address)
            .field("ready", &self.ready)
            .finish_non_exhaustive()
    }
}
