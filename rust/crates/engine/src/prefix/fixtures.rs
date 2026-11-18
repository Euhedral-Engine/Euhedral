use std::sync::Arc;

use euhedral_memory::{LocationKind, LocationSpec, OwnerSpec, Planner, Priority, Scope, SlotArena};

/// An arena of `count` slots of `slot_bytes`, addressed as offsets from 0, accounted on a host location.
pub fn arena(count: u32, slot_bytes: u64) -> Arc<SlotArena> {
    let planner = Planner::new([LocationSpec {
        kind: LocationKind::HostPinned,
        capacity: u64::from(count.max(1)) * slot_bytes,
        reserve: 0,
    }]);
    let location = planner.location(LocationKind::HostPinned).expect("the host location");
    let owner = planner.open_owner(OwnerSpec::new("prefix", Priority(10), Scope::Epoch(1)));
    let claim = planner
        .claim(owner, location, u64::from(count) * slot_bytes, "prefix slots")
        .expect("the claim fits");
    SlotArena::new(0, slot_bytes, count, claim)
}

pub fn tokens(from: u32, count: u32) -> Vec<u32> {
    (from..from + count).collect()
}
