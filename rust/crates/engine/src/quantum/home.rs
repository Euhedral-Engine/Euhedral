//! Where a quantum's stages run among the stage lanes.
//!
//! The plan gives each stage an offset from the quantum's home lane (a chain of lane continuations has one offset;
//! the branches of a fan-out have different ones). Each quantum has its own home, taken from a hash of its id and the time it starts, so the root of
//! every quantum does not land on one lane. Only the stage lanes take part: the owner's, the capture and the
//! transfer lanes are never chosen.

use euhedral_hashing::hasher_api::hash_bytes_with_seed;

/// The home lane of the quantum with `id` started at `clock` (a monotonic reading in nanoseconds), among
/// `stage_lanes` stage lanes: the high half of the product of the xxHash64 of `id`, seeded with `clock`, and
/// `stage_lanes`, which is uniform over `0..stage_lanes` without a division. The reading is an argument so the
/// function is pure.
pub fn home_of(id: u64, clock: u64, stage_lanes: usize) -> usize {
    let hash = hash_bytes_with_seed(&id.to_le_bytes(), clock);
    ((u128::from(hash) * stage_lanes as u128) >> 64) as usize
}

/// The lane a stage with plan lane `offset` runs on for a quantum homed on `home`: `(home + offset)` modulo the
/// number of stage lanes. Stages of one chain share an offset and so a lane; branches have different offsets (the
/// plan is no wider than the stage lanes) and so different lanes.
pub fn lane_for(home: usize, offset: usize, stage_lanes: usize) -> usize {
    (home + offset) % stage_lanes
}
