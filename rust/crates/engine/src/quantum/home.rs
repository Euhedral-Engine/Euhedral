//! Where a quantum's stages run among the stage lanes.
//!
//! The plan gives each stage an offset from the quantum's home lane (a chain of lane continuations has one offset;
//! the branches of a fan-out have different ones). Each quantum has its own home, taken from its id, so the root of
//! every quantum does not land on one lane. Only the stage lanes take part: the owner's, the capture and the
//! transfer lanes are never chosen.

/// The home lane of the quantum with `id`, among `stage_lanes` stage lanes: the high half of the product of a
/// 64-bit hash of `id` and `stage_lanes`, which is uniform over `0..stage_lanes` without a division.
pub fn home_of(id: u64, stage_lanes: usize) -> usize {
    let mut hash = id.wrapping_add(0x9E37_79B9_7F4A_7C15);
    hash = (hash ^ (hash >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    hash ^= hash >> 31;
    ((u128::from(hash) * stage_lanes as u128) >> 64) as usize
}

/// The lane a stage with plan lane `offset` runs on for a quantum homed on `home`: `(home + offset)` modulo the
/// number of stage lanes. Stages of one chain share an offset and so a lane; branches have different offsets (the
/// plan is no wider than the stage lanes) and so different lanes.
pub fn lane_for(home: usize, offset: usize, stage_lanes: usize) -> usize {
    (home + offset) % stage_lanes
}
