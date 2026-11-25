//! The storages a stage reads and writes, which the plan orders (`ExecutionPlan.Buffer`, the instruction's input and
//! output buffers, and the expansion scratch).

use crate::workspace::BufferSlot;

/// What one stage touches: the workspace slots it reads and writes, and the scratch region it writes, if it takes one.
/// The input record (`TokenIds`) is not tracked: the quantum's preparation writes it before any stage runs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Access {
    pub reads: Vec<BufferSlot>,
    pub writes: Vec<BufferSlot>,
    /// The scratch region the stage takes, one at a time per region.
    pub scratch: Option<usize>,
    /// The staging slot the stage reads (a staged weight's consumer).
    pub slot_read: Option<usize>,
    /// The staging slot the stage fills (a weight transfer).
    pub slot_write: Option<usize>,
}

/// The number of storages a plan with `regions` scratch regions and `slots` staging slots tracks: every workspace slot,
/// then each region, then each staging slot.
pub(crate) fn storage_count(regions: usize, slots: usize) -> usize {
    BufferSlot::ALL.len() + regions + slots
}

/// The storage index of staging slot `slot` in a plan with `regions` scratch regions.
pub(crate) fn ring_storage(regions: usize, slot: usize) -> usize {
    BufferSlot::ALL.len() + regions + slot
}

/// The storage index of a slot, or `None` for the input record.
pub(crate) fn slot_storage(slot: BufferSlot) -> Option<usize> {
    (slot != BufferSlot::TokenIds).then(|| slot.index())
}

/// The storage index of scratch region `region`.
pub(crate) fn scratch_storage(region: usize) -> usize {
    BufferSlot::ALL.len() + region
}
