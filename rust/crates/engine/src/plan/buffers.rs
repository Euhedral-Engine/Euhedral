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
}

/// The number of storages a plan with `regions` scratch regions tracks: every workspace slot, then each region.
pub(crate) fn storage_count(regions: usize) -> usize {
    BufferSlot::ALL.len() + regions
}

/// The storage index of a slot, or `None` for the input record.
pub(crate) fn slot_storage(slot: BufferSlot) -> Option<usize> {
    (slot != BufferSlot::TokenIds).then(|| slot.index())
}

/// The storage index of scratch region `region`.
pub(crate) fn scratch_storage(region: usize) -> usize {
    BufferSlot::ALL.len() + region
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_input_record_is_not_a_tracked_storage() {
        assert_eq!(slot_storage(BufferSlot::TokenIds), None);
        assert_eq!(slot_storage(BufferSlot::HiddenState), Some(0));
        assert_eq!(scratch_storage(1), BufferSlot::ALL.len() + 1);
        assert_eq!(storage_count(2), BufferSlot::ALL.len() + 2);
    }
}
