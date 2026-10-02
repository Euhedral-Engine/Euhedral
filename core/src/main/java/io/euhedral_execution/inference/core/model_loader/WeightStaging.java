package io.euhedral_execution.inference.core.model_loader;

/// The device staging ring for host-backed weights: `slots` slots of `slotBytes` each from
/// `baseAddress`.
///
/// Each view assigns the uses of host-backed weights to slots round-robin, in instruction order. A
/// weight-transfer stage copies the weight into its slot after the use that
/// last read the slot, and the consumer reads the slot after the transfer. Quanta that stage weights
/// hold the ring from preparation until their last stage submitted, so their slot uses never interleave.
public record WeightStaging(long baseAddress, long slotBytes, int slots) {

    /// Slots start on 256-byte boundaries, like device allocations.
    public static final long SLOT_ALIGNMENT = 256;

    public WeightStaging {
        if (baseAddress == 0 || slotBytes <= 0 || slotBytes % SLOT_ALIGNMENT != 0 || slots < 2)
            throw new IllegalArgumentException("staging needs at least two aligned, non-empty slots");
    }

    public long slotAddress(int slot) {
        return this.baseAddress + slot * this.slotBytes;
    }

    /// Slot size that holds `largestWeight` bytes.
    public static long slotBytesFor(long largestWeight) {
        return (largestWeight + SLOT_ALIGNMENT - 1) / SLOT_ALIGNMENT * SLOT_ALIGNMENT;
    }
}
