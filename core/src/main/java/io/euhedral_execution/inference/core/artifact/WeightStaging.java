package io.euhedral_execution.inference.core.artifact;

/// The device staging slots for host-backed weights: `slots` slots of `slotBytes` each from `baseAddress`.
///
/// Each view assigns the uses of host-backed weights to slots round-robin, in instruction order. A
/// weight-transfer stage copies the weight into its slot after the use that last read the slot, and the
/// consumer reads the slot after the transfer. Each slot is a workspace buffer, so across graphs too a transfer
/// follows the slot's last reader in the graph admitted before: nothing holds the slots.
public record WeightStaging(long baseAddress, long slotBytes, int slots) {

    /// Slots in a staging ring: enough for copies to queue ahead of their consumers.
    public static final int SLOTS = 4;

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
