package io.euhedral_execution.inference.core.model.qwen38.offload;

import io.euhedral_execution.inference.core.artifact.WeightStaging;

/// The pinned host window between the artifact file and the device staging ring: [#SLOTS] slots, two for each
/// staging slot, each read whole from the file before the device copies it out.
public final class DiskWindow {

    /// Window slots: two per staging slot, so the next read into a staging slot's window runs while the device copies
    /// the last one.
    public static final int SLOTS = 2 * WeightStaging.SLOTS;

    /// Reads are whole aligned blocks of the file.
    public static final long ALIGNMENT = 4096;

    private DiskWindow() {}

    /// Slot size that holds the aligned blocks around a payload of `largest` bytes at any offset.
    public static long slotBytesFor(long largest) {
        return align(largest) + ALIGNMENT;
    }

    static long align(long bytes) {
        return (bytes + ALIGNMENT - 1) & -ALIGNMENT;
    }
}
