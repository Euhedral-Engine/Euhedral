package io.euhedral_execution.inference.core.model.qwen38.offload;

import io.euhedral_execution.inference.core.artifact.WeightStaging;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;

/// The pinned host window between the artifact file and the device staging ring: [#SLOTS] slots, two for each
/// staging slot, each with a 32-bit ready flag in host-mapped memory.
///
/// A slot's flag is 0 while the slot is free and 1 once a read filled it. The [DiskReader] raises it; the device
/// waits for it, copies the slot into its staging slot and lowers it on its stream. Each side touches a slot only in
/// its turn: the reader fills a slot only after the device lowered its flag, the device copies it only after the
/// reader raised it.
public final class DiskWindow {

    /// Window slots: two per staging slot, so the next read into a staging slot's window runs while the device copies
    /// the last one.
    public static final int SLOTS = 2 * WeightStaging.SLOTS;

    /// Reads are whole aligned blocks of the file.
    public static final long ALIGNMENT = 4096;

    /// Flags sit a cache line apart.
    private static final long FLAG_STRIDE = 64;

    private static final VarHandle INT = ValueLayout.JAVA_INT.varHandle();

    private final MemorySegment slots;
    private final MemorySegment flags;
    private final long slotBytes;
    private final long flagsDevice;
    /// A window of [#SLOTS] slots of `slotBytes` each in pinned, host-mapped memory of `memory`, with its flag page.
    /// The memory belongs to `memory`'s owner, which frees it.
    public DiskWindow(GpuMemory memory, long slotBytes) {
        if (slotBytes <= 0 || slotBytes % ALIGNMENT != 0)
            throw new IllegalArgumentException("window slots are whole blocks");
        long bytes = Math.addExact(Math.multiplyExact(slotBytes, SLOTS), ALIGNMENT);
        long allocation = memory.allocateHostWeights(bytes);
        MemorySegment all = MemorySegment.ofAddress(allocation).reinterpret(bytes);
        this.flags = all.asSlice(0, ALIGNMENT);
        this.slots = all.asSlice(ALIGNMENT, bytes - ALIGNMENT);
        this.slotBytes = slotBytes;
        this.flags.fill((byte) 0);
        this.flagsDevice = memory.hostWeightsDeviceAddress(allocation);
    }

    /// A window over `slots` and `flags` that no device sees (tests).
    public DiskWindow(MemorySegment slots, MemorySegment flags, long slotBytes) {
        if (slots.byteSize() < slotBytes * SLOTS || flags.byteSize() < FLAG_STRIDE * SLOTS)
            throw new IllegalArgumentException("the window's memory is too small");
        this.slots = slots;
        this.flags = flags;
        this.slotBytes = slotBytes;
        this.flagsDevice = 0;
    }

    /// Slot size that holds the aligned blocks around a payload of `largest` bytes at any offset.
    public static long slotBytesFor(long largest) {
        return align(largest) + ALIGNMENT;
    }

    static long align(long bytes) {
        return (bytes + ALIGNMENT - 1) & -ALIGNMENT;
    }

    public long slotBytes() {
        return this.slotBytes;
    }

    /// The host address of slot `slot`.
    public long slotHost(int slot) {
        return this.slots.address() + slot * this.slotBytes;
    }

    /// Slot `slot`'s memory.
    MemorySegment slotSegment(int slot) {
        return this.slots.asSlice(slot * this.slotBytes, this.slotBytes);
    }

    /// The host address of slot `slot`'s flag.
    public long flagHost(int slot) {
        return this.flags.address() + slot * FLAG_STRIDE;
    }

    /// The device address of slot `slot`'s flag, which stream memory operations wait on and write.
    public long flagDevice(int slot) {
        if (this.flagsDevice == 0) throw new IllegalStateException("this window is not mapped into a device");
        return this.flagsDevice + slot * FLAG_STRIDE;
    }

    /// Slot `slot`'s flag, read with acquire semantics.
    public int flag(int slot) {
        return (int) INT.getAcquire(this.flags, slot * FLAG_STRIDE);
    }

    /// Marks slot `slot` filled: its bytes are written before the flag.
    public void raise(int slot) {
        INT.setRelease(this.flags, slot * FLAG_STRIDE, 1);
    }

    /// Marks slot `slot` free from the host: the device's role, which tests play.
    public void lower(int slot) {
        INT.setRelease(this.flags, slot * FLAG_STRIDE, 0);
    }
}
