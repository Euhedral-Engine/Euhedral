package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/// Device side of the expert tests: a few expert slots, the descriptor and scratch of a chunk, and the loop that
/// plays a chunk the way the model does: every active expert on its own (its record in a slot, its address in the
/// descriptor, its kernels), in a chosen order, then the ordered combine.
final class ExpertHarness implements AutoCloseable {

    /// Where an expert's record comes from.
    interface Records {
        /// Copies the record of `expert` into `destination` (at least `RECORD_BYTES` long).
        void read(int expert, MemorySegment destination) throws IOException;
    }

    /// Called after every expert ran, before the combine, with the device addresses of the scratch.
    interface ExpertsObserver {
        void experts(ExpertRouting plan, long act, long weighted) throws IOException;
    }

    /// The order the experts run in.
    enum Order {
        ASCENDING,
        DESCENDING
    }

    static final long SLOT_BYTES = (ExpertReference.RECORD_BYTES + 4095L) / 4096 * 4096;

    final CudaGpuMemory gpu;
    final ExpertOps.Geometry geometry = ExpertOps.Geometry.flashNext();
    final ExpertRouting plan;
    private final Arena arena = Arena.ofConfined();
    private final int slots;
    private final long slab;
    private final long descriptor;
    private final long scratch;
    private final MemorySegment hostDescriptor;
    private final MemorySegment hostRecord;
    private final int[] slotExpert;
    private final int maxRows;
    private long xAddress;
    private long outAddress;
    long bytesAllocated;

    ExpertHarness(CudaGpuMemory gpu, int experts, int topK, int maxRows, int slots) {
        this.gpu = gpu;
        this.maxRows = maxRows;
        this.slots = slots;
        this.plan = new ExpertRouting(experts, topK, maxRows);
        this.slab = gpu.allocate(SLOT_BYTES * slots);
        this.descriptor = gpu.allocate(this.plan.descriptorBytes());
        this.scratch = gpu.allocate(ExpertRouting.scratchBytes(this.plan.maxPairs()));
        this.xAddress = gpu.allocate(2L * maxRows * 2560);
        this.outAddress = gpu.allocate(2L * maxRows * 2560);
        this.bytesAllocated = SLOT_BYTES * slots
                + this.plan.descriptorBytes()
                + ExpertRouting.scratchBytes(this.plan.maxPairs())
                + 4L * maxRows * 2560;
        this.hostDescriptor = this.arena.allocate(this.plan.descriptorBytes(), 16);
        this.hostRecord = this.arena.allocate(SLOT_BYTES, 4096);
        this.slotExpert = new int[slots];
        java.util.Arrays.fill(this.slotExpert, -1);
    }

    /// Runs the chunk's experts in ascending order and returns `[rows][2560]` BF16 bits of the routed sums.
    short[] run(int rows, int[] ids, short[] weights, short[] x, Records records, ExpertsObserver observer)
            throws IOException {
        return run(rows, ids, weights, x, records, Order.ASCENDING, observer);
    }

    /// Runs the chunk's experts in `order` and returns `[rows][2560]` BF16 bits of the routed sums.
    short[] run(int rows, int[] ids, short[] weights, short[] x, Records records, Order order, ExpertsObserver observer)
            throws IOException {
        if (rows > this.maxRows) throw new IllegalArgumentException("rows");
        MemorySegment hostX = this.arena.allocate(2L * rows * 2560, 16);
        MemorySegment.copy(x, 0, hostX, ValueLayout.JAVA_SHORT, 0, rows * 2560);
        this.gpu.copyHostToDevice(this.xAddress, hostX, 2L * rows * 2560);
        // the combine overwrites every row, so the start value is irrelevant
        this.gpu.zeroDeviceMemory(this.outAddress, 2L * rows * 2560);
        this.plan.plan(rows, ids, weights);
        int active = this.plan.activeExperts();
        if (active == 0) return new short[rows * 2560];
        this.plan.fill(this.hostDescriptor);
        this.gpu.copyHostToDevice(this.descriptor, this.hostDescriptor, this.plan.descriptorBytes());
        for (int n = 0; n < active; n++) {
            int index = order == Order.ASCENDING ? n : active - 1 - n;
            int expert = this.plan.activeExpert(index);
            int slot = index % this.slots;
            if (this.slotExpert[slot] != expert) {
                records.read(expert, this.hostRecord);
                this.gpu.copyHostToDevice(this.slab + SLOT_BYTES * slot, this.hostRecord, SLOT_BYTES);
                this.slotExpert[slot] = expert;
            }
            this.plan.setSlot(this.hostDescriptor, index, this.slab + SLOT_BYTES * slot);
            long entry = this.plan.slotsOffset() + 8L * index;
            this.gpu.copyHostToDevice(this.descriptor + entry, this.hostDescriptor.asSlice(entry, 8), 8);
            ExpertOps.runExpert(
                    this.gpu, this.geometry, this.plan, index, this.descriptor, this.xAddress, this.scratch);
            // The slot may be reloaded for the next expert: the kernels that read it must be done.
            this.gpu.synchronize();
        }
        if (observer != null)
            observer.experts(
                    this.plan,
                    this.scratch,
                    this.scratch + ExpertOps.Scratch.weightedOffset(this.plan.pairCount(), this.geometry.inter()));
        ExpertOps.combineExperts(this.gpu, this.geometry, this.plan, this.descriptor, this.scratch, this.outAddress);
        return download(this.outAddress, rows * 2560);
    }

    short[] download(long address, int count) {
        MemorySegment host = this.arena.allocate(2L * count, 16);
        this.gpu.copyDeviceToHost(host, address, 2L * count);
        short[] values = new short[count];
        MemorySegment.copy(host, ValueLayout.JAVA_SHORT, 0, values, 0, count);
        return values;
    }

    /// Wipes the slots, so the next run reloads every record.
    void forgetSlots() {
        java.util.Arrays.fill(this.slotExpert, -1);
    }

    @Override
    public void close() {
        this.arena.close();
    }
}
