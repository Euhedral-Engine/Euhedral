package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/// Device side of the expert tests: a slab of expert slots, the descriptor and scratch of a wave, and the loop that
/// plays a chunk through its waves the way the model will (fill the slots of a wave, describe it, run it).
final class Qwen4ExpertHarness implements AutoCloseable {

    /// Where an expert's record comes from.
    interface Records {
        /// Copies the record of `expert` into `destination` (at least `RECORD_BYTES` long).
        void read(int expert, MemorySegment destination) throws IOException;
    }

    /// Called after a wave ran, with its index and the device addresses of its scratch.
    interface WaveObserver {
        void wave(int wave, Qwen4ExpertWave plan, long act, long weighted) throws IOException;
    }

    static final long SLOT_BYTES = (Qwen4ExpertReference.RECORD_BYTES + 4095L) / 4096 * 4096;

    final CudaGpuMemory gpu;
    final Qwen4ExpertOps.Geometry geometry = Qwen4ExpertOps.Geometry.flashNext();
    final Qwen4ExpertWave plan;
    private final Arena arena = Arena.ofConfined();
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

    Qwen4ExpertHarness(CudaGpuMemory gpu, int experts, int topK, int maxRows, int maxWaveExperts, int maxWavePairs) {
        this.gpu = gpu;
        this.maxRows = maxRows;
        this.plan = new Qwen4ExpertWave(experts, topK, maxRows, maxWaveExperts, maxWavePairs);
        this.slab = gpu.allocate(SLOT_BYTES * maxWaveExperts);
        this.descriptor = gpu.allocate(this.plan.descriptorBytes());
        this.scratch = gpu.allocate(Qwen4ExpertWave.scratchBytes(maxWavePairs));
        this.xAddress = gpu.allocate(2L * maxRows * 2560);
        this.outAddress = gpu.allocate(2L * maxRows * 2560);
        this.bytesAllocated = SLOT_BYTES * maxWaveExperts
                + this.plan.descriptorBytes()
                + Qwen4ExpertWave.scratchBytes(maxWavePairs)
                + 4L * maxRows * 2560;
        this.hostDescriptor = this.arena.allocate(this.plan.descriptorBytes(), 16);
        this.hostRecord = this.arena.allocate(SLOT_BYTES, 4096);
        this.slotExpert = new int[maxWaveExperts];
        java.util.Arrays.fill(this.slotExpert, -1);
    }

    /// Runs the chunk's waves in order and returns `[rows][2560]` BF16 bits of the routed sums.
    short[] run(int rows, int[] ids, short[] weights, short[] x, Records records, WaveObserver observer)
            throws IOException {
        if (rows > this.maxRows) throw new IllegalArgumentException("rows");
        MemorySegment hostX = this.arena.allocate(2L * rows * 2560, 16);
        MemorySegment.copy(x, 0, hostX, ValueLayout.JAVA_SHORT, 0, rows * 2560);
        this.gpu.copyHostToDevice(this.xAddress, hostX, 2L * rows * 2560);
        // the first wave overwrites every row, so the start value is irrelevant
        this.gpu.zeroDeviceMemory(this.outAddress, 2L * rows * 2560);
        this.plan.plan(rows, ids, weights);
        for (int wave = 0; wave < this.plan.waveCount(); wave++) {
            this.plan.fill(wave, this.hostDescriptor);
            for (int i = 0; i < this.plan.waveExpertCount(wave); i++) {
                int expert = this.plan.waveExpert(wave, i);
                if (this.slotExpert[i] != expert) {
                    records.read(expert, this.hostRecord);
                    this.gpu.copyHostToDevice(this.slab + SLOT_BYTES * i, this.hostRecord, SLOT_BYTES);
                    this.slotExpert[i] = expert;
                }
                this.plan.setSlot(this.hostDescriptor, i, this.slab + SLOT_BYTES * i);
            }
            this.gpu.copyHostToDevice(this.descriptor, this.hostDescriptor, this.plan.descriptorBytes());
            Qwen4ExpertOps.runWave(
                    this.gpu,
                    this.geometry,
                    this.plan,
                    wave,
                    this.descriptor,
                    this.xAddress,
                    this.scratch,
                    this.outAddress,
                    wave == 0);
            if (observer != null) {
                int pairs = this.plan.wavePairCount(wave);
                observer.wave(
                        wave,
                        this.plan,
                        this.scratch,
                        this.scratch + Qwen4ExpertOps.Scratch.weightedOffset(pairs, this.geometry.inter()));
            }
        }
        if (this.plan.waveCount() == 0) return new short[rows * 2560];
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
