package io.euhedral_execution.inference.core.model_loader.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/// A test GPU whose "device" and pinned memory are native host allocations, so raw addresses behave as they do with
/// CUDA, copies are synchronous, and live allocations are counted for leak checks.
public final class HostMemoryGpu extends ExecutionGpu {

    private record Allocation(Arena arena, long size) {}

    private final Map<Long, Allocation> device = new ConcurrentHashMap<>();
    private final Map<Long, Allocation> pinned = new ConcurrentHashMap<>();
    private final AtomicLong deviceBytes = new AtomicLong();
    private final AtomicLong peakDeviceBytes = new AtomicLong();
    private final AtomicLong hostBytes = new AtomicLong();
    private final AtomicLong hostToDeviceBytes = new AtomicLong();

    @Override
    public long allocate(long byteSize) {
        Arena arena = Arena.ofShared();
        long address = arena.allocate(byteSize, 256).address();
        this.device.put(address, new Allocation(arena, byteSize));
        peakDeviceBytes.accumulateAndGet(deviceBytes.addAndGet(byteSize), Math::max);
        return address;
    }

    @Override
    public void free(long address) {
        if (address == 0) return;
        Allocation allocation = this.device.remove(address);
        if (allocation == null) throw new IllegalArgumentException("not a device allocation: " + address);
        this.deviceBytes.addAndGet(-allocation.size());
        allocation.arena().close();
    }

    @Override
    public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
        MemorySegment.copy(source, 0, MemorySegment.ofAddress(destination).reinterpret(byteSize), 0, byteSize);
        this.hostToDeviceBytes.addAndGet(byteSize);
    }

    @Override
    public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
        MemorySegment.copy(MemorySegment.ofAddress(source).reinterpret(byteSize), 0, destination, 0, byteSize);
    }

    @Override
    public long allocateHostWeights(long byteSize) {
        Arena arena = Arena.ofShared();
        long address = arena.allocate(byteSize, 4096).address();
        this.pinned.put(address, new Allocation(arena, byteSize));
        this.hostBytes.addAndGet(byteSize);
        return address;
    }

    @Override
    public void freeHostWeights(long address) {
        Allocation allocation = this.pinned.remove(address);
        if (allocation == null) throw new IllegalArgumentException("not a host weight allocation: " + address);
        this.hostBytes.addAndGet(-allocation.size());
        allocation.arena().close();
    }

    @Override
    public long hostWeightsDeviceAddress(long hostAddress) {
        return hostAddress;
    }

    @Override
    public void copyHostWeightsToDevice(long destination, long source, long byteSize) {
        MemorySegment.copy(
                MemorySegment.ofAddress(source).reinterpret(byteSize),
                0,
                MemorySegment.ofAddress(destination).reinterpret(byteSize),
                0,
                byteSize);
        this.hostToDeviceBytes.addAndGet(byteSize);
    }

    @Override
    public void synchronize() {}

    @Override
    public void embedQ3(
            long tokenIdsAddress,
            long embeddingAddress,
            long embeddingByteSize,
            long hiddenStateAddress,
            int tokenCount,
            int vocabularySize,
            int hiddenSize) {
        throw new UnsupportedOperationException("no kernels on the host-memory test GPU");
    }

    public long liveDeviceBytes() {
        return this.deviceBytes.get();
    }

    public long peakDeviceBytes() {
        return this.peakDeviceBytes.get();
    }

    public int liveDeviceAllocations() {
        return this.device.size();
    }

    public long livePinnedBytes() {
        return this.hostBytes.get();
    }

    public int livePinnedAllocations() {
        return this.pinned.size();
    }

    public long hostToDeviceBytes() {
        return this.hostToDeviceBytes.get();
    }
}
