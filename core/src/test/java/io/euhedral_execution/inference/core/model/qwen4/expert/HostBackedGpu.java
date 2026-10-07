package io.euhedral_execution.inference.core.model.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.gpu.InlineGpuStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

/// A GPU whose "device" and "pinned host" memory are ordinary native memory, with exact bookkeeping of every
/// allocation: copies are range-checked against live allocations (a write outside a slab, or into freed memory,
/// fails), double frees fail, and the live allocations can be asserted to be gone.
final class HostBackedGpu extends ExecutionGpu {

    private record Region(long address, long size, Arena arena) {}

    private final ConcurrentSkipListMap<Long, Region> device = new ConcurrentSkipListMap<>();
    private final ConcurrentSkipListMap<Long, Region> host = new ConcurrentSkipListMap<>();
    private final List<Long> deviceAllocationSizes = new ArrayList<>();
    private final List<Long> hostAllocationSizes = new ArrayList<>();
    private final AtomicInteger deviceFrees = new AtomicInteger();
    private final AtomicInteger hostFrees = new AtomicInteger();
    private final AtomicInteger hostCopies = new AtomicInteger();
    private final AtomicLong hostCopyBytes = new AtomicLong();
    private final AtomicInteger copiesToFail = new AtomicInteger();
    private final AtomicInteger streamsOpened = new AtomicInteger();
    private final AtomicInteger streamsClosed = new AtomicInteger();
    private volatile Supplier<GpuStream> streams = InlineGpuStream::new;
    private volatile boolean provenCompletion = true;

    /// Streams `openStream` hands out, each counted as opened and (through the returned wrapper's close) closed.
    void streamFactory(Supplier<GpuStream> factory) {
        this.streams = factory;
    }

    /// The next `count` host-to-device copies throw from the submitting call.
    void failNextHostCopies(int count) {
        this.copiesToFail.set(count);
    }

    void completionProven(boolean proven) {
        this.provenCompletion = proven;
    }

    @Override
    public boolean completionProven() {
        return this.provenCompletion;
    }

    @Override
    public GpuStream openStream() {
        this.streamsOpened.incrementAndGet();
        GpuStream stream = this.streams.get();
        return new CountingStream(stream);
    }

    int streamsOpened() {
        return this.streamsOpened.get();
    }

    int streamsClosed() {
        return this.streamsClosed.get();
    }

    // ------------------------------------------------------------ device memory

    @Override
    public long allocate(long byteSize) {
        Arena arena = Arena.ofShared();
        long address = arena.allocate(byteSize, 256).address();
        this.device.put(address, new Region(address, byteSize, arena));
        synchronized (this.deviceAllocationSizes) {
            this.deviceAllocationSizes.add(byteSize);
        }
        return address;
    }

    @Override
    public void free(long address) {
        Region region = this.device.remove(address);
        if (region == null) throw new IllegalStateException("not a live device allocation: " + address);
        this.deviceFrees.incrementAndGet();
        region.arena().close();
    }

    @Override
    public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
        checkRange(this.device, destination, byteSize, "device");
        MemorySegment.ofAddress(destination).reinterpret(byteSize).copyFrom(source.asSlice(0, byteSize));
    }

    @Override
    public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
        checkRange(this.device, source, byteSize, "device");
        destination
                .asSlice(0, byteSize)
                .copyFrom(MemorySegment.ofAddress(source).reinterpret(byteSize));
    }

    // ------------------------------------------------------------ pinned host weights

    @Override
    public long allocateHostWeights(long byteSize) {
        Arena arena = Arena.ofShared();
        long address = arena.allocate(byteSize, 4096).address();
        this.host.put(address, new Region(address, byteSize, arena));
        synchronized (this.hostAllocationSizes) {
            this.hostAllocationSizes.add(byteSize);
        }
        return address;
    }

    @Override
    public void freeHostWeights(long address) {
        Region region = this.host.remove(address);
        if (region == null) throw new IllegalStateException("not a live host weight allocation: " + address);
        this.hostFrees.incrementAndGet();
        region.arena().close();
    }

    @Override
    public void copyHostWeightsToDevice(long destination, long source, long byteSize) {
        if (this.copiesToFail.getAndUpdate(count -> Math.max(0, count - 1)) > 0)
            throw new IllegalStateException("injected host copy failure");
        checkRange(this.device, destination, byteSize, "device");
        checkRange(this.host, source, byteSize, "host");
        FakeStream stream = FakeStream.selected();
        Runnable copy = () -> {
            // The bookkeeping is checked again when the copy runs: the memory must still be live then.
            checkRange(this.device, destination, byteSize, "device");
            checkRange(this.host, source, byteSize, "host");
            MemorySegment.ofAddress(destination)
                    .reinterpret(byteSize)
                    .copyFrom(MemorySegment.ofAddress(source).reinterpret(byteSize));
            this.hostCopies.incrementAndGet();
            this.hostCopyBytes.addAndGet(byteSize);
        };
        if (stream == null) copy.run();
        else stream.enqueue(copy);
    }

    private static void checkRange(ConcurrentSkipListMap<Long, Region> regions, long address, long size, String what) {
        Map.Entry<Long, Region> entry = regions.floorEntry(address);
        if (entry == null
                || address + size
                        > entry.getValue().address() + entry.getValue().size())
            throw new IllegalStateException(
                    what + " range [" + address + ", +" + size + ") is outside every live allocation");
    }

    @Override
    public void embedQ3(
            long tokenIdsAddress,
            long embeddingAddress,
            long embeddingByteSize,
            long hiddenStateAddress,
            int tokenCount,
            int vocabularySize,
            int hiddenSize) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void synchronize() {}

    // ------------------------------------------------------------ inspection

    byte[] readDevice(long address, long byteSize) {
        checkRange(this.device, address, byteSize, "device");
        return MemorySegment.ofAddress(address).reinterpret(byteSize).toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
    }

    int liveDeviceAllocations() {
        return this.device.size();
    }

    long liveDeviceBytes() {
        return this.device.values().stream().mapToLong(Region::size).sum();
    }

    int liveHostAllocations() {
        return this.host.size();
    }

    long liveHostBytes() {
        return this.host.values().stream().mapToLong(Region::size).sum();
    }

    List<Long> deviceAllocationSizes() {
        synchronized (this.deviceAllocationSizes) {
            return List.copyOf(this.deviceAllocationSizes);
        }
    }

    List<Long> hostAllocationSizes() {
        synchronized (this.hostAllocationSizes) {
            return List.copyOf(this.hostAllocationSizes);
        }
    }

    int deviceFrees() {
        return this.deviceFrees.get();
    }

    int hostFrees() {
        return this.hostFrees.get();
    }

    int hostCopies() {
        return this.hostCopies.get();
    }

    long hostCopyBytes() {
        return this.hostCopyBytes.get();
    }

    /// Fails unless every allocation made was freed exactly once.
    void assertAllReleased() {
        if (!this.device.isEmpty() || !this.host.isEmpty())
            throw new AssertionError("live allocations remain: " + this.device.size() + " device (" + liveDeviceBytes()
                    + " bytes), " + this.host.size() + " host (" + liveHostBytes() + " bytes)");
        if (this.deviceFrees.get() != deviceAllocationSizes().size()
                || this.hostFrees.get() != hostAllocationSizes().size())
            throw new AssertionError("allocations and frees differ");
    }

    /// Counts closes of a delegate stream.
    private final class CountingStream implements GpuStream {
        private final GpuStream delegate;
        private boolean closed;

        private CountingStream(GpuStream delegate) {
            this.delegate = delegate;
        }

        @Override
        public void submit(Runnable launches, boolean overlapPredecessor) {
            this.delegate.submit(launches, overlapPredecessor);
        }

        @Override
        public long notifyRetired(RetirementListener listener) {
            return this.delegate.notifyRetired(listener);
        }

        @Override
        public Throwable confirmRetired(long ticket) {
            return this.delegate.confirmRetired(ticket);
        }

        @Override
        public long openMarker() {
            return this.delegate.openMarker();
        }

        @Override
        public void mark(long marker) {
            this.delegate.mark(marker);
        }

        @Override
        public void await(long marker) {
            this.delegate.await(marker);
        }

        @Override
        public void closeMarker(long marker) {
            this.delegate.closeMarker(marker);
        }

        @Override
        public void synchronize() {
            this.delegate.synchronize();
        }

        @Override
        public void recover(Throwable failure) {
            this.delegate.recover(failure);
        }

        @Override
        public synchronized void close() {
            if (this.closed) return;
            this.closed = true;
            this.delegate.close();
            HostBackedGpu.this.streamsClosed.incrementAndGet();
        }
    }
}
