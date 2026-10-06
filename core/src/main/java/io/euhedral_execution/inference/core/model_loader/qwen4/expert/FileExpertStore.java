package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

/// Cold storage: records stay in the artifact file (or in the tier a [RecordSource] puts in front
/// of it), and a lane's pinned staging slot holds one between the read and the end of its device
/// copy.
///
/// All slots are carved from one pinned arena, each as large as the biggest record rounded up to a
/// page. [#open] has the [RecordSource] fill the lane's slot: the one host copy of the record is
/// from the page cache into pinned memory, with no heap array in between. The lane owns its slot,
/// so there is no pool, no wait and no lock; memory is bounded by `lanes * slotBytes` however many
/// records are requested.
public final class FileExpertStore implements HostExpertStore {
    private static final long PAGE = 4096;

    private final ExpertBank[] banks;
    private final ExpertKeys keys;
    private final RecordSource source;
    private final HostArena arena;
    private final long slotBytes;
    private final int slotCount;
    private final AtomicInteger readsNow = new AtomicInteger();
    private final AtomicInteger readsHighWater = new AtomicInteger();
    private final LongAdder readNanos = new LongAdder();
    private final LongAdder opens = new LongAdder();

    /// A store that reads `file` directly.
    public FileExpertStore(GpuMemory memory, Path file, ExpertBank[] banks, int lanes) throws IOException {
        this(memory, new FileRecordSource(file, banks), banks, lanes);
    }

    /// A store over `source`, which it owns from here on: it is closed with the store, and also
    /// when this constructor fails.
    public FileExpertStore(GpuMemory memory, RecordSource source, ExpertBank[] banks, int lanes) {
        Objects.requireNonNull(memory, "memory");
        this.source = Objects.requireNonNull(source, "source");
        try {
            if (lanes < 1) throw new IllegalArgumentException("lanes must be positive");
            this.banks = banks.clone();
            this.keys = new ExpertKeys(this.banks);
            this.slotCount = lanes;
            this.slotBytes = ExpertFiles.alignUp(ExpertFiles.maxRecordBytes(this.banks), PAGE);
            this.arena = new HostArena(memory, Math.multiplyExact(this.slotBytes, (long) lanes));
        } catch (RuntimeException | Error failure) {
            try {
                source.close();
            } catch (RuntimeException closeFailure) {
                failure.addSuppressed(closeFailure);
            }
            throw failure;
        }
    }

    @Override
    public ExpertBank[] banks() {
        return this.banks.clone();
    }

    @Override
    public int lanes() {
        return this.slotCount;
    }

    @Override
    public HostRecord open(int bank, int expert, int lane) throws IOException, InterruptedException {
        this.keys.key(bank, expert);
        java.util.Objects.checkIndex(lane, this.slotCount);
        long size = this.banks[bank].recordBytes(expert);
        long address = this.arena.address() + this.slotBytes * lane;
        int now = this.readsNow.incrementAndGet();
        this.readsHighWater.accumulateAndGet(now, Math::max);
        long begin = System.nanoTime();
        try {
            this.source.read(
                    this.banks[bank], expert, MemorySegment.ofAddress(address).reinterpret(size));
        } finally {
            this.readsNow.decrementAndGet();
            this.readNanos.add(System.nanoTime() - begin);
        }
        this.opens.increment();
        return new StagedRecord(address, size);
    }

    @Override
    public long bytesRead() {
        return this.source.bytesRead();
    }

    @Override
    public long recordOpens() {
        return this.opens.sum();
    }

    /// Time spent in reads, summed over reads (parallel reads add up).
    public long readNanos() {
        return this.readNanos.sum();
    }

    /// The most reads that ran at once.
    public int concurrentReadsHighWater() {
        return this.readsHighWater.get();
    }

    /// Bytes of one staging slot.
    public long slotBytes() {
        return this.slotBytes;
    }

    /// Closes the source and frees the staging arena. No lane may be in use.
    @Override
    public void close() {
        try {
            this.source.close();
        } finally {
            this.arena.close();
        }
    }

    private record StagedRecord(long hostAddress, long byteSize) implements HostRecord {}
}
