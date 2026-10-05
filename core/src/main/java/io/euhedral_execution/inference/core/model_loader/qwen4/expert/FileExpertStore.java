package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/// Cold storage: records stay in the artifact file, and a bounded pool of pinned staging slots holds one between
/// the file read and the end of its device copy.
///
/// All slots are carved from one pinned arena, each as large as the biggest record rounded up to a page. [#open]
/// waits (interruptibly, with an optional timeout) for a free slot, then has the [RecordSource] fill it: the one
/// host copy of the record is from the page cache into pinned memory, with no heap array in between. The slot
/// returns to the pool when the record is closed. Memory is bounded by `stagingSlots * slotBytes` however many
/// records are requested.
public final class FileExpertStore implements HostExpertStore {
    private static final long PAGE = 4096;

    private final ExpertBank[] banks;
    private final ExpertKeys keys;
    private final RecordSource source;
    private final HostArena arena;
    private final long slotBytes;
    private final int slotCount;
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition slotFree = this.lock.newCondition();
    private final int[] freeSlots;
    private int freeCount;
    private boolean closed;
    private final LongAdder opens = new LongAdder();
    private final LongAdder blockedOpens = new LongAdder();

    /// A store that reads `file` directly.
    public FileExpertStore(GpuMemory memory, Path file, ExpertBank[] banks, int stagingSlots) throws IOException {
        this(memory, new FileRecordSource(file, banks), banks, stagingSlots);
    }

    /// A store over `source`, which it owns from here on: it is closed with the store, and also when this
    /// constructor fails.
    public FileExpertStore(GpuMemory memory, RecordSource source, ExpertBank[] banks, int stagingSlots) {
        Objects.requireNonNull(memory, "memory");
        this.source = Objects.requireNonNull(source, "source");
        try {
            if (stagingSlots < 1) throw new IllegalArgumentException("stagingSlots must be positive");
            this.banks = banks.clone();
            this.keys = new ExpertKeys(this.banks);
            this.slotCount = stagingSlots;
            this.slotBytes = ExpertFiles.alignUp(ExpertFiles.maxRecordBytes(this.banks), PAGE);
            this.arena = new HostArena(memory, Math.multiplyExact(this.slotBytes, (long) stagingSlots));
        } catch (RuntimeException | Error failure) {
            try {
                source.close();
            } catch (RuntimeException closeFailure) {
                failure.addSuppressed(closeFailure);
            }
            throw failure;
        }
        this.freeSlots = new int[stagingSlots];
        for (int slot = 0; slot < stagingSlots; slot++) this.freeSlots[slot] = stagingSlots - 1 - slot;
        this.freeCount = stagingSlots;
    }

    @Override
    public ExpertBank[] banks() {
        return this.banks.clone();
    }

    @Override
    public HostRecord open(int bank, int expert, long timeoutNanos)
            throws IOException, InterruptedException, TimeoutException {
        this.keys.key(bank, expert);
        int slot = takeSlot(timeoutNanos);
        boolean opened = false;
        try {
            long size = this.banks[bank].recordBytes(expert);
            long address = this.arena.address() + this.slotBytes * slot;
            this.arena.borrow();
            try {
                this.source.read(
                        this.banks[bank],
                        expert,
                        MemorySegment.ofAddress(address).reinterpret(size));
            } catch (IOException | InterruptedException | RuntimeException | Error failure) {
                this.arena.giveBack();
                throw failure;
            }
            this.opens.increment();
            opened = true;
            return new StagedRecord(slot, address, size);
        } finally {
            if (!opened) returnSlot(slot);
        }
    }

    private int takeSlot(long timeoutNanos) throws InterruptedException, TimeoutException {
        long remaining = timeoutNanos;
        this.lock.lockInterruptibly();
        try {
            boolean blocked = false;
            while (true) {
                if (this.closed) throw new IllegalStateException("the expert store is closed");
                if (this.freeCount > 0) return this.freeSlots[--this.freeCount];
                if (!blocked) {
                    blocked = true;
                    this.blockedOpens.increment();
                }
                if (timeoutNanos < 0) this.slotFree.await();
                else if (remaining <= 0) throw new TimeoutException("no staging slot became free");
                else remaining = this.slotFree.awaitNanos(remaining);
            }
        } finally {
            this.lock.unlock();
        }
    }

    private void returnSlot(int slot) {
        this.lock.lock();
        try {
            this.freeSlots[this.freeCount++] = slot;
            this.slotFree.signal();
        } finally {
            this.lock.unlock();
        }
    }

    @Override
    public long bytesRead() {
        return this.source.bytesRead();
    }

    @Override
    public long recordOpens() {
        return this.opens.sum();
    }

    /// Opens that found every staging slot taken and had to wait.
    public long blockedOpens() {
        return this.blockedOpens.sum();
    }

    /// Staging slots not holding a record.
    public int freeSlots() {
        this.lock.lock();
        try {
            return this.freeCount;
        } finally {
            this.lock.unlock();
        }
    }

    public int stagingSlots() {
        return this.slotCount;
    }

    /// Bytes of one staging slot.
    public long slotBytes() {
        return this.slotBytes;
    }

    /// Rejects further opens, wakes the waiting ones, and frees the staging arena once every open record is
    /// closed.
    @Override
    public void close() {
        this.lock.lock();
        try {
            if (this.closed) return;
            this.closed = true;
            this.slotFree.signalAll();
        } finally {
            this.lock.unlock();
        }
        try {
            this.source.close();
        } finally {
            this.arena.close();
        }
    }

    private final class StagedRecord implements HostRecord {
        private final int slot;
        private final long address;
        private final long byteSize;
        private final AtomicBoolean closed = new AtomicBoolean();

        private StagedRecord(int slot, long address, long byteSize) {
            this.slot = slot;
            this.address = address;
            this.byteSize = byteSize;
        }

        @Override
        public long hostAddress() {
            return this.address;
        }

        @Override
        public long byteSize() {
            return this.byteSize;
        }

        @Override
        public void close() {
            if (!this.closed.compareAndSet(false, true)) return;
            returnSlot(this.slot);
            FileExpertStore.this.arena.giveBack();
        }
    }
}
