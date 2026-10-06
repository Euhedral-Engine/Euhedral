package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.host.HostFrames;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/// Cold storage: records stay in the artifact file, and a bounded pool of pinned staging slots holds one
/// between the file read and the end of its device copy.
///
/// All slots are carved from one pinned arena, each as large as the biggest record rounded up to a page.
/// [#open] waits (interruptibly, with an optional timeout) for a free slot, then has the [RecordSource] fill
/// it: the one host copy of the record is from the page cache into pinned memory, with no heap array in
/// between. The slot returns to the pool when the record is closed. Memory is bounded by `stagingSlots *
/// slotBytes` however many records are requested.
///
/// [#openAsync] never blocks its caller. A request that finds no staging slot parks as a continuation, and
/// the record that closes next hands its slot to the oldest one; the read itself is dispatched as a host
/// frame, so simultaneous requests read on different workers, and at most `stagingSlots` reads run at once:
/// the staging pool is the bound on concurrent reads.
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
    private final AtomicInteger readsNow = new AtomicInteger();
    private final AtomicInteger readsHighWater = new AtomicInteger();
    private final LongAdder readNanos = new LongAdder();
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

    @Override
    public void openAsync(int bank, int expert, HostFrames frames, OpenListener listener, int tag) {
        this.keys.key(bank, expert);
        AsyncOpen open = new AsyncOpen(bank, expert, frames, listener, tag);
        Throwable refused = null;
        int slot = -1;
        this.lock.lock();
        try {
            if (this.closed) refused = new IllegalStateException("the expert store is closed");
            else if (this.freeCount > 0) slot = this.freeSlots[--this.freeCount];
            else {
                // The requester bounds its outstanding opens by the staging slots; one past that fails, it never
                // queues.
                this.blockedOpens.increment();
                refused = new IllegalStateException(
                        "all " + this.slotCount + " staging slots hold records: more opens than slots are outstanding");
            }
        } finally {
            this.lock.unlock();
        }
        if (refused != null) listener.opened(tag, null, refused);
        else open.dispatch(slot);
    }

    /// One asynchronous open: a staging slot taken at once, then read frames, then the listener.
    private final class AsyncOpen implements HostFrames.Task {
        private final int bank;
        private final int expert;
        private final HostFrames frames;
        private final OpenListener listener;
        private final int tag;
        private int slot = -1;

        AsyncOpen(int bank, int expert, HostFrames frames, OpenListener listener, int tag) {
            this.bank = bank;
            this.expert = expert;
            this.frames = frames;
            this.listener = listener;
            this.tag = tag;
        }

        /// Staging slot `slot` is ours: read the record as a frame of its own.
        void dispatch(int slot) {
            this.slot = slot;
            try {
                this.frames.run(this);
            } catch (RuntimeException | Error rejected) {
                fail(rejected);
            }
        }

        @Override
        public void run() {
            ExpertBank source = FileExpertStore.this.banks[this.bank];
            long size = source.recordBytes(this.expert);
            long address = FileExpertStore.this.arena.address() + FileExpertStore.this.slotBytes * this.slot;
            int now = FileExpertStore.this.readsNow.incrementAndGet();
            FileExpertStore.this.readsHighWater.accumulateAndGet(now, Math::max);
            long begin = System.nanoTime();
            FileExpertStore.this.arena.borrow();
            try {
                FileExpertStore.this.source.read(
                        source, this.expert, MemorySegment.ofAddress(address).reinterpret(size));
            } catch (IOException | InterruptedException | RuntimeException | Error failure) {
                FileExpertStore.this.arena.giveBack();
                fail(failure);
                return;
            } finally {
                FileExpertStore.this.readsNow.decrementAndGet();
                FileExpertStore.this.readNanos.add(System.nanoTime() - begin);
            }
            FileExpertStore.this.opens.increment();
            this.listener.opened(this.tag, new StagedRecord(this.slot, address, size), null);
        }

        @Override
        public void failed(Throwable cause) {
            fail(cause);
        }

        private void fail(Throwable cause) {
            int held = this.slot;
            this.slot = -1;
            if (held >= 0) returnSlot(held);
            this.listener.opened(this.tag, null, cause);
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

    /// Time spent in positional reads, summed over reads (parallel reads add up).
    public long readNanos() {
        return this.readNanos.sum();
    }

    /// The most reads that ran at once.
    public int concurrentReadsHighWater() {
        return this.readsHighWater.get();
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
