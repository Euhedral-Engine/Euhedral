package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.host.HostFrames;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongSupplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// A bounded cache of routed experts in device memory, keyed by `(bank ordinal, expert)`.
///
/// The device holds exactly `slotCount * slotBytes` bytes for it: one slab, allocated once through
/// [GpuMemory#allocate] and sliced into equal slots, each large enough for any record. An expert is used
/// through a lease ([#acquire]), which pins its slot: a slot that is loading or leased is never evicted or
/// refilled, so the address a lease exposes is valid until it is closed. Everything else is replaced least
/// recently used first (a slot's recency is the last time a lease on it was closed).
///
/// Misses are loaded by the acquirer that found them: it reserves a slot (a free one, else the least recently
/// used unpinned resident), opens the record in the [HostExpertStore] and starts the [ExpertTransfer], all
/// outside the cache's lock, then waits for completion like everyone else. Concurrent requests for an expert
/// that is already loading join that transfer instead of starting another. An acquirer that is interrupted or
/// times out gives up only its own claim: a transfer that was started completes and leaves the expert
/// resident, whoever is left waiting for it.
///
/// A failed transfer fails every acquirer waiting for it with an [ExpertTransferException], returns the slot
/// to the free list and leaves nothing behind; the next request for the expert starts a new transfer.
///
/// The directory and the slot metadata are primitive arrays sized by the number of experts and slots; the
/// only objects are one small record per transfer in flight and one lease per open lease.
///
/// ## Closing
///
/// [#close()] rejects new requests and aborts the waiting ones, waits (up to the close timeout) for transfers
/// in flight to complete, then invalidates every lease still open (they report `isValid() == false`; a caller
/// that still has device work reading a slot must have ordered it before closing, as the slab is freed),
/// closes the transfer and the store, and frees the slab, each exactly once. If transfers are still in flight
/// at the timeout nothing is released and `close()` throws; calling it again retries.
///
/// A caller holding leases while acquiring another can exhaust the slots and wait for itself: a thread should
/// not hold more leases than there are slots less those other threads need, or should use a timeout.
///
/// ## Asynchronous requests
///
/// [#acquireAsync] and [#prefetch] never block the caller, and nothing on their path waits on a thread or in
/// a queue. A hit answers at once; a miss reserves a slot and the load proceeds as a chain of frames: the
/// record's open ([HostExpertStore#openAsync], whose read runs as frames of its own), the copy
/// ([ExpertTransfer#startAsync]), and the copy's completion as a frame, which ends by answering every request
/// that asked for the expert, each with its own lease. Each completion is what makes the next piece of work
/// available. A requester bounds what it has outstanding by the slots it may use; a request that finds every
/// slot pinned or loading broke that bound and fails at once, instead of queueing to hide it. A request
/// cannot be withdrawn: a requester that no longer wants its lease closes it. The blocking [#acquire] shares
/// all of the state and remains for callers that are not lattice workers.
public final class ExpertCache implements AutoCloseable {
    private static final Logger LOG = LoggerFactory.getLogger(ExpertCache.class);

    /// Slot size granularity: the alignment of a device allocation, so every slot starts aligned.
    public static final long SLOT_ALIGNMENT = 256;

    public static final long DEFAULT_CLOSE_TIMEOUT_NANOS = TimeUnit.SECONDS.toNanos(30);

    private static final int NONE = -1;
    private static final byte EMPTY = 0;
    private static final byte LOADING = 1;
    private static final byte RESIDENT = 2;
    private static final byte IN_USE = 3;

    private final HostExpertStore store;
    private final ExpertTransfer transfer;
    private final GpuMemory memory;
    private final ExpertBank[] banks;
    private final ExpertKeys keys;
    private final long deviceBase;
    private final int slotCount;
    private final long slotBytes;
    private final LongSupplier clock;
    private final long closeTimeoutNanos;
    private final ExpertCacheStats stats;
    private final HostFrames frames;

    private final ReentrantLock lock = new ReentrantLock();
    private final Condition slotAvailable = this.lock.newCondition();
    private final Condition loadFinished = this.lock.newCondition();

    // Everything below up to `closeMonitor` is guarded by `lock`.
    private final int[] directory;
    private final byte[] state;
    private final int[] slotKey;
    private final int[] pins;
    private final int[] generation;
    private final DeviceFence[] fence;
    private final Load[] load;
    private final boolean[] prefetched;
    private final int[] prev;
    private final int[] next;
    private int freeHead = NONE;
    private int lruHead = NONE;
    private int lruTail = NONE;
    private int loading;
    private int pinned;
    private int leasesOpen;
    private boolean leasesForced;
    private volatile boolean closing;

    private final Object closeMonitor = new Object();
    private boolean transferClosed;
    private boolean storeClosed;
    private boolean slabFreed;

    /// A cache of `slotCount` slots of `slotBytes` bytes over `store`, whose records `transfer` moves into a
    /// slab allocated from `memory`. The cache owns `store` and `transfer` once this constructor returns, and
    /// closes them; if it throws, the caller still owns them.
    public ExpertCache(
            HostExpertStore store, ExpertTransfer transfer, GpuMemory memory, int slotCount, long slotBytes) {
        this(store, transfer, memory, slotCount, slotBytes, System::nanoTime, DEFAULT_CLOSE_TIMEOUT_NANOS, null);
    }

    /// As above, whose asynchronous requests ([#acquireAsync], [#prefetch]) run their continuations as host
    /// work on `frames`.
    public ExpertCache(
            HostExpertStore store,
            ExpertTransfer transfer,
            GpuMemory memory,
            int slotCount,
            long slotBytes,
            HostFrames frames) {
        this(
                store,
                transfer,
                memory,
                slotCount,
                slotBytes,
                System::nanoTime,
                DEFAULT_CLOSE_TIMEOUT_NANOS,
                Objects.requireNonNull(frames, "frames"));
    }

    /// As above, with the nanosecond clock that times the statistics and the time [#close()] waits for
    /// transfers in flight.
    ///
    /// @throws IllegalArgumentException when `slotBytes` is smaller than the largest record of any bank, is not
    ///     a multiple of [#SLOT_ALIGNMENT], or the cache would need more than 2^63 bytes
    public ExpertCache(
            HostExpertStore store,
            ExpertTransfer transfer,
            GpuMemory memory,
            int slotCount,
            long slotBytes,
            LongSupplier clock,
            long closeTimeoutNanos) {
        this(store, transfer, memory, slotCount, slotBytes, clock, closeTimeoutNanos, null);
    }

    /// As above, with the host frames asynchronous requests run on (null: only the blocking API is
    /// available).
    public ExpertCache(
            HostExpertStore store,
            ExpertTransfer transfer,
            GpuMemory memory,
            int slotCount,
            long slotBytes,
            LongSupplier clock,
            long closeTimeoutNanos,
            HostFrames frames) {
        this.frames = frames;
        this.store = Objects.requireNonNull(store, "store");
        this.transfer = Objects.requireNonNull(transfer, "transfer");
        this.memory = Objects.requireNonNull(memory, "memory");
        this.clock = Objects.requireNonNull(clock, "clock");
        if (slotCount < 1) throw new IllegalArgumentException("slotCount must be positive");
        this.banks = store.banks();
        this.keys = new ExpertKeys(this.banks);
        long largest = ExpertFiles.maxRecordBytes(this.banks);
        if (slotBytes < largest)
            throw new IllegalArgumentException(
                    "slotBytes " + slotBytes + " is smaller than the largest record " + largest);
        if (slotBytes % SLOT_ALIGNMENT != 0)
            throw new IllegalArgumentException("slotBytes " + slotBytes + " is not a multiple of " + SLOT_ALIGNMENT);
        long capacity = Math.multiplyExact(slotBytes, (long) slotCount);
        this.slotCount = slotCount;
        this.slotBytes = slotBytes;
        this.closeTimeoutNanos = closeTimeoutNanos;
        this.stats = new ExpertCacheStats(slotCount, slotBytes);
        this.directory = new int[this.keys.keyCount()];
        Arrays.fill(this.directory, NONE);
        this.state = new byte[slotCount];
        this.slotKey = new int[slotCount];
        this.pins = new int[slotCount];
        this.generation = new int[slotCount];
        this.fence = new DeviceFence[slotCount];
        this.load = new Load[slotCount];
        this.prefetched = new boolean[slotCount];
        this.prev = new int[slotCount];
        this.next = new int[slotCount];
        for (int slot = 0; slot < slotCount; slot++) {
            this.slotKey[slot] = NONE;
            this.next[slot] = slot + 1 < slotCount ? slot + 1 : NONE;
            this.prev[slot] = NONE;
        }
        this.freeHead = 0;
        this.deviceBase = memory.allocate(capacity);
        if (this.deviceBase == 0) throw new IllegalStateException("the device slab allocation returned a null address");
    }

    /// The smallest slot size that holds every record of `banks`.
    public static long slotBytesFor(ExpertBank[] banks) {
        return ExpertFiles.alignUp(ExpertFiles.maxRecordBytes(banks), SLOT_ALIGNMENT);
    }

    // ---------------------------------------------------------------- acquiring

    /// Returns a lease on the expert, loading it when it is not resident and waiting for a slot when every
    /// slot is leased or loading.
    ///
    /// @throws IndexOutOfBoundsException when `bank` or `expert` is out of range
    /// @throws ExpertTransferException when loading the expert failed
    /// @throws IllegalStateException when the cache is closed, or closes while waiting
    /// @throws InterruptedException when interrupted while waiting; the cache is left as if the call was never made
    public ExpertLease acquire(int bank, int expert) throws InterruptedException {
        try {
            return acquire(bank, expert, HostExpertStore.NO_TIMEOUT);
        } catch (TimeoutException unreachable) {
            throw new IllegalStateException("an untimed acquire timed out", unreachable);
        }
    }

    /// As [#acquire(int, int)], giving up after `timeoutNanos` (negative: never) of waiting.
    ///
    /// @throws TimeoutException when the expert was not available in time; the cache is left as if the call was
    ///     never made, except that a transfer already started completes
    public ExpertLease acquire(int bank, int expert, long timeoutNanos) throws InterruptedException, TimeoutException {
        int key = this.keys.key(bank, expert);
        boolean timed = timeoutNanos >= 0;
        long deadline = timed ? System.nanoTime() + timeoutNanos : 0;
        while (true) {
            Claim claim = claim(bank, expert, key, deadline, timed);
            if (claim.lease != null) return claim.lease;
            long begin = this.clock.getAsLong();
            if (claim.loader) runLoader(claim.load, bank, expert, deadline, timed);
            ExpertLease lease = awaitLoad(claim.load, bank, expert, deadline, timed, begin);
            if (lease != null) return lease;
            // The acquirer that began this load gave up before it started a transfer: claim the expert again.
        }
    }

    /// Returns a lease on the expert only when it is resident, without waiting and without starting a load.
    ///
    /// @throws IllegalStateException when the cache is closed
    public Optional<ExpertLease> tryAcquire(int bank, int expert) {
        int key = this.keys.key(bank, expert);
        this.lock.lock();
        try {
            ensureOpen();
            int slot = this.directory[key];
            if (slot == NONE || this.state[slot] == LOADING) return Optional.empty();
            this.stats.hit();
            return Optional.of(pin(slot, bank, expert));
        } finally {
            this.lock.unlock();
        }
    }

    /// What one pass through the directory decided: a lease (hit), a load to run and wait for (miss, as its
    /// loader), or a load to wait for (joined).
    private static final class Claim {
        private final ExpertLease lease;
        private final Load load;
        private final boolean loader;

        private Claim(ExpertLease lease, Load load, boolean loader) {
            this.lease = lease;
            this.load = load;
            this.loader = loader;
        }
    }

    /// One transfer, from the reservation of its slot to its completion. For an asynchronous load it is also
    /// the listener of its record's opening and the completion of its copy, so a load allocates nothing else.
    private final class Load implements HostExpertStore.OpenListener, ExpertTransfer.Completion {
        private final int slot;
        private final int key;
        private final int generation;
        private final long bytes;
        private final int bank;
        private final int expert;
        /// Started by [#prefetch]: the load itself holds the slot's first pin, released when it finishes.
        private boolean prefetch;
        /// The slot's pending fence, taken for the transfer to wait behind; guarded by the cache lock.
        private DeviceFence fence;

        private boolean finished;
        private Throwable failure;
        private boolean retriable;
        private volatile long startNanos;

        /// Asynchronous requests waiting for this load, each holding one pin; guarded by the cache lock.
        private ExpertListener[] listeners = new ExpertListener[2];
        private int[] tags = new int[2];
        private int listenerCount;

        private Load(int slot, int key, int generation, long bytes, int bank, int expert) {
            this.slot = slot;
            this.key = key;
            this.generation = generation;
            this.bytes = bytes;
            this.bank = bank;
            this.expert = expert;
        }

        void addListener(ExpertListener listener, int tag) {
            if (this.listenerCount == this.listeners.length) {
                this.listeners = Arrays.copyOf(this.listeners, this.listenerCount * 2);
                this.tags = Arrays.copyOf(this.tags, this.listenerCount * 2);
            }
            this.listeners[this.listenerCount] = listener;
            this.tags[this.listenerCount++] = tag;
        }

        /// The record is staged (or could not be): start its copy.
        @Override
        public void opened(int tag, HostRecord record, Throwable failure) {
            if (failure != null) {
                finishLoad(this, failure, false);
                return;
            }
            try {
                if (ExpertCache.this.closing) {
                    closeQuietly(record);
                    finishLoad(this, new IllegalStateException("the expert cache is closed"), true);
                    return;
                }
                if (record.byteSize() != this.bytes) {
                    long size = record.byteSize();
                    closeQuietly(record);
                    finishLoad(
                            this,
                            new IllegalStateException("the store returned " + size + " bytes for a record of "
                                    + this.bytes + " (bank " + this.bank + ", expert " + this.expert + ")"),
                            false);
                    return;
                }
                this.startNanos = ExpertCache.this.clock.getAsLong();
                ExpertCache.this.transfer.startAsync(record, slotAddress(this.slot), this.fence, this);
            } catch (RuntimeException | Error thrown) {
                closeQuietly(record);
                finishLoad(this, thrown, false);
            }
        }

        /// The copy ended.
        @Override
        public void complete(Throwable failure) {
            finishLoad(this, failure, false);
        }
    }

    private Claim claim(int bank, int expert, int key, long deadline, boolean timed)
            throws InterruptedException, TimeoutException {
        this.lock.lockInterruptibly();
        long waitStart = -1;
        try {
            while (true) {
                ensureOpen();
                int slot = this.directory[key];
                if (slot != NONE) {
                    if (this.state[slot] == LOADING) {
                        this.pins[slot]++;
                        this.stats.coalesced();
                        return new Claim(null, this.load[slot], false);
                    }
                    this.stats.hit();
                    return new Claim(pin(slot, bank, expert), null, false);
                }
                if (this.freeHead != NONE || this.lruHead != NONE) {
                    int victim = this.freeHead != NONE ? this.freeHead : this.lruHead;
                    // Allocate before changing anything: a failure here leaves the cache untouched.
                    Load fresh = new Load(
                            victim,
                            key,
                            this.generation[victim] + 1,
                            this.banks[bank].recordBytes(expert),
                            bank,
                            expert);
                    reserve(victim, fresh);
                    this.stats.miss();
                    return new Claim(null, fresh, true);
                }
                if (waitStart < 0) waitStart = this.clock.getAsLong();
                if (!timed) this.slotAvailable.await();
                else {
                    long remaining = deadline - System.nanoTime();
                    if (remaining <= 0) throw new TimeoutException("every expert slot is in use");
                    this.slotAvailable.awaitNanos(remaining);
                }
            }
        } finally {
            if (waitStart >= 0) this.stats.waitedForSlot(this.clock.getAsLong() - waitStart);
            this.lock.unlock();
        }
    }

    /// Pins a resident slot for one more lease. Called with the lock held.
    private ExpertLease pin(int slot, int bank, int expert) {
        if (this.prefetched[slot]) {
            this.prefetched[slot] = false;
            this.stats.prefetchUsed();
        }
        if (this.state[slot] == RESIDENT) {
            lruRemove(slot);
            this.state[slot] = IN_USE;
            nonEvictable();
        }
        this.pins[slot]++;
        return newLease(slot, this.generation[slot], bank, expert);
    }

    private ExpertLease newLease(int slot, int generation, int bank, int expert) {
        this.leasesOpen++;
        return new ExpertLease(this, this.banks[bank], bank, expert, slot, generation, slotAddress(slot));
    }

    /// Takes the free or least recently used slot `victim` for `fresh`. Called with the lock held; changes
    /// only primitives, so it cannot fail halfway.
    private void reserve(int victim, Load fresh) {
        if (this.state[victim] == RESIDENT) {
            lruRemove(victim);
            this.directory[this.slotKey[victim]] = NONE;
            this.stats.eviction();
            if (this.prefetched[victim]) {
                this.prefetched[victim] = false;
                this.stats.prefetchWasted();
            }
        } else {
            this.freeHead = this.next[victim];
        }
        this.generation[victim] = fresh.generation;
        this.slotKey[victim] = fresh.key;
        this.directory[fresh.key] = victim;
        this.state[victim] = LOADING;
        this.pins[victim] = 1;
        this.load[victim] = fresh;
        fresh.fence = this.fence[victim];
        this.fence[victim] = null;
        this.loading++;
        nonEvictable();
    }

    private void nonEvictable() {
        this.pinned++;
        this.stats.slotsInUse(this.pinned);
    }

    // ---------------------------------------------------------------- loading

    /// Opens the record and starts its transfer, on the acquirer that reserved the slot and outside the lock.
    /// Returns once the transfer is under way (its completion may already have run); throws when it never
    /// started, after resetting the slot.
    private void runLoader(Load load, int bank, int expert, long deadline, boolean timed)
            throws InterruptedException, TimeoutException {
        HostRecord record = null;
        boolean closedMeanwhile = false;
        try {
            long remaining = timed ? Math.max(0, deadline - System.nanoTime()) : HostExpertStore.NO_TIMEOUT;
            record = this.store.open(bank, expert, remaining);
            if (this.closing) {
                closeQuietly(record);
                closedMeanwhile = true;
            } else {
                if (record.byteSize() != load.bytes)
                    throw new IllegalStateException("the store returned " + record.byteSize()
                            + " bytes for a record of " + load.bytes + " (bank " + bank + ", expert " + expert + ")");
                HostRecord handed = record;
                record = null;
                load.startNanos = this.clock.getAsLong();
                this.transfer.start(
                        handed, slotAddress(load.slot), load.fence, failure -> finishLoad(load, failure, false));
            }
        } catch (InterruptedException | TimeoutException abandoned) {
            closeQuietly(record);
            finishLoad(load, abandoned, true);
            throw abandoned;
        } catch (IOException | RuntimeException | Error failure) {
            closeQuietly(record);
            boolean applied = finishLoad(load, failure, false);
            if (failure instanceof Error error) throw error;
            // A completion that ran before start threw already decided the load.
            if (applied)
                throw new ExpertTransferException(
                        "expert " + expert + " of bank " + bank + " (" + this.banks[bank].name() + ") failed to load",
                        failure);
        }
        if (closedMeanwhile) {
            IllegalStateException closed = new IllegalStateException("the expert cache is closed");
            finishLoad(load, closed, true);
            throw closed;
        }
    }

    private static void closeQuietly(HostRecord record) {
        if (record == null) return;
        try {
            record.close();
        } catch (RuntimeException failure) {
            LOG.warn("closing an expert record failed", failure);
        }
    }

    /// Ends `load`. `abandoned` marks a load dropped before any transfer started, which its waiters retry.
    /// Returns false when the load had already ended. The asynchronous requests that waited for it are
    /// answered here, after the lock is released.
    private boolean finishLoad(Load load, Throwable failure, boolean abandoned) {
        DeviceFence consumed = null;
        ExpertLease[] granted = null;
        ExpertListener[] listeners = null;
        int[] tags = null;
        int answers = 0;
        Throwable answer = null;
        this.lock.lock();
        try {
            if (load.finished) {
                LOG.warn("a transfer reported its completion twice (slot {})", load.slot);
                return false;
            }
            load.finished = true;
            load.failure = failure;
            load.retriable = abandoned;
            int slot = load.slot;
            this.load[slot] = null;
            this.loading--;
            answers = load.listenerCount;
            listeners = load.listeners;
            tags = load.tags;
            if (failure == null) {
                consumed = load.fence;
                load.fence = null;
                if (load.prefetch) {
                    // The load's own pin goes; a prefetch nobody joined waits, unused, to be asked for.
                    this.pins[slot]--;
                    if (answers == 0) this.prefetched[slot] = true;
                }
                if (answers > 0) {
                    granted = new ExpertLease[answers];
                    for (int i = 0; i < answers; i++)
                        granted[i] = newLease(slot, load.generation, load.bank, load.expert);
                }
                if (this.pins[slot] > 0) this.state[slot] = IN_USE;
                else evictable(slot);
                this.stats.transferred(load.bytes, this.clock.getAsLong() - load.startNanos);
            } else {
                this.directory[load.key] = NONE;
                this.state[slot] = EMPTY;
                this.pins[slot] = 0;
                // The device work the fence orders still has to precede the next refill.
                this.fence[slot] = load.fence;
                load.fence = null;
                this.slotKey[slot] = NONE;
                this.next[slot] = this.freeHead;
                this.freeHead = slot;
                this.pinned--;
                if (abandoned) this.stats.abandonedLoad();
                else this.stats.failedTransfer();
                this.slotAvailable.signalAll();
                if (answers > 0)
                    answer = abandoned
                            ? new IllegalStateException("the expert cache is closed")
                            : new ExpertTransferException(
                                    "expert " + load.expert + " of bank " + load.bank + " ("
                                            + this.banks[load.bank].name() + ") failed to load",
                                    failure);
            }
            this.loadFinished.signalAll();
            return true;
        } finally {
            this.lock.unlock();
            releaseFence(consumed);
            for (int i = 0; i < answers; i++)
                notifyListener(listeners[i], tags[i], granted == null ? null : granted[i], answer);
        }
    }

    private static void notifyListener(ExpertListener listener, int tag, ExpertLease lease, Throwable failure) {
        try {
            listener.ready(tag, lease, failure);
        } catch (RuntimeException | Error listenerFailure) {
            // The request's owner failed to take the lease: nobody else can close it.
            if (lease != null) lease.close();
            LOG.error("an expert listener failed", listenerFailure);
        }
    }

    // ---------------------------------------------------------------- asynchronous requests

    /// Asks for the expert without blocking: `listener` is called exactly once with a lease (which the
    /// requester then owns and must close) or the reason there is none. See the class documentation.
    ///
    /// A hit calls the listener on the calling thread before this returns; anything else, later on a worker.
    /// The listener must not block.
    ///
    /// @throws IllegalStateException when the cache is closed, was built without host frames, or more requests wait
    ///     for a slot than the cache has slots; the listener is not called
    /// @throws IndexOutOfBoundsException when `bank` or `expert` is out of range
    public void acquireAsync(int bank, int expert, ExpertListener listener, int tag) {
        Objects.requireNonNull(listener, "listener");
        request(bank, expert, listener, tag);
    }

    private void request(int bank, int expert, ExpertListener listener, int tag) {
        if (this.frames == null) throw new IllegalStateException("the cache was built without host frames");
        int key = this.keys.key(bank, expert);
        ExpertLease lease = null;
        Load start = null;
        this.lock.lock();
        try {
            ensureOpen();
            int slot = this.directory[key];
            if (slot != NONE) {
                if (this.state[slot] == LOADING) {
                    Load joined = this.load[slot];
                    this.pins[slot]++;
                    this.stats.coalesced();
                    if (joined.prefetch) {
                        joined.prefetch = false;
                        this.stats.prefetchUsed();
                    }
                    joined.addListener(listener, tag);
                    return;
                }
                this.stats.hit();
                lease = pin(slot, bank, expert);
            } else if (this.freeHead != NONE || this.lruHead != NONE) {
                int victim = this.freeHead != NONE ? this.freeHead : this.lruHead;
                Load fresh = new Load(
                        victim, key, this.generation[victim] + 1, this.banks[bank].recordBytes(expert), bank, expert);
                fresh.addListener(listener, tag);
                reserve(victim, fresh);
                this.stats.miss();
                start = fresh;
            } else {
                // The requester bounds what it has outstanding by the slots it may use: a request that finds every slot
                // pinned or loading broke that bound, and nothing queues to hide it.
                throw new IllegalStateException("every expert slot is in use or loading: " + this.pinned + " of "
                        + this.slotCount + " (bank " + bank + ", expert " + expert + ")");
            }
        } finally {
            this.lock.unlock();
        }
        if (lease != null) notifyListener(listener, tag, lease, null);
        else startLoad(start);
    }

    /// Starts the load of the expert if it is neither resident nor loading and a slot is free or evictable
    /// right now. Once the load ends it holds no lease and no pin: the expert is an ordinary evictable
    /// resident, counted as a used prefetch if a request finds it before it is evicted. Never waits.
    ///
    /// @return whether a load was started
    /// @throws IllegalStateException when the cache is closed or was built without host frames
    public boolean prefetch(int bank, int expert) {
        if (this.frames == null) throw new IllegalStateException("the cache was built without host frames");
        int key = this.keys.key(bank, expert);
        Load fresh;
        this.lock.lock();
        try {
            ensureOpen();
            if (this.directory[key] != NONE) return false;
            if (this.freeHead == NONE && this.lruHead == NONE) return false;
            int victim = this.freeHead != NONE ? this.freeHead : this.lruHead;
            fresh = new Load(
                    victim, key, this.generation[victim] + 1, this.banks[bank].recordBytes(expert), bank, expert);
            fresh.prefetch = true;
            reserve(victim, fresh);
            this.stats.prefetchStarted();
        } finally {
            this.lock.unlock();
        }
        startLoad(fresh);
        return true;
    }

    /// Begins a reserved load: its record is opened without blocking, and the open's completion starts the
    /// copy.
    private void startLoad(Load load) {
        try {
            this.store.openAsync(load.bank, load.expert, this.frames, load, 0);
        } catch (RuntimeException | Error failure) {
            finishLoad(load, failure, false);
        }
    }

    /// Waits for `load`, whose slot is pinned for this acquirer, and returns the lease; null when the load
    /// was abandoned and the request should start over.
    private ExpertLease awaitLoad(Load load, int bank, int expert, long deadline, boolean timed, long begin)
            throws InterruptedException, TimeoutException {
        this.lock.lock();
        try {
            while (!load.finished) {
                if (this.closing) {
                    unpinWaiter(load);
                    throw new IllegalStateException("the expert cache is closed");
                }
                try {
                    if (!timed) this.loadFinished.await();
                    else {
                        long remaining = deadline - System.nanoTime();
                        if (remaining <= 0) {
                            unpinWaiter(load);
                            throw new TimeoutException("expert " + expert + " of bank " + bank + " is still loading");
                        }
                        this.loadFinished.awaitNanos(remaining);
                    }
                } catch (InterruptedException interrupted) {
                    unpinWaiter(load);
                    throw interrupted;
                }
            }
            if (load.failure != null) {
                if (load.retriable) return null;
                throw new ExpertTransferException(
                        "expert " + expert + " of bank " + bank + " (" + this.banks[bank].name() + ") failed to load",
                        load.failure);
            }
            return newLease(load.slot, load.generation, bank, expert);
        } finally {
            this.stats.waitedForLoad(this.clock.getAsLong() - begin);
            this.lock.unlock();
        }
    }

    /// Drops one waiter's claim on `load`. Called with the lock held.
    private void unpinWaiter(Load load) {
        if (!load.finished) this.pins[load.slot]--;
        else if (load.failure == null) unpin(load.slot);
    }

    // ---------------------------------------------------------------- releasing

    boolean isCurrent(int slot, int generation) {
        this.lock.lock();
        try {
            return this.generation[slot] == generation;
        } finally {
            this.lock.unlock();
        }
    }

    void release(int slot, int generation, DeviceFence fence) {
        List<DeviceFence> unused = new ArrayList<>(2);
        this.lock.lock();
        try {
            if (this.generation[slot] != generation) {
                // The cache closed with this lease open: the slab is gone, so there is nothing to order.
                if (fence != null) unused.add(fence);
            } else {
                this.leasesOpen--;
                if (fence != null) this.fence[slot] = DeviceFences.merge(this.fence[slot], fence, unused);
                unpin(slot);
            }
        } finally {
            this.lock.unlock();
        }
        for (DeviceFence redundant : unused) releaseFence(redundant);
    }

    /// Removes one pin from a slot that has leases. Called with the lock held.
    private void unpin(int slot) {
        if (--this.pins[slot] > 0) return;
        evictable(slot);
    }

    /// A slot with no claims becomes the most recently used evictable resident. Called with the lock held.
    private void evictable(int slot) {
        this.state[slot] = RESIDENT;
        lruAppend(slot);
        this.pinned--;
        this.slotAvailable.signalAll();
    }

    private static void releaseFence(DeviceFence fence) {
        if (fence == null) return;
        try {
            fence.release();
        } catch (RuntimeException failure) {
            LOG.warn("releasing a device fence failed", failure);
        }
    }

    // ---------------------------------------------------------------- recency list

    private void lruAppend(int slot) {
        this.prev[slot] = this.lruTail;
        this.next[slot] = NONE;
        if (this.lruTail == NONE) this.lruHead = slot;
        else this.next[this.lruTail] = slot;
        this.lruTail = slot;
    }

    private void lruRemove(int slot) {
        int before = this.prev[slot];
        int after = this.next[slot];
        if (before == NONE) this.lruHead = after;
        else this.next[before] = after;
        if (after == NONE) this.lruTail = before;
        else this.prev[after] = before;
        this.prev[slot] = NONE;
        this.next[slot] = NONE;
    }

    private long slotAddress(int slot) {
        return this.deviceBase + (long) slot * this.slotBytes;
    }

    private void ensureOpen() {
        if (this.closing) throw new IllegalStateException("the expert cache is closed");
    }

    // ---------------------------------------------------------------- inspection

    public ExpertCacheStats stats() {
        return this.stats;
    }

    public int slotCount() {
        return this.slotCount;
    }

    public long slotBytes() {
        return this.slotBytes;
    }

    /// Device bytes the cache holds: `slotCount * slotBytes`, in one allocation.
    public long capacityBytes() {
        return (long) this.slotCount * this.slotBytes;
    }

    /// The device address of the slab, whose slot `i` starts at `slabAddress() + i * slotBytes()`.
    public long slabAddress() {
        return this.deviceBase;
    }

    public ExpertBank[] banks() {
        return this.banks.clone();
    }

    public boolean isClosed() {
        return this.closing;
    }

    /// Leases open now.
    public int openLeaseCount() {
        this.lock.lock();
        try {
            return this.leasesOpen;
        } finally {
            this.lock.unlock();
        }
    }

    /// Whether the expert's record is in a slot now (resident, leased or being refreshed by no one). A
    /// loading expert is not resident.
    public boolean isResident(int bank, int expert) {
        int key = this.keys.key(bank, expert);
        this.lock.lock();
        try {
            int slot = this.directory[key];
            return slot != NONE && this.state[slot] != LOADING;
        } finally {
            this.lock.unlock();
        }
    }

    /// Slots holding a loaded expert that no lease pins, which a miss may evict.
    public int evictableSlots() {
        this.lock.lock();
        try {
            int count = 0;
            for (int slot = this.lruHead; slot != NONE; slot = this.next[slot]) count++;
            return count;
        } finally {
            this.lock.unlock();
        }
    }

    /// [#checkInvariants()], and also that no request is in progress: nothing is loading, and the pins are
    /// exactly the open leases.
    ///
    /// @throws IllegalStateException naming the first violation
    public void checkQuiescent() {
        checkInvariants();
        this.lock.lock();
        try {
            long pinSum = 0;
            for (int slot = 0; slot < this.slotCount; slot++) pinSum += this.pins[slot];
            if (this.loading != 0) throw new IllegalStateException(this.loading + " transfers are still loading");
            if (!this.leasesForced && pinSum != this.leasesOpen)
                throw new IllegalStateException("pins " + pinSum + " differ from open leases " + this.leasesOpen);
        } finally {
            this.lock.unlock();
        }
    }

    /// Verifies the cache's bookkeeping, which holds at any moment, and throws [IllegalStateException] naming
    /// the first inconsistency: the directory and the slots agree, every slot is in exactly the list its
    /// state says, no lease is without a pin.
    public void checkInvariants() {
        this.lock.lock();
        try {
            int free = 0;
            for (int slot = this.freeHead; slot != NONE; slot = this.next[slot]) {
                if (++free > this.slotCount) throw new IllegalStateException("free list loops");
                if (this.state[slot] != EMPTY)
                    throw new IllegalStateException("slot " + slot + " is free but not empty");
            }
            int evictable = 0;
            int before = NONE;
            for (int slot = this.lruHead; slot != NONE; slot = this.next[slot]) {
                if (++evictable > this.slotCount) throw new IllegalStateException("recency list loops");
                if (this.state[slot] != RESIDENT || this.pins[slot] != 0)
                    throw new IllegalStateException("slot " + slot + " is evictable but pinned or not resident");
                if (this.prev[slot] != before)
                    throw new IllegalStateException("slot " + slot + " has a broken back link");
                before = slot;
            }
            if (before != this.lruTail) throw new IllegalStateException("the recency tail is wrong");
            int empty = 0;
            int resident = 0;
            int inUse = 0;
            int loadingSlots = 0;
            long pinSum = 0;
            for (int slot = 0; slot < this.slotCount; slot++) {
                pinSum += this.pins[slot];
                switch (this.state[slot]) {
                    case EMPTY -> {
                        empty++;
                        if (this.pins[slot] != 0 || this.slotKey[slot] != NONE || this.load[slot] != null)
                            throw new IllegalStateException("empty slot " + slot + " holds state");
                    }
                    case LOADING -> {
                        loadingSlots++;
                        if (this.load[slot] == null || this.load[slot].finished || this.pins[slot] < 0)
                            throw new IllegalStateException("loading slot " + slot + " has no live load");
                        if (this.directory[this.slotKey[slot]] != slot)
                            throw new IllegalStateException("loading slot " + slot + " is not in the directory");
                    }
                    case RESIDENT -> {
                        resident++;
                        if (this.directory[this.slotKey[slot]] != slot)
                            throw new IllegalStateException("resident slot " + slot + " is not in the directory");
                    }
                    case IN_USE -> {
                        inUse++;
                        if (this.pins[slot] <= 0)
                            throw new IllegalStateException("leased slot " + slot + " has no pins");
                        if (this.directory[this.slotKey[slot]] != slot)
                            throw new IllegalStateException("leased slot " + slot + " is not in the directory");
                    }
                    default -> throw new IllegalStateException("slot " + slot + " has an unknown state");
                }
            }
            if (empty != free)
                throw new IllegalStateException("free list holds " + free + " of " + empty + " empty slots");
            if (resident != evictable)
                throw new IllegalStateException("recency list holds " + evictable + " of " + resident + " residents");
            if (loadingSlots != this.loading) throw new IllegalStateException("loading count is wrong");
            if (inUse + loadingSlots != this.pinned) throw new IllegalStateException("pinned count is wrong");
            int mapped = 0;
            for (int key = 0; key < this.directory.length; key++) {
                int slot = this.directory[key];
                if (slot == NONE) continue;
                mapped++;
                if (this.slotKey[slot] != key || this.state[slot] == EMPTY)
                    throw new IllegalStateException("directory entry " + key + " points at slot " + slot);
            }
            if (mapped != this.slotCount - empty) throw new IllegalStateException("directory size is wrong");
            // Acquirers whose transfer just finished hold a pin before their lease exists, never the reverse.
            if (!this.leasesForced && pinSum < this.leasesOpen)
                throw new IllegalStateException("pins " + pinSum + " are fewer than open leases " + this.leasesOpen);
        } finally {
            this.lock.unlock();
        }
    }

    // ---------------------------------------------------------------- closing

    /// Closes the cache as described in the class documentation. Safe to call more than once and from any
    /// thread.
    ///
    /// @throws IllegalStateException when transfers were still in flight at the close timeout (nothing was
    ///     released; call again to retry) or releasing a resource failed
    @Override
    public void close() {
        List<DeviceFence> pending = new ArrayList<>();
        this.lock.lock();
        try {
            if (!this.closing) {
                this.closing = true;
                this.slotAvailable.signalAll();
                this.loadFinished.signalAll();
            }
            if (!this.leasesForced) {
                long remaining = this.closeTimeoutNanos;
                boolean interrupted = false;
                while (this.loading > 0 && remaining > 0) {
                    try {
                        remaining = this.loadFinished.awaitNanos(remaining);
                    } catch (InterruptedException interrupt) {
                        interrupted = true;
                        break;
                    }
                }
                if (interrupted) Thread.currentThread().interrupt();
                if (this.loading > 0)
                    throw new IllegalStateException(
                            this.loading
                                    + " expert transfers are still in flight; the cache keeps its device slab and stores, close it again to retry");
                this.stats.forcedLeases(this.leasesOpen);
                this.leasesOpen = 0;
                this.leasesForced = true;
                for (int slot = 0; slot < this.slotCount; slot++) {
                    // A bumped generation makes every open lease invalid and every later close of one a no-op.
                    this.generation[slot]++;
                    if (this.fence[slot] != null) pending.add(this.fence[slot]);
                    this.fence[slot] = null;
                }
            }
        } finally {
            this.lock.unlock();
        }
        for (DeviceFence fence : pending) releaseFence(fence);
        releaseResources();
    }

    /// Closes the transfer, then the store, then frees the slab, each at most once even when one of them
    /// fails: only a failing transfer close stops the sequence, as the slab and the host records may still be
    /// in use by a copy.
    private void releaseResources() {
        synchronized (this.closeMonitor) {
            if (!this.transferClosed) {
                this.transfer.close();
                this.transferClosed = true;
            }
            Throwable failure = null;
            if (!this.storeClosed) {
                this.storeClosed = true;
                try {
                    this.store.close();
                } catch (RuntimeException | Error closeFailure) {
                    failure = closeFailure;
                }
            }
            if (!this.slabFreed) {
                this.slabFreed = true;
                try {
                    this.memory.free(this.deviceBase);
                } catch (RuntimeException | Error freeFailure) {
                    if (failure == null) failure = freeFailure;
                    else failure.addSuppressed(freeFailure);
                }
            }
            if (failure != null) throw new IllegalStateException("releasing the expert cache failed", failure);
        }
    }
}
