package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// One partition of the device cache of routed experts: a contiguous range of slots of the slab and
/// the directory of the experts that hash to it.
///
/// A shard has one owner, and every method but [#isCurrent] and the inspection methods is called
/// only by it. The owner is a serial source
/// ([io.euhedral_execution.inference.core.scheduling.graph.SerialSource]): the lattice runs its
/// `request` and `pull` one thread at a time, so the shard's state is plain fields with no lock, no
/// atomic and no condition. A lease closed on any thread reports to its [ExpertLease.Owner], which
/// posts the release to the owner's mailbox; nothing here waits or is waited for.
///
/// An expert is used through a lease, which pins its slot: a slot that is loading or leased is
/// never evicted or refilled, so the address a lease exposes is valid until it is closed.
/// Everything else is replaced least recently used first (a slot's recency is the last time a lease
/// on it was closed).
///
/// A miss is a [Load]: [#claim] reserves a slot (a free one, else the least recently used unpinned
/// resident) and the owner has the record read and its copy submitted on a copy stream. The copy
/// records a marker behind itself, and [Load#submitted] answers with a lease that carries that
/// marker: device work that reads the slot waits for the marker on its own stream, and no thread
/// waits for the bytes. The slot stays unevictable until [Load#retired], even when the lease was
/// closed already behind the kernels that read it. A failed load leaves nothing behind; the next
/// request starts a new one.
///
/// ## Closing
///
/// [#close] invalidates every lease still open (they report `isValid() == false`; a caller that
/// still has device work reading a slot must have ordered it before closing, as the slab is freed)
/// and releases the fences still pending.
public final class ExpertCacheShard implements ExpertLease.Owner {
    private static final Logger LOG = LoggerFactory.getLogger(ExpertCacheShard.class);

    private static final int NONE = -1;
    private static final byte EMPTY = 0;
    private static final byte LOADING = 1;
    private static final byte RESIDENT = 2;
    private static final byte IN_USE = 3;

    private final int index;
    private final ExpertBank[] banks;
    private final ExpertKeys keys;
    private final long deviceBase;
    private final int slotCount;
    private final long slotBytes;
    private final long[] readyMarker;
    private final ExpertCacheStats stats;

    private final int[] directory;
    private final byte[] state;
    private final int[] slotKey;
    private final int[] pins;
    private final int[] generation;
    private final DeviceFence[] fence;
    private final Load[] load;
    private final int[] prev;
    private final int[] next;
    private int freeHead = NONE;
    private int lruHead = NONE;
    private int lruTail = NONE;
    private int loading;
    private int pinned;
    private int leasesOpen;
    private boolean closed;
    private ExpertLease.Owner owner = this;

    /// A shard of `slotCount` slots of `slotBytes` bytes starting at device address `deviceBase`,
    /// each with the marker its copy records (`readyMarkers[slot]`).
    ExpertCacheShard(
            int index,
            ExpertBank[] banks,
            long deviceBase,
            int slotCount,
            long slotBytes,
            long[] readyMarkers,
            ExpertCacheStats stats) {
        if (slotCount < 1) throw new IllegalArgumentException("a shard needs at least one slot");
        this.index = index;
        this.banks = banks;
        this.keys = new ExpertKeys(banks);
        this.deviceBase = deviceBase;
        this.slotCount = slotCount;
        this.slotBytes = slotBytes;
        this.readyMarker = readyMarkers;
        this.stats = stats;
        this.directory = new int[this.keys.keyCount()];
        java.util.Arrays.fill(this.directory, NONE);
        this.state = new byte[slotCount];
        this.slotKey = new int[slotCount];
        this.pins = new int[slotCount];
        this.generation = new int[slotCount];
        this.fence = new DeviceFence[slotCount];
        this.load = new Load[slotCount];
        this.prev = new int[slotCount];
        this.next = new int[slotCount];
        for (int slot = 0; slot < slotCount; slot++) {
            this.slotKey[slot] = NONE;
            this.next[slot] = slot + 1 < slotCount ? slot + 1 : NONE;
            this.prev[slot] = NONE;
        }
        this.freeHead = 0;
    }

    /// Who the leases this shard hands out report to (itself by default: a release is then applied
    /// inline).
    public void owner(ExpertLease.Owner owner) {
        this.owner = java.util.Objects.requireNonNull(owner, "owner");
    }

    public int index() {
        return this.index;
    }

    // ---------------------------------------------------------------- claiming

    /// The answer to [#claim]: a lease (a hit, or an expert whose copy was already submitted), a
    /// load for the owner to run, or neither when the expert is absent and there is no room (or no
    /// loading was allowed).
    public static final class Ticket {
        private ExpertLease lease;
        private Load load;
        private boolean waiting;

        public ExpertLease lease() {
            return this.lease;
        }

        public Load load() {
            return this.load;
        }

        /// Neither a lease nor a load: ask again after a slot frees.
        public boolean waiting() {
            return this.waiting;
        }
    }

    /// Asks for the expert. A resident expert is a lease at once. An absent one is a load when
    /// `mayLoad` and a slot is free or evictable; otherwise the ticket says to wait. Nothing is
    /// changed when the ticket waits.
    ///
    /// @throws IllegalStateException when the shard is closed, or the expert is being loaded by a
    ///     request whose copy was not submitted yet
    public void claim(int bank, int expert, boolean mayLoad, Ticket ticket) {
        ticket.lease = null;
        ticket.load = null;
        ticket.waiting = false;
        ensureOpen();
        int key = this.keys.key(bank, expert);
        int slot = this.directory[key];
        if (slot != NONE) {
            if (this.state[slot] == LOADING) {
                Load joined = this.load[slot];
                if (!joined.streamed)
                    throw new IllegalStateException(
                            "expert " + expert + " of bank " + bank + " is loading and its copy was not submitted");
                this.pins[slot]++;
                this.stats.coalesced();
                ticket.lease = newLease(slot, this.generation[slot], bank, expert, this.readyMarker[slot]);
                return;
            }
            this.stats.hit();
            ticket.lease = pin(slot, bank, expert);
            return;
        }
        if (!mayLoad || (this.freeHead == NONE && this.lruHead == NONE)) {
            ticket.waiting = true;
            return;
        }
        int victim = this.freeHead != NONE ? this.freeHead : this.lruHead;
        Load fresh =
                new Load(victim, key, this.generation[victim] + 1, this.banks[bank].recordBytes(expert), bank, expert);
        reserve(victim, fresh);
        this.stats.miss();
        ticket.load = fresh;
    }

    /// Whether a miss could reserve a slot now.
    public boolean hasRoom() {
        return this.freeHead != NONE || this.lruHead != NONE;
    }

    /// One transfer, from the reservation of its slot to its end.
    public final class Load {
        private final int slot;
        private final int key;
        private final int generation;
        private final long bytes;
        private final int bank;
        private final int expert;
        /// The slot's pending fence, taken for the copy to wait behind.
        private DeviceFence fence;
        private boolean streamed;
        private boolean finished;
        private long startNanos;

        private Load(int slot, int key, int generation, long bytes, int bank, int expert) {
            this.slot = slot;
            this.key = key;
            this.generation = generation;
            this.bytes = bytes;
            this.bank = bank;
            this.expert = expert;
        }

        public int bank() {
            return this.bank;
        }

        public int expert() {
            return this.expert;
        }

        /// Bytes of the record.
        public long bytes() {
            return this.bytes;
        }

        /// Where the copy writes.
        public long deviceAddress() {
            return slotAddress(this.slot);
        }

        /// The marker the copy records behind itself.
        public long readyMarker() {
            return ExpertCacheShard.this.readyMarker[this.slot];
        }

        /// What the copy must order itself behind on the device (the kernels that read the slot's
        /// previous expert), or null. The shard releases it once the load ended.
        public DeviceFence fence() {
            return this.fence;
        }

        /// The copy was submitted: the load's own claim on the slot becomes the returned lease,
        /// which carries the copy's marker.
        public ExpertLease submitted() {
            this.streamed = true;
            this.startNanos = System.nanoTime();
            return newLease(this.slot, this.generation, this.bank, this.expert, readyMarker());
        }

        /// The copy retired: the expert is resident once nobody holds it. `failure` is the device's
        /// report, or null.
        public void retired(Throwable failure) {
            if (failure == null) finishLoad(this, null);
            else finishStreamed(this, failure);
        }

        /// Ends a load whose copy was not submitted (the record could not be read, or the copy was
        /// not armed): the slot returns to the cache.
        public void failed(Throwable failure) {
            finishLoad(this, failure);
        }
    }

    /// Pins a resident slot for one more lease.
    private ExpertLease pin(int slot, int bank, int expert) {
        if (this.state[slot] == RESIDENT) {
            lruRemove(slot);
            this.state[slot] = IN_USE;
            nonEvictable();
        }
        this.pins[slot]++;
        return newLease(slot, this.generation[slot], bank, expert, 0);
    }

    private ExpertLease newLease(int slot, int generation, int bank, int expert, long readyMarker) {
        this.leasesOpen++;
        return new ExpertLease(
                this.owner, this.banks[bank], bank, expert, slot, generation, slotAddress(slot), readyMarker);
    }

    /// Takes the free or least recently used slot `victim` for `fresh`.
    private void reserve(int victim, Load fresh) {
        if (this.state[victim] == RESIDENT) {
            lruRemove(victim);
            if (this.directory[this.slotKey[victim]] == victim) this.directory[this.slotKey[victim]] = NONE;
            this.stats.eviction();
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

    /// Ends `load` that never submitted, or whose copy retired well.
    private void finishLoad(Load load, Throwable failure) {
        if (load.finished) {
            LOG.warn("a load ended twice (slot {})", load.slot);
            return;
        }
        load.finished = true;
        int slot = load.slot;
        this.load[slot] = null;
        this.loading--;
        DeviceFence consumed = load.fence;
        load.fence = null;
        if (failure == null) {
            if (this.pins[slot] > 0) this.state[slot] = IN_USE;
            else evictable(slot);
            this.stats.transferred(load.bytes, System.nanoTime() - load.startNanos);
        } else {
            this.directory[load.key] = NONE;
            this.state[slot] = EMPTY;
            this.pins[slot] = 0;
            // The device work the fence orders still has to precede the next refill.
            this.fence[slot] = consumed;
            consumed = null;
            this.slotKey[slot] = NONE;
            this.next[slot] = this.freeHead;
            this.freeHead = slot;
            this.pinned--;
            this.stats.failedTransfer();
        }
        releaseFence(consumed);
    }

    /// Ends a streamed load whose copy failed at retirement. The holders' leases keep the slot
    /// pinned; the expert is no longer found by later requests, and the slot is returned as an
    /// ordinary evictable one when they close.
    private void finishStreamed(Load load, Throwable failure) {
        if (load.finished) return;
        load.finished = true;
        this.load[load.slot] = null;
        this.loading--;
        if (this.directory[load.key] == load.slot) this.directory[load.key] = NONE;
        DeviceFence consumed = load.fence;
        load.fence = null;
        this.stats.failedTransfer();
        releaseFence(consumed);
        // The holders' pins stay; with none left the slot is an ordinary evictable one.
        if (this.pins[load.slot] > 0) this.state[load.slot] = IN_USE;
        else evictable(load.slot);
    }

    // ---------------------------------------------------------------- releasing

    @Override
    public boolean isCurrent(int slot, int generation) {
        return this.generation[slot] == generation;
    }

    /// A lease on `slot` closed. The owner calls this, in its own turn.
    @Override
    public void release(int slot, int generation, DeviceFence fence) {
        List<DeviceFence> unused = new ArrayList<>(2);
        if (this.generation[slot] != generation) {
            // The shard closed with this lease open: the slab is gone, so there is nothing to order.
            if (fence != null) unused.add(fence);
        } else {
            this.leasesOpen--;
            if (fence != null) this.fence[slot] = DeviceFences.merge(this.fence[slot], fence, unused);
            unpin(slot);
        }
        for (DeviceFence redundant : unused) releaseFence(redundant);
    }

    /// Removes one pin from a slot that has leases.
    private void unpin(int slot) {
        if (--this.pins[slot] > 0) return;
        // A streamed load whose copy has not retired keeps its slot: the leases may all be closed (the kernels that
        // read
        // it were submitted behind the copy's marker), but the copy is still writing it. Its retirement makes the slot
        // evictable.
        if (this.state[slot] == LOADING) return;
        evictable(slot);
    }

    /// A slot with no claims becomes the most recently used evictable resident.
    private void evictable(int slot) {
        this.state[slot] = RESIDENT;
        lruAppend(slot);
        this.pinned--;
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
        if (this.closed) throw new IllegalStateException("the expert cache is closed");
    }

    // ---------------------------------------------------------------- inspection

    public ExpertCacheStats stats() {
        return this.stats;
    }

    public int slotCount() {
        return this.slotCount;
    }

    /// Leases open now.
    public int openLeaseCount() {
        return this.leasesOpen;
    }

    /// Loads in flight now.
    public int loadingCount() {
        return this.loading;
    }

    /// Whether the expert's record is in a slot now (resident or leased). A loading expert is not
    /// resident.
    public boolean isResident(int bank, int expert) {
        int slot = this.directory[this.keys.key(bank, expert)];
        return slot != NONE && this.state[slot] != LOADING;
    }

    /// Slots holding a loaded expert that no lease pins, which a miss may evict.
    public int evictableSlots() {
        int count = 0;
        for (int slot = this.lruHead; slot != NONE; slot = this.next[slot]) count++;
        return count;
    }

    /// [#checkInvariants()], and also that no load is in progress and the pins are exactly the open
    /// leases.
    ///
    /// @throws IllegalStateException naming the first violation
    public void checkQuiescent() {
        checkInvariants();
        long pinSum = 0;
        for (int slot = 0; slot < this.slotCount; slot++) pinSum += this.pins[slot];
        if (this.loading != 0) throw new IllegalStateException(this.loading + " transfers are still loading");
        if (!this.closed && pinSum != this.leasesOpen)
            throw new IllegalStateException("pins " + pinSum + " differ from open leases " + this.leasesOpen);
    }

    /// Verifies the bookkeeping, which holds at any moment between two calls of the owner, and
    /// throws [IllegalStateException] naming the first inconsistency: the directory and the slots
    /// agree, every slot is in exactly the list its state says.
    public void checkInvariants() {
        int free = 0;
        for (int slot = this.freeHead; slot != NONE; slot = this.next[slot]) {
            if (++free > this.slotCount) throw new IllegalStateException("free list loops");
            if (this.state[slot] != EMPTY) throw new IllegalStateException("slot " + slot + " is free but not empty");
        }
        int evictable = 0;
        int before = NONE;
        for (int slot = this.lruHead; slot != NONE; slot = this.next[slot]) {
            if (++evictable > this.slotCount) throw new IllegalStateException("recency list loops");
            if (this.state[slot] != RESIDENT || this.pins[slot] != 0)
                throw new IllegalStateException("slot " + slot + " is evictable but pinned or not resident");
            if (this.prev[slot] != before) throw new IllegalStateException("slot " + slot + " has a broken back link");
            before = slot;
        }
        if (before != this.lruTail) throw new IllegalStateException("the recency tail is wrong");
        int empty = 0;
        int resident = 0;
        int inUse = 0;
        int loadingSlots = 0;
        for (int slot = 0; slot < this.slotCount; slot++) {
            switch (this.state[slot]) {
                case EMPTY -> empty++;
                case RESIDENT -> resident++;
                case IN_USE -> {
                    inUse++;
                    if (this.pins[slot] < 1)
                        throw new IllegalStateException("slot " + slot + " is in use but unpinned");
                }
                case LOADING -> {
                    loadingSlots++;
                    if (this.load[slot] == null) throw new IllegalStateException("slot " + slot + " loads nothing");
                }
                default -> throw new IllegalStateException("slot " + slot + " has an unknown state");
            }
            if (this.state[slot] != EMPTY && this.slotKey[slot] != NONE) {
                int mapped = this.directory[this.slotKey[slot]];
                // A slot whose streamed load failed is unmapped on purpose until its leases close.
                if (mapped != slot && this.state[slot] != IN_USE && this.state[slot] != RESIDENT)
                    throw new IllegalStateException("slot " + slot + " is not in the directory");
            }
        }
        if (empty != free) throw new IllegalStateException("free list holds " + free + " of " + empty + " empty slots");
        if (resident != evictable)
            throw new IllegalStateException("recency list holds " + evictable + " of " + resident + " residents");
        if (loadingSlots != this.loading) throw new IllegalStateException("loading count is wrong");
        if (inUse + loadingSlots != this.pinned) throw new IllegalStateException("pinned count is wrong");
    }

    // ---------------------------------------------------------------- closing

    /// Invalidates every open lease and releases the fences still pending. No load may be in
    /// flight.
    ///
    /// @throws IllegalStateException when loads are still in flight
    public void close() {
        if (this.closed) return;
        if (this.loading > 0) throw new IllegalStateException(this.loading + " expert transfers are still in flight");
        this.closed = true;
        this.stats.forcedLeases(this.leasesOpen);
        this.leasesOpen = 0;
        for (int slot = 0; slot < this.slotCount; slot++) {
            // A bumped generation makes every open lease invalid and every later close of one a no-op.
            this.generation[slot]++;
            releaseFence(this.fence[slot]);
            this.fence[slot] = null;
        }
    }

    public boolean isClosed() {
        return this.closed;
    }
}
