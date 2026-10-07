package io.euhedral_execution.inference.core.model.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The device cache of routed experts: one slab of `slotCount * slotBytes` bytes, allocated once
/// through [GpuMemory#allocate], split into shards, each an [ExpertCacheShard] with its own
/// directory and its own slots. An expert belongs to the shard its key hashes to, so the experts of
/// one layer spread over every shard.
///
/// The shards are independent. Each has one owner that confines its bookkeeping, with no state
/// shared between them but the read-only configuration. This class holds what the shards share: the
/// slab, the markers, the host store and the transfer, and the shard routing. The staging buffers
/// and the copy streams belong to no shard. It has no lock and does not wait for anything.
///
/// ## Closing
///
/// [#close()] closes the shards (every open lease becomes invalid; a caller that still has device
/// work reading a slot must have ordered it before closing, as the slab is freed), the transfer and
/// the store, and frees the slab, each exactly once. No load may be in flight.
public final class ExpertCache implements AutoCloseable {
    private static final Logger LOG = LoggerFactory.getLogger(ExpertCache.class);

    /// Slot size granularity: the alignment of a device allocation, so every slot starts aligned.
    public static final long SLOT_ALIGNMENT = 256;

    private final HostExpertStore store;
    private final ExpertTransfer transfer;
    private final GpuMemory memory;
    private final ExpertBank[] banks;
    private final ExpertKeys keys;
    private final long deviceBase;
    private final int slotCount;
    private final long slotBytes;
    private final long[] markers;
    private final ExpertCacheShard[] shards;
    private final ExpertCacheStats stats;
    private boolean closed;

    /// A cache of `slotCount` slots of `slotBytes` bytes in `shards` shards over `store`, whose
    /// records `transfer` moves into a slab allocated from `memory`. The cache owns `store` and
    /// `transfer` once this constructor returns, and closes them; if it throws, the caller still
    /// owns them.
    ///
    /// @throws IllegalArgumentException when `slotBytes` is smaller than the largest record of any
    ///     bank or is not a multiple of [#SLOT_ALIGNMENT], or the shards cannot each have a slot
    public ExpertCache(
            HostExpertStore store,
            ExpertTransfer transfer,
            GpuMemory memory,
            int slotCount,
            long slotBytes,
            int shards) {
        this(store, transfer, memory, slotCount, slotBytes, shards, ReplacementPolicy.GLOBAL_LRU);
    }

    /// As above, with the shards replacing their experts by `policy`.
    public ExpertCache(
            HostExpertStore store,
            ExpertTransfer transfer,
            GpuMemory memory,
            int slotCount,
            long slotBytes,
            int shards,
            ReplacementPolicy policy) {
        this.store = Objects.requireNonNull(store, "store");
        this.transfer = Objects.requireNonNull(transfer, "transfer");
        this.memory = Objects.requireNonNull(memory, "memory");
        if (slotCount < 1) throw new IllegalArgumentException("slotCount must be positive");
        this.banks = store.banks();
        this.keys = new ExpertKeys(this.banks);
        long largest = ExpertFiles.maxRecordBytes(this.banks);
        if (slotBytes < largest)
            throw new IllegalArgumentException(
                    "slotBytes " + slotBytes + " is smaller than the largest record " + largest);
        if (slotBytes % SLOT_ALIGNMENT != 0)
            throw new IllegalArgumentException("slotBytes " + slotBytes + " is not a multiple of " + SLOT_ALIGNMENT);
        if (shards < 1 || shards > slotCount)
            throw new IllegalArgumentException(shards + " shards need a slot each: " + slotCount + " slots");
        long capacity = Math.multiplyExact(slotBytes, (long) slotCount);
        this.slotCount = slotCount;
        this.slotBytes = slotBytes;
        this.deviceBase = memory.allocate(capacity);
        if (this.deviceBase == 0) throw new IllegalStateException("the device slab allocation returned a null address");
        this.markers = new long[slotCount];
        this.shards = new ExpertCacheShard[shards];
        ExpertCacheStats[] parts = new ExpertCacheStats[shards];
        try {
            for (int slot = 0; slot < slotCount; slot++) this.markers[slot] = transfer.openMarker();
            int firstSlot = 0;
            for (int shard = 0; shard < shards; shard++) {
                int slots = slotCount / shards + (shard < slotCount % shards ? 1 : 0);
                parts[shard] = new ExpertCacheStats(slots, slotBytes);
                this.shards[shard] = new ExpertCacheShard(
                        shard,
                        this.banks,
                        this.deviceBase + (long) firstSlot * slotBytes,
                        slots,
                        slotBytes,
                        java.util.Arrays.copyOfRange(this.markers, firstSlot, firstSlot + slots),
                        parts[shard],
                        policy,
                        shards);
                firstSlot += slots;
            }
        } catch (RuntimeException | Error failure) {
            closeMarkers();
            memory.free(this.deviceBase);
            throw failure;
        }
        this.stats = new ExpertCacheStats(parts, slotCount, slotBytes);
    }

    /// The smallest slot size that holds every record of `banks`.
    public static long slotBytesFor(ExpertBank[] banks) {
        return ExpertFiles.alignUp(ExpertFiles.maxRecordBytes(banks), SLOT_ALIGNMENT);
    }

    // ---------------------------------------------------------------- shards

    public int shardCount() {
        return this.shards.length;
    }

    public ExpertCacheShard shard(int shard) {
        return this.shards[shard];
    }

    /// The shard whose slots can hold the expert: the same for every request for it.
    public int shardOf(int bank, int expert) {
        return ExpertKeys.shardOf(this.keys.key(bank, expert), this.shards.length);
    }

    public HostExpertStore store() {
        return this.store;
    }

    public ExpertTransfer transfer() {
        return this.transfer;
    }

    // ---------------------------------------------------------------- inspection

    /// The counters of all shards, read as one.
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

    /// The device address of the slab.
    public long slabAddress() {
        return this.deviceBase;
    }

    public ExpertBank[] banks() {
        return this.banks.clone();
    }

    public boolean isClosed() {
        return this.closed;
    }

    /// Leases open now, over all shards. Read it with the cache quiescent.
    public int openLeaseCount() {
        int count = 0;
        for (ExpertCacheShard shard : this.shards) count += shard.openLeaseCount();
        return count;
    }

    /// Whether the expert's record is in a slot now. Read it with the cache quiescent.
    public boolean isResident(int bank, int expert) {
        return this.shards[shardOf(bank, expert)].isResident(bank, expert);
    }

    /// Slots holding a loaded expert that no lease pins. Read it with the cache quiescent.
    public int evictableSlots() {
        int count = 0;
        for (ExpertCacheShard shard : this.shards) count += shard.evictableSlots();
        return count;
    }

    /// [ExpertCacheShard#checkInvariants()] of every shard. Read it with the cache quiescent.
    public void checkInvariants() {
        for (ExpertCacheShard shard : this.shards) shard.checkInvariants();
    }

    /// [ExpertCacheShard#checkQuiescent()] of every shard.
    public void checkQuiescent() {
        for (ExpertCacheShard shard : this.shards) shard.checkQuiescent();
    }

    // ---------------------------------------------------------------- closing

    private void closeMarkers() {
        for (int slot = 0; slot < this.markers.length; slot++) {
            if (this.markers[slot] == 0) continue;
            this.transfer.closeMarker(this.markers[slot]);
            this.markers[slot] = 0;
        }
    }

    /// Closes the cache as described in the class documentation. Safe to call more than once.
    ///
    /// @throws IllegalStateException when loads were still in flight (nothing was released; call
    ///     again to retry) or ///     releasing a resource failed
    @Override
    public void close() {
        if (this.closed) return;
        for (ExpertCacheShard shard : this.shards) shard.close();
        this.closed = true;
        Throwable failure = null;
        closeMarkers();
        try {
            this.transfer.close();
        } catch (RuntimeException | Error transferFailure) {
            failure = transferFailure;
        }
        try {
            this.store.close();
        } catch (RuntimeException | Error storeFailure) {
            if (failure == null) failure = storeFailure;
            else failure.addSuppressed(storeFailure);
        }
        try {
            this.memory.free(this.deviceBase);
        } catch (RuntimeException | Error freeFailure) {
            if (failure == null) failure = freeFailure;
            else failure.addSuppressed(freeFailure);
        }
        if (failure != null) {
            LOG.warn("releasing the expert cache failed", failure);
            throw new IllegalStateException("releasing the expert cache failed", failure);
        }
    }
}
