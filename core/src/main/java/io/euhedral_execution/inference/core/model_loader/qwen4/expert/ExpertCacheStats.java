package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

/// The counters of one [ExpertCache], updated without contention and read at any time with [#snapshot].
public final class ExpertCacheStats {
    private final int slotCount;
    private final long slotBytes;
    private final LongAdder hits = new LongAdder();
    private final LongAdder misses = new LongAdder();
    private final LongAdder evictions = new LongAdder();
    private final LongAdder coalesced = new LongAdder();
    private final LongAdder transferBytes = new LongAdder();
    private final LongAdder transferNanos = new LongAdder();
    private final LongAdder slotWaitNanos = new LongAdder();
    private final LongAdder loadWaitNanos = new LongAdder();
    private final LongAdder failedTransfers = new LongAdder();
    private final LongAdder abandonedLoads = new LongAdder();
    private final LongAdder forcedLeases = new LongAdder();
    private final LongAdder prefetchesStarted = new LongAdder();
    private final LongAdder prefetchesUsed = new LongAdder();
    private final LongAdder prefetchesWasted = new LongAdder();
    private final AtomicInteger peakSlotsInUse = new AtomicInteger();

    ExpertCacheStats(int slotCount, long slotBytes) {
        this.slotCount = slotCount;
        this.slotBytes = slotBytes;
    }

    void hit() {
        this.hits.increment();
    }

    void miss() {
        this.misses.increment();
    }

    void eviction() {
        this.evictions.increment();
    }

    void coalesced() {
        this.coalesced.increment();
    }

    void transferred(long bytes, long nanos) {
        this.transferBytes.add(bytes);
        this.transferNanos.add(nanos);
    }

    void waitedForSlot(long nanos) {
        this.slotWaitNanos.add(nanos);
    }

    void waitedForLoad(long nanos) {
        this.loadWaitNanos.add(nanos);
    }

    void failedTransfer() {
        this.failedTransfers.increment();
    }

    void abandonedLoad() {
        this.abandonedLoads.increment();
    }

    void forcedLeases(long count) {
        this.forcedLeases.add(count);
    }

    void prefetchStarted() {
        this.prefetchesStarted.increment();
    }

    void prefetchUsed() {
        this.prefetchesUsed.increment();
    }

    void prefetchWasted() {
        this.prefetchesWasted.increment();
    }

    void slotsInUse(int count) {
        this.peakSlotsInUse.accumulateAndGet(count, Math::max);
    }

    public Snapshot snapshot() {
        return new Snapshot(
                this.hits.sum(),
                this.misses.sum(),
                this.evictions.sum(),
                this.coalesced.sum(),
                this.transferBytes.sum(),
                this.transferNanos.sum(),
                this.slotWaitNanos.sum(),
                this.loadWaitNanos.sum(),
                this.failedTransfers.sum(),
                this.abandonedLoads.sum(),
                this.forcedLeases.sum(),
                this.peakSlotsInUse.get(),
                this.slotCount,
                this.slotBytes,
                this.prefetchesStarted.sum(),
                this.prefetchesUsed.sum(),
                this.prefetchesWasted.sum());
    }

    /// An immutable reading of the counters.
    ///
    /// @param hits requests answered by a resident expert
    /// @param misses requests that started a transfer
    /// @param evictions residents replaced to make room
    /// @param coalescedRequests requests that joined a transfer already in flight
    /// @param transferBytes record bytes that reached the device
    /// @param transferNanos time from starting a transfer to its completion, summed over successful transfers
    /// @param slotWaitNanos time acquirers spent blocked because every slot was in use
    /// @param loadWaitNanos time acquirers spent blocked on a transfer, their own or another's
    /// @param failedTransfers transfers that failed
    /// @param abandonedLoads loads dropped before any transfer started, because the acquirer that began them was
    ///     interrupted or timed out
    /// @param forcedLeases leases still open when the cache closed
    /// @param peakSlotsInUse most slots at once that were loading or leased
    /// @param slotCount slots in the cache
    /// @param slotBytes bytes of one slot
    /// @param prefetchesStarted loads started by [ExpertCache#prefetch]
    /// @param prefetchesUsed prefetched experts that a request later found resident (or joined while loading)
    /// @param prefetchesWasted prefetched experts evicted before any request used them
    public record Snapshot(
            long hits,
            long misses,
            long evictions,
            long coalescedRequests,
            long transferBytes,
            long transferNanos,
            long slotWaitNanos,
            long loadWaitNanos,
            long failedTransfers,
            long abandonedLoads,
            long forcedLeases,
            int peakSlotsInUse,
            int slotCount,
            long slotBytes,
            long prefetchesStarted,
            long prefetchesUsed,
            long prefetchesWasted) {

        public long capacityBytes() {
            return (long) this.slotCount * this.slotBytes;
        }

        /// All time acquirers spent blocked.
        public long waitNanos() {
            return this.slotWaitNanos + this.loadWaitNanos;
        }

        public long requests() {
            return this.hits + this.misses + this.coalescedRequests;
        }

        /// Host-to-device rate over the transfers' own durations, in bytes per second (0 before any transfer).
        public double transferBytesPerSecond() {
            return this.transferNanos == 0 ? 0 : this.transferBytes * 1e9 / this.transferNanos;
        }
    }
}
