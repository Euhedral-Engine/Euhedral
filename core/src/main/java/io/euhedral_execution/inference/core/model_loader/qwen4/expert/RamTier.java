package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/// The expert tier in ordinary (pageable) host memory, between the device cache and the artifact:
///
/// ```
/// gpu cache <- h2d <- pinned staging <- ram tier <- artifact
/// ```
///
/// A device eviction does not evict from the tier, so once it is warm a device miss is a copy out of RAM
/// and no artifact read. The memory is one block, [PageableMemory], committed as records arrive, cut into
/// equal page-aligned slots. The slots are shared out to shards by the same hash that shards the device
/// cache, and each [RamTierShard] is owned by the owner of the device shard with its index: the
/// bookkeeping has no lock because nothing else touches it. The tier object holds what the shards share
/// (the memory and the key space) and the startup preload.
///
/// When every record has a slot the tier is resident: [#preload] reads each record once, at load, by
/// several readers, and inference reads no routed expert from the artifact. Otherwise it is bounded, filled
/// lazily by the loads that miss it, and replaced by the shards' policy.
public final class RamTier implements AutoCloseable {

    private static final long PAGE = 4096;

    private final ExpertBank[] banks;
    private final ExpertKeys keys;
    private final ReplacementPolicy policy;
    private final int shards;
    private final long slotBytes;
    private final int slotCount;
    private final PageableMemory memory;
    private final RamTierShard[] shardList;
    private final boolean resident;
    private volatile long preloadBytes;
    private volatile long preloadNanos;

    /// A tier of at most `slotCount` slots over `banks`, in `shards` shards. A tier with a slot for every
    /// record is resident, and holds exactly that many.
    public RamTier(ExpertBank[] banks, int slotCount, int shards, ReplacementPolicy policy) {
        this.banks = banks.clone();
        this.keys = new ExpertKeys(this.banks);
        this.policy = Objects.requireNonNull(policy, "policy");
        if (shards < 1) throw new IllegalArgumentException("shards must be positive");
        if (slotCount < shards) throw new IllegalArgumentException("every shard needs a slot");
        this.shards = shards;
        this.slotBytes = ExpertFiles.alignUp(ExpertFiles.maxRecordBytes(this.banks), PAGE);
        this.resident = slotCount >= this.keys.keyCount();
        int[] owned = new int[shards];
        for (int key = 0; key < this.keys.keyCount(); key++) owned[ExpertKeys.shardOf(key, shards)]++;
        int[] sizes = new int[shards];
        int total = 0;
        for (int shard = 0; shard < shards; shard++) {
            sizes[shard] = this.resident
                    ? owned[shard]
                    : Math.min(owned[shard], slotCount / shards + (shard < slotCount % shards ? 1 : 0));
            total += sizes[shard];
        }
        this.slotCount = total;
        this.memory = PageableMemory.allocate(Math.multiplyExact(this.slotBytes, (long) Math.max(1, total)));
        this.shardList = new RamTierShard[shards];
        int first = 0;
        for (int shard = 0; shard < shards; shard++) {
            this.shardList[shard] = new RamTierShard(
                    this, this.keys, shard, first, sizes[shard], policy == ReplacementPolicy.BANK_PARTITIONED);
            first += sizes[shard];
        }
        if (this.resident) for (int shard = 0; shard < shards; shard++) this.shardList[shard].assignResident(shard);
    }

    public int shards() {
        return this.shards;
    }

    /// The shard whose owner serves the experts that hash to `shard`.
    public RamTierShard shard(int shard) {
        return this.shardList[shard];
    }

    public ReplacementPolicy policy() {
        return this.policy;
    }

    public boolean isResident() {
        return this.resident;
    }

    public int slotCount() {
        return this.slotCount;
    }

    public long slotBytes() {
        return this.slotBytes;
    }

    /// Bytes of ordinary memory the tier may commit.
    public long capacityBytes() {
        return this.slotBytes * this.slotCount;
    }

    long address(int slot) {
        return this.memory.segment().address() + this.slotBytes * slot;
    }

    // ---------------------------------------------------------------- preload

    /// Reads every record into its slot, by `readers` threads that each take consecutive records of one
    /// bank up to about 32 MiB, so at most `readers` chunks are in flight. Returns when every record is in
    /// memory. Only for a resident tier, before the model serves, on the loading thread: the readers end
    /// with the call.
    public void preload(RecordSource source, int readers) throws IOException, InterruptedException {
        if (!this.resident) throw new IllegalStateException("only a tier with a slot per record preloads");
        long begin = System.nanoTime();
        this.preloadBytes = RamTierPreload.run(this, source, readers);
        this.preloadNanos = System.nanoTime() - begin;
    }

    ExpertBank[] banks() {
        return this.banks;
    }

    ExpertKeys keys() {
        return this.keys;
    }

    /// Where the resident record `key` lives.
    long residentAddress(int key) {
        return address(this.shardList[ExpertKeys.shardOf(key, this.shards)].residentSlot(key));
    }

    // ---------------------------------------------------------------- inspection

    /// The tier's counters by layer, summed over the shards. Counters are read without stopping the
    /// shards, so a reading taken under load is approximate.
    public Stats stats() {
        int count = this.banks.length;
        long[] h = new long[count];
        long[] m = new long[count];
        long[] e = new long[count];
        long[] b = new long[count];
        int[] ready = new int[count];
        for (RamTierShard shard : this.shardList) shard.addTo(h, m, e, b, ready);
        int residentExperts = Arrays.stream(ready).sum();
        return new Stats(
                h,
                m,
                e,
                b,
                ready,
                residentExperts,
                (long) residentExperts * this.slotBytes,
                this.preloadBytes,
                this.preloadNanos);
    }

    /// [RamTierShard#checkInvariants()] of every shard. Read it with the tier quiescent.
    public void checkInvariants() {
        for (RamTierShard shard : this.shardList) shard.checkInvariants();
    }

    /// An immutable reading. `hits`, `misses` (loads that filled a slot), `evictions` and `bypasses` (loads
    /// that read the artifact without a slot) are per bank, as is `residentPerBank`.
    public record Stats(
            long[] hits,
            long[] misses,
            long[] evictions,
            long[] bypasses,
            int[] residentPerBank,
            int residentExperts,
            long residentBytes,
            long preloadBytes,
            long preloadNanos) {

        public long totalHits() {
            return Arrays.stream(this.hits).sum();
        }

        public long totalMisses() {
            return Arrays.stream(this.misses).sum();
        }

        public long totalEvictions() {
            return Arrays.stream(this.evictions).sum();
        }

        public long totalBypasses() {
            return Arrays.stream(this.bypasses).sum();
        }

        /// Hits over loads for the bank, 0 before the first load.
        public double hitRate(int bank) {
            long loads = this.hits[bank] + this.misses[bank] + this.bypasses[bank];
            return loads == 0 ? 0 : (double) this.hits[bank] / loads;
        }

        public double hitRate() {
            long loads = totalHits() + totalMisses() + totalBypasses();
            return loads == 0 ? 0 : (double) totalHits() / loads;
        }

        /// Startup throughput of the preload in bytes per second (0 when the tier did not preload).
        public double preloadBytesPerSecond() {
            return this.preloadNanos == 0 ? 0 : this.preloadBytes * 1e9 / this.preloadNanos;
        }
    }

    /// Releases the memory. No load may be using a slot.
    @Override
    public void close() {
        this.memory.close();
    }
}
