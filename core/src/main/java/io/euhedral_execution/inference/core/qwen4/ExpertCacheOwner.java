package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.hashing.HasherApi;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.DeviceFence;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCache;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCacheShard;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertLease;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.RamTierShard;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.TierDirective;
import io.euhedral_execution.inference.core.scheduling.graph.FrameLake;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

/// The owner of the expert cache's bookkeeping (the device cache's shards and the host tier's): the directory,
/// the slots' states, pins and recency, the tier's slots.
///
/// The bookkeeping is plain fields. Every frame that touches it carries the owner's routing hash ([#HASH]): a
/// fetch stage, the submit and retire frames of a load, the release of a lease. The lattice runs frames of one
/// hash in order, one at a time, so these are the only places the state changes and they need nothing else.
/// Everything else the bookkeeping must learn becomes such a frame: a lease closed on a wave's thread publishes
/// a release, and a copy's retirement (a driver callback) publishes its retire frame.
public final class ExpertCacheOwner {

    /// The routing hash of every frame that touches the bookkeeping.
    public static final long HASH = HasherApi.mix(0x0e_c4c8_0d0eL);

    /// What asked for an expert: a fetch stage of a graph.
    public interface Fetch {
        /// The quantum stopped: nothing more is loaded for it.
        boolean stopped();

        /// The expert is held by `lease` (carrying its copy's marker when the copy was only submitted).
        void arrived(ExpertLease lease);

        /// The expert will not come: its load failed.
        void failed(Throwable failure);
    }

    /// What [#fetch] did.
    public enum Outcome {
        /// The expert was resident: the fetch has its lease.
        LEASED,
        /// The expert is being loaded: the fetch hears from the load.
        LOADING,
        /// Every slot that could hold it is pinned, every staging buffer is in use, or the disk has all the
        /// reads in flight that keep it busy: nothing changed.
        FULL
    }

    /// Where the loads spent their time, summed over loads (loads overlap, so the sums exceed wall time): from the
    /// fetch to the first frame running, reading (or copying out of RAM), from the record being ready to the
    /// submit frame running, the copy on the device, and from the retirement callback to its frame running.
    public record Timings(long loads, long dispatch, long read, long submitHop, long copy, long retireHop) {
        public Timings plus(Timings other) {
            return new Timings(
                    this.loads + other.loads,
                    this.dispatch + other.dispatch,
                    this.read + other.read,
                    this.submitHop + other.submitHop,
                    this.copy + other.copy,
                    this.retireHop + other.retireHop);
        }
    }

    /// Records read from the artifact at once at most: the depth of the disk's queue that keeps it busy, past which
    /// a read only waits behind the others and every record arrives late. `EUHEDRAL_QWEN4_READS` overrides it for
    /// benchmarks.
    static final int READS = Integer.parseInt(System.getenv().getOrDefault("EUHEDRAL_QWEN4_READS", "32"));

    final ExpertCache cache;
    final FrameLake lake;
    /// Loads reading the artifact now; owner frames only.
    private int reading;
    final int readParts;
    private final ExpertCacheShard.Ticket ticket = new ExpertCacheShard.Ticket();
    private final AtomicInteger inFlight = new AtomicInteger();
    private long nextSeed = HasherApi.mix(0x1a4e_0000L);
    private final LongAdder loads = new LongAdder();
    private final LongAdder dispatchNanos = new LongAdder();
    private final LongAdder readNanos = new LongAdder();
    private final LongAdder submitNanos = new LongAdder();
    private final LongAdder copyNanos = new LongAdder();
    private final LongAdder retireNanos = new LongAdder();
    private final LongAdder fullFetches = new LongAdder();

    public ExpertCacheOwner(ExpertCache cache, FrameLake lake) {
        this.cache = cache;
        this.lake = lake;
        this.readParts = cache.store().readParts();
        for (int shard = 0; shard < cache.shardCount(); shard++)
            cache.shard(shard).owner(new Releases(shard));
    }

    /// The leases of one shard report here: a close publishes a release frame routed to the owner.
    private final class Releases implements ExpertLease.Owner {
        private final int shard;

        Releases(int shard) {
            this.shard = shard;
        }

        @Override
        public void release(int slot, int generation, DeviceFence fence) {
            ExpertCacheOwner.this.lake.publish(new Release(this.shard, slot, generation, fence));
        }

        @Override
        public boolean isCurrent(int slot, int generation) {
            return ExpertCacheOwner.this.cache.shard(this.shard).isCurrent(slot, generation);
        }
    }

    /// A lease closed: the owner's frame that unpins its slot.
    private final class Release extends AbstractFrame {
        private final int shard;
        private final int slot;
        private final int generation;
        private final DeviceFence fence;

        Release(int shard, int slot, int generation, DeviceFence fence) {
            super(HASH);
            this.shard = shard;
            this.slot = slot;
            this.generation = generation;
            this.fence = fence;
        }

        @Override
        public void execute() {
            ExpertCacheOwner.this.cache.shard(this.shard).release(this.slot, this.generation, this.fence);
        }

        @Override
        public void doFinally() {}

        /// The lattice rejected the frame without running it; the slot is still released, on the rejecting
        /// thread, which is not a driver callback.
        @Override
        public void doFinallyWithError(Throwable rejection) {
            execute();
        }
    }

    // ---------------------------------------------------------------- called by owner frames only

    /// Asks for `expert` of `bank` for `target`. Called by a frame routed with [#HASH].
    public Outcome fetch(Fetch target, int bank, int expert) {
        int shardIndex = this.cache.shardOf(bank, expert);
        ExpertCacheShard shard = this.cache.shard(shardIndex);
        RamTierShard tier = this.cache.store().tier(shardIndex);
        // A miss needs a slot and, unless its record is in a pinned tier, a staging buffer: both, or neither.
        boolean pinnedTier = tier != null && tier.pinned();
        int buffer = pinnedTier ? -1 : this.cache.store().acquireStaging();
        shard.claim(bank, expert, pinnedTier || buffer >= 0, this.ticket);
        ExpertLease lease = this.ticket.lease();
        ExpertCacheShard.Load reserved = this.ticket.load();
        if (reserved == null && buffer >= 0) this.cache.store().releaseStaging(buffer);
        if (lease != null) {
            target.arrived(lease);
            return Outcome.LEASED;
        }
        if (reserved == null) {
            this.fullFetches.increment();
            return Outcome.FULL;
        }
        TierDirective directive = new TierDirective();
        if (tier != null) tier.plan(bank, expert, directive);
        boolean reads = directive.mode() != TierDirective.Mode.HIT;
        if (reads && this.reading >= READS) {
            // The disk has as many records in flight as keep it busy: this one is read once one of them is in.
            ExpertLoad.giveBack(tier, directive);
            reserved.cancel();
            if (buffer >= 0) this.cache.store().releaseStaging(buffer);
            this.fullFetches.increment();
            return Outcome.FULL;
        }
        if (buffer < 0 && this.cache.store().stagesThrough(directive)) {
            // A pinned tier with every slot in use: the record bypasses it through a staging buffer.
            buffer = this.cache.store().acquireStaging();
            if (buffer < 0) {
                ExpertLoad.giveBack(tier, directive);
                reserved.cancel();
                this.fullFetches.increment();
                return Outcome.FULL;
            }
        }
        if (reads) this.reading++;
        ExpertLoad load = new ExpertLoad(this, target, reserved, buffer, tier, directive, reads, this.nextSeed);
        this.nextSeed += 64;
        // The load is the quantum's continuation until its copy retired: the lake cannot finish without it.
        this.lake.admitDuringDrain();
        this.inFlight.incrementAndGet();
        load.start();
        return Outcome.LOADING;
    }

    /// A load ended (its copy retired, or it failed). Called by a frame routed with [#HASH].
    void ended(ExpertLoad load, boolean submitted) {
        if (load.reads) this.reading--;
        if (submitted) {
            this.loads.increment();
            this.dispatchNanos.add(load.startedAt - load.fetchedAt);
            this.readNanos.add(load.readAt - load.startedAt);
            this.submitNanos.add(load.submittingAt - load.readAt);
            this.copyNanos.add(load.retiredAt - load.submittedAt);
            this.retireNanos.add(load.confirmedAt - load.retiredAt);
        }
        this.inFlight.decrementAndGet();
        this.lake.terminated();
    }

    // ---------------------------------------------------------------- inspection, any thread

    /// Loads fetched whose copies have not retired yet.
    public int loadsInFlight() {
        return this.inFlight.get();
    }

    /// Fetches that found every slot pinned and were published again.
    public long fullFetches() {
        return this.fullFetches.sum();
    }

    public Timings timings() {
        return new Timings(
                this.loads.sum(),
                this.dispatchNanos.sum(),
                this.readNanos.sum(),
                this.submitNanos.sum(),
                this.copyNanos.sum(),
                this.retireNanos.sum());
    }
}
