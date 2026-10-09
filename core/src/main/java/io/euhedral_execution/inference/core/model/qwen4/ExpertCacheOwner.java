package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.ingest.AbstractIngestSink;
import io.euhedral_execution.data_structures.queues.MpscQueue;
import io.euhedral_execution.hashing.HasherApi;
import io.euhedral_execution.inference.core.model.qwen4.expert.DeviceFence;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertCache;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertCacheShard;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertLease;
import io.euhedral_execution.inference.core.model.qwen4.expert.RamTierShard;
import io.euhedral_execution.inference.core.model.qwen4.expert.TierDirective;
import io.euhedral_execution.inference.core.model.qwen4.loader.ResidencyPlanner;
import io.euhedral_execution.inference.core.runtime.graph.FrameLake;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Consumer;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The owner of the expert cache's bookkeeping (the device cache's shards and the host tier's): the directory,
/// the slots' states, pins and recency, the tier's slots. It is a source of the lattice, like the artifact's reads.
///
/// The bookkeeping is plain fields, changed only while a worker polls the source: the lattice never calls `request`
/// or `pull` of one source twice at a time, so the source is the owner and nothing else guards the state. Whatever
/// the bookkeeping must learn becomes a [Record] posted to a lock-free queue, from any thread: a fetch stage's request,
/// a lease closed on a wave's thread, a copy's retirement (a driver callback, which may only enqueue), a read's
/// completion, a prediction. A poll applies the records in the order they were posted and then asks again the fetches
/// that found the cache full, but only when something that could let one proceed has changed since it last asked
/// ([#changed]): a fetch that cannot be served is not a frame that runs again, it waits in the source, and the poll of
/// a source with nothing to do returns at once.
public final class ExpertCacheOwner extends AbstractIngestSink {

    private static final Logger LOG = LoggerFactory.getLogger(ExpertCacheOwner.class);

    /// Something the source applies to the cache's bookkeeping, in the order it was posted. It never throws.
    interface Record {
        void apply();
    }

    /// What asked for an expert: a fetch stage of a graph.
    public interface Fetch {
        /// The quantum stopped: nothing more is loaded for it.
        boolean stopped();

        /// The fetch will not be served: the quantum stopped before the expert came.
        void abandoned();

        /// The expert is held by `lease` (carrying its copy's marker when the copy was only submitted).
        void arrived(ExpertLease lease);

        /// The expert will not come: its load failed.
        void failed(Throwable failure);

        /// Whether the asking block is a prefill chunk (more than one row), which visits nearly every expert of a
        /// layer: the tier admits its records only in place of records asked for less often.
        default boolean scan() {
            return false;
        }
    }

    /// What [#fetch] did.
    public enum Outcome {
        /// The expert was resident: the fetch has its lease.
        LEASED,
        /// The expert is being loaded: the fetch hears from the load.
        LOADING,
        /// Every slot that could hold it is pinned, every staging buffer is in use, the disk has all the
        /// reads in flight that keep it busy, or the expert is being filled or loaded and not ready to lease:
        /// nothing changed, and [ExpertCacheOwner#fullCause] says what the fetch waits for.
        FULL
    }

    /// What a fetch that found the cache full waits for.
    enum Cause {
        /// A slot of the expert's shard.
        SLOTS,
        /// A staging buffer.
        STAGING,
        /// One of the disk's reads.
        READS,
        /// The expert's own fill or load, which is not ready to lease yet.
        PENDING
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

    /// Records read from the artifact at once at most ([ResidencyPlanner#DISK_READS]).
    static final int READS = ResidencyPlanner.DISK_READS;

    final ExpertCache cache;
    final FrameLake lake;
    /// Loads reading the artifact now: a physical resource, like the staging pool. The owner's fetch takes a read;
    /// the frame that has the record (its read complete) gives it back, on whichever worker runs it, so the disk
    /// is offered the next record as soon as one is in, not once its copy retired.
    private final AtomicInteger reading = new AtomicInteger();
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
    private final LongAdder fullSlots = new LongAdder();
    private final LongAdder fullStaging = new LongAdder();
    private final LongAdder fullReads = new LongAdder();

    /// What the bookkeeping must learn, from any thread; applied by the poll, in this order.
    private final MpscQueue<Record> records = new MpscQueue<>(1024, 4);
    /// Fetches that found the cache full, oldest first, and what each waits for. Touched by the poll only.
    private final ArrayList<Request> blocked = new ArrayList<>();
    /// Counts what could let a blocked fetch proceed: a slot released, a staging buffer or a read given back, a tier
    /// slot settled, a copy submitted. Any thread.
    private final AtomicLong epoch = new AtomicLong();
    /// The epoch the blocked fetches were last asked at, and when the poll last looked for blocked fetches whose
    /// quantum stopped. Touched by the poll only.
    private long askedAt;
    private long sweptAt;
    /// What the last [#fetch] that returned FULL waited for. Touched by the poll only.
    private Cause cause;
    private int causeShard;
    private final Delegate delegate = new Delegate();

    public ExpertCacheOwner(ExpertCache cache, FrameLake lake) {
        this.cache = cache;
        this.lake = lake;
        this.readParts = cache.store().readParts();
        for (int shard = 0; shard < cache.shardCount(); shard++)
            cache.shard(shard).owner(new Releases(shard));
    }

    /// The leases of one shard report here: a close posts the release.
    private final class Releases implements ExpertLease.Owner {
        private final int shard;

        Releases(int shard) {
            this.shard = shard;
        }

        @Override
        public void release(int slot, int generation, DeviceFence fence) {
            post(() -> {
                ExpertCacheOwner.this.cache.shard(this.shard).release(slot, generation, fence);
                changed();
            });
        }

        @Override
        public boolean isCurrent(int slot, int generation) {
            return ExpertCacheOwner.this.cache.shard(this.shard).isCurrent(slot, generation);
        }
    }

    // ---------------------------------------------------------------- any thread

    /// Posts `record` for the poll to apply. Never blocks, and is the only thing a driver callback may do.
    void post(Record record) {
        if (!this.records.offer(record)) throw new IllegalStateException("the expert cache's records refused one");
    }

    /// Something that could let a blocked fetch proceed happened: the next poll asks them again.
    void changed() {
        this.epoch.incrementAndGet();
    }

    /// Asks for `expert` of `bank` for `target`: the source serves it when the cache can, now or once something
    /// frees a slot, a staging buffer or a read. The fetch hears of the outcome through `target`.
    public void request(Fetch target, int bank, int expert) {
        post(new Request(target, bank, expert));
    }

    /// A fetch's request, and what it waits for while the cache is full.
    private final class Request implements Record {
        final Fetch target;
        final int bank;
        final int expert;
        Cause cause;
        int shard;
        /// The epoch the cache was last asked at for this request.
        long askedAt;

        Request(Fetch target, int bank, int expert) {
            this.target = target;
            this.bank = bank;
            this.expert = expert;
        }

        @Override
        public void apply() {
            if (!ask(this)) ExpertCacheOwner.this.blocked.add(this);
        }
    }

    // ---------------------------------------------------------------- the poll only

    /// Asks the cache for `expert` of `bank` for `target` once. Called by the poll.
    public Outcome fetch(Fetch target, int bank, int expert) {
        int shardIndex = this.cache.shardOf(bank, expert);
        ExpertCacheShard shard = this.cache.shard(shardIndex);
        RamTierShard tier = this.cache.store().tier(shardIndex);
        // A miss needs a slot and, unless a pinned tier holds or takes its record, a staging buffer: both are reserved
        // before the tier is asked, so a fetch that has to wait leaves the tier as it was (a fill's plan takes a slot
        // and evicts its record, and admission counts the request).
        boolean pinnedTier = tier != null && tier.pinned();
        boolean inTier = tier != null && tier.isResident(bank, expert);
        boolean staged = !(pinnedTier && inTier);
        int buffer = staged ? this.cache.store().acquireStaging() : -1;
        shard.claim(bank, expert, !staged || buffer >= 0, this.ticket);
        ExpertLease lease = this.ticket.lease();
        ExpertCacheShard.Load reserved = this.ticket.load();
        if (reserved == null && buffer >= 0) this.cache.store().releaseStaging(buffer);
        if (lease != null) {
            target.arrived(lease);
            return Outcome.LEASED;
        }
        if (reserved == null) {
            this.fullFetches.increment();
            boolean noStaging = staged && buffer < 0;
            (noStaging ? this.fullStaging : this.fullSlots).increment();
            this.cause = noStaging ? Cause.STAGING : this.ticket.waiting() ? Cause.SLOTS : Cause.PENDING;
            this.causeShard = shardIndex;
            return Outcome.FULL;
        }
        if (tier != null && tier.isFilling(bank, expert)) {
            // A prefetch is reading the record into the tier: once it is in, the load copies it from there.
            reserved.cancel();
            if (buffer >= 0) this.cache.store().releaseStaging(buffer);
            this.fullFetches.increment();
            this.fullReads.increment();
            this.cause = Cause.PENDING;
            return Outcome.FULL;
        }
        if (!inTier && this.reading.get() >= READS) {
            // The disk has as many records in flight as keep it busy: this one is read once one of them is in.
            reserved.cancel();
            if (buffer >= 0) this.cache.store().releaseStaging(buffer);
            this.fullFetches.increment();
            this.fullReads.increment();
            this.cause = Cause.READS;
            return Outcome.FULL;
        }
        TierDirective directive = new TierDirective();
        if (tier != null) tier.plan(bank, expert, target.scan(), directive);
        boolean reads = directive.mode() != TierDirective.Mode.HIT;
        if (buffer >= 0 && !this.cache.store().stagesThrough(directive)) {
            // A pinned tier takes the record into its slot: the copy reads it there.
            this.cache.store().releaseStaging(buffer);
            buffer = -1;
        }
        if (reads) this.reading.incrementAndGet();
        ExpertLoad load = new ExpertLoad(this, target, reserved, buffer, tier, directive, reads, this.nextSeed);
        this.nextSeed += 64;
        // The load is the quantum's continuation until its copy retired: the lake cannot finish without it.
        this.lake.admitDuringDrain();
        this.inFlight.incrementAndGet();
        load.start();
        return Outcome.LOADING;
    }

    /// The prefetch (`EUHEDRAL_QWEN4_PREFETCH=K,B[,D]`: the router of the layer `D` ahead (1) applied to a decode
    /// step's input, its best `K` experts, at most `B` of them read per layer; `0`: none). By default the next
    /// layer's best 10, one read per layer (docs/FLASH_NEXT_PREFETCH.md).
    static final int[] PREFETCH = prefetchSetting(System.getenv("EUHEDRAL_QWEN4_PREFETCH"));

    private static int[] prefetchSetting(String value) {
        if (value == null || value.isBlank()) value = "10,1,1";
        if (value.equals("0")) return new int[] {0, 0, 1};
        String[] parts = value.split(",");
        return new int[] {
            Integer.parseInt(parts[0].trim()),
            Integer.parseInt(parts[1].trim()),
            parts.length > 2 ? Integer.parseInt(parts[2].trim()) : 1
        };
    }

    /// How many layers ahead the prefetch predicts.
    public static int prefetchDistance() {
        return PREFETCH.length > 2 ? PREFETCH[2] : 1;
    }

    /// The best experts of the next layer's router to consider, and how many of them a prefetch reads at most.
    public static int prefetchCandidates() {
        return PREFETCH[0];
    }

    private final LongAdder prefetches = new LongAdder();
    private final LongAdder prefetchFailures = new LongAdder();

    /// Posts a prefetch of `experts` of bank `bank` (best first): the poll reads up to the configured number of
    /// them that neither the device nor the host tier holds into the tier, while the disk has a read to spare. Any
    /// thread.
    public void publishPrefetch(int bank, int[] experts) {
        if (PREFETCH[1] <= 0) return;
        post(() -> prefetch(bank, experts, PREFETCH[1]));
    }

    /// Reads up to `budget` of `experts` into the host tier, speculatively: it reads only what the disk can take now
    /// and never waits. Called by the poll.
    public void prefetch(int bank, int[] experts, int budget) {
        for (int expert : experts) {
            if (budget <= 0 || this.reading.get() >= READS) return;
            int shardIndex = this.cache.shardOf(bank, expert);
            RamTierShard tier = this.cache.store().tier(shardIndex);
            if (tier == null || !tier.pinned()) return;
            if (this.cache.shard(shardIndex).holds(bank, expert)) continue;
            TierDirective directive = new TierDirective();
            tier.planPrefetch(bank, expert, directive);
            if (directive.mode() != TierDirective.Mode.FILL) continue;
            budget--;
            this.reading.incrementAndGet();
            this.lake.admitDuringDrain();
            this.inFlight.incrementAndGet();
            TierFill fill = new TierFill(this, tier, directive, bank, expert, this.nextSeed);
            this.nextSeed += 64;
            if (!fill.start()) {
                this.reading.decrementAndGet();
                tier.abandoned(directive);
                prefetchEnded(false);
                return;
            }
        }
    }

    /// A prefetch ended, its record in the tier (`read`) or not. Called by the poll.
    void prefetchEnded(boolean read) {
        (read ? this.prefetches : this.prefetchFailures).increment();
        this.inFlight.decrementAndGet();
        this.lake.terminated();
        changed();
    }

    /// Prefetches that read their record, and those that failed. Any thread.
    public long[] prefetchCounts() {
        return new long[] {this.prefetches.sum(), this.prefetchFailures.sum()};
    }

    /// The load's artifact read is over: the disk may take another. Any thread, once per load.
    void readEnded() {
        this.reading.decrementAndGet();
        changed();
    }

    /// A load ended (its copy retired, or it failed). Called by the poll.
    void ended(ExpertLoad load, boolean submitted) {
        load.readDone();
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
        changed();
    }

    // ---------------------------------------------------------------- the source

    /// Applies what was posted, in order, then asks the blocked fetches again if something could let one proceed.
    /// The lattice calls it from one worker at a time.
    public void poll() {
        for (Record record = this.records.poll(); record != null; record = this.records.poll()) {
            try {
                record.apply();
            } catch (RuntimeException failure) {
                LOG.error("an expert cache record failed", failure);
            }
        }
        if (!this.blocked.isEmpty()) askBlocked();
    }

    /// A stopped quantum's fetches are found at least this often, even when nothing changed.
    private static final long SWEEP_NANOS = 250_000L;

    /// Asks the blocked fetches once more, oldest first, when the epoch moved since they were last asked, and drops
    /// those of stopped quanta. Once a fetch finds the staging buffers, the disk's reads or a shard's slots
    /// exhausted, the younger fetches that wait for the same are not asked: they would find the same.
    private void askBlocked() {
        long epoch = this.epoch.get();
        long now = System.nanoTime();
        boolean moved = epoch != this.askedAt;
        boolean sweep = now - this.sweptAt >= SWEEP_NANOS;
        if (!moved && !sweep) return;
        this.askedAt = epoch;
        if (sweep) this.sweptAt = now;
        boolean noStaging = false;
        boolean noReads = false;
        long noSlots = 0;
        int kept = 0;
        for (int i = 0; i < this.blocked.size(); i++) {
            Request request = this.blocked.get(i);
            if (request.target.stopped()) {
                request.target.abandoned();
                continue;
            }
            if (moved && request.askedAt != epoch) {
                boolean skip =
                        switch (request.cause) {
                            case STAGING -> noStaging;
                            case READS -> noReads;
                            case SLOTS -> (noSlots >> request.shard & 1) != 0;
                            case PENDING -> false;
                        };
                if (!skip) request.askedAt = epoch;
                if (!skip && ask(request)) continue;
                if (!skip) {
                    noStaging |= request.cause == Cause.STAGING;
                    noReads |= request.cause == Cause.READS;
                    if (request.cause == Cause.SLOTS) noSlots |= 1L << request.shard;
                }
            }
            this.blocked.set(kept++, request);
        }
        this.blocked.subList(kept, this.blocked.size()).clear();
    }

    /// Asks the cache once for `request`; false, with what it waits for recorded, when the cache is full.
    private boolean ask(Request request) {
        if (request.target.stopped()) {
            request.target.abandoned();
            return true;
        }
        request.askedAt = this.epoch.get();
        if (fetch(request.target, request.bank, request.expert) != Outcome.FULL) return true;
        request.cause = this.cause;
        request.shard = this.causeShard;
        return false;
    }

    /// The source: a worker that polls it applies the posted records. A poll that finds nothing to do returns at
    /// once and emits no frame; the frames a record leads to (a load's reads) are published as before.
    @Override
    public LatticeSource getDelegate() {
        return this.delegate;
    }

    @Override
    public void complete() {
        this.delegate.complete();
    }

    @Override
    public boolean isComplete() {
        return this.delegate.isComplete();
    }

    /// The lattice calls `request` and `pull` on a registered source one thread at a time, so the bookkeeping
    /// needs no other protection.
    private final class Delegate extends AbstractIngestSink.Delegate {
        @Override
        public long hookOnPull(
                Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stopCondition, long demand) {
            poll();
            return 0;
        }

        @Override
        public void hookOnRequest(LatticeReceiver terminal, long demand) {
            poll();
        }
    }

    /// Fetches waiting for the cache. For tests, which play the worker that polls: the list is the poll's.
    public int blockedFetches() {
        return this.blocked.size();
    }

    /// Whether a record is posted and not yet applied. Any thread.
    public boolean hasRecords() {
        return !this.records.isEmpty();
    }

    // ---------------------------------------------------------------- inspection, any thread

    /// Loads fetched whose copies have not retired yet.
    public int loadsInFlight() {
        return this.inFlight.get();
    }

    /// Fetches that found the cache full, by what was missing: a device slot, a staging buffer, or one of the
    /// disk's reads. Any thread.
    public long[] fullCauses() {
        return new long[] {this.fullSlots.sum(), this.fullStaging.sum(), this.fullReads.sum()};
    }

    /// Loads whose artifact read has not ended. Any thread.
    public int readsInFlight() {
        return this.reading.get();
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
