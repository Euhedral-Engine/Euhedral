package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.hashing.HasherApi;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.DeviceFence;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCache;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCacheShard;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertLease;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.HostRecord;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.RamTierShard;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.TierDirective;
import io.euhedral_execution.inference.core.scheduling.graph.SerialSource;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.LongAdder;

/// The owner of one [ExpertCacheShard], as a lattice source that is attached to the lattice on its
/// own: the cache of a model is several of these, side by side.
///
/// The shard's state is touched only here, from `request` and `pull`, which Euhedral runs one
/// thread at a time: no lock, no atomic. Everything that concerns the shard reaches it as a message
/// in the source's mailbox: the share of a block that the plan stage posts, a lease that closed
/// (from the wave stage's thread), a copy that was submitted (from a load frame) or retired (from a
/// driver callback).
///
/// The source walks its share of the block in order. For each expert it claims the cache: a
/// resident one is a lease at once, with no frame at all; a miss that finds a slot and a copy lane
/// free becomes a load frame, which the source generates and Euhedral routes to the lane's worker.
/// When nothing is free the walk waits; a lease closing or a copy retiring is a message that
/// resumes it. In-order claiming with the shard's slots as the only bound is what keeps a block
/// from pinning more than the shard holds: an expert is claimed only when a slot is there.
///
/// The load frame does the one blocking thing, the read of the record into the lane's staging slot,
/// submits the copy on the lane's copy stream (which records the marker the lease will carry), and
/// posts that it did. The source then answers the block with the lease. The copy's retirement is a
/// driver callback that posts too; the lane is free again after it.
///
/// The host tier's shard with the same index is owned here too. Before a load frame is published
/// the source asks it what the load does (copy the record out of RAM, read the artifact through a
/// tier slot, or read the artifact alone) and hands the answer to the frame in the lane's
/// directive; the slot stays pinned until the load frame has posted that it is done with it.
public final class ExpertSource extends SerialSource implements ExpertLease.Owner {

    /// A share of a block for this shard: the positions (in order) of the block's experts that hash
    /// here.
    public static final class Work {
        ExpertBlock block;
        int[] positions;
        int count;

        public Work set(ExpertBlock block, int[] positions, int count) {
            this.block = block;
            this.positions = positions;
            this.count = count;
            return this;
        }
    }

    /// Where the loads of this source spent their time, summed over loads (loads of different lanes
    /// overlap, so the sums exceed wall time): `dispatch` from the claim to a worker starting the
    /// frame, `open` in the host store (the read or the RAM copy), `submit` from the frame's end to
    /// the source handling it, `copy` from the copy's submission to its retirement callback, and
    /// `retire` from that callback to the source handling it.
    public record Timings(long loads, long dispatch, long open, long submit, long copy, long retire) {
        public Timings plus(Timings other) {
            return new Timings(
                    this.loads + other.loads,
                    this.dispatch + other.dispatch,
                    this.open + other.open,
                    this.submit + other.submit,
                    this.copy + other.copy,
                    this.retire + other.retire);
        }
    }

    private final LongAdder loadCount = new LongAdder();
    private final LongAdder dispatchNanos = new LongAdder();
    private final LongAdder openNanos = new LongAdder();
    private final LongAdder submitNanos = new LongAdder();
    private final LongAdder copyNanos = new LongAdder();
    private final LongAdder retireNanos = new LongAdder();

    public Timings timings() {
        return new Timings(
                this.loadCount.sum(),
                this.dispatchNanos.sum(),
                this.openNanos.sum(),
                this.submitNanos.sum(),
                this.copyNanos.sum(),
                this.retireNanos.sum());
    }

    private record Release(int slot, int generation, DeviceFence fence) {}

    /// What a lane reports; the message is the lane's own object, so reporting allocates nothing.
    private enum Kind {
        SUBMITTED,
        RETIRED,
        FAILED
    }

    private final ExpertCache cache;
    private final ExpertCacheShard shard;
    private final RamTierShard tier;
    private final int laneBase;
    private final Lane[] lanes;
    private final ExpertCacheShard.Ticket ticket = new ExpertCacheShard.Ticket();

    // Serial state: the share of the block being walked.
    private Work work;
    private int next;
    private int inFlight;

    public ExpertSource(ExpertCache cache, int shard) {
        this.cache = cache;
        this.shard = cache.shard(shard);
        this.tier = cache.store().tier(shard);
        this.laneBase = cache.laneBase(shard);
        this.lanes = new Lane[cache.laneCount(shard)];
        for (int lane = 0; lane < this.lanes.length; lane++) this.lanes[lane] = new Lane(lane);
        this.shard.owner(this);
    }

    public ExpertCacheShard shard() {
        return this.shard;
    }

    /// Posts this shard's share of a block.
    public void submit(Work share) {
        post(share);
    }

    // ---------------------------------------------------------------- lease owner

    @Override
    public void release(int slot, int generation, DeviceFence fence) {
        post(new Release(slot, generation, fence));
    }

    @Override
    public boolean isCurrent(int slot, int generation) {
        return this.shard.isCurrent(slot, generation);
    }

    // ---------------------------------------------------------------- serial

    @Override
    protected void handle(Object message) {
        if (message instanceof Work share) begin(share);
        else if (message instanceof Release release) {
            this.shard.release(release.slot(), release.generation(), release.fence());
            advance();
        } else if (message instanceof Lane.Report report) report.lane().handle(report.kind());
    }

    private void begin(Work share) {
        if (this.work != null) throw new IllegalStateException("an expert source serves one block at a time");
        this.work = share;
        this.next = 0;
        this.inFlight = 0;
        advance();
    }

    /// Takes the experts in order while the shard has room for them.
    private void advance() {
        Work current = this.work;
        if (current == null) return;
        ExpertBlock block = current.block;
        while (this.next < current.count) {
            int position = current.positions[this.next];
            if (block.stopped()) {
                // Nothing more is loaded for a stopped quantum, but every expert arrives so that its stages end.
                while (this.next < current.count) block.arrive(current.positions[this.next++], null);
                break;
            }
            Lane lane = freeLane();
            this.shard.claim(block.bank(), block.expertAt(position), lane != null, this.ticket);
            if (this.ticket.lease() != null) {
                block.arrive(position, this.ticket.lease());
                this.next++;
            } else if (this.ticket.load() != null) {
                lane.start(block, position, this.ticket.load());
                this.inFlight++;
                this.next++;
            } else break;
        }
        if (this.next == current.count && this.inFlight == 0) {
            this.work = null;
            block.shardDone();
        }
    }

    private Lane freeLane() {
        for (Lane lane : this.lanes) if (!lane.busy) return lane;
        return null;
    }

    // ---------------------------------------------------------------- lanes

    /// One copy lane: a pinned staging slot and a copy stream, and the load frame that works them.
    /// A lane serves one load at a time, from the claim to the retirement of its copy.
    private final class Lane extends AbstractFrame implements GpuStream.RetirementListener {

        /// What a lane tells the source.
        private record Report(Lane lane, Kind kind) {}

        private final int local;
        private final int global;
        private final Report submitted = new Report(this, Kind.SUBMITTED);
        private final Report retired = new Report(this, Kind.RETIRED);
        private final Report failedReport = new Report(this, Kind.FAILED);

        // Serial state, also handed to the load frame by the message that makes it ready.
        boolean busy;
        private ExpertBlock block;
        private int position;
        private ExpertCacheShard.Load load;
        private final TierDirective directive = new TierDirective();
        private boolean submittedSeen;
        private boolean retiredEarly;
        // Written by the load frame before it posts.
        private HostRecord record;
        private boolean touched;
        private Throwable failure;
        private volatile long ticket;
        // Timestamps for the load's timings: the owner writes the first, the frame the next three, the
        // driver callback the last.
        private long claimedAt;
        private long startedAt;
        private long openedAt;
        private long submittedAt;
        private volatile long retiredAt;
        // The parts of an artifact read, when it is split: the frames that read them, how many have yet
        // to end, and the first failure.
        private final Part[] parts;
        private final AtomicInteger pending = new AtomicInteger();
        private final AtomicReference<Throwable> partFailure = new AtomicReference<>();

        private Lane(int local) {
            // The lane's hash is its identity: every load of the lane is routed to the same worker.
            super(HasherApi.mix(0x1a4e_0000L + ExpertSource.this.laneBase + local));
            this.local = local;
            this.global = ExpertSource.this.laneBase + local;
            this.parts = new Part[ExpertSource.this.cache.store().readParts()];
            for (int part = 0; part < this.parts.length; part++) this.parts[part] = new Part(this, part);
        }

        /// One part of an artifact read: a frame of its own, so that the parts of one record are read
        /// side by side on different workers. The part that ends last carries the load on.
        private final class Part extends AbstractFrame {
            private final Lane lane;
            private final int index;

            Part(Lane lane, int index) {
                super(HasherApi.mix(0x2b5f_0000L + 64L * (ExpertSource.this.laneBase + lane.local) + index));
                this.lane = lane;
                this.index = index;
            }

            @Override
            public void execute() {
                this.lane.readPart(this.index);
            }

            @Override
            public void doFinallyWithError(Throwable rejection) {
                this.lane.partFailure.compareAndSet(
                        null, new IllegalStateException("the lattice rejected an expert read", rejection));
                this.lane.partDone();
            }
        }

        /// Serial: the claim found a miss; the load frame is ready.
        void start(ExpertBlock block, int position, ExpertCacheShard.Load load) {
            this.busy = true;
            this.block = block;
            this.position = position;
            this.load = load;
            this.submittedSeen = false;
            this.retiredEarly = false;
            this.record = null;
            this.touched = false;
            this.failure = null;
            this.claimedAt = System.nanoTime();
            if (ExpertSource.this.tier == null) this.directive.clear();
            else ExpertSource.this.tier.plan(this.load.bank(), this.load.expert(), this.directive);
            if (this.parts.length > 1 && this.directive.mode() != TierDirective.Mode.HIT) {
                this.partFailure.set(null);
                this.pending.set(this.parts.length);
                for (Part part : this.parts) ready(part);
            } else ready(this);
        }

        /// Runs on a worker: one part of the artifact read.
        private void readPart(int index) {
            ExpertSource source = ExpertSource.this;
            if (index == 0) this.startedAt = System.nanoTime();
            try {
                source.cache
                        .store()
                        .readPart(
                                this.load.bank(),
                                this.load.expert(),
                                this.global,
                                this.directive,
                                index,
                                this.parts.length);
            } catch (Throwable thrown) {
                this.partFailure.compareAndSet(null, thrown);
            }
            partDone();
        }

        private void partDone() {
            if (this.pending.decrementAndGet() != 0) return;
            Throwable thrown = this.partFailure.get();
            try {
                if (thrown != null) throw thrown;
                this.record = ExpertSource.this
                        .cache
                        .store()
                        .completeOpen(this.load.bank(), this.load.expert(), this.global, this.directive);
                submitCopy();
            } catch (Throwable failed) {
                this.failure = failed;
                ExpertSource.this.post(this.failedReport);
            }
        }

        /// The record is addressable in the lane's staging slot: submit its copy and tell the owner.
        private void submitCopy() {
            ExpertSource source = ExpertSource.this;
            this.touched = true;
            this.openedAt = System.nanoTime();
            source.cache.transfer().stream(
                    this.global,
                    this.record,
                    this.load.deviceAddress(),
                    this.load.fence(),
                    this.load.readyMarker(),
                    this);
            this.submittedAt = System.nanoTime();
            source.post(this.submitted);
        }

        /// Runs on a worker of the lattice: the blocking read, then the copy's submission.
        @Override
        public void execute() {
            ExpertSource source = ExpertSource.this;
            this.startedAt = System.nanoTime();
            try {
                this.record =
                        source.cache.store().open(this.load.bank(), this.load.expert(), this.global, this.directive);
                submitCopy();
            } catch (Throwable thrown) {
                this.failure = thrown;
                source.post(this.failedReport);
            }
        }

        @Override
        public void doFinallyWithError(Throwable rejection) {
            this.failure = new IllegalStateException("the lattice rejected an expert load", rejection);
            post(this.failedReport);
        }

        /// The copy retired (a driver thread: it only posts).
        @Override
        public void retired(long ticket, boolean driverThread) {
            this.ticket = ticket;
            this.retiredAt = System.nanoTime();
            post(this.retired);
        }

        /// Serial.
        void handle(Kind kind) {
            switch (kind) {
                case SUBMITTED -> {
                    long now = System.nanoTime();
                    ExpertSource.this.loadCount.increment();
                    ExpertSource.this.dispatchNanos.add(this.startedAt - this.claimedAt);
                    ExpertSource.this.openNanos.add(this.openedAt - this.startedAt);
                    ExpertSource.this.submitNanos.add(now - this.submittedAt);
                    settleTier(true);
                    this.submittedSeen = true;
                    this.block.arrive(this.position, this.load.submitted());
                    if (this.retiredEarly) finish();
                }
                case RETIRED -> {
                    // The copy may retire before the message that says it was submitted is handled.
                    if (!this.submittedSeen) this.retiredEarly = true;
                    else finish();
                }
                case FAILED -> {
                    ExpertSource source = ExpertSource.this;
                    settleTier(this.touched);
                    if (this.touched) source.cache.transfer().recover(this.global, this.failure);
                    this.load.failed(this.failure);
                    this.block.failed(this.failure);
                    this.block.arrive(this.position, null);
                    release();
                }
            }
        }

        /// Serial: the load frame is done with its tier slot. A fill whose record was read (`read`) keeps
        /// it for the next miss; one that was not gives the slot back.
        private void settleTier(boolean read) {
            RamTierShard tier = ExpertSource.this.tier;
            if (tier == null) return;
            switch (this.directive.mode()) {
                case HIT -> tier.used(this.directive);
                case FILL -> {
                    if (read) tier.filled(this.directive);
                    else tier.abandoned(this.directive);
                }
                case NONE, BYPASS -> {}
            }
            this.directive.clear();
        }

        /// The copy retired and was submitted: the load is over and the lane is free.
        private void finish() {
            ExpertSource.this.copyNanos.add(this.retiredAt - this.submittedAt);
            ExpertSource.this.retireNanos.add(System.nanoTime() - this.retiredAt);
            Throwable device = ExpertSource.this.cache.transfer().confirm(this.global, this.ticket);
            this.load.retired(device);
            if (device != null) this.block.failed(device);
            release();
        }

        private void release() {
            this.busy = false;
            this.block = null;
            this.load = null;
            this.record = null;
            ExpertSource.this.inFlight--;
            advance();
        }
    }
}
