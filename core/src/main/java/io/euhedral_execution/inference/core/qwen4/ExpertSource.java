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

        private Lane(int local) {
            // The lane's hash is its identity: every load of the lane is routed to the same worker.
            super(HasherApi.mix(0x1a4e_0000L + ExpertSource.this.laneBase + local));
            this.local = local;
            this.global = ExpertSource.this.laneBase + local;
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
            if (ExpertSource.this.tier == null) this.directive.clear();
            else ExpertSource.this.tier.plan(this.load.bank(), this.load.expert(), this.directive);
            ready(this);
        }

        /// Runs on a worker of the lattice: the blocking read, then the copy's submission.
        @Override
        public void execute() {
            ExpertSource source = ExpertSource.this;
            try {
                this.record =
                        source.cache.store().open(this.load.bank(), this.load.expert(), this.global, this.directive);
                this.touched = true;
                source.cache.transfer().stream(
                        this.global,
                        this.record,
                        this.load.deviceAddress(),
                        this.load.fence(),
                        this.load.readyMarker(),
                        this);
                source.post(this.submitted);
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
            post(this.retired);
        }

        /// Serial.
        void handle(Kind kind) {
            switch (kind) {
                case SUBMITTED -> {
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
