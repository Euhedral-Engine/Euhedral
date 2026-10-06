package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCache;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertLease;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.HostRecord;
import io.euhedral_execution.inference.core.scheduling.graph.FrameSeeds;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraph;
import java.util.concurrent.atomic.AtomicInteger;

/// One expert of a layer's MoE block, as a frame that the plan stage spawns.
///
/// The block's experts form a stream: expert `p` runs on lane `p % lanes`, one expert at a time per
/// lane, and the lanes spread over the lattice's workers. An item finds its expert in the cache (a
/// resident one is held at once) or reads the record into a pinned staging slot, submits its
/// host-to-device copy on its lane's own copy stream, and ends its part of the wave: the lease it
/// holds carries the marker the copy records, so the wave's kernels wait for the bytes on the
/// device and no host thread waits for them. Only when the copy retired (a driver callback that
/// enqueues a second frame of the same lane) is the staging slot free again, and the lane's next
/// expert becomes runnable.
///
/// An item is runnable when its incoming edges arrived: its lane's previous expert ended (the
/// staging slot is free) and the wave `window` before its own was submitted (the cache's slots are
/// free). Both bounds therefore hold by construction, and nothing queues. Items are recycled
/// through a [FrameManager]; they share one id hash and a lane routes by the seed of the lane, so a
/// lane stays on one worker.
final class ExpertItem extends AbstractFrame {

    /// What the load stage of a wave offers its items: how many leases to expect, and each arrival.
    interface Arrivals {
        void expect(int count);

        void arrived();
    }

    private static final long LANE_SEED = 0x5eed_1a4eL;

    private final AtomicInteger gate = new AtomicInteger();
    private final AtomicInteger ends = new AtomicInteger();
    private final ExpertCache.Ticket ticket = new ExpertCache.Ticket();
    private final Retire retire = new Retire();

    private StageGraph graph;
    private Qwen4Quantum quantum;
    private Qwen4GraphStorage storage;
    private Qwen4ExecutionPlan plan;
    private Arrivals arrivals;
    private ExpertItem next;
    private int bank;
    private int expert;
    private int position;
    private int wave;
    private int lane;
    private boolean ran;
    private ExpertCache.Load load;
    private HostRecord record;
    private volatile long copyTicket;

    private ExpertItem(FrameManager<ExpertItem, ExpertItem> pool) {
        super(FrameSeeds.ID_HASH, pool, null);
    }

    /// An item from `pool` (which the one spawning thread at a time owns) or a new one.
    static ExpertItem obtain(FrameManager<ExpertItem, ExpertItem> pool, long password) {
        ExpertItem item = pool.get(password);
        return item != null ? item : new ExpertItem(pool);
    }

    /// Readies the item for expert `expert` of `bank` at `position` of the block, on `lane`, which
    /// runs after `gates` arrivals.
    void bind(
            StageGraph graph,
            Qwen4Quantum quantum,
            Qwen4GraphStorage storage,
            Qwen4ExecutionPlan plan,
            Arrivals arrivals,
            int bank,
            int expert,
            int position,
            int wave,
            int lane,
            int gates) {
        this.graph = graph;
        this.quantum = quantum;
        this.storage = storage;
        this.plan = plan;
        this.arrivals = arrivals;
        this.bank = bank;
        this.expert = expert;
        this.position = position;
        this.wave = wave;
        this.lane = lane;
        this.next = null;
        this.ran = false;
        this.ends.set(1);
        this.gate.set(gates);
        resetHash();
        randomizeHash(LANE_SEED + lane);
        this.retire.resetHash();
        this.retire.randomizeHash(LANE_SEED + lane);
    }

    /// `following` runs on this item's lane after this item ended.
    void linkTo(ExpertItem following) {
        this.next = following;
    }

    int gate() {
        return this.gate.get();
    }

    /// One incoming edge arrived; the last one makes the item runnable.
    void arrive() {
        if (this.gate.decrementAndGet() == 0) publish();
    }

    /// Makes the item runnable now (it has no incoming edge left to wait for).
    void publish() {
        try {
            this.graph.spawn(this, false);
        } catch (RuntimeException | Error failure) {
            // Never published: it can neither run nor wait, so it ends here.
            this.quantum.fail(failure);
            this.ran = true;
            this.arrivals.arrived();
            end();
        }
    }

    @Override
    public void execute() {
        this.ran = true;
        try {
            if (!this.quantum.stopRequested()) {
                ExpertLease lease;
                this.plan.expertCache().claim(this.bank, this.expert, this.ticket);
                lease = this.ticket.lease();
                if (lease == null) {
                    this.load = this.ticket.load();
                    lease = submit();
                }
                try {
                    this.storage.moe().hold(this.position, lease);
                } catch (RuntimeException | Error held) {
                    lease.close();
                    throw held;
                }
            }
        } catch (Throwable failure) {
            this.quantum.fail(failure);
        }
        this.arrivals.arrived();
    }

    /// Reads the record and submits its copy on the lane. Returns the lease, which carries the
    /// copy's marker.
    private ExpertLease submit() throws Exception {
        boolean touched = false;
        try {
            this.record = this.load.open();
            // This frame's end and the retire frame's end both finish the item.
            this.ends.incrementAndGet();
            touched = true;
            try {
                return this.load.submit(this.lane, this.record, this.retire);
            } catch (Throwable notArmed) {
                this.ends.decrementAndGet();
                throw notArmed;
            }
        } catch (Throwable failure) {
            HostRecord opened = this.record;
            this.record = null;
            ExpertCache.Load failed = this.load;
            this.load = null;
            failed.abandon(this.lane, opened, failure, touched);
            throw failure;
        }
    }

    @Override
    public void doFinally() {
        end();
    }

    /// The lattice rejected the frame without running it.
    @Override
    public void doFinallyWithError(Throwable rejection) {
        if (!this.ran) {
            this.quantum.fail(new IllegalStateException("the lattice rejected an expert load", rejection));
            this.arrivals.arrived();
        }
        end();
    }

    private void end() {
        if (this.ends.decrementAndGet() != 0) return;
        StageGraph owner = this.graph;
        Qwen4GraphStorage block = this.storage;
        int window = this.plan.window();
        int ofWave = this.wave;
        ExpertItem following = this.next;
        this.graph = null;
        this.quantum = null;
        this.storage = null;
        this.plan = null;
        this.arrivals = null;
        this.next = null;
        this.load = null;
        this.record = null;
        recycle();
        // The next expert of the lane may run now that this one's staging slot is free.
        if (following != null) following.arrive();
        // The wave's slots are free once all its items ended.
        block.itemEnded(ofWave, window);
        owner.finishChild();
    }

    /// The copy retired: the staging slot is free and the load is over.
    private final class Retire extends AbstractFrame implements GpuStream.RetirementListener {

        Retire() {
            super(FrameSeeds.ID_HASH);
        }

        @Override
        public void retired(long ticket, boolean driverThread) {
            ExpertItem.this.copyTicket = ticket;
            try {
                if (driverThread) ExpertItem.this.graph.lake().publishFromCallback(this);
                else ExpertItem.this.graph.lake().publish(this);
            } catch (RuntimeException | Error lost) {
                // The graph cannot run it: finish here so that the quantum can still retire.
                finishLoad();
                end();
            }
        }

        @Override
        public void execute() {
            finishLoad();
        }

        private void finishLoad() {
            ExpertItem item = ExpertItem.this;
            try {
                item.load.retire(item.lane, item.copyTicket);
            } catch (Throwable failure) {
                item.quantum.fail(failure);
            }
            HostRecord staged = item.record;
            try {
                if (staged != null) staged.close();
            } catch (RuntimeException closeFailure) {
                item.quantum.fail(closeFailure);
            }
        }

        @Override
        public void doFinally() {
            end();
        }

        @Override
        public void doFinallyWithError(Throwable rejection) {
            finishLoad();
            end();
        }
    }
}
