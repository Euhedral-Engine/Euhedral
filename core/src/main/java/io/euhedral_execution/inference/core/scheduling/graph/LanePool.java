package io.euhedral_execution.inference.core.scheduling.graph;

import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;

/// Device lanes shared by every stage graph of a runtime.
///
/// A stage chooses its lane each time it runs, so independent frames spread over many device
/// orderings and the worker threads that submit them rarely share one. A dependency between stages
/// that ran on different lanes is ordered on the device by a marker; stages on the same lane rely on
/// that lane's order.
public final class LanePool implements AutoCloseable {

    /// How a stage chooses its lane.
    public enum Placement {
        /// A uniformly random lane for every stage.
        RANDOM,
        /// The lane of the submitting worker thread (one lane per worker, modulo the pool size).
        WORKER,
        /// A stage that only continues a linear chain (one predecessor whose only successor it is) stays
        /// on its predecessor's lane; every other stage takes a random lane.
        CHAIN,
        /// The first successor of a stage to run continues that stage's lane; the other branches of a
        /// fan-out fork to random lanes. A join continues the lane of a predecessor no other successor
        /// continued. Cross-lane edges then occur only where branches split or meet.
        FORK,
        /// A stage continues the lane of the predecessor whose longest remaining path runs through it, so
        /// each graph's critical chain keeps one lane in every quantum; the other branches of a fan-out
        /// take random lanes. Static, so no successor races for a lane.
        PATH
    }

    /// Markers record which lanes a quantum used in one 64-bit mask.
    public static final int MAX_LANES = 64;

    private final GpuStream[] lanes;
    /// Lanes that stages choose among; a transfer lane, when present, follows them.
    private final int computeLanes;
    private final Placement placement;
    private final AtomicInteger nextWorker = new AtomicInteger();
    private final AtomicInteger nextHome = new AtomicInteger();
    private final ThreadLocal<int[]> workerLane = ThreadLocal.withInitial(() -> new int[] {-1});
    private final java.util.function.IntUnaryOperator fixed;
    private boolean closed;

    public LanePool(GpuStream[] lanes, Placement placement) {
        this(lanes, placement, null);
    }

    /// A pool whose last lane is reserved for host-to-device transfers: only transfer stages use it, so
    /// copies queue behind copies and never behind compute.
    public static LanePool withTransferLane(GpuStream[] compute, GpuStream transfer, Placement placement) {
        GpuStream[] lanes = java.util.Arrays.copyOf(compute, compute.length + 1);
        lanes[compute.length] = Objects.requireNonNull(transfer, "transfer");
        return new LanePool(lanes, placement, null, compute.length);
    }

    /// Test placement: stage `s` always runs on lane `fixed.applyAsInt(s)`.
    LanePool(GpuStream[] lanes, Placement placement, java.util.function.IntUnaryOperator fixed) {
        this(lanes, placement, fixed, Objects.requireNonNull(lanes, "lanes").length);
    }

    private LanePool(
            GpuStream[] lanes, Placement placement, java.util.function.IntUnaryOperator fixed, int computeLanes) {
        Objects.requireNonNull(lanes, "lanes");
        if (lanes.length == 0 || lanes.length > MAX_LANES || computeLanes < 1 || computeLanes > lanes.length) {
            throw new IllegalArgumentException("a lane pool needs 1 to " + MAX_LANES + " lanes");
        }
        this.computeLanes = computeLanes;
        this.lanes = lanes.clone();
        for (GpuStream lane : this.lanes) Objects.requireNonNull(lane, "lane");
        this.placement = Objects.requireNonNull(placement, "placement");
        this.fixed = fixed;
    }

    /// A pool of one lane: every stage of every graph keeps that lane's order.
    public static LanePool single(GpuStream stream) {
        return new LanePool(new GpuStream[] {stream}, Placement.CHAIN);
    }

    public int size() {
        return this.lanes.length;
    }

    /// The lane reserved for transfer stages, or -1.
    public int transferLane() {
        return this.computeLanes < this.lanes.length ? this.computeLanes : -1;
    }

    public Placement placement() {
        return this.placement;
    }

    public GpuStream lane(int lane) {
        return this.lanes[lane];
    }

    /// The home lane for a newly built graph: its quantum's preparation and retirement boundary.
    int nextHome() {
        return Math.floorMod(this.nextHome.getAndIncrement(), this.computeLanes);
    }

    /// The lane for `stage`, given the lane it may continue (-1 for none): its chain predecessor's
    /// under CHAIN, an unclaimed predecessor's (or the graph's home lane for a root) under FORK, its
    /// path predecessor's (or the home lane for a root) under PATH.
    int choose(int stage, int continued) {
        if (this.computeLanes == 1) return 0;
        if (this.fixed != null) return this.fixed.applyAsInt(stage);
        if (continued >= this.computeLanes) continued = -1;
        return switch (this.placement) {
            case RANDOM -> ThreadLocalRandom.current().nextInt(this.computeLanes);
            case WORKER -> worker();
            case CHAIN, FORK, PATH ->
                continued >= 0 ? continued : ThreadLocalRandom.current().nextInt(this.computeLanes);
        };
    }

    private int worker() {
        int[] slot = this.workerLane.get();
        if (slot[0] < 0) slot[0] = Math.floorMod(this.nextWorker.getAndIncrement(), this.computeLanes);
        return slot[0];
    }

    /// Closes every lane. Only after every graph using the pool has retired its work and closed.
    @Override
    public void close() {
        if (this.closed) return;
        RuntimeException failure = null;
        for (GpuStream lane : this.lanes) {
            try {
                lane.close();
            } catch (RuntimeException closeFailure) {
                if (failure == null) failure = closeFailure;
                else failure.addSuppressed(closeFailure);
            }
        }
        this.closed = true;
        if (failure != null) throw failure;
    }
}
