package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;

/// Device lanes shared by every stage graph of a runtime.
///
/// A stage chooses its lane each time it runs, so independent frames spread over many device
/// orderings and the worker threads that submit them rarely share one. A stage continues the lane of
/// the predecessor whose longest remaining path runs through it (or its graph's home lane for a root),
/// so each graph's critical chain keeps one lane in every quantum; the other branches of a fan-out take
/// random lanes. A dependency between stages that ran on different lanes is ordered on the device by a
/// marker; stages on the same lane rely on that lane's order.
public final class LanePool implements AutoCloseable {

    /// Markers record which lanes a quantum used in one 64-bit mask.
    public static final int MAX_LANES = 64;

    private final GpuStream[] lanes;
    /// Lanes that stages choose among; a transfer lane, when present, follows them.
    private final int computeLanes;
    private final AtomicInteger nextHome = new AtomicInteger();
    private final java.util.function.IntUnaryOperator fixed;
    /// Markers of closed graphs that later graphs may still await (their stages' and their joins'): kept, and
    /// reused by the graphs built after, instead of destroyed while an external edge may still name one. Awaiting
    /// one a newer graph re-recorded waits for that graph's earlier work too, never for anything after the wait.
    private final java.util.concurrent.ConcurrentLinkedQueue<Long> spareMarkers =
            new java.util.concurrent.ConcurrentLinkedQueue<>();
    private boolean closed;

    public LanePool(GpuStream[] lanes) {
        this(lanes, null);
    }

    /// A pool whose last lane is reserved for host-to-device transfers: only transfer stages use it, so
    /// copies queue behind copies and never behind compute.
    public static LanePool withTransferLane(GpuStream[] compute, GpuStream transfer) {
        GpuStream[] lanes = java.util.Arrays.copyOf(compute, compute.length + 1);
        lanes[compute.length] = Objects.requireNonNull(transfer, "transfer");
        return new LanePool(lanes, null, compute.length);
    }

    /// Test placement: stage `s` always runs on lane `fixed.applyAsInt(s)`.
    LanePool(GpuStream[] lanes, java.util.function.IntUnaryOperator fixed) {
        this(lanes, fixed, Objects.requireNonNull(lanes, "lanes").length);
    }

    private LanePool(GpuStream[] lanes, java.util.function.IntUnaryOperator fixed, int computeLanes) {
        Objects.requireNonNull(lanes, "lanes");
        if (lanes.length == 0 || lanes.length > MAX_LANES || computeLanes < 1 || computeLanes > lanes.length) {
            throw new IllegalArgumentException("a lane pool needs 1 to " + MAX_LANES + " lanes");
        }
        this.computeLanes = computeLanes;
        this.lanes = lanes.clone();
        for (GpuStream lane : this.lanes) Objects.requireNonNull(lane, "lane");
        this.fixed = fixed;
    }

    /// A pool of one lane: every stage of every graph keeps that lane's order.
    public static LanePool single(GpuStream stream) {
        return new LanePool(new GpuStream[] {stream});
    }

    public int size() {
        return this.lanes.length;
    }

    /// The lane reserved for transfer stages, or -1.
    public int transferLane() {
        return this.computeLanes < this.lanes.length ? this.computeLanes : -1;
    }

    public GpuStream lane(int lane) {
        return this.lanes[lane];
    }

    /// A marker another graph's external edge may name: a spare one of a closed graph, or a new one on `lane`.
    long openSharedMarker(int lane) {
        Long spare = this.spareMarkers.poll();
        return spare != null ? spare : this.lanes[lane].openMarker();
    }

    /// Keeps a closed graph's marker that another graph's external edge may name, for reuse.
    void retireSharedMarker(long marker) {
        if (marker != 0) this.spareMarkers.add(marker);
    }

    /// The home lane for a newly built graph: its quantum's preparation and retirement boundary.
    int nextHome() {
        return Math.floorMod(this.nextHome.getAndIncrement(), this.computeLanes);
    }

    /// The lane for `stage`, given the lane it may continue (-1 for none): its path predecessor's, or the
    /// graph's home lane for a root.
    int choose(int stage, int continued) {
        if (this.computeLanes == 1) return 0;
        if (this.fixed != null) return this.fixed.applyAsInt(stage);
        if (continued >= this.computeLanes) continued = -1;
        return continued >= 0 ? continued : ThreadLocalRandom.current().nextInt(this.computeLanes);
    }

    /// Closes every lane. Only after every graph using the pool has retired its work and closed.
    @Override
    public void close() {
        if (this.closed) return;
        RuntimeException failure = null;
        for (Long spare = this.spareMarkers.poll(); spare != null; spare = this.spareMarkers.poll()) {
            try {
                this.lanes[0].closeMarker(spare);
            } catch (RuntimeException closeFailure) {
                if (failure == null) failure = closeFailure;
                else failure.addSuppressed(closeFailure);
            }
        }
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
