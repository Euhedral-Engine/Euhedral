package io.euhedral_execution.inference.core.scheduling.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.ArrayDeque;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// One reusable runtime instance of a static stage DAG.
///
/// The graph is built once: its stage frames, their successor references, and one frame per
/// device-completion edge are wired at construction and rebound to each quantum.
///
/// Stages submit to the lanes of a [LanePool], choosing a lane each time they run. The graph's home
/// lane prepares the quantum and carries its retirement boundary. Every stage with successors records
/// a reusable marker after submitting; a successor that runs on another lane awaits it on the device,
/// and a root on another lane awaits the quantum's preparation marker. Before the retirement boundary
/// every other lane the quantum used joins the home lane the same way, so that boundary covers all of
/// the quantum's device work.
///
/// Admission publishes only the root stages. A stage that submitted successfully satisfies its
/// outgoing edges; a submission edge is satisfied at once and a device-completion edge when the
/// producer's work retires. The final arrival at a successor publishes it to the source. The graph
/// never runs a stage, scans for ready work, or walks its topology after admission.
///
/// `live` counts published frames that have not finished plus armed device-completion edges. When it
/// reaches zero, no stage of this quantum can run again, whether it succeeded, failed, or was
/// cancelled. The graph then arms the quantum's single retirement boundary. The retirement frame
/// confirms it, runs each attempted stage's retirement hook and the quantum's terminal work, and
/// returns the graph to its recycler before publishing the outcome.
public final class StageGraph implements AutoCloseable {

    /// Creates the frame for one stage while the graph is built.
    @FunctionalInterface
    public interface StageFactory {
        StageFrame create(StageGraph graph, int stage);
    }

    /// Receives a graph whose quantum has retired, for the next quantum.
    @FunctionalInterface
    public interface Recycler {
        void recycle(StageGraph graph);
    }

    private static final Logger LOG = LoggerFactory.getLogger(StageGraph.class);
    private static final long NO_TICKET = 0L;

    private final StageTopology topology;
    private final LanePool pool;
    private final boolean ownsPool;
    private final int home;
    private final long prepared;
    private final long[] tails;
    private final AtomicLong usedLanes = new AtomicLong();
    private final boolean spread;
    private final QwenExecutionSource source;
    private final Recycler recycler;
    private final long routingSeed = ThreadLocalRandom.current().nextLong();
    private final StageFrame[] stages;
    private final StageFrame[] roots;
    private final Retirement retirement;
    private final AtomicInteger live = new AtomicInteger();
    private StageQuantum quantum;
    private boolean overlap;

    /// Builds a graph that owns one stream: every stage keeps that stream's order.
    public StageGraph(
            StageTopology topology,
            StageFactory factory,
            GpuStream stream,
            QwenExecutionSource source,
            Recycler recycler) {
        this(
                topology,
                factory,
                source,
                recycler,
                LanePool.single(Objects.requireNonNull(stream, "stream")),
                true,
                true);
    }

    /// Builds a graph whose stages run on the lanes of a shared `pool`, which outlives the graph. With
    /// `spread`, stages are placed over the pool's lanes; otherwise every stage keeps the graph's home
    /// lane, so the graph still has its own device ordering but no cross-lane edges.
    public StageGraph(
            StageTopology topology,
            StageFactory factory,
            LanePool pool,
            boolean spread,
            QwenExecutionSource source,
            Recycler recycler) {
        this(topology, factory, source, recycler, pool, false, spread);
    }

    private StageGraph(
            StageTopology topology,
            StageFactory factory,
            QwenExecutionSource source,
            Recycler recycler,
            LanePool pool,
            boolean ownsPool,
            boolean spread) {
        this.spread = spread;
        this.topology = Objects.requireNonNull(topology, "topology");
        this.pool = Objects.requireNonNull(pool, "pool");
        this.ownsPool = ownsPool;
        this.home = pool.nextHome();
        this.tails = new long[pool.size()];
        this.source = Objects.requireNonNull(source, "source");
        this.recycler = Objects.requireNonNull(recycler, "recycler");
        Objects.requireNonNull(factory, "factory");
        this.stages = new StageFrame[topology.size()];
        for (int stage = 0; stage < this.stages.length; stage++) {
            StageFrame frame = Objects.requireNonNull(factory.create(this, stage), "stage frame");
            if (frame.graph() != this || frame.stage() != stage) {
                throw new IllegalArgumentException("stage factory returned a frame for another stage");
            }
            this.stages[stage] = frame;
        }
        int[] predecessorCounts = new int[this.stages.length];
        for (int stage = 0; stage < this.stages.length; stage++) {
            int[] submitted = topology.submittedSuccessors(stage);
            StageFrame[] successors = new StageFrame[submitted.length];
            for (int index = 0; index < submitted.length; index++) {
                successors[index] = this.stages[submitted[index]];
                predecessorCounts[submitted[index]]++;
            }
            this.stages[stage].submittedSuccessors = successors;
            int[] retired = topology.retiredSuccessors(stage);
            RetiredEdge[] edges = new RetiredEdge[retired.length];
            for (int index = 0; index < retired.length; index++) {
                edges[index] = new RetiredEdge(this, this.stages[stage], this.stages[retired[index]]);
            }
            this.stages[stage].retiredEdges = edges;
        }
        for (int stage = 0; stage < this.stages.length; stage++) {
            this.stages[stage].submittedPredecessors = new StageFrame[predecessorCounts[stage]];
            predecessorCounts[stage] = 0;
        }
        for (StageFrame producer : this.stages) {
            for (StageFrame successor : producer.submittedSuccessors) {
                successor.submittedPredecessors[predecessorCounts[successor.stage()]++] = producer;
            }
        }
        markPaths(this.stages);
        long preparedMarker = 0;
        try {
            if (pool.size() > 1) {
                preparedMarker = pool.lane(this.home).openMarker();
                for (StageFrame stage : this.stages) {
                    if (stage.submittedSuccessors.length > 0)
                        stage.marker = pool.lane(this.home).openMarker();
                }
                for (int lane = 0; lane < this.tails.length; lane++) {
                    if (lane != this.home) this.tails[lane] = pool.lane(lane).openMarker();
                }
            }
        } catch (RuntimeException | Error failure) {
            this.prepared = preparedMarker;
            closeMarkers();
            throw failure;
        }
        this.prepared = preparedMarker;
        int[] rootStages = topology.roots();
        this.roots = new StageFrame[rootStages.length];
        for (int index = 0; index < rootStages.length; index++) this.roots[index] = this.stages[rootStages[index]];
        this.retirement = new Retirement(this);
    }

    public StageTopology topology() {
        return this.topology;
    }

    /// The home lane: the device ordering that prepares and retires the bound quantum.
    public GpuStream stream() {
        return this.pool.lane(this.home);
    }

    LanePool pool() {
        return this.pool;
    }

    int home() {
        return this.home;
    }

    /// Whether stages are placed over the pool's lanes; otherwise they all run on the home lane.
    /// Links each stage to the successor that continues its longest submitted path to a sink (the first
    /// listed on ties), so lane placement keeps a graph's critical chain on one lane.
    private static void markPaths(StageFrame[] stages) {
        int[] height = new int[stages.length];
        int[] pending = new int[stages.length];
        ArrayDeque<StageFrame> ready = new ArrayDeque<>();
        for (StageFrame stage : stages) {
            pending[stage.stage()] = stage.submittedSuccessors.length;
            if (pending[stage.stage()] == 0) ready.add(stage);
        }
        while (!ready.isEmpty()) {
            StageFrame stage = ready.poll();
            StageFrame next = null;
            for (StageFrame successor : stage.submittedSuccessors) {
                // A transfer never continues a compute chain, and no chain continues a transfer's lane.
                if (successor.transfers()) continue;
                if (next == null || height[successor.stage()] > height[next.stage()]) next = successor;
            }
            height[stage.stage()] = next == null ? 1 : height[next.stage()] + 1;
            if (next != null && !stage.transfers()) next.pathPredecessor = stage;
            for (StageFrame predecessor : stage.submittedPredecessors) {
                if (--pending[predecessor.stage()] == 0) ready.add(predecessor);
            }
        }
    }

    boolean spread() {
        return this.spread;
    }

    long prepared() {
        return this.prepared;
    }

    void used(int lane) {
        long bit = 1L << lane;
        if ((this.usedLanes.get() & bit) == 0) this.usedLanes.getAndUpdate(mask -> mask | bit);
    }

    /// Proves every lane this quantum used idle after a failed submission or boundary registration.
    void recover(Throwable failure) {
        long lanes = this.usedLanes.get() | (1L << this.home);
        while (lanes != 0) {
            int lane = Long.numberOfTrailingZeros(lanes);
            lanes &= lanes - 1;
            this.pool.lane(lane).recover(failure);
        }
    }

    /// The quantum currently bound to this graph.
    public StageQuantum quantum() {
        return this.quantum;
    }

    public StageFrame stage(int stage) {
        return this.stages[stage];
    }

    long routingSeed() {
        return this.routingSeed;
    }

    boolean overlapLaunches() {
        return this.overlap;
    }

    boolean stopRequested() {
        return this.quantum.stopRequested();
    }

    /// Binds a quantum whose resources are already prepared and publishes the root stages.
    public void start(StageQuantum quantum) {
        Objects.requireNonNull(quantum, "quantum");
        if (this.quantum != null) throw new IllegalStateException("stage graph already runs a quantum");
        this.quantum = quantum;
        this.overlap = quantum.overlapLaunches();
        for (StageFrame stage : this.stages) stage.reset();
        this.retirement.reset();
        this.usedLanes.set(1L << this.home);
        // Roots on other lanes order behind the preparation already submitted to the home lane.
        if (this.prepared != 0) this.pool.lane(this.home).mark(this.prepared);
        // Admission holds one count so that fast roots cannot retire the quantum before all publish.
        this.live.set(this.roots.length + 1);
        for (StageFrame root : this.roots) this.source.publish(root);
        stageFinished();
    }

    /// Publishes the successors that `stage` makes ready after its successful submission.
    void release(StageFrame stage) {
        try {
            for (StageFrame successor : stage.submittedSuccessors) {
                if (stopRequested()) break;
                if (successor.arrive()) {
                    this.live.incrementAndGet();
                    this.source.publish(successor);
                }
            }
            for (RetiredEdge edge : stage.retiredEdges) {
                if (stopRequested()) break;
                edge.arm();
            }
        } catch (RuntimeException | Error failure) {
            fail(failure);
        } finally {
            stageFinished();
        }
    }

    void stageFailed(Throwable failure) {
        fail(failure);
        stageFinished();
    }

    void fail(Throwable failure) {
        this.quantum.fail(failure);
    }

    /// Drops one live count. Callers must not touch their frame afterwards: the last drop arms the
    /// retirement boundary, after which the graph and its frames may be rebound to another quantum.
    void stageFinished() {
        if (this.live.decrementAndGet() == 0) quiesce();
    }

    void addLive() {
        this.live.incrementAndGet();
    }

    /// The source through which this graph's frames reach Euhedral.
    public QwenExecutionSource source() {
        return this.source;
    }

    /// No stage of this quantum can run again. Arms the quantum's single device-completion boundary.
    private void quiesce() {
        Retirement terminal = this.retirement;
        GpuStream home = this.pool.lane(this.home);
        try {
            // Every other lane this quantum used joins the home lane on the device.
            long others = this.usedLanes.get() & ~(1L << this.home);
            while (others != 0) {
                int lane = Long.numberOfTrailingZeros(others);
                others &= others - 1;
                this.pool.lane(lane).mark(this.tails[lane]);
                home.await(this.tails[lane]);
            }
            this.quantum.lanesJoined(home);
            home.notifyRetired(terminal);
        } catch (RuntimeException | Error failure) {
            this.quantum.fail(failure);
            recover(failure);
            this.quantum.lanesJoined(null);
            this.source.publish(terminal);
        }
    }

    /// Terminal work for the bound quantum; runs on an ordinary worker after device retirement. Every
    /// failure before recycling is recorded on the quantum, whose outcome is always published.
    void retire(long ticket) {
        StageQuantum retiring = this.quantum;
        Throwable deviceFailure =
                ticket == NO_TICKET ? null : this.pool.lane(this.home).confirmRetired(ticket);
        if (deviceFailure != null) retiring.fail(deviceFailure);
        boolean committed = !retiring.stopRequested();
        for (StageFrame stage : this.stages) {
            if (!stage.attempted) continue;
            try {
                stage.retired(committed && stage.submitted);
            } catch (RuntimeException | Error failure) {
                retiring.fail(failure);
                committed = false;
            }
        }
        try {
            retiring.retire(deviceFailure);
        } catch (RuntimeException | Error failure) {
            retiring.fail(failure);
        }
        // The next quantum may reuse this graph before this one's outcome is observed.
        this.quantum = null;
        this.recycler.recycle(this);
        // The quantum stops counting as active before its outcome becomes visible.
        try {
            this.source.terminated();
        } finally {
            retiring.publishOutcome();
        }
    }

    /// Releases the graph's markers, and its stream when it owns one. Only an unbound graph whose work
    /// has retired may be closed.
    @Override
    public void close() {
        if (this.quantum != null) throw new IllegalStateException("stage graph still runs a quantum");
        closeMarkers();
        if (this.ownsPool) this.pool.close();
    }

    private void closeMarkers() {
        GpuStream any = this.pool.lane(this.home);
        for (StageFrame stage : this.stages) {
            if (stage == null || stage.marker == 0) continue;
            any.closeMarker(stage.marker);
            stage.marker = 0;
        }
        for (int lane = 0; lane < this.tails.length; lane++) {
            if (this.tails[lane] == 0) continue;
            any.closeMarker(this.tails[lane]);
            this.tails[lane] = 0;
        }
        if (this.prepared != 0) any.closeMarker(this.prepared);
    }

    /// The single device-completion boundary of a quantum. The driver callback only publishes it.
    ///
    /// Its `execute` never throws: the graph may serve another quantum as soon as it is recycled, so
    /// `doFinallyWithError` means only that the lattice rejected the frame without running it.
    static final class Retirement extends AbstractFrame implements GpuStream.RetirementListener {
        private final StageGraph graph;
        private long ticket;

        Retirement(StageGraph graph) {
            super(0L);
            this.graph = graph;
            randomizeHash(graph.routingSeed);
        }

        void reset() {
            this.ticket = NO_TICKET;
        }

        @Override
        public void retired(long ticket, boolean driverThread) {
            this.ticket = ticket;
            if (driverThread) this.graph.source.publishFromCallback(this);
            else this.graph.source.publish(this);
        }

        @Override
        public void execute() {
            try {
                this.graph.retire(this.ticket);
            } catch (RuntimeException | Error failure) {
                // Only admission accounting can fail here, after the outcome was published.
                LOG.error("Qwen quantum retirement failed", failure);
            }
        }

        /// The graph may already serve another quantum; this frame's state is not touched again.
        @Override
        public void doFinally() {}

        /// The lattice rejected the frame without running it: the worker's cache retired, or no
        /// downstream was routable. The quantum still retires exactly once, and the rejecting thread is an
        /// ordinary worker or admission thread, never a driver callback: retire it here.
        @Override
        public void doFinallyWithError(Throwable rejection) {
            execute();
        }
    }

    /// A device-completion edge. Its producer arms it after submission; the driver callback publishes
    /// it, and an ordinary worker confirms retirement before satisfying the consumer's edge.
    static final class RetiredEdge extends AbstractFrame implements GpuStream.RetirementListener {
        private final StageGraph graph;
        private final StageFrame producer;
        private final StageFrame consumer;
        private GpuStream stream;
        private long ticket;

        RetiredEdge(StageGraph graph, StageFrame producer, StageFrame consumer) {
            super(0L);
            this.graph = graph;
            this.producer = producer;
            this.consumer = consumer;
            randomizeHash(graph.routingSeed);
        }

        /// Arms the boundary on the lane the producer submitted to.
        void arm() {
            this.graph.addLive();
            this.stream = this.graph.pool.lane(this.producer.lane);
            try {
                this.stream.notifyRetired(this);
            } catch (RuntimeException | Error failure) {
                this.graph.fail(failure);
                this.graph.recover(failure);
                this.graph.stageFinished();
            }
        }

        @Override
        public void retired(long ticket, boolean driverThread) {
            this.ticket = ticket;
            if (driverThread) this.graph.source.publishFromCallback(this);
            else this.graph.source.publish(this);
        }

        @Override
        public void execute() {
            Throwable failure = this.stream.confirmRetired(this.ticket);
            if (failure != null) {
                this.graph.fail(failure);
                return;
            }
            if (!this.graph.stopRequested() && this.consumer.arrive()) {
                this.graph.addLive();
                this.graph.source.publish(this.consumer);
            }
        }

        @Override
        public void doFinally() {
            this.graph.stageFinished();
        }

        /// `execute` never throws, so the lattice rejected this frame without running it. The producer's
        /// boundary must still be confirmed and the edge resolved once; the rejecting thread is never a
        /// driver callback, so the edge is finished here.
        @Override
        public void doFinallyWithError(Throwable rejection) {
            execute();
            doFinally();
        }
    }
}
