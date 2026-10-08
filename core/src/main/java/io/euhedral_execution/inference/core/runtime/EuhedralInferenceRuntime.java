package io.euhedral_execution.inference.core.runtime;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.data_structures.queues.MpmcQueue;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.runtime.graph.FrameLake;
import io.euhedral_execution.inference.core.runtime.graph.GraphShape;
import io.euhedral_execution.inference.core.runtime.graph.GraphStorage;
import io.euhedral_execution.inference.core.runtime.graph.InferenceLake;
import io.euhedral_execution.inference.core.runtime.graph.LanePool;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;
import io.euhedral_execution.inference.core.runtime.graph.StageQuantum;
import io.euhedral_execution.inference.core.runtime.graph.WorkspaceOwner;
import io.euhedral_execution.inference.core.runtime.graph.WorkspaceUse;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// Admits quanta of any model into reusable frame graphs.
///
/// Admission acquires an idle graph of the quantum's shape, binds the graph's storage, prepares the quantum on the
/// graph's home lane, and publishes the root stages into the lake. After that it is out of the execution path: the
/// stages publish their successors, workers take the published frames (first come, first served, or routed through
/// the lattice by the frame's hash), and the quantum's retirement frame recycles the graph, with its storage, before
/// publishing the outcome. No central scheduler decides where a stage runs. The runtime knows no model: a model's
/// plan supplies the shapes and quanta, and the model's runtime owns the lake and the host work.
public final class EuhedralInferenceRuntime implements AutoCloseable {

    /// Ingest sinks of a lake (upstream sources of the lattice) and the producer partitions of each.
    public static final int LAKE_SINKS = Integer.parseInt(System.getenv().getOrDefault("EUHEDRAL_LAKE_SINKS", "4"));
    public static final int LAKE_PARTITIONS =
            Integer.parseInt(System.getenv().getOrDefault("EUHEDRAL_LAKE_PARTITIONS", "2"));

    /// Decode, verification and draft quanta replay CUDA graphs captured from earlier quanta
    /// (docs/CUDA_GRAPHS.md); `EUHEDRAL_CUDA_GRAPHS=0` submits every quantum stage by stage instead.
    public static final boolean CAPTURE_GRAPHS = !"0".equals(System.getenv("EUHEDRAL_CUDA_GRAPHS"));

    /// The device lanes of a runtime: `compute` lanes the stages spread over, an optional lane reserved for
    /// host-to-device weight transfers, and whether quanta that carry a capture key are captured into CUDA graphs.
    public record Lanes(int compute, boolean transferLane, boolean captureGraphs) {
        public Lanes {
            if (compute < 1 || compute > LanePool.MAX_LANES - (transferLane ? 1 : 0))
                throw new IllegalArgumentException("compute lanes must be 1 to " + LanePool.MAX_LANES);
        }

        /// `compute` lanes, no transfer lane, no capture.
        public static Lanes of(int compute) {
            return new Lanes(compute, false, false);
        }
    }

    private final ExecutionGpu gpu;
    private static final Logger LOG = LoggerFactory.getLogger(EuhedralInferenceRuntime.class);
    private final ConcurrentHashMap<GraphShape, GraphPool> pools = new ConcurrentHashMap<>();
    /// The workspace's owner: confined to frames ordered on [WorkspaceOwner#HASH], as every admission is.
    private final WorkspaceOwner owner;
    private final Object closeLock = new Object();
    /// The pool of ready work the graphs' stages and the driver callbacks throw frames into. The model's runtime owns
    /// it, with the host work that also publishes into it.
    private final InferenceLake lake;
    /// What the graphs publish through: the lake, with this runtime's quanta counted.
    private final QuantumLake quanta = new QuantumLake();
    private volatile boolean closed;
    /// Device lanes shared by every graph, opened with the first graph.
    private volatile LanePool lanes;
    private final Object laneLock = new Object();
    private final Lanes lanesConfig;

    /// Lanes in the shared pool: one per available processor, at most [LanePool#MAX_LANES].
    public static int laneCount() {
        return Math.min(Runtime.getRuntime().availableProcessors(), LanePool.MAX_LANES);
    }

    /// A lake of the default shape (sinks and producer partitions) attached to `lattice`.
    public static InferenceLake newLake(LatticeTerminal lattice) {
        return new InferenceLake(lattice, LAKE_SINKS, LAKE_PARTITIONS);
    }

    /// A runtime that runs the [GraphShape]s its callers admit through [#admit] on `lanes`, throwing every ready
    /// frame into `lake`, which the caller owns, attaches and completes after closing this runtime.
    public EuhedralInferenceRuntime(InferenceLake lake, ExecutionGpu gpu, Lanes lanes) {
        this(lake, gpu, lanes, 0);
    }

    /// A runtime whose quanta share a workspace of `workspaceBuffers` buffers, the reuse of which its owner orders
    /// across graphs ([WorkspaceOwner]).
    public EuhedralInferenceRuntime(InferenceLake lake, ExecutionGpu gpu, Lanes lanes, int workspaceBuffers) {
        this.lake = Objects.requireNonNull(lake, "lake");
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.lanesConfig = Objects.requireNonNull(lanes, "lanes");
        this.owner = new WorkspaceOwner(workspaceBuffers);
    }

    /// The workspace's owner. Frames ordered on [WorkspaceOwner#HASH] only.
    public WorkspaceOwner workspaceOwner() {
        return this.owner;
    }

    /// The shared lanes, opened by the first graph built.
    private LanePool lanes() {
        LanePool open = this.lanes;
        return open != null ? open : openLanes();
    }

    /// Lifecycle: opens the shared pool once. Streams open outside `closeLock`; a pool that finds the runtime
    /// closed releases its streams itself, because close() never saw it.
    private LanePool openLanes() {
        synchronized (this.laneLock) {
            synchronized (this.closeLock) {
                if (this.lanes != null) return this.lanes;
            }
            boolean transfers = this.lanesConfig.transferLane();
            int compute = this.lanesConfig.compute();
            GpuStream[] streams = new GpuStream[compute + (transfers ? 1 : 0)];
            LanePool pool;
            try {
                for (int lane = 0; lane < streams.length; lane++) streams[lane] = this.gpu.openStream();
                if (transfers) {
                    pool = LanePool.withTransferLane(java.util.Arrays.copyOf(streams, compute), streams[compute]);
                } else pool = new LanePool(streams);
            } catch (RuntimeException | Error failure) {
                for (GpuStream stream : streams) {
                    if (stream == null) continue;
                    try {
                        stream.close();
                    } catch (RuntimeException | Error closeFailure) {
                        failure.addSuppressed(closeFailure);
                    }
                }
                throw failure;
            }
            synchronized (this.closeLock) {
                if (!this.closed) {
                    this.lanes = pool;
                    return pool;
                }
            }
            pool.close();
            throw new IllegalStateException("inference runtime is closed");
        }
    }

    /// Prepares an admitted quantum on the graph's home stream, with that stream selected, so that
    /// whatever it queues precedes every stage. Returns whether the quantum proceeds to its stages;
    /// otherwise it has prepared its terminal outcome, which the runtime publishes once the stream
    /// is no longer selected.
    @FunctionalInterface
    public interface Preparation {
        boolean prepare(GpuStream stream, GraphStorage storage);
    }

    /// Runs `admission` on the workspace's owner, for a caller outside a frame ordered on it (tests and tools,
    /// or a completion chain): publishes an [Admission] whose `execute` runs it. `admission` calls [#admit]; a
    /// refusal it throws is logged, because the runtime already published the quantum's failure through its
    /// continuation.
    public void publishOnOwner(Runnable admission) {
        // A lake that no longer takes frames (it is closing) runs the admission here, which then refuses it.
        this.lake.publishOrRun(new Admission(admission));
    }

    /// One admission run on the workspace's owner: its `idHash` is [WorkspaceOwner#HASH] and it stays ordered, so
    /// admissions run one at a time. Not necessarily in the order they were published: the lake's queues are
    /// partitioned ([#LAKE_PARTITIONS], more than one by default), so only frames one admission publishes after it
    /// ran are ordered behind it.
    static final class Admission extends AbstractFrame {
        private final Runnable admission;

        Admission(Runnable admission) {
            super(WorkspaceOwner.HASH);
            this.admission = Objects.requireNonNull(admission, "admission");
        }

        @Override
        public void execute() {
            try {
                this.admission.run();
            } catch (RuntimeException | Error refused) {
                LOG.debug("an admission was refused; its quantum's continuation carries the failure", refused);
            }
        }

        /// A rejected owner frame still admits (and so still concludes) its quantum.
        @Override
        public void doFinallyWithError(Throwable rejection) {
            execute();
        }
    }

    /// Admits one quantum of any shape: acquires an idle graph of `shape` (building one when every
    /// graph is in use), runs `prepare` on its home stream, binds the graph behind the
    /// last accessors of the workspace buffers it touches, and publishes its root stages. After that
    /// the runtime is out of the execution path: stages publish their successors, workers run them,
    /// and the quantum's retirement recycles the graph before the outcome is published. A quantum is
    /// admitted at most once: when admission itself fails the quantum is retired as failed and the
    /// failure is thrown.
    ///
    /// Runs only on frames ordered on [WorkspaceOwner#HASH] (the generation's `Admit`, or an [Admission]): the
    /// workspace's owner state is confined to them.
    public void admit(GraphShape shape, StageQuantum quantum, Preparation prepare) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(quantum, "quantum");
        Objects.requireNonNull(prepare, "prepare");
        GraphPool pool;
        PooledGraph pooled;
        try {
            this.gpu.ensureHealthy();
            pool = pool(shape);
            pooled = pool.idleOrNew();
        } catch (RuntimeException | Error failure) {
            quantum.lanesJoined(null);
            quantum.fail(failure);
            quantum.concludeUnstarted();
            throw failure;
        }
        StageGraph graph = pooled.graph();
        try {
            this.quanta.admit();
        } catch (RuntimeException | Error failure) {
            pool.recycle(pooled);
            quantum.lanesJoined(null);
            quantum.fail(failure);
            quantum.concludeUnstarted();
            throw failure;
        }
        boolean started = false;
        try {
            GpuStream stream = graph.stream();
            boolean[] proceeds = new boolean[1];
            try {
                stream.submit(() -> proceeds[0] = prepare.prepare(stream, pooled.storage()), false);
            } catch (RuntimeException | Error failure) {
                // The stream failed around the preparation; prove it idle before storage is released.
                stream.recover(failure);
                quantum.fail(failure);
                throw failure;
            }
            if (proceeds[0]) {
                // From here the graph owns the quantum; its retirement recycles the graph.
                started = true;
                this.owner.bind(graph, shape, pool.use, quantum);
            }
        } finally {
            if (!started) {
                quantum.lanesJoined(null);
                // The quantum concludes in its owner's order; the graph and its storage stay held until its
                // retirement released the workspace binding.
                quantum.concludeInOrder(() -> {
                    try {
                        quantum.retire(null);
                    } finally {
                        pool.recycle(pooled);
                        this.quanta.terminated();
                        // Outcome callbacks never run with the graph's stream selected.
                        quantum.publishOutcome();
                    }
                });
            }
        }
    }

    private GraphPool pool(GraphShape view) {
        GraphPool pool = this.pools.get(view);
        return pool != null ? pool : openPool(view);
    }

    /// Lifecycle: the first graph pool of a shape.
    private GraphPool openPool(GraphShape view) {
        synchronized (this.closeLock) {
            ensureOpen();
            return this.pools.computeIfAbsent(view, GraphPool::new);
        }
    }

    private void ensureOpen() {
        if (this.closed) throw new IllegalStateException("inference runtime is closed");
    }

    /// Stops admission, waits for every accepted quantum to retire, and releases the graphs, their storage and the
    /// lanes. The lake stays the owner's to complete.
    @Override
    public void close() {
        synchronized (this.closeLock) {
            if (this.closed) return;
            this.closed = true;
        }
        RuntimeException failure = null;
        // Every accepted quantum retires on the lattice before its graph, workspace and lanes are released.
        this.quanta.awaitIdle();
        for (GraphPool pool : this.pools.values()) {
            try {
                pool.close();
            } catch (RuntimeException closeFailure) {
                if (failure == null) failure = closeFailure;
                else failure.addSuppressed(closeFailure);
            }
        }
        LanePool lanes;
        synchronized (this.closeLock) {
            lanes = this.lanes;
        }
        if (lanes != null) {
            try {
                lanes.close();
            } catch (RuntimeException closeFailure) {
                if (failure == null) failure = closeFailure;
                else failure.addSuppressed(closeFailure);
            }
        }
        if (failure != null) throw failure;
    }

    /// Device bytes the graphs' reusable workspace storage retains between quanta.
    public long retainedWorkspaceBytes() {
        long bytes = 0;
        for (GraphPool pool : this.pools.values()) bytes += pool.retainedWorkspaceBytes();
        return bytes;
    }

    /// Quanta replayed from captured CUDA graphs.
    public long replayedQuanta() {
        long replayed = 0;
        for (GraphPool pool : this.pools.values()) replayed += pool.replayedQuanta();
        return replayed;
    }

    /// Admitted quanta whose graphs have not yet retired.
    public int activeQuanta() {
        return this.quanta.active();
    }

    /// Whether the lake's sinks are still attached to the lattice.
    public boolean isAttached() {
        return this.lake.isAttached();
    }

    /// The runtime's lake: the root source of the lattice, which every producer throws ready frames
    /// into.
    public InferenceLake lake() {
        return this.lake;
    }

    /// The lake as the graphs see it: publishing goes to the lake, and the quanta they carry are
    /// counted so that a closing runtime can wait for them whatever else the lake carries.
    private final class QuantumLake implements FrameLake {
        private static final int CLOSED = Integer.MIN_VALUE;
        private final java.util.concurrent.atomic.AtomicInteger count = new java.util.concurrent.atomic.AtomicInteger();
        private final CompletableFuture<Void> idle = new CompletableFuture<>();

        @Override
        public void publish(AbstractFrame frame) {
            EuhedralInferenceRuntime.this.lake.publish(frame);
        }

        @Override
        public void publishFromCallback(AbstractFrame frame) {
            EuhedralInferenceRuntime.this.lake.publishFromCallback(frame);
        }

        @Override
        public void admit() {
            EuhedralInferenceRuntime.this.lake.admit();
            this.count.incrementAndGet();
        }

        @Override
        public void admitDuringDrain() {
            EuhedralInferenceRuntime.this.lake.admitDuringDrain();
            this.count.incrementAndGet();
        }

        @Override
        public void terminated() {
            try {
                EuhedralInferenceRuntime.this.lake.terminated();
            } finally {
                if (this.count.decrementAndGet() == CLOSED) this.idle.complete(null);
            }
        }

        int active() {
            return this.count.get() & Integer.MAX_VALUE;
        }

        /// Waits for every admitted quantum to terminate; new quanta are refused in the meantime by
        /// the runtime's closed flag.
        void awaitIdle() {
            int old = this.count.getAndUpdate(value -> value | CLOSED);
            if ((old & Integer.MAX_VALUE) == 0) this.idle.complete(null);
            this.idle.join();
        }
    }

    /// A reusable graph and the workspace storage its quanta bind. Both are recycled together, only
    /// after the graph's quantum retired, so storage is never shared by two live quanta.
    /// A built graph and its storage; released once, by whichever of the runtime's close and a build that raced it
    /// claims it.
    private static final class PooledGraph {
        private final StageGraph graph;
        private final GraphStorage storage;
        private final AtomicBoolean released = new AtomicBoolean();

        PooledGraph(StageGraph graph, GraphStorage storage) {
            this.graph = graph;
            this.storage = storage;
        }

        StageGraph graph() {
            return this.graph;
        }

        GraphStorage storage() {
            return this.storage;
        }

        boolean claimRelease() {
            return this.released.compareAndSet(false, true);
        }
    }

    /// Idle graphs of one shape. Graphs are built only when every existing one is in use.
    private final class GraphPool {
        private final GraphShape view;
        /// Each buffer's first and last accessors in this shape, computed once.
        final WorkspaceUse use;
        private final MpmcQueue<PooledGraph> idle = new MpmcQueue<>(16, 2);
        private final ConcurrentLinkedQueue<PooledGraph> built = new ConcurrentLinkedQueue<>();

        private GraphPool(GraphShape view) {
            this.view = view;
            this.use = WorkspaceUse.of(view);
        }

        PooledGraph idleOrNew() {
            PooledGraph graph = this.idle.poll();
            return graph != null ? graph : build();
        }

        void recycle(PooledGraph graph) {
            if (!this.idle.offer(graph)) throw new IllegalStateException("idle graph pool rejected a graph");
        }

        private PooledGraph build() {
            ensureOpen();
            ExecutionGpu gpu = EuhedralInferenceRuntime.this.gpu;
            LanePool lanes = lanes();
            GraphStorage storage = this.view.newStorage(gpu);
            PooledGraph[] pooled = new PooledGraph[1];
            StageGraph.StageFactory frames = (owner, stage) -> this.view.createStage(owner, stage, gpu);
            // Independent branches of every view spread over lanes: decode leaves the GPU idle between
            // dependent kernels, and a prefill side branch fills the tail waves of the chain's GEMMs.
            StageGraph graph = EuhedralInferenceRuntime.this.lanesConfig.captureGraphs()
                            && lanes.lane(0).capturesGraphs()
                    ? new StageGraph(
                            this.view.topology(),
                            frames,
                            lanes,
                            true,
                            EuhedralInferenceRuntime.this.quanta,
                            retired -> recycle(pooled[0]),
                            gpu::openStream)
                    : new StageGraph(
                            this.view.topology(),
                            frames,
                            lanes,
                            true,
                            EuhedralInferenceRuntime.this.quanta,
                            retired -> recycle(pooled[0]));
            pooled[0] = new PooledGraph(graph, storage);
            this.built.add(pooled[0]);
            // A close that ran during this build may not have seen the graph: whichever claims it releases it.
            if (EuhedralInferenceRuntime.this.closed && pooled[0].claimRelease()) {
                this.built.remove(pooled[0]);
                graph.close();
                throw new IllegalStateException("inference runtime is closed");
            }
            return pooled[0];
        }

        long replayedQuanta() {
            long replayed = 0;
            for (PooledGraph pooled : this.built) replayed += pooled.graph().replayedQuanta();
            return replayed;
        }

        long retainedWorkspaceBytes() {
            long bytes = 0;
            for (PooledGraph pooled : this.built) bytes += pooled.storage().retainedBytes();
            return bytes;
        }

        /// Releases each retired graph's stream and storage. A GPU that cannot prove its submitted
        /// work stopped keeps the storage, which stays counted: queued kernels may still reference
        /// it.
        void close() {
            RuntimeException failure = null;
            boolean proven = EuhedralInferenceRuntime.this.gpu.completionProven();
            for (PooledGraph pooled : this.built) {
                if (!pooled.claimRelease()) continue;
                try {
                    pooled.graph().close();
                } catch (RuntimeException closeFailure) {
                    if (failure == null) failure = closeFailure;
                    else failure.addSuppressed(closeFailure);
                }
                if (!proven) continue;
                try {
                    pooled.storage().close();
                } catch (RuntimeException closeFailure) {
                    if (failure == null) failure = closeFailure;
                    else failure.addSuppressed(closeFailure);
                }
            }
            this.built.removeIf(pooled -> pooled.storage().isClosed());
            if (failure != null) throw failure;
        }
    }
}
