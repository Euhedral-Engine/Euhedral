package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.data_structures.queues.MpmcQueue;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.scheduling.frames.QwenStageFrame;
import io.euhedral_execution.inference.core.scheduling.graph.LanePool;
import io.euhedral_execution.inference.core.scheduling.graph.QwenExecutionSource;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraph;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

/// Admits Qwen quanta into reusable frame graphs and owns the Euhedral sources that run them.
///
/// Admission acquires an idle graph for the quantum's plan view, binds the graph's workspace storage and
/// prepares quantum-owned resources on that graph's stream, and publishes the root stages. After that it
/// is out of the execution path: the stages publish their successors, workers take the published frames
/// (first come, first served, or routed through the lattice by the frame's hash), and the quantum's
/// retirement frame recycles the graph, with its storage, before publishing the outcome. No central
/// scheduler decides where a stage runs.
///
/// Each graph publishes through its own source, attached to the lattice once when the graph is built.
/// Independent quanta therefore stay independent: a worker draining one graph's source never holds
/// another quantum's ready frames.
public final class EuhedralInferenceRuntime implements AutoCloseable {

    private static final Consumer<QwenExecutionContext> NO_TERMINAL_CONSUMER = ignored -> {};

    private final LatticeTerminal lattice;
    private final QwenExecutionPlan plan;
    private final ExecutionGpu gpu;
    private final ConcurrentHashMap<QwenExecutionPlan, GraphPool> pools = new ConcurrentHashMap<>();
    private final Object closeLock = new Object();
    private boolean closed;
    /// Device lanes shared by every graph, opened with the first graph.
    private LanePool lanes;
    private final Object laneLock = new Object();
    private final int laneCount;
    /// Held by the quantum that stages weights, from its admission until its last stage submitted.
    private final java.util.concurrent.Semaphore stagingHold = new java.util.concurrent.Semaphore(1);
    /// Recorded on the releasing quantum's home lane once its lanes joined; the next staging quantum's
    /// preparation awaits it, so its first copies follow every earlier read of the ring.
    private long stagingIdle;

    /// Decode, verification and draft quanta replay CUDA graphs captured from earlier quanta
    /// (docs/CUDA_GRAPHS.md); `EUHEDRAL_CUDA_GRAPHS=0` submits every quantum stage by stage instead.
    static final boolean CAPTURE_GRAPHS = !"0".equals(System.getenv("EUHEDRAL_CUDA_GRAPHS"));

    private final boolean captureGraphs;

    /// Lanes in the shared pool: one per available processor, at most [LanePool#MAX_LANES].
    static int laneCount() {
        return Math.min(Runtime.getRuntime().availableProcessors(), LanePool.MAX_LANES);
    }

    /// Opens the shared pool once. Streams open outside `closeLock`; a pool that finds the runtime closed
    /// releases its streams itself, because close() never saw it.
    private LanePool lanes() {
        synchronized (this.laneLock) {
            synchronized (this.closeLock) {
                if (this.lanes != null) return this.lanes;
            }
            boolean transfers = this.plan.staging() != null;
            int compute = transfers ? Math.min(this.laneCount, LanePool.MAX_LANES - 1) : this.laneCount;
            GpuStream[] streams = new GpuStream[compute + (transfers ? 1 : 0)];
            LanePool pool;
            try {
                for (int lane = 0; lane < streams.length; lane++) streams[lane] = this.gpu.openStream();
                if (transfers) {
                    pool = LanePool.withTransferLane(java.util.Arrays.copyOf(streams, compute), streams[compute]);
                    this.stagingIdle = streams[compute].openMarker();
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

    /// A runtime whose lane pool has one lane per available processor.
    public EuhedralInferenceRuntime(LatticeTerminal lattice, QwenExecutionPlan plan, ExecutionGpu gpu) {
        this(lattice, plan, gpu, laneCount());
    }

    /// A runtime whose graphs share `laneCount` device lanes.
    public EuhedralInferenceRuntime(LatticeTerminal lattice, QwenExecutionPlan plan, ExecutionGpu gpu, int laneCount) {
        this(lattice, plan, gpu, laneCount, CAPTURE_GRAPHS);
    }

    /// A runtime that captures decode, verification and draft quanta into CUDA graphs only with `captureGraphs`.
    EuhedralInferenceRuntime(
            LatticeTerminal lattice, QwenExecutionPlan plan, ExecutionGpu gpu, int laneCount, boolean captureGraphs) {
        this.captureGraphs = captureGraphs;
        this.lattice = Objects.requireNonNull(lattice, "lattice");
        this.plan = Objects.requireNonNull(plan, "plan").executionOwner();
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (laneCount < 1 || laneCount > LanePool.MAX_LANES)
            throw new IllegalArgumentException("laneCount must be 1 to " + LanePool.MAX_LANES);
        this.laneCount = laneCount;
    }

    /// Executes quanta and waits for all of their outcomes.
    public List<QwenExecutionContext.Outcome> execute(List<QwenExecutionContext> contexts)
            throws InterruptedException, ExecutionException {
        return execute(contexts, NO_TERMINAL_CONSUMER);
    }

    /// Executes quanta while their terminal callback can still inspect the live workspace.
    public List<QwenExecutionContext.Outcome> execute(
            List<QwenExecutionContext> contexts, Consumer<? super QwenExecutionContext> terminalConsumer)
            throws InterruptedException, ExecutionException {
        List<QwenExecutionContext> accepted = List.copyOf(Objects.requireNonNull(contexts, "contexts"));
        Objects.requireNonNull(terminalConsumer, "terminalConsumer");
        List<CompletableFuture<QwenExecutionContext.Outcome>> completions = new ArrayList<>(accepted.size());
        try {
            for (QwenExecutionContext context : accepted) completions.add(submit(context, terminalConsumer));
        } catch (RuntimeException | Error admissionFailure) {
            // Quanta admitted before the failure still own device work; wait for them to retire.
            for (var completion : completions) {
                try {
                    completion.join();
                } catch (RuntimeException ignored) {
                    // Their outcome is reported through their own futures.
                }
            }
            throw admissionFailure;
        }
        CompletableFuture.allOf(completions.toArray(CompletableFuture[]::new)).get();
        return completions.stream().map(CompletableFuture::join).toList();
    }

    public CompletableFuture<QwenExecutionContext.Outcome> submit(QwenExecutionContext context) {
        return submit(context, NO_TERMINAL_CONSUMER);
    }

    /// Admits one quantum. The returned future completes after its device work retired and the
    /// quantum's graph was recycled. A quantum is admitted at most once: when admission itself fails,
    /// the failure is thrown and the quantum's outcome is failed as well.
    public CompletableFuture<QwenExecutionContext.Outcome> submit(
            QwenExecutionContext context, Consumer<? super QwenExecutionContext> terminalConsumer) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(terminalConsumer, "terminalConsumer");
        QwenExecutionPlan view = context.plan();
        if (view.executionOwner() != this.plan) {
            throw new IllegalArgumentException("quantum belongs to another execution plan");
        }
        context.claim();
        GraphPool pool;
        PooledGraph pooled;
        try {
            this.gpu.ensureHealthy();
            pool = pool(view);
            pooled = pool.acquire();
        } catch (RuntimeException | Error failure) {
            context.fail(failure);
            context.finish();
            throw failure;
        }
        StageGraph graph = pooled.graph();
        try {
            graph.source().admit();
        } catch (RuntimeException | Error failure) {
            pool.recycle(pooled);
            context.fail(failure);
            context.finish();
            throw failure;
        }
        boolean started = false;
        if (view.stagesWeights()) {
            // Blocks only this admitting thread, until the previous staging quantum has submitted its
            // last stage; its device work is ordered by `stagingIdle`, not waited for here.
            this.stagingHold.acquireUninterruptibly();
            context.holdStaging(this::releaseStaging);
        }
        try {
            GpuStream stream = graph.stream();
            if (view.stagesWeights()) stream.await(this.stagingIdle);
            try {
                stream.submit(() -> context.begin(this.gpu, stream, terminalConsumer, pooled.storage()), false);
            } catch (RuntimeException | Error failure) {
                // The stream failed around the preparation; prove it idle before storage is released.
                stream.recover(failure);
                context.fail(failure);
                context.retire(null);
                throw failure;
            }
            if (!context.terminal()) {
                // From here the graph owns the quantum; its retirement recycles the graph.
                started = true;
                graph.start(context);
            }
            return context.completion().copy();
        } finally {
            if (!started) {
                context.lanesJoined(null);
                pool.recycle(pooled);
                graph.source().terminated();
                // Outcome callbacks never run with the graph's stream selected.
                context.publishOutcome();
            }
        }
    }

    /// Releases the staging ring. `home` has joined every lane of the releasing quantum; null when its
    /// lanes were proven idle instead.
    private void releaseStaging(GpuStream home) {
        try {
            if (home != null) home.mark(this.stagingIdle);
        } finally {
            this.stagingHold.release();
        }
    }

    private GraphPool pool(QwenExecutionPlan view) {
        GraphPool pool = this.pools.get(view);
        if (pool != null) return pool;
        synchronized (this.closeLock) {
            ensureOpen();
            return this.pools.computeIfAbsent(view, GraphPool::new);
        }
    }

    private void ensureOpen() {
        synchronized (this.closeLock) {
            if (this.closed) throw new IllegalStateException("inference runtime is closed");
        }
    }

    /// Stops admission, waits for every accepted quantum to retire, detaches the graphs' sources from
    /// Euhedral, and releases their streams.
    @Override
    public void close() {
        synchronized (this.closeLock) {
            if (this.closed) return;
            this.closed = true;
        }
        RuntimeException failure = null;
        for (GraphPool pool : this.pools.values()) {
            try {
                pool.completeSources();
            } catch (RuntimeException completionFailure) {
                if (failure == null) failure = completionFailure;
                else failure.addSuppressed(completionFailure);
            }
        }
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
            if (this.stagingIdle != 0) {
                try {
                    lanes.lane(lanes.transferLane()).closeMarker(this.stagingIdle);
                } catch (RuntimeException closeFailure) {
                    if (failure == null) failure = closeFailure;
                    else failure.addSuppressed(closeFailure);
                }
            }
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
        int active = 0;
        for (GraphPool pool : this.pools.values()) active += pool.activeQuanta();
        return active;
    }

    /// Whether any graph's source is still attached to Euhedral.
    public boolean isAttached() {
        for (GraphPool pool : this.pools.values()) if (pool.attached()) return true;
        return false;
    }

    /// A reusable graph and the workspace storage its quanta bind. Both are recycled together, only
    /// after the graph's quantum retired, so storage is never shared by two live quanta.
    private record PooledGraph(StageGraph graph, QwenWorkspaceStorage storage) {}

    /// Idle graphs of one plan view. Graphs are built only when every existing one is in use.
    private final class GraphPool {
        private final QwenExecutionPlan view;
        private final MpmcQueue<PooledGraph> idle = new MpmcQueue<>(16, 2);
        private final List<PooledGraph> built = new ArrayList<>();

        private GraphPool(QwenExecutionPlan view) {
            this.view = view;
        }

        PooledGraph acquire() {
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
            QwenWorkspaceStorage storage = new QwenWorkspaceStorage(gpu);
            PooledGraph[] pooled = new PooledGraph[1];
            List<QwenExecutionPlan.Instruction> instructions = this.view.instructions();
            StageGraph.StageFactory frames =
                    (owner, stage) -> QwenStageFrame.create(owner, instructions.get(stage), gpu);
            // Independent branches of every view spread over lanes: decode leaves the GPU idle between
            // dependent kernels, and a prefill side branch fills the tail waves of the chain's GEMMs.
            StageGraph graph =
                    EuhedralInferenceRuntime.this.captureGraphs && lanes.lane(0).capturesGraphs()
                            ? new StageGraph(
                                    this.view.stageTopology(),
                                    frames,
                                    lanes,
                                    true,
                                    new QwenExecutionSource(),
                                    retired -> recycle(pooled[0]),
                                    gpu::openStream)
                            : new StageGraph(
                                    this.view.stageTopology(),
                                    frames,
                                    lanes,
                                    true,
                                    new QwenExecutionSource(),
                                    retired -> recycle(pooled[0]));
            pooled[0] = new PooledGraph(graph, storage);
            synchronized (EuhedralInferenceRuntime.this.closeLock) {
                // A close that ran during this build saw no such graph; it would never release it.
                if (EuhedralInferenceRuntime.this.closed) {
                    graph.close();
                    throw new IllegalStateException("inference runtime is closed");
                }
                synchronized (this.built) {
                    this.built.add(pooled[0]);
                }
            }
            // Attached once per reusable graph, never per quantum. A close from here on completes it.
            EuhedralInferenceRuntime.this.lattice.addUpstream(graph.source());
            return pooled[0];
        }

        /// Closes each graph's admission, then waits until its accepted quantum retired and Euhedral
        /// detached the source. The first failure is reported after every graph was completed.
        void completeSources() {
            List<StageGraph> graphs;
            synchronized (this.built) {
                graphs = this.built.stream().map(PooledGraph::graph).toList();
            }
            RuntimeException failure = null;
            for (StageGraph graph : graphs) {
                try {
                    graph.source().completeGracefully();
                } catch (RuntimeException completionFailure) {
                    if (failure == null) failure = completionFailure;
                    else failure.addSuppressed(completionFailure);
                }
            }
            for (StageGraph graph : graphs) {
                try {
                    graph.source().awaitTermination();
                } catch (java.util.concurrent.CompletionException terminationFailure) {
                    // A downstream completion failure was already reported by completeGracefully.
                    if (failure == null) failure = terminationFailure;
                }
            }
            if (failure != null) throw failure;
        }

        int activeQuanta() {
            int active = 0;
            synchronized (this.built) {
                for (PooledGraph pooled : this.built)
                    active += pooled.graph().source().activeGraphs();
            }
            return active;
        }

        long replayedQuanta() {
            long replayed = 0;
            synchronized (this.built) {
                for (PooledGraph pooled : this.built) replayed += pooled.graph().replayedQuanta();
            }
            return replayed;
        }

        boolean attached() {
            synchronized (this.built) {
                for (PooledGraph pooled : this.built)
                    if (pooled.graph().source().isAttached()) return true;
            }
            return false;
        }

        long retainedWorkspaceBytes() {
            long bytes = 0;
            synchronized (this.built) {
                for (PooledGraph pooled : this.built) bytes += pooled.storage().retainedBytes();
            }
            return bytes;
        }

        /// Releases each retired graph's stream and storage. A GPU that cannot prove its submitted work
        /// stopped keeps the storage, which stays counted: queued kernels may still reference it.
        void close() {
            RuntimeException failure = null;
            boolean proven = EuhedralInferenceRuntime.this.gpu.completionProven();
            synchronized (this.built) {
                for (PooledGraph pooled : this.built) {
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
            }
            if (failure != null) throw failure;
        }
    }
}
