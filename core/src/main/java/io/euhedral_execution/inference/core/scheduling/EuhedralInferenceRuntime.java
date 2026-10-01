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
/// is out of the execution path: the stages publish their successors, Euhedral schedules every stage,
/// and the quantum's retirement frame recycles the graph, with its storage, before publishing the
/// outcome.
///
/// Each graph publishes through its own source, attached to the lattice once when the graph is built.
/// Independent quanta therefore stay independently schedulable: a worker draining one graph's source
/// never holds another quantum's ready frames.
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
    private final LanePool.Placement placement;

    /// Lanes in the shared pool: `EUHEDRAL_LANES`, by default one per available processor (at most 64).
    static int laneCount() {
        String configured = System.getenv("EUHEDRAL_LANES");
        int count = configured == null || configured.isBlank()
                ? Runtime.getRuntime().availableProcessors()
                : Integer.parseInt(configured.strip());
        if (count < 1) throw new IllegalArgumentException("EUHEDRAL_LANES must be positive");
        return Math.min(count, LanePool.MAX_LANES);
    }

    /// Stage placement over the lanes: `EUHEDRAL_LANE_PLACEMENT` (RANDOM, WORKER, CHAIN, FORK or PATH; PATH
    /// by default).
    static LanePool.Placement lanePlacement() {
        String configured = System.getenv("EUHEDRAL_LANE_PLACEMENT");
        return configured == null || configured.isBlank()
                ? LanePool.Placement.PATH
                : LanePool.Placement.valueOf(configured.strip().toUpperCase(java.util.Locale.ROOT));
    }

    /// Opens the shared pool once. Streams open outside `closeLock`; a pool that finds the runtime closed
    /// releases its streams itself, because close() never saw it.
    private LanePool lanes() {
        synchronized (this.laneLock) {
            synchronized (this.closeLock) {
                if (this.lanes != null) return this.lanes;
            }
            GpuStream[] streams = new GpuStream[this.laneCount];
            LanePool pool;
            try {
                for (int lane = 0; lane < streams.length; lane++) streams[lane] = this.gpu.openStream();
                pool = new LanePool(streams, this.placement);
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

    /// A runtime whose lane pool takes its size and placement from the environment.
    public EuhedralInferenceRuntime(LatticeTerminal lattice, QwenExecutionPlan plan, ExecutionGpu gpu) {
        this(lattice, plan, gpu, laneCount(), lanePlacement());
    }

    /// A runtime whose graphs share `laneCount` device lanes, placing stages by `placement`.
    public EuhedralInferenceRuntime(
            LatticeTerminal lattice,
            QwenExecutionPlan plan,
            ExecutionGpu gpu,
            int laneCount,
            LanePool.Placement placement) {
        this.lattice = Objects.requireNonNull(lattice, "lattice");
        this.plan = Objects.requireNonNull(plan, "plan").executionOwner();
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (laneCount < 1 || laneCount > LanePool.MAX_LANES)
            throw new IllegalArgumentException("laneCount must be 1 to " + LanePool.MAX_LANES);
        this.laneCount = laneCount;
        this.placement = Objects.requireNonNull(placement, "placement");
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
        try {
            GpuStream stream = graph.stream();
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
                pool.recycle(pooled);
                graph.source().terminated();
                // Outcome callbacks never run with the graph's stream selected.
                context.publishOutcome();
            }
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
            StageGraph graph = new StageGraph(
                    this.view.stageTopology(),
                    (owner, stage) -> QwenStageFrame.create(owner, instructions.get(stage), gpu),
                    lanes,
                    // Independent branches of every view spread over lanes: decode leaves the GPU idle
                    // between dependent kernels, and a prefill side branch fills the tail waves of the
                    // chain's GEMMs.
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
