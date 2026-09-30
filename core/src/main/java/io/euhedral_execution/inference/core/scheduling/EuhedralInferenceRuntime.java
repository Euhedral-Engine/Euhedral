package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.data_structures.queues.MpmcQueue;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.scheduling.frames.QwenStageFrame;
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
/// Admission acquires an idle graph for the quantum's plan view, prepares quantum-owned resources on
/// that graph's stream, and publishes the root stages. After that it is out of the execution path:
/// the stages publish their successors, Euhedral schedules every stage, and the quantum's retirement
/// frame recycles the graph before publishing the outcome.
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

    public EuhedralInferenceRuntime(LatticeTerminal lattice, QwenExecutionPlan plan, ExecutionGpu gpu) {
        this.lattice = Objects.requireNonNull(lattice, "lattice");
        this.plan = Objects.requireNonNull(plan, "plan").executionOwner();
        this.gpu = Objects.requireNonNull(gpu, "gpu");
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
    /// quantum's graph was recycled.
    public CompletableFuture<QwenExecutionContext.Outcome> submit(
            QwenExecutionContext context, Consumer<? super QwenExecutionContext> terminalConsumer) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(terminalConsumer, "terminalConsumer");
        QwenExecutionPlan view = context.plan();
        if (view.executionOwner() != this.plan) {
            throw new IllegalArgumentException("quantum belongs to another execution plan");
        }
        this.gpu.ensureHealthy();
        GraphPool pool = pool(view);
        StageGraph graph = pool.acquire();
        try {
            graph.source().admit();
        } catch (RuntimeException | Error failure) {
            pool.recycle(graph);
            throw failure;
        }
        boolean started = false;
        try {
            GpuStream stream = graph.stream();
            try {
                stream.submit(() -> context.begin(this.gpu, stream, terminalConsumer), false);
            } catch (RuntimeException | Error failure) {
                if (!(failure instanceof QwenExecutionContext.DuplicateAdmissionException)
                        && !context.completion().isDone()) {
                    stream.recover(failure);
                    context.fail(failure);
                    context.finish();
                }
                throw failure;
            }
            if (!context.completion().isDone()) {
                // From here the graph owns the quantum; its retirement recycles the graph.
                started = true;
                graph.start(context);
            }
            return context.completion().copy();
        } finally {
            if (!started) {
                pool.recycle(graph);
                graph.source().terminated();
            }
        }
    }

    private GraphPool pool(QwenExecutionPlan view) {
        GraphPool pool = this.pools.get(view);
        if (pool != null) return pool;
        synchronized (this.closeLock) {
            if (this.closed) throw new IllegalStateException("inference runtime is closed");
            return this.pools.computeIfAbsent(view, GraphPool::new);
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
        if (failure != null) throw failure;
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

    /// Idle graphs of one plan view. Graphs are built only when every existing one is in use.
    private final class GraphPool {
        private final QwenExecutionPlan view;
        private final MpmcQueue<StageGraph> idle = new MpmcQueue<>(16, 2);
        private final List<StageGraph> built = new ArrayList<>();

        private GraphPool(QwenExecutionPlan view) {
            this.view = view;
        }

        StageGraph acquire() {
            StageGraph graph = this.idle.poll();
            return graph != null ? graph : build();
        }

        void recycle(StageGraph graph) {
            if (!this.idle.offer(graph)) throw new IllegalStateException("idle graph pool rejected a graph");
        }

        private StageGraph build() {
            synchronized (EuhedralInferenceRuntime.this.closeLock) {
                if (EuhedralInferenceRuntime.this.closed)
                    throw new IllegalStateException("inference runtime is closed");
            }
            ExecutionGpu gpu = EuhedralInferenceRuntime.this.gpu;
            GpuStream stream = gpu.openStream();
            StageGraph graph;
            try {
                List<QwenExecutionPlan.Instruction> instructions = this.view.instructions();
                graph = new StageGraph(
                        this.view.stageTopology(),
                        (owner, stage) -> QwenStageFrame.create(owner, instructions.get(stage), gpu),
                        stream,
                        new QwenExecutionSource(),
                        this::recycle);
            } catch (RuntimeException | Error failure) {
                try {
                    stream.close();
                } catch (RuntimeException | Error closeFailure) {
                    failure.addSuppressed(closeFailure);
                }
                throw failure;
            }
            synchronized (this.built) {
                this.built.add(graph);
            }
            // Attached once per reusable graph, never per quantum.
            EuhedralInferenceRuntime.this.lattice.addUpstream(graph.source());
            synchronized (EuhedralInferenceRuntime.this.closeLock) {
                // A close that raced this build has already completed every source it saw.
                if (EuhedralInferenceRuntime.this.closed) graph.source().completeGracefully();
            }
            return graph;
        }

        /// Closes each graph's admission, then waits until its accepted quantum retired and Euhedral
        /// detached the source. The first failure is reported after every graph was completed.
        void completeSources() {
            List<StageGraph> graphs;
            synchronized (this.built) {
                graphs = List.copyOf(this.built);
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
                for (StageGraph graph : this.built) active += graph.source().activeGraphs();
            }
            return active;
        }

        boolean attached() {
            synchronized (this.built) {
                for (StageGraph graph : this.built) if (graph.source().isAttached()) return true;
            }
            return false;
        }

        void close() {
            RuntimeException failure = null;
            synchronized (this.built) {
                for (StageGraph graph : this.built) {
                    try {
                        graph.close();
                    } catch (RuntimeException closeFailure) {
                        if (failure == null) failure = closeFailure;
                        else failure.addSuppressed(closeFailure);
                    }
                }
                this.built.clear();
            }
            if (failure != null) throw failure;
        }
    }
}
