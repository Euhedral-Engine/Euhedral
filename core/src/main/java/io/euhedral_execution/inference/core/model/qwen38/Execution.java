package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.prefix.PrefixFrames;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.runtime.HostTasks;
import io.euhedral_execution.inference.core.runtime.graph.InferenceLake;
import io.euhedral_execution.inference.core.runtime.graph.LanePool;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;
import java.util.function.Supplier;

/// Runs the dense model's quanta: the runtime it admits them to, the lake that runtime publishes into, the host
/// work around a request, and the weight staging ring's hand-over between quanta.
///
/// Quanta whose view stages weights hold the ring from admission until their last stage submitted (a later change
/// replaces this hold with frames routed to the ring's owner).
public final class Execution implements AutoCloseable {

    private static final Consumer<Quantum> NO_TERMINAL_CONSUMER = ignored -> {};

    private final ExecutionPlan plan;
    private final ExecutionGpu gpu;
    private final InferenceLake lake;
    private final HostTasks hostTasks;
    private final EuhedralInferenceRuntime runtime;
    /// Held by the quantum that stages weights, from its admission until its last stage submitted.
    private final java.util.concurrent.Semaphore stagingHold = new java.util.concurrent.Semaphore(1);
    /// Whether the staging ring holds the decode view's first slots ([ExecutionPlan#prefetchesRing]). Read and
    /// written only under `stagingHold`.
    private boolean ringPreloaded;
    private final Object closeLock = new Object();
    private boolean closed;

    /// An execution whose lane pool has one lane per available processor.
    public Execution(LatticeTerminal lattice, ExecutionPlan plan, ExecutionGpu gpu) {
        this(lattice, plan, gpu, EuhedralInferenceRuntime.laneCount());
    }

    /// An execution whose graphs share `laneCount` device lanes.
    public Execution(LatticeTerminal lattice, ExecutionPlan plan, ExecutionGpu gpu, int laneCount) {
        this(lattice, plan, gpu, laneCount, EuhedralInferenceRuntime.CAPTURE_GRAPHS);
    }

    /// An execution that captures decode, verification and draft quanta into CUDA graphs only with `captureGraphs`.
    /// A plan that stages weights gets one lane fewer for compute and a lane for its transfers.
    public Execution(
            LatticeTerminal lattice, ExecutionPlan plan, ExecutionGpu gpu, int laneCount, boolean captureGraphs) {
        this.plan = Objects.requireNonNull(plan, "plan").executionOwner();
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (laneCount < 1 || laneCount > LanePool.MAX_LANES)
            throw new IllegalArgumentException("laneCount must be 1 to " + LanePool.MAX_LANES);
        boolean transfers = this.plan.staging() != null;
        int compute = transfers ? Math.min(laneCount, LanePool.MAX_LANES - 1) : laneCount;
        this.lake = EuhedralInferenceRuntime.newLake(lattice);
        this.hostTasks = new HostTasks(this.lake);
        this.runtime = new EuhedralInferenceRuntime(
                this.lake, gpu, new EuhedralInferenceRuntime.Lanes(compute, transfers, captureGraphs));
    }

    /// Executes quanta and waits for all of their outcomes.
    public List<Quantum.Outcome> execute(List<Quantum> contexts) throws InterruptedException, ExecutionException {
        return execute(contexts, NO_TERMINAL_CONSUMER);
    }

    /// Executes quanta while their terminal callback can still inspect the live workspace.
    public List<Quantum.Outcome> execute(List<Quantum> contexts, Consumer<? super Quantum> terminalConsumer)
            throws InterruptedException, ExecutionException {
        List<Quantum> accepted = List.copyOf(Objects.requireNonNull(contexts, "contexts"));
        Objects.requireNonNull(terminalConsumer, "terminalConsumer");
        List<CompletableFuture<Quantum.Outcome>> completions = new ArrayList<>(accepted.size());
        try {
            for (Quantum context : accepted) completions.add(submit(context, terminalConsumer));
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

    public CompletableFuture<Quantum.Outcome> submit(Quantum context) {
        return submit(context, NO_TERMINAL_CONSUMER);
    }

    /// Admits one quantum. The returned future completes after its device work retired and the
    /// quantum's graph was recycled. A quantum is admitted at most once: when admission itself
    /// fails, the failure is thrown and the quantum's outcome is failed as well.
    public CompletableFuture<Quantum.Outcome> submit(Quantum context, Consumer<? super Quantum> terminalConsumer) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(terminalConsumer, "terminalConsumer");
        ExecutionPlan view = context.plan();
        if (view.executionOwner() != this.plan) {
            throw new IllegalArgumentException("quantum belongs to another execution plan");
        }
        context.claim();
        boolean staging = view.stagesWeights();
        if (staging) {
            // Blocks only this admitting thread, until the previous staging quantum has submitted its
            // last stage; its device work is ordered by the transfer marker, not waited for here. A decode view
            // that finds its first slots prefetched runs without their transfers.
            this.stagingHold.acquireUninterruptibly();
            if (this.ringPreloaded) view = view.preloadedVariant();
            this.ringPreloaded = false;
            boolean prefetches = view.prefetchesRing();
            context.holdStaging(
                    home -> releaseStaging(home, prefetches && home != null && !context.hasFailureOrCancellation()));
        }
        boolean ordered = staging;
        this.runtime.admit(
                view,
                context,
                ordered ? stream -> stream.await(this.runtime.transferMarker()) : null,
                (stream, storage) -> context.begin(this.gpu, stream, terminalConsumer, (WorkspaceStorage) storage));
        return context.completion().copy();
    }

    /// Releases the staging ring. `home` has joined every lane of the releasing quantum; null when its lanes
    /// were proven idle instead. `preloaded`: every stage of a prefetching quantum ran, so the ring holds the
    /// decode view's first slots once the transfer marker is reached.
    private void releaseStaging(GpuStream home, boolean preloaded) {
        try {
            if (home != null) home.mark(this.runtime.transferMarker());
            this.ringPreloaded = preloaded;
        } finally {
            this.stagingHold.release();
        }
    }

    /// Tokenizes `text` on the lattice's workers (PromptTokenization), with the BOS/EOS tokens of
    /// tokenizer_config.json when `modelSpecialTokens`. The future completes on a worker.
    public CompletableFuture<int[]> tokenize(QwenTokenizer tokenizer, String text, boolean modelSpecialTokens) {
        return this.hostTasks.tokenize(tokenizer, text, modelSpecialTokens);
    }

    /// Runs `work` as one frame on the lattice's workers; the future completes on that worker.
    public <T> CompletableFuture<T> onWorker(Supplier<T> work) {
        return this.hostTasks.onWorker(work);
    }

    /// Host work for the prefix cache: each piece runs as one frame on the lattice's workers.
    public PrefixFrames frames() {
        return this.hostTasks.prefixFrames();
    }

    public EuhedralInferenceRuntime runtime() {
        return this.runtime;
    }

    /// The lake the runtime publishes into, which this execution owns.
    public InferenceLake lake() {
        return this.lake;
    }

    public HostTasks hostTasks() {
        return this.hostTasks;
    }

    /// Device bytes the graphs' reusable workspace storage retains between quanta.
    public long retainedWorkspaceBytes() {
        return this.runtime.retainedWorkspaceBytes();
    }

    /// Quanta replayed from captured CUDA graphs.
    public long replayedQuanta() {
        return this.runtime.replayedQuanta();
    }

    /// Admitted quanta whose graphs have not yet retired.
    public int activeQuanta() {
        return this.runtime.activeQuanta();
    }

    /// Whether the lake's sinks are still attached to the lattice.
    public boolean isAttached() {
        return this.lake.isAttached();
    }

    /// Stops the host work, waits for every accepted quantum to retire and releases the graphs and lanes, then
    /// completes the lake and waits for it: the order the dense runtime closed in.
    @Override
    public void close() {
        synchronized (this.closeLock) {
            if (this.closed) return;
            this.closed = true;
        }
        RuntimeException failure = null;
        try {
            this.hostTasks.close();
        } catch (RuntimeException completionFailure) {
            failure = completionFailure;
        }
        try {
            this.runtime.close();
        } catch (RuntimeException closeFailure) {
            if (failure == null) failure = closeFailure;
            else failure.addSuppressed(closeFailure);
        }
        try {
            this.lake.completeGracefully();
            this.lake.awaitTermination();
        } catch (RuntimeException completionFailure) {
            if (failure == null) failure = completionFailure;
            else failure.addSuppressed(completionFailure);
        }
        if (failure != null) throw failure;
    }
}
