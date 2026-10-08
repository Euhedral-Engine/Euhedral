package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.runtime.HostTasks;
import io.euhedral_execution.inference.core.runtime.PromptSink;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import io.euhedral_execution.inference.core.runtime.graph.FrameSeeds;
import io.euhedral_execution.inference.core.runtime.graph.GraphShape;
import io.euhedral_execution.inference.core.runtime.graph.InferenceLake;
import io.euhedral_execution.inference.core.runtime.graph.LanePool;
import io.euhedral_execution.inference.core.runtime.graph.WorkspaceOwner;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;
import java.util.function.IntFunction;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// Runs the dense model's quanta: the runtime it admits them to, the lake that runtime publishes into, its one
/// workspace, and the host work around a request.
///
/// Host-backed weights reach the device through staging slots, which are workspace buffers: a transfer into a slot
/// follows the slot's last reader, in its own graph or the one admitted before, so nothing holds the slots.
public final class Execution implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(Execution.class);

    private static final Consumer<Quantum> NO_TERMINAL_CONSUMER = ignored -> {};

    private final ExecutionPlan plan;
    private final ExecutionGpu gpu;
    private final InferenceLake lake;
    private final HostTasks hostTasks;
    private final EuhedralInferenceRuntime runtime;
    private final Object closeLock = new Object();
    /// The runtime's one workspace, sized at load.
    private final SharedWorkspace shared;
    /// The prompt graphs' shapes; confined to the workspace's owner.
    private final PromptShapes prompts;
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
        this(lattice, plan, gpu, laneCount, captureGraphs, InferenceConfig.PREFILL_CHUNK_TOKENS);
    }

    /// As above, with one workspace sized at load for quanta of up to `maxRows` rows.
    public Execution(
            LatticeTerminal lattice,
            ExecutionPlan plan,
            ExecutionGpu gpu,
            int laneCount,
            boolean captureGraphs,
            int maxRows) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (laneCount < 1 || laneCount > LanePool.MAX_LANES)
            throw new IllegalArgumentException("laneCount must be 1 to " + LanePool.MAX_LANES);
        boolean transfers = this.plan.staging() != null;
        int compute = transfers ? Math.min(laneCount, LanePool.MAX_LANES - 1) : laneCount;
        this.lake = EuhedralInferenceRuntime.newLake(lattice);
        this.hostTasks = new HostTasks(this.lake);
        this.shared = new SharedWorkspace(gpu, this.plan, maxRows);
        this.runtime = new EuhedralInferenceRuntime(
                this.lake,
                gpu,
                new EuhedralInferenceRuntime.Lanes(compute, transfers, captureGraphs),
                SharedWorkspace.bufferCount(this.plan));
        this.prompts = new PromptShapes(this.runtime);
    }

    /// The rows the workspace holds: the largest quantum it admits.
    public int maxRows() {
        return this.shared.maxRows();
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

    /// Admits one quantum from outside the workspace's owner (tests and tools): the admission runs as an owner
    /// frame. The returned future completes after its device work retired and the quantum's graph was recycled. A
    /// quantum of another plan, or one already admitted, throws here; any other failure to admit it fails its
    /// outcome.
    public CompletableFuture<Quantum.Outcome> submit(Quantum context, Consumer<? super Quantum> terminalConsumer) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(terminalConsumer, "terminalConsumer");
        claim(context);
        Completion done = new Completion();
        context.continueWith(this.lake, done);
        this.runtime.publishOnOwner(() -> accept(context, terminalConsumer));
        return done.outcome;
    }

    /// The continuation of a quantum submitted by a caller that waits (tests, tools): completes a future with its
    /// outcome.
    private static final class Completion extends AbstractFrame implements AbstractQuantum.Continuation {
        final CompletableFuture<Quantum.Outcome> outcome = new CompletableFuture<>();
        private Quantum quantum;

        Completion() {
            super(FrameSeeds.ID_HASH);
            randomizeHash(FrameSeeds.forHostWork().next());
        }

        @Override
        public void concluded(AbstractQuantum quantum) {
            this.quantum = (Quantum) quantum;
        }

        @Override
        public void execute() {
            this.outcome.complete(this.quantum.conclusion());
        }

        @Override
        public void doFinallyWithError(Throwable rejection) {
            execute();
        }
    }

    /// Admits `context` on the generation path, from its `Admit`, which runs ordered on the workspace's owner: once
    /// its outcome is published it throws `continuation` into the lake. Throws only when the quantum never reached
    /// the runtime (another plan's, already admitted); then `continuation` is never thrown. The runtime's refusal
    /// is not thrown: the runtime published the quantum's failed outcome, which `continuation` carries.
    public void admit(Quantum context, AbstractFrame continuation) {
        Objects.requireNonNull(context, "context");
        claim(context);
        context.continueWith(this.lake, continuation);
        accept(context, NO_TERMINAL_CONSUMER);
    }

    /// Checks that `context` belongs to this plan and claims its single admission.
    private void claim(Quantum context) {
        if (context.plan() != this.plan) {
            throw new IllegalArgumentException("quantum belongs to another execution plan");
        }
        context.claim();
    }

    /// Admits a claimed quantum. Runs on the workspace's owner.
    private void accept(Quantum context, Consumer<? super Quantum> terminalConsumer) {
        Shape view = context.shape();
        // A decode view whose first staging slots this sequence's DFlash2 block just filled runs without their
        // transfers; its first readers of those slots then follow the block's prefetch transfers (their last writer).
        Shape preloaded = view.preloadedVariant();
        if (preloaded != view) {
            WorkspaceOwner owner = this.runtime.workspaceOwner();
            if (preloadHolds(
                    this.plan.dflashBlockShape(),
                    context.sequenceState(),
                    this.plan.prefetchedSlots(),
                    slot -> owner.last(SharedWorkspace.stagingBuffer(this.plan, slot)))) view = preloaded;
        }
        // A prompt of several chunks runs as one graph of them.
        GraphShape admitted = context.chunkCount() > 1 ? this.prompts.shape(context) : view;
        try {
            this.runtime.admit(
                    admitted,
                    context,
                    (stream, storage) ->
                            context.begin(this.gpu, stream, terminalConsumer, (WorkspaceStorage) storage, this.shared));
        } catch (RuntimeException | Error refused) {
            LOG.debug("a quantum was refused; its continuation carries the failure", refused);
        }
    }

    /// Whether the first `slots` staging slots hold what `block` (a DFlash2 block view that prefetches) loads for
    /// `sequence`: the graph that last used each slot ran `block` for that sequence, and concluded successfully.
    /// `last` gives each slot's record on the workspace's owner, whose frames call this.
    static boolean preloadHolds(GraphShape block, Sequence sequence, int slots, IntFunction<WorkspaceOwner.Last> last) {
        if (block == null || slots == 0) return false;
        for (int slot = 0; slot < slots; slot++) {
            WorkspaceOwner.Last written = last.apply(slot);
            if (written == null
                    || written.shape() != block
                    || !(written.quantum() instanceof Quantum quantum)
                    || quantum.sequenceState() != sequence
                    || quantum.conclusion() == null
                    || quantum.conclusion().status() != Quantum.Status.SUCCESS) return false;
        }
        return true;
    }

    /// Tokenizes `text` on the lattice's workers (PromptTokenization), with the BOS/EOS tokens of
    /// tokenizer_config.json when `modelSpecialTokens`. The future completes on a worker.
    public CompletableFuture<int[]> tokenize(QwenTokenizer tokenizer, String text, boolean modelSpecialTokens) {
        return this.hostTasks.tokenize(tokenizer, text, modelSpecialTokens);
    }

    /// As [#tokenize(QwenTokenizer, String, boolean)]; the worker that joins the tokenization hands the IDs to `sink`.
    public void tokenize(QwenTokenizer tokenizer, String text, boolean modelSpecialTokens, PromptSink sink) {
        this.hostTasks.tokenize(tokenizer, text, modelSpecialTokens, sink);
    }

    /// Runs `work` as one frame on the lattice's workers; the future completes on that worker.
    public <T> CompletableFuture<T> onWorker(Supplier<T> work) {
        return this.hostTasks.onWorker(work);
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
        return this.runtime.retainedWorkspaceBytes() + this.shared.retainedBytes();
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
        // Every quantum retired: the workspace is idle. A poisoned GPU keeps it, as it keeps the graphs' storage.
        if (this.gpu.completionProven()) {
            try {
                this.shared.close();
            } catch (RuntimeException freeFailure) {
                if (failure == null) failure = freeFailure;
                else failure.addSuppressed(freeFailure);
            }
        }
        if (failure != null) throw failure;
    }
}
