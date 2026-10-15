package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.offload.DiskReader;
import io.euhedral_execution.inference.core.model.qwen38.offload.DiskUse;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.runtime.HostTasks;
import io.euhedral_execution.inference.core.runtime.PromptSink;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import io.euhedral_execution.inference.core.runtime.graph.AsyncReads;
import io.euhedral_execution.inference.core.runtime.graph.ChunkedShape;
import io.euhedral_execution.inference.core.runtime.graph.FrameSeeds;
import io.euhedral_execution.inference.core.runtime.graph.GraphShape;
import io.euhedral_execution.inference.core.runtime.graph.InferenceLake;
import io.euhedral_execution.inference.core.runtime.graph.LanePool;
import io.euhedral_execution.inference.core.runtime.graph.WorkspaceOwner;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicLongArray;
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

    private static final Logger PASSES = LoggerFactory.getLogger(Execution.class.getName() + ".passes");

    private static final Consumer<Quantum> NO_TERMINAL_CONSUMER = ignored -> {};

    /// How long the device must leave the window's flags alone at shutdown, and the longest the release runs.
    private static final long RELEASE_QUIET_NANOS = 100_000_000L;
    private static final long RELEASE_BOUND_NANOS = 10_000_000_000L;

    private static final int COUNTERS = 6;
    private static final int QUANTA = 0;
    private static final int BATCHED = 1;
    private static final int SEGMENTS = 2;
    private static final int ROWS = 3;
    private static final int DISK = 4;
    private static final int TRANSFER = 5;

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
    /// The reader of disk-backed weights and the reads it submits; null when no weight is disk-backed.
    private final DiskReader diskReader;
    private final AsyncReads asyncReads;
    /// The number of the next admitted quantum with disk uses; confined to the workspace's owner.
    private long nextDiskTicket;
    /// Per execution kind, [#COUNTERS] counts of the admitted quanta; read from other threads.
    private final AtomicLongArray counters = new AtomicLongArray(Quantum.ExecutionKind.values().length * COUNTERS);
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
        this(lattice, plan, gpu, laneCount, captureGraphs, maxRows, null);
    }

    /// As above, reading the plan's disk-backed weights from `artifact` (null when none is disk-backed): a
    /// [DiskReader] and the reads it submits are sources of the lake.
    public Execution(
            LatticeTerminal lattice,
            ExecutionPlan plan,
            ExecutionGpu gpu,
            int laneCount,
            boolean captureGraphs,
            int maxRows,
            Path artifact) {
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
        if (this.plan.window() == null) {
            this.asyncReads = null;
            this.diskReader = null;
            return;
        }
        if (artifact == null) throw new IllegalArgumentException("disk-backed weights need their artifact");
        AsyncReads reads = openAsyncReads();
        try {
            this.diskReader = new DiskReader(this.plan.window(), reads, artifact);
        } catch (IOException | RuntimeException failure) {
            if (reads != null) reads.close();
            this.runtime.close();
            throw new IllegalStateException("the artifact cannot be read for its disk-backed weights", failure);
        }
        this.asyncReads = reads;
        if (reads != null) this.lake.attach(reads.getDelegate());
        this.lake.attach(this.diskReader.getDelegate());
    }

    /// The machine's asynchronous reads, or null: the disk reader then reads positionally on its poll.
    private static AsyncReads openAsyncReads() {
        if (!AsyncReads.available()) return null;
        try {
            return AsyncReads.open();
        } catch (RuntimeException unavailable) {
            LOG.warn("disk-backed weights are read positionally: {}", unavailable.getMessage());
            return null;
        }
    }

    /// The reads of the disk uses of the quantum that runs `shape`, in the order its device work consumes each
    /// window slot's: a prompt's chunks one after another.
    static List<DiskUse> diskUses(GraphShape shape) {
        if (shape instanceof Shape view) return view.diskUses();
        if (shape instanceof CheckpointView checkpointed)
            return checkpointed.view().diskUses();
        if (!(shape instanceof ChunkedShape chunked)) return List.of();
        List<DiskUse> uses = new ArrayList<>();
        for (int chunk = 0; chunk < chunked.chunks(); chunk++) uses.addAll(diskUses(chunked.template(chunk)));
        return uses;
    }

    /// The bytes the quantum that runs `shape` copies into the staging ring, or reads from disk when `disk`.
    static long weightBytes(GraphShape shape, boolean disk) {
        if (shape instanceof Shape view) return disk ? view.diskReadBytes() : view.transferBytes();
        if (shape instanceof CheckpointView checkpointed) return weightBytes(checkpointed.view(), disk);
        if (!(shape instanceof ChunkedShape chunked)) return 0;
        long bytes = 0;
        for (int chunk = 0; chunk < chunked.chunks(); chunk++) bytes += weightBytes(chunked.template(chunk), disk);
        return bytes;
    }

    /// The counts of the quanta of `kind` admitted so far; each counter is read independently, so they are not a
    /// consistent snapshot.
    public PassStatistics statistics(Quantum.ExecutionKind kind) {
        int base = kind.ordinal() * COUNTERS;
        return new PassStatistics(
                this.counters.get(base + QUANTA),
                this.counters.get(base + BATCHED),
                this.counters.get(base + SEGMENTS),
                this.counters.get(base + ROWS),
                this.counters.get(base + DISK),
                this.counters.get(base + TRANSFER));
    }

    /// What the admitted quanta of one kind did: `segments` counts a quantum of one sequence as one segment, `rows`
    /// the input tokens, and the byte counts the disk reads and the staging transfers their graphs issue.
    public record PassStatistics(
            long quanta, long batchedQuanta, long segments, long rows, long diskReadBytes, long transferBytes) {

        public double diskBytesPerQuantum() {
            return this.quanta == 0 ? 0 : (double) this.diskReadBytes / this.quanta;
        }

        public double transferBytesPerQuantum() {
            return this.quanta == 0 ? 0 : (double) this.transferBytes / this.quanta;
        }
    }

    private void count(Quantum context, GraphShape admitted) {
        int base = context.kind().ordinal() * COUNTERS;
        boolean batched = context.batched();
        long segments = batched ? context.segments().length : 1;
        long rows = context.inputTokenCount();
        long disk = this.diskReader == null ? 0 : weightBytes(admitted, true);
        long transfer = weightBytes(admitted, false);
        this.counters.incrementAndGet(base + QUANTA);
        if (batched) this.counters.incrementAndGet(base + BATCHED);
        this.counters.addAndGet(base + SEGMENTS, segments);
        this.counters.addAndGet(base + ROWS, rows);
        this.counters.addAndGet(base + DISK, disk);
        this.counters.addAndGet(base + TRANSFER, transfer);
        if (PASSES.isDebugEnabled())
            PASSES.debug(
                    "{} segments={} rows={} disk={} MiB h2d={} MiB",
                    context.kind(),
                    segments,
                    rows,
                    disk >> 20,
                    transfer >> 20);
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

    /// Admits `context` ahead of a later quantum of its sequence whose continuation reads both: no frame follows
    /// its own conclusion. The later quantum concludes after it and fails when it failed, so its continuation
    /// sees the outcome of both. Returns why the runtime refused `context`, or null. Runs where [#admit] does.
    public Throwable admitAhead(Quantum context) {
        Objects.requireNonNull(context, "context");
        claim(context);
        return accept(context, NO_TERMINAL_CONSUMER);
    }

    /// Checks that `context` belongs to this plan and claims its single admission.
    private void claim(Quantum context) {
        if (context.plan() != this.plan) {
            throw new IllegalArgumentException("quantum belongs to another execution plan");
        }
        context.claim();
    }

    /// Admits a claimed quantum, and returns why the runtime refused it, or null. Runs on the workspace's owner.
    private Throwable accept(Quantum context, Consumer<? super Quantum> terminalConsumer) {
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
        GraphShape admitted =
                context.chunkCount() > 1 || !context.checkpointChunks().isEmpty() ? this.prompts.shape(context) : view;
        // Its disk uses are registered once it started, in admission order: the order of each window slot's copies.
        List<DiskUse> uses = this.diskReader == null ? List.of() : diskUses(admitted);
        long ticket = uses.isEmpty() ? -1 : this.nextDiskTicket++;
        if (ticket >= 0) context.readsFromDisk(this.diskReader, ticket);
        boolean[] began = new boolean[1];
        try {
            this.runtime.admit(
                    admitted,
                    context,
                    (stream, storage) -> began[0] =
                            context.begin(this.gpu, stream, terminalConsumer, (WorkspaceStorage) storage, this.shared));
        } catch (RuntimeException | Error refused) {
            LOG.debug("a quantum was refused; its continuation carries the failure", refused);
            // Refused after its graph started publishing: the failure drains what its stages leave, so the
            // reader must read its uses. Refused before: no stage of it runs, and nothing may be read for it.
            if (ticket >= 0 && context.stagesPublished()) this.diskReader.enqueue(ticket, uses, context::fail);
            if (context.stagesPublished()) count(context, admitted);
            return refused;
        }
        if (ticket >= 0 && context.stagesPublished()) this.diskReader.enqueue(ticket, uses, context::fail);
        count(context, admitted);
        return null;
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
        // Every quantum retired, so every read its device work waited for completed; the reader closes the file.
        if (this.diskReader != null) {
            try {
                this.diskReader.close();
            } catch (RuntimeException closeFailure) {
                if (failure == null) failure = closeFailure;
                else failure.addSuppressed(closeFailure);
            }
        }
        try {
            this.lake.completeGracefully();
            this.lake.awaitTermination();
        } catch (RuntimeException completionFailure) {
            if (failure == null) failure = completionFailure;
            else failure.addSuppressed(completionFailure);
        }
        if (this.asyncReads != null && (this.diskReader == null || this.diskReader.readsInFlight() == 0)) {
            try {
                this.asyncReads.close();
            } catch (RuntimeException closeFailure) {
                if (failure == null) failure = closeFailure;
                else failure.addSuppressed(closeFailure);
            }
        }
        // Every quantum retired, so the device holds no flag wait; should one remain, the window's flags are raised
        // until the device leaves them alone, so freeing the workspace never waits on a read that will not come.
        if (this.diskReader != null && this.gpu.completionProven()) {
            try {
                this.diskReader.releaseDevice(RELEASE_QUIET_NANOS, RELEASE_BOUND_NANOS);
            } catch (RuntimeException releaseFailure) {
                if (failure == null) failure = releaseFailure;
                else failure.addSuppressed(releaseFailure);
            }
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
