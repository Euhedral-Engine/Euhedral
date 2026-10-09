package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import io.euhedral_execution.inference.core.runtime.graph.GraphStorage;
import java.lang.foreign.ValueLayout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// One prefill chunk or decode token as the quantum of a [Shape] graph: what the stages read
/// (the sequence, the tokens, where the logits go), whether it stopped, and what happens when its
/// device work has retired. The graph never inspects what the stages compute; this class owns the
/// quantum's terminal work: committing the sequence and closing the leases a stopped block still
/// holds. Its continuation, thrown when the outcome is published, carries the result to the generation.
///
/// Quanta of a plan run side by side: each is admitted on the workspace's owner, which orders its graph behind the
/// graphs admitted before it wherever they share a buffer.
final class Quantum extends AbstractQuantum implements ExecutionPlan.Handle {

    private static final Logger LOG = LoggerFactory.getLogger(Quantum.class);
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED;

    private final ExecutionPlan plan;
    private final Shape shape;
    private final Sequence sequence;
    private final int[] tokens;
    private final int offset;
    private final int rows;
    private final ExecutionPlan.LogitsSink sink;
    private final ExecutionPlan.StateExchange exchange;
    private Workspace storage;
    private MoeLayer moe;
    private boolean prepared;
    /// The quantum's rows in chunks: a prompt's chunks, or one chunk of all its rows.
    private final Chunk[] chunks;
    /// A prompt's own pinned copy of its tokens (one chunk's go through the workspace's), and the n-gram rows of
    /// each of its chunks, whose host stages may run ahead of the previous chunk.
    private ExecutionGpu.UploadBuffer promptUpload;
    private PleRows[] pleRows;

    /// One chunk of a quantum's rows: its place, the index of its first token in the quantum's token array, its
    /// rows. Only the last chunk produces logits.
    record Chunk(int index, int offset, int rows, boolean last) {}

    /// A chunk's n-gram rows, their staging buffer and how many there are: written by the stage that computes the
    /// ids, read by the stages that gather and copy them.
    static final class PleRows {
        final long[] rowIds;
        ExecutionGpu.UploadBuffer upload;
        int count;

        PleRows(int rows) {
            this.rowIds = new long[rows];
        }
    }

    // The diagnostic shape's clock: the component running and when it began.
    private int component = -1;
    private long begun;

    Quantum(
            ExecutionPlan plan,
            Shape shape,
            Sequence sequence,
            int[] tokens,
            int offset,
            int rows,
            ExecutionPlan.LogitsSink sink,
            ExecutionPlan.StateExchange exchange) {
        this.plan = plan;
        this.shape = shape;
        this.sequence = sequence;
        this.tokens = tokens;
        this.offset = offset;
        this.rows = rows;
        this.sink = sink;
        this.exchange = exchange;
        this.chunks = new Chunk[] {new Chunk(0, offset, rows, true)};
    }

    /// A prompt quantum: `rows` tokens from `offset` in chunks of `chunkRows` (the remainder last), run as one
    /// graph whose chunks are copies of `shape`'s chunk template.
    Quantum(
            ExecutionPlan plan,
            Shape shape,
            Sequence sequence,
            int[] tokens,
            int offset,
            int rows,
            int chunkRows,
            ExecutionPlan.LogitsSink sink) {
        if (chunkRows <= 0) throw new IllegalArgumentException("chunkRows must be positive");
        this.plan = plan;
        this.shape = shape;
        this.sequence = sequence;
        this.tokens = tokens;
        this.offset = offset;
        this.rows = rows;
        this.sink = sink;
        this.exchange = null;
        int count = (rows + chunkRows - 1) / chunkRows;
        this.chunks = new Chunk[count];
        for (int index = 0; index < count; index++) {
            int first = index * chunkRows;
            this.chunks[index] =
                    new Chunk(index, offset + first, Math.min(chunkRows, rows - first), index == count - 1);
        }
    }

    Chunk chunk(int index) {
        return this.chunks[index];
    }

    int chunkCount() {
        return this.chunks.length;
    }

    /// Chunk `chunk`'s n-gram rows, the quantum's own: quanta of other sessions run beside it on the one workspace.
    PleRows pleRows(int chunk) {
        return this.pleRows[chunk];
    }

    /// The pinned host address of chunk `chunk`'s tokens in the quantum's own copy.
    long promptTokens(int chunk) {
        return this.promptUpload.segment().address() + 4L * (this.chunks[chunk].offset() - this.offset);
    }

    // ---------------------------------------------------------------- what the stages read

    ExecutionPlan plan() {
        return this.plan;
    }

    Workspace storage() {
        return this.storage;
    }

    /// The MoE block resources of the quantum's row capacity.
    MoeLayer moe() {
        return this.moe;
    }

    Sequence sequence() {
        return this.sequence;
    }

    int[] tokens() {
        return this.tokens;
    }

    int offset() {
        return this.offset;
    }

    int rows() {
        return this.rows;
    }

    ExecutionPlan.LogitsSink sink() {
        return this.sink;
    }

    // ---------------------------------------------------------------- starting

    /// Admits this quantum from a caller that is not on the workspace's owner (tests, tools): the admission runs
    /// as an owner frame.
    void enter() {
        this.plan.quantumStarted();
        this.plan.runtime().publishOnOwner(this::admit);
    }

    /// Admits this quantum from a frame ordered on the workspace's owner (a generation's `Admit`): the owner orders
    /// its graph behind the graphs admitted before it, buffer by buffer, so quanta of the plan (several sessions'
    /// steps) need no chain of their own. A refusal is not thrown: the runtime published the quantum's failed
    /// outcome, which its continuation carries.
    void enterOnOwner() {
        this.plan.quantumStarted();
        admit();
    }

    /// A prompt of several chunks runs as one graph of them, whose shape the owner looks up.
    private void admit() {
        try {
            this.plan
                    .runtime()
                    .admit(this.chunks.length == 1 ? this.shape : this.plan.promptShape(this), this, this::prepare);
        } catch (RuntimeException | Error refused) {
            LOG.debug("a quantum was refused; its continuation carries the failure", refused);
        }
    }

    Shape shape() {
        return this.shape;
    }

    /// Runs on the graph's home lane before any stage: binds the graph's workspace, hands over the
    /// tokens, and queues the state a layer test starts from.
    private boolean prepare(GpuStream stream, GraphStorage graphStorage) {
        Workspace.Lease lease = (Workspace.Lease) graphStorage;
        this.storage = lease.storage();
        this.moe = lease.moe();
        this.prepared = true;
        // The quantum's own tokens and n-gram rows: another session's quantum may be admitted, and its host stages
        // run, before this one's device copies and gathers ran.
        this.promptUpload = this.plan.gpu().allocateUploadBuffer(4L * this.rows);
        for (int i = 0; i < this.rows; i++) this.promptUpload.segment().set(INT, 4L * i, this.tokens[this.offset + i]);
        this.pleRows = new PleRows[this.chunks.length];
        int perToken = this.plan.ple().rowsPerToken();
        for (Chunk chunk : this.chunks) this.pleRows[chunk.index()] = new PleRows(chunk.rows() * perToken);
        if (stopRequested()) {
            // Stopped before it began: no stage will run, so the runtime publishes the outcome this prepares.
            retire(null);
            return false;
        }
        if (this.exchange != null) this.exchange.provide(this.plan.gpu(), this.storage.state());
        return true;
    }

    @Override
    public boolean overlapLaunches() {
        return false;
    }

    /// Leases of waves that never ran (the quantum stopped); no load is outstanding and no kernel reads them.
    @Override
    protected void release() {
        if (!this.prepared) return;
        tick(-1);
        this.moe.abandon();
        if (ExpertCacheOwner.aheadDistance() > 0) this.plan.expertOwner().aheadReset();
        // The device stopped reading the tokens, every chunk's embedding retired, unless the GPU cannot prove its
        // work stopped: then a queued copy may still read them, and they are kept.
        if (this.promptUpload != null && this.plan.gpu().completionProven()) {
            this.promptUpload.close();
            this.promptUpload = null;
        }
    }

    @Override
    protected void commit() {
        if (!this.prepared) return;
        if (this.exchange != null) this.exchange.collect(this.plan.gpu(), this.storage.state());
        if (this.shape.advances()) this.sequence.advance(this.rows);
        this.plan.stepCompleted(this.rows);
    }

    /// The graph is recycled: the next quantum of the plan's chain starts before the continuation is thrown.
    @Override
    protected void published() {
        this.plan.quantumEnded();
    }

    // ---------------------------------------------------------------- diagnostics

    /// Charges the time since the last tick to the component that was running and starts
    /// `component` (a negative component only closes the running one). Called by the diagnostic
    /// shape's stages, whose edges are device completions, so it is never concurrent.
    void tick(int component) {
        if (!this.plan.timingsOn()) return;
        long now = System.nanoTime();
        if (this.component >= 0) this.plan.charge(this.component, now - this.begun);
        this.component = component;
        this.begun = now;
    }
}
