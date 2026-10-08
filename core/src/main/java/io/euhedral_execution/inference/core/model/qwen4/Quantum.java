package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import io.euhedral_execution.inference.core.runtime.graph.GraphStorage;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// One prefill chunk or decode token as the quantum of a [Shape] graph: what the stages read
/// (the sequence, the tokens, where the logits go), whether it stopped, and what happens when its
/// device work has retired. The graph never inspects what the stages compute; this class owns the
/// quantum's terminal work: committing the sequence and closing the leases a stopped block still
/// holds. Its continuation, thrown when the outcome is published, carries the result to the generation.
///
/// Quanta of a plan form a chain. A graph's expert window is sized to the cache's slots and staging
/// slots, so one quantum runs at a time; a quantum that is admitted while another runs registers as
/// the other's successor, and the predecessor's conclusion publishes it. Nothing waits: registering
/// is a compare-and-set.
final class Quantum extends AbstractQuantum implements ExecutionPlan.Handle {

    private static final Logger LOG = LoggerFactory.getLogger(Quantum.class);
    private static final Object FINISHED = new Object();
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED;

    private final ExecutionPlan plan;
    private final Shape shape;
    private final Sequence sequence;
    private final int[] tokens;
    private final int offset;
    private final int rows;
    private final ExecutionPlan.LogitsSink sink;
    private final ExecutionPlan.StateExchange exchange;
    private final AtomicReference<Object> successor = new AtomicReference<>();
    private Workspace storage;
    private boolean prepared;

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
    }

    // ---------------------------------------------------------------- what the stages read

    ExecutionPlan plan() {
        return this.plan;
    }

    Workspace storage() {
        return this.storage;
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

    // ---------------------------------------------------------------- starting: the completion chain

    /// Joins the plan's chain: the quantum is admitted when the one before it concludes, or at once
    /// when none runs.
    void enter() {
        Quantum previous = this.plan.chainTail().getAndSet(this);
        if (previous == null || !previous.attach(this)) go();
    }

    /// Registers `next` to be admitted by this quantum's conclusion; false when this quantum
    /// already concluded.
    private boolean attach(Quantum next) {
        return this.successor.compareAndSet(null, next);
    }

    /// Admits this quantum on the workspace's owner: the plan's completion chain calls this from its predecessor's
    /// publication, outside the owner, so the admission is published as an owner frame.
    private void go() {
        this.plan.quantumStarted();
        this.plan.runtime().publishOnOwner(() -> this.plan.runtime().admit(this.shape, this, null, this::prepare));
    }

    /// Runs on the graph's home lane before any stage: binds the graph's workspace, hands over the
    /// tokens, and queues the state a layer test starts from.
    private boolean prepare(GpuStream stream, GraphStorage graphStorage) {
        this.storage = ((Workspace.Lease) graphStorage).storage();
        this.prepared = true;
        for (int i = 0; i < this.rows; i++)
            this.storage.tokenUpload().segment().set(INT, 4L * i, this.tokens[this.offset + i]);
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
        this.storage.moe().abandon();
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
        Object next = this.successor.getAndSet(FINISHED);
        if (next != null) ((Quantum) next).go();
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
