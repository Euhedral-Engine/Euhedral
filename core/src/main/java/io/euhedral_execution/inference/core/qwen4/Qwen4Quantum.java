package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.scheduling.graph.GraphStorage;
import io.euhedral_execution.inference.core.scheduling.graph.StageQuantum;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// One prefill chunk or decode token as the quantum of a [Qwen4Shape] graph: what the stages read
/// (the sequence, the tokens, where the logits go), whether it stopped, and what happens when its
/// device work has retired. The graph never inspects what the stages compute; this class owns the
/// quantum's terminal work: committing the sequence, closing the leases a stopped block still
/// holds, and reporting to the listener.
///
/// Quanta of a plan form a chain. A graph's expert window is sized to the cache's slots and staging
/// slots, so one quantum runs at a time; a quantum that is admitted while another runs registers as
/// the other's successor, and the predecessor's conclusion publishes it. Nothing waits: registering
/// is a compare-and-set.
final class Qwen4Quantum implements StageQuantum, Qwen4ExecutionPlan.Handle {

    private static final Logger LOG = LoggerFactory.getLogger(Qwen4Quantum.class);
    private static final Object FINISHED = new Object();
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED;

    private final Qwen4ExecutionPlan plan;
    private final Qwen4Shape shape;
    private final Qwen4Sequence sequence;
    private final int[] tokens;
    private final int offset;
    private final int rows;
    private final Qwen4ExecutionPlan.LogitsSink sink;
    private final Qwen4ExecutionPlan.StateExchange exchange;
    private final Qwen4ExecutionPlan.Listener listener;
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private final AtomicReference<Object> successor = new AtomicReference<>();
    private volatile boolean cancelled;
    private Qwen4GraphStorage storage;
    private Throwable outcome;
    private boolean prepared;

    // The diagnostic shape's clock: the component running and when it began.
    private int component = -1;
    private long begun;

    Qwen4Quantum(
            Qwen4ExecutionPlan plan,
            Qwen4Shape shape,
            Qwen4Sequence sequence,
            int[] tokens,
            int offset,
            int rows,
            Qwen4ExecutionPlan.LogitsSink sink,
            Qwen4ExecutionPlan.StateExchange exchange,
            Qwen4ExecutionPlan.Listener listener) {
        this.plan = plan;
        this.shape = shape;
        this.sequence = sequence;
        this.tokens = tokens;
        this.offset = offset;
        this.rows = rows;
        this.sink = sink;
        this.exchange = exchange;
        this.listener = listener;
    }

    // ---------------------------------------------------------------- what the stages read

    Qwen4ExecutionPlan plan() {
        return this.plan;
    }

    Qwen4GraphStorage storage() {
        return this.storage;
    }

    Qwen4Sequence sequence() {
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

    Qwen4ExecutionPlan.LogitsSink sink() {
        return this.sink;
    }

    // ---------------------------------------------------------------- starting: the completion chain

    /// Joins the plan's chain: the quantum is admitted when the one before it concludes, or at once
    /// when none runs.
    void enter() {
        Qwen4Quantum previous = this.plan.chainTail().getAndSet(this);
        if (previous == null || !previous.attach(this)) go();
    }

    /// Registers `next` to be admitted by this quantum's conclusion; false when this quantum
    /// already concluded.
    private boolean attach(Qwen4Quantum next) {
        return this.successor.compareAndSet(null, next);
    }

    private void go() {
        this.plan.quantumStarted();
        try {
            this.plan.runtime().admit(this.shape, this, null, this::prepare);
        } catch (RuntimeException | Error refused) {
            // The runtime retired the quantum as failed and published its outcome before it threw.
            LOG.debug("a quantum was refused", refused);
        }
    }

    /// Runs on the graph's home lane before any stage: binds the graph's workspace, hands over the
    /// tokens, and queues the state a layer test starts from.
    private boolean prepare(GpuStream stream, GraphStorage graphStorage) {
        this.storage = ((Qwen4GraphStorage.Lease) graphStorage).storage();
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

    /// Asks the quantum to stop at its next stage.
    @Override
    public void cancel() {
        this.cancelled = true;
    }

    // ---------------------------------------------------------------- StageQuantum

    @Override
    public boolean stopRequested() {
        return this.cancelled || this.failure.get() != null;
    }

    @Override
    public void fail(Throwable cause) {
        if (this.failure.compareAndSet(null, cause)) return;
        Throwable first = this.failure.get();
        if (first != cause) first.addSuppressed(cause);
    }

    @Override
    public boolean overlapLaunches() {
        return false;
    }

    /// Runs on an ordinary worker after the device work retired and every stage's retirement hook
    /// ran.
    @Override
    public void retire(Throwable deviceFailure) {
        if (deviceFailure != null) fail(deviceFailure);
        if (this.cancelled && this.failure.get() == null) fail(new CancellationException("the step was cancelled"));
        if (this.prepared) {
            tick(-1);
            try {
                // Leases of waves that never ran (the quantum stopped); no load is outstanding and no kernel reads
                // them.
                this.storage.moe().abandon();
            } catch (RuntimeException | Error cleanup) {
                fail(cleanup);
            }
        }
        Throwable error = this.failure.get();
        if (error == null && this.prepared) {
            try {
                if (this.exchange != null) this.exchange.collect(this.plan.gpu(), this.storage.state());
                if (this.shape.advances()) this.sequence.advance(this.rows);
                this.plan.stepCompleted(this.rows);
            } catch (RuntimeException | Error commit) {
                fail(commit);
                error = commit;
            }
        }
        this.outcome = error != null ? this.failure.get() : null;
    }

    /// The graph is recycled: the next quantum of the chain starts, then the listener runs.
    @Override
    public void publishOutcome() {
        this.plan.quantumEnded();
        Object next = this.successor.getAndSet(FINISHED);
        if (next != null) ((Qwen4Quantum) next).go();
        try {
            this.listener.finished(this.outcome);
        } catch (Throwable listenerFailure) {
            LOG.error("a step's listener failed", listenerFailure);
        }
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
