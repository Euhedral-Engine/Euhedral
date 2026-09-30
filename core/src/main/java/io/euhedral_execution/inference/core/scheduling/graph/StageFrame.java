package io.euhedral_execution.inference.core.scheduling.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

/// One execution stage of a reusable [StageGraph], scheduled by Euhedral like any other frame.
///
/// A stage submits its device work to its quantum's stream and never runs a successor. After a
/// successful submission it satisfies its outgoing edges; the edge that completes a successor's
/// incoming set publishes that successor to the source, and Euhedral decides when and where it runs.
///
/// Once published, a frame is handled by one thread at a time. Its incoming-edge count is the only
/// state that concurrent predecessors touch before publication.
public abstract class StageFrame extends AbstractFrame implements Runnable {

    private static final VarHandle ARRIVALS;

    static {
        try {
            ARRIVALS = MethodHandles.lookup().findVarHandle(StageFrame.class, "arrivals", int.class);
        } catch (ReflectiveOperationException failure) {
            throw new ExceptionInInitializerError(failure);
        }
    }

    private final StageGraph graph;
    private final int stage;
    private final int inDegree;
    StageFrame[] submittedSuccessors;
    StageGraph.RetiredEdge[] retiredEdges;

    @SuppressWarnings("unused")
    private volatile int arrivals;

    boolean attempted;
    boolean submitted;

    protected StageFrame(StageGraph graph, int stage) {
        super(0L);
        this.graph = graph;
        this.stage = stage;
        this.inDegree = graph.topology().inDegree(stage);
        randomizeHash(graph.routingSeed());
    }

    /// Submits this stage's device work. The quantum's stream is selected on the calling thread.
    protected abstract void submit();

    /// Runs once per quantum on the retiring worker for a stage that attempted submission, after the
    /// quantum's device work has retired. `committed` is true only when the whole quantum succeeded:
    /// publish externally visible state then, and release temporary resources either way.
    protected void retired(boolean committed) {}

    protected final StageGraph graph() {
        return this.graph;
    }

    public final int stage() {
        return this.stage;
    }

    @Override
    public final void execute() {
        this.attempted = false;
        this.submitted = false;
        StageGraph owner = this.graph;
        if (owner.stopRequested()) return;
        this.attempted = true;
        try {
            owner.stream().submit(this, owner.overlapLaunches());
        } catch (Error fatal) {
            // Euhedral finalizes Exceptions only; account for this frame before the Error escapes.
            owner.stageFailed(fatal);
            throw fatal;
        }
        this.submitted = true;
    }

    /// The stream's submission body; runs on the thread that executes this frame.
    @Override
    public final void run() {
        submit();
    }

    @Override
    public final void doFinally() {
        if (this.submitted) this.graph.release(this);
        else this.graph.stageFinished();
    }

    @Override
    public final void doFinallyWithError(Throwable failure) {
        this.graph.stageFailed(failure);
    }

    /// Satisfies one incoming edge. Exactly one caller sees the final arrival.
    final boolean arrive() {
        return (int) ARRIVALS.getAndAdd(this, 1) + 1 == this.inDegree;
    }

    final void reset() {
        ARRIVALS.set(this, 0);
        this.attempted = false;
        this.submitted = false;
    }
}
