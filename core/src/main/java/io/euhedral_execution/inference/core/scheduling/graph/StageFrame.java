package io.euhedral_execution.inference.core.scheduling.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.concurrent.atomic.AtomicInteger;

/// One execution stage of a reusable [StageGraph], an Euhedral frame like any other.
///
/// A stage submits its device work to a lane of its graph's pool, chosen when it runs, and never runs
/// a successor. It first awaits, on that lane, the markers of predecessors that ran on other lanes. After a
/// successful submission it satisfies its outgoing edges; the edge that completes a successor's
/// incoming set publishes that successor to the source. No authority places it: a worker with capacity
/// takes it first come, first served, or the lattice routes it by the frame's hash.
///
/// Once published, a frame is handled by one thread at a time. Its incoming-edge count is the only
/// state that concurrent predecessors touch before publication.
///
/// `execute` never throws. A failed submission is recorded on the quantum, because an `Error`
/// escaping into Euhedral would complete this graph's source or end the worker. A stage therefore
/// reaches `doFinallyWithError` only when the lattice rejected the frame without running it.
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
    StageFrame[] submittedPredecessors;
    /// The predecessor whose longest remaining path continues through this stage; null otherwise.
    StageFrame pathPredecessor;
    /// Recorded on this stage's lane after it submits when it has successors (multi-lane pools only).
    long marker;
    /// The marker's mirror on the lane's shadow while the graph records a captured quantum.
    long shadowMarker;
    /// The lane this stage submitted to in the current quantum. Successors read it after their final
    /// arrival, which orders it after this stage's submission.
    int lane;

    @SuppressWarnings("unused")
    private volatile int arrivals;

    /// Set by a stage whose completion is not its submission: its work finishes later,
    /// asynchronously (an expert load that ends on a transfer's completion frame). Its outgoing
    /// edges are then satisfied by [#completeDeferred], not by the end of `submit()`.
    private volatile boolean deferred;

    private final AtomicInteger deferral = new AtomicInteger();

    boolean attempted;
    boolean submitted;

    protected StageFrame(StageGraph graph, int stage) {
        super(FrameSeeds.ID_HASH);
        this.graph = graph;
        this.stage = stage;
        this.inDegree = graph.topology().inDegree(stage);
        randomizeHash(graph.nextRoutingSeed());
    }

    /// Submits this stage's device work. The quantum's stream is selected on the calling thread.
    protected abstract void submit();

    /// Called from `submit()` by a stage that starts asynchronous work and completes when that work
    /// does: the stage stays live, and satisfies its outgoing edges only when [#completeDeferred]
    /// is called, from any thread that may run frames. A completion that arrives before the frame's
    /// own end is held until it.
    protected final void deferCompletion() {
        this.deferred = true;
    }

    /// The asynchronous work of a deferring stage ended: satisfy its outgoing edges, as the end of
    /// `submit()` does for other stages. Called exactly once per deferring run.
    protected final void completeDeferred() {
        if (this.deferral.incrementAndGet() == 2) finishDeferred();
    }

    private void finishDeferred() {
        if (this.submitted) this.graph.release(this);
        else this.graph.stageFinished();
    }

    /// Whether this stage only copies host memory to the device. Such a stage runs on the pool's
    /// transfer lane when it has one, and never carries a compute chain.
    protected boolean transfers() {
        return false;
    }

    /// Whether this stage does host work only: it submits nothing to a lane, records no marker and
    /// orders nothing on the device. A device stage that must follow a host stage's effect names
    /// the device stage that produced it as a predecessor too.
    protected boolean host() {
        return false;
    }

    /// Whether this stage has nothing to do in the current quantum (decided when its last incoming
    /// edge arrives): it then only joins its predecessors' device order, so that its successors
    /// still wait for what it stood after, and it runs in place on the thread that satisfied its
    /// last edge instead of being published.
    protected boolean skips() {
        return false;
    }

    /// Runs once per quantum on the retiring worker for a stage that attempted submission, after
    /// the quantum's device work has retired. `committed` is true only when the whole quantum
    /// succeeded: publish externally visible state then, and release temporary resources either
    /// way.
    protected void retired(boolean committed) {}

    protected final StageGraph graph() {
        return this.graph;
    }

    /// The lane stream this stage submits to, selected while [#submit()] runs.
    protected final GpuStream laneStream() {
        return this.graph.pool().lane(this.lane);
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
        boolean skipped = skips();
        this.attempted = !skipped;
        if (host()) {
            this.lane = owner.home();
            try {
                if (!skipped) submit();
            } catch (Throwable failure) {
                owner.fail(failure);
                return;
            }
            this.submitted = true;
            return;
        }
        LanePool pool = owner.pool();
        int lane = transfers() && pool.transferLane() >= 0
                ? pool.transferLane()
                : owner.spread() ? pool.choose(this.stage, continuedLane(pool, owner)) : owner.home();
        this.lane = lane;
        GpuStream stream = pool.lane(lane);
        try {
            boolean awaited = false;
            boolean recording = owner.recording();
            if (pool.size() > 1) {
                if (this.submittedPredecessors.length == 0 && lane != owner.home()) {
                    stream.await(owner.prepared());
                    if (recording) owner.shadowAwait(lane, owner.shadowPrepared());
                    awaited = true;
                }
                for (StageFrame predecessor : this.submittedPredecessors) {
                    if (predecessor.lane == lane || predecessor.host()) continue;
                    stream.await(predecessor.marker);
                    if (recording) owner.shadowAwait(lane, predecessor.shadowMarker);
                    awaited = true;
                }
                owner.used(lane);
            }
            // A launch that waits on another lane does not overlap its stream predecessor.
            if (!skipped) owner.submit(this, stream, lane, owner.overlapLaunches() && !awaited, !awaited);
            if (this.marker != 0) {
                stream.mark(this.marker);
                if (recording) owner.shadowMark(lane, this.shadowMarker);
            }
        } catch (Throwable failure) {
            // doFinally drops this frame's live count and publishes no successor.
            owner.fail(failure);
            return;
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
        if (this.deferred) {
            // The edges are satisfied by whichever of the frame's end and the asynchronous completion comes second.
            if (this.deferral.incrementAndGet() == 2) finishDeferred();
            return;
        }
        if (this.submitted) this.graph.release(this);
        else this.graph.stageFinished();
    }

    /// The lattice rejected this frame without running it: the worker's cache retired, or no downstream
    /// was routable. The stage never submitted, so the quantum fails. The rejection may be an instance
    /// the lattice shares, so it is only referenced as the cause.
    @Override
    public final void doFinallyWithError(Throwable rejection) {
        this.graph.stageFailed(new IllegalStateException("the lattice rejected stage " + this.stage, rejection));
    }

    /// Satisfies one incoming edge. Exactly one caller sees the final arrival.
    final boolean arrive() {
        return (int) ARRIVALS.getAndAdd(this, 1) + 1 == this.inDegree;
    }

    /// The lane this stage continues, or -1: its path predecessor's, or the graph's home lane for a root.
    private int continuedLane(LanePool pool, StageGraph owner) {
        if (pool.size() == 1) return -1;
        if (this.submittedPredecessors.length == 0) return owner.home();
        return this.pathPredecessor == null ? -1 : this.pathPredecessor.lane;
    }

    final void reset() {
        ARRIVALS.set(this, 0);
        this.deferred = false;
        this.deferral.set(0);
        this.attempted = false;
        this.submitted = false;
    }
}
