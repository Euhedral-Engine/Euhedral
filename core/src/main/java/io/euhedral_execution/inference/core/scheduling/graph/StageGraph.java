package io.euhedral_execution.inference.core.scheduling.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;

/// One reusable runtime instance of a static stage DAG.
///
/// The graph is built once: its stage frames, their successor references, and one frame per
/// device-completion edge are wired at construction and rebound to each quantum. A quantum owns the
/// graph's stream while it runs, so every stage submits to the same device ordering whichever worker
/// Euhedral schedules it on.
///
/// Admission publishes only the root stages. A stage that submitted successfully satisfies its
/// outgoing edges; a submission edge is satisfied at once and a device-completion edge when the
/// producer's work retires. The final arrival at a successor publishes it to the source. The graph
/// never runs a stage, scans for ready work, or walks its topology after admission.
///
/// `live` counts published frames that have not finished plus armed device-completion edges. When it
/// reaches zero, no stage of this quantum can run again, whether it succeeded, failed, or was
/// cancelled. The graph then arms the quantum's single retirement boundary. The retirement frame
/// confirms it, runs each attempted stage's retirement hook and the quantum's terminal work, and
/// returns the graph to its recycler before publishing the outcome.
public final class StageGraph implements AutoCloseable {

    /// Creates the frame for one stage while the graph is built.
    @FunctionalInterface
    public interface StageFactory {
        StageFrame create(StageGraph graph, int stage);
    }

    /// Receives a graph whose quantum has retired, for the next quantum.
    @FunctionalInterface
    public interface Recycler {
        void recycle(StageGraph graph);
    }

    private static final long NO_TICKET = 0L;

    private final StageTopology topology;
    private final GpuStream stream;
    private final QwenExecutionSource source;
    private final Recycler recycler;
    private final long routingSeed = ThreadLocalRandom.current().nextLong();
    private final StageFrame[] stages;
    private final StageFrame[] roots;
    private final Retirement retirement;
    private final AtomicInteger live = new AtomicInteger();
    private StageQuantum quantum;
    private boolean overlap;

    public StageGraph(
            StageTopology topology,
            StageFactory factory,
            GpuStream stream,
            QwenExecutionSource source,
            Recycler recycler) {
        this.topology = Objects.requireNonNull(topology, "topology");
        this.stream = Objects.requireNonNull(stream, "stream");
        this.source = Objects.requireNonNull(source, "source");
        this.recycler = Objects.requireNonNull(recycler, "recycler");
        Objects.requireNonNull(factory, "factory");
        this.stages = new StageFrame[topology.size()];
        for (int stage = 0; stage < this.stages.length; stage++) {
            StageFrame frame = Objects.requireNonNull(factory.create(this, stage), "stage frame");
            if (frame.graph() != this || frame.stage() != stage) {
                throw new IllegalArgumentException("stage factory returned a frame for another stage");
            }
            this.stages[stage] = frame;
        }
        for (int stage = 0; stage < this.stages.length; stage++) {
            int[] submitted = topology.submittedSuccessors(stage);
            StageFrame[] successors = new StageFrame[submitted.length];
            for (int index = 0; index < submitted.length; index++) successors[index] = this.stages[submitted[index]];
            this.stages[stage].submittedSuccessors = successors;
            int[] retired = topology.retiredSuccessors(stage);
            RetiredEdge[] edges = new RetiredEdge[retired.length];
            for (int index = 0; index < retired.length; index++) {
                edges[index] = new RetiredEdge(this, this.stages[retired[index]]);
            }
            this.stages[stage].retiredEdges = edges;
        }
        int[] rootStages = topology.roots();
        this.roots = new StageFrame[rootStages.length];
        for (int index = 0; index < rootStages.length; index++) this.roots[index] = this.stages[rootStages[index]];
        this.retirement = new Retirement(this);
    }

    public StageTopology topology() {
        return this.topology;
    }

    /// The device ordering that every stage of the bound quantum submits to.
    public GpuStream stream() {
        return this.stream;
    }

    /// The quantum currently bound to this graph.
    public StageQuantum quantum() {
        return this.quantum;
    }

    public StageFrame stage(int stage) {
        return this.stages[stage];
    }

    long routingSeed() {
        return this.routingSeed;
    }

    boolean overlapLaunches() {
        return this.overlap;
    }

    boolean stopRequested() {
        return this.quantum.stopRequested();
    }

    /// Binds a quantum whose resources are already prepared and publishes the root stages.
    public void start(StageQuantum quantum) {
        Objects.requireNonNull(quantum, "quantum");
        if (this.quantum != null) throw new IllegalStateException("stage graph already runs a quantum");
        this.quantum = quantum;
        this.overlap = quantum.overlapLaunches();
        for (StageFrame stage : this.stages) stage.reset();
        this.retirement.reset();
        // Admission holds one count so that fast roots cannot retire the quantum before all publish.
        this.live.set(this.roots.length + 1);
        for (StageFrame root : this.roots) this.source.publish(root);
        stageFinished();
    }

    /// Publishes the successors that `stage` makes ready after its successful submission.
    void release(StageFrame stage) {
        try {
            for (StageFrame successor : stage.submittedSuccessors) {
                if (stopRequested()) break;
                if (successor.arrive()) {
                    this.live.incrementAndGet();
                    this.source.publish(successor);
                }
            }
            for (RetiredEdge edge : stage.retiredEdges) {
                if (stopRequested()) break;
                edge.arm();
            }
        } catch (RuntimeException | Error failure) {
            fail(failure);
        } finally {
            stageFinished();
        }
    }

    void stageFailed(Throwable failure) {
        fail(failure);
        stageFinished();
    }

    void fail(Throwable failure) {
        this.quantum.fail(failure);
    }

    /// Drops one live count. Callers must not touch their frame afterwards: the last drop arms the
    /// retirement boundary, after which the graph and its frames may be rebound to another quantum.
    void stageFinished() {
        if (this.live.decrementAndGet() == 0) quiesce();
    }

    void addLive() {
        this.live.incrementAndGet();
    }

    /// The source through which this graph's frames reach Euhedral.
    public QwenExecutionSource source() {
        return this.source;
    }

    /// No stage of this quantum can run again. Arms the quantum's single device-completion boundary.
    private void quiesce() {
        Retirement terminal = this.retirement;
        try {
            this.stream.notifyRetired(terminal);
        } catch (RuntimeException | Error failure) {
            this.quantum.fail(failure);
            this.stream.recover(failure);
            this.source.publish(terminal);
        }
    }

    /// Terminal work for the bound quantum; runs on an ordinary worker after device retirement.
    void retire(long ticket) {
        StageQuantum retiring = this.quantum;
        Throwable deviceFailure = ticket == NO_TICKET ? null : this.stream.confirmRetired(ticket);
        if (deviceFailure != null) retiring.fail(deviceFailure);
        boolean committed = !retiring.stopRequested();
        for (StageFrame stage : this.stages) {
            if (!stage.attempted) continue;
            try {
                stage.retired(committed && stage.submitted);
            } catch (RuntimeException | Error failure) {
                retiring.fail(failure);
                committed = false;
            }
        }
        try {
            retiring.retire(deviceFailure);
        } catch (RuntimeException | Error failure) {
            retiring.fail(failure);
        }
        // The next quantum may reuse this graph before this one's outcome is observed.
        this.quantum = null;
        this.recycler.recycle(this);
        try {
            retiring.publishOutcome();
        } finally {
            this.source.terminated();
        }
    }

    /// Releases the graph's stream. Only an unbound graph whose work has retired may be closed.
    @Override
    public void close() {
        if (this.quantum != null) throw new IllegalStateException("stage graph still runs a quantum");
        this.stream.close();
    }

    /// The single device-completion boundary of a quantum. The driver callback only publishes it.
    static final class Retirement extends AbstractFrame implements GpuStream.RetirementListener {
        private final StageGraph graph;
        private long ticket;

        Retirement(StageGraph graph) {
            super(0L);
            this.graph = graph;
            randomizeHash(graph.routingSeed);
        }

        void reset() {
            this.ticket = NO_TICKET;
        }

        @Override
        public void retired(long ticket, boolean driverThread) {
            this.ticket = ticket;
            if (driverThread) this.graph.source.publishFromCallback(this);
            else this.graph.source.publish(this);
        }

        @Override
        public void execute() {
            this.graph.retire(this.ticket);
        }

        /// The graph may already serve another quantum; this frame's state is not touched again.
        @Override
        public void doFinally() {}

        @Override
        public void doFinallyWithError(Throwable failure) {}
    }

    /// A device-completion edge. Its producer arms it after submission; the driver callback publishes
    /// it, and an ordinary worker confirms retirement before satisfying the consumer's edge.
    static final class RetiredEdge extends AbstractFrame implements GpuStream.RetirementListener {
        private final StageGraph graph;
        private final StageFrame consumer;
        private long ticket;

        RetiredEdge(StageGraph graph, StageFrame consumer) {
            super(0L);
            this.graph = graph;
            this.consumer = consumer;
            randomizeHash(graph.routingSeed);
        }

        void arm() {
            this.graph.addLive();
            try {
                this.graph.stream.notifyRetired(this);
            } catch (RuntimeException | Error failure) {
                this.graph.fail(failure);
                this.graph.stream.recover(failure);
                this.graph.stageFinished();
            }
        }

        @Override
        public void retired(long ticket, boolean driverThread) {
            this.ticket = ticket;
            if (driverThread) this.graph.source.publishFromCallback(this);
            else this.graph.source.publish(this);
        }

        @Override
        public void execute() {
            Throwable failure = this.graph.stream.confirmRetired(this.ticket);
            if (failure != null) {
                this.graph.fail(failure);
                return;
            }
            if (!this.graph.stopRequested() && this.consumer.arrive()) {
                this.graph.addLive();
                this.graph.source.publish(this.consumer);
            }
        }

        @Override
        public void doFinally() {
            this.graph.stageFinished();
        }

        @Override
        public void doFinallyWithError(Throwable failure) {
            this.graph.stageFailed(failure);
        }
    }
}
