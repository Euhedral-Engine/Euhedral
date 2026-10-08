package io.euhedral_execution.inference.core.runtime.graph;

import static io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.dependencies;
import static io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.drain;
import static io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.stage;
import static io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.take;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.RecordingStream;
import io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.Recycler;
import io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.TestQuantum;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// A graph admitted after another orders its first accessors of a workspace buffer behind the other's last
/// accessors of it: an external edge, satisfied when that stage submits, or when its graph quiesces without it.
@Timeout(10)
class CrossGraphEdgesTest {

    /// P: stage 0 writes buffer 0, stage 1 reads it.
    private static final StageTopology WRITE_READ = StageTopology.submitted(dependencies(new int[0], new int[] {0}));
    /// G: one stage.
    private static final StageTopology ONE = StageTopology.submitted(dependencies(new int[0]));

    private final TestLake lake = new TestLake();
    private final Recycler recycler = new Recycler();
    private final RecordingStream home = new RecordingStream();
    private final RecordingStream side = new RecordingStream();
    private final WorkspaceOwner owner = new WorkspaceOwner(2);
    private final LanePool pool;

    CrossGraphEdgesTest() {
        this.side.base = 100;
        this.pool = new LanePool(new RecordingStream[] {this.home, this.side});
    }

    private StageGraph graph(StageTopology topology) {
        return new StageGraph(topology, StageGraphFixtures.TestStage::new, this.pool, true, this.lake, this.recycler);
    }

    private void bind(StageGraph graph, GraphShape shape, TestQuantum quantum) {
        this.owner.bind(graph, shape, WorkspaceUse.of(shape), quantum);
    }

    /// Runs the published frames of `graph` that are its stage `index`, leaving the others published.
    private void runStage(StageGraph graph, int index) {
        List<AbstractFrame> frames = take(this.lake);
        for (AbstractFrame frame : frames) {
            if (frame == graph.stage(index)) StageGraphFixtures.run(frame);
            else this.lake.publish(frame);
        }
    }

    private boolean published(AbstractFrame frame) {
        List<AbstractFrame> frames = take(this.lake);
        frames.forEach(this.lake::publish);
        return frames.contains(frame);
    }

    /// Retires every armed boundary on both lanes and drains what that publishes.
    private void retireAll() {
        drain(this.lake);
        while (this.home.armed() > 0 || this.side.armed() > 0) {
            if (this.home.armed() > 0) this.home.retireNext(false);
            if (this.side.armed() > 0) this.side.retireNext(false);
            drain(this.lake);
        }
    }

    @Test
    void aSecondGraphsFirstAccessorWaitsForTheFirstGraphsLastAccessor() {
        StageGraph p = graph(WRITE_READ);
        StageGraph g = graph(ONE);
        var pQuantum = new TestQuantum();
        var gQuantum = new TestQuantum();
        bind(p, TestShapes.of(WRITE_READ, new int[][] {{0}, {0}}, 2), pQuantum);
        runStage(p, 0);
        bind(g, TestShapes.of(ONE, new int[][] {{0}}, 2), gQuantum);
        assertFalse(published(g.stage(0)), "G's writer waits for P's reader");
        runStage(p, 1);
        assertTrue(published(g.stage(0)), "P's reader submitted: G's writer may run");
        runStage(g, 0);
        RecordingStream lane = (RecordingStream) this.pool.lane(g.stage(0).lane);
        int kernel = lane.kernels.indexOf("k0");
        assertTrue(
                lane.kernels.subList(0, kernel).contains("await:" + p.stage(1).marker),
                "G's writer awaited P's reader on the device before it launched: " + lane.kernels);
        retireAll();
        assertEquals("SUCCESS", pQuantum.outcome.join());
        assertEquals("SUCCESS", gQuantum.outcome.join());
    }

    @Test
    void aClosedGraphsMarkerThatALaterGraphAwaitsIsKeptForReuse() {
        StageGraph p = graph(WRITE_READ);
        StageGraph g = graph(ONE);
        var pQuantum = new TestQuantum();
        var gQuantum = new TestQuantum();
        bind(p, TestShapes.of(WRITE_READ, new int[][] {{0}, {0}}, 2), pQuantum);
        runStage(p, 0);
        runStage(p, 1);
        long marker = p.stage(1).marker;
        bind(g, TestShapes.of(ONE, new int[][] {{0}}, 2), gQuantum);
        // P retires and is closed (its shape's pool was released) before G's writer, which took P's reader's
        // marker, submits.
        drainExcept(g.stage(0));
        while (this.home.armed() > 0 || this.side.armed() > 0) {
            if (this.home.armed() > 0) this.home.retireNext(false);
            if (this.side.armed() > 0) this.side.retireNext(false);
            drainExcept(g.stage(0));
        }
        assertEquals("SUCCESS", pQuantum.outcome.join());
        p.close();
        assertFalse(this.home.closedMarkers.contains(marker), "a later graph may still await it");
        runStage(g, 0);
        RecordingStream lane = (RecordingStream) this.pool.lane(g.stage(0).lane);
        assertTrue(lane.kernels.contains("await:" + marker), lane.kernels.toString());
        retireAll();
        assertEquals("SUCCESS", gQuantum.outcome.join());
        this.pool.close();
        assertTrue(this.home.closedMarkers.contains(marker), "the pool's close destroys the spare markers");
    }

    /// Runs every published frame but `kept`, which stays published.
    private void drainExcept(AbstractFrame kept) {
        for (List<AbstractFrame> frames = take(this.lake); !frames.isEmpty(); frames = take(this.lake)) {
            boolean ranAny = false;
            for (AbstractFrame frame : frames) {
                if (frame == kept) this.lake.publish(frame);
                else {
                    StageGraphFixtures.run(frame);
                    ranAny = true;
                }
            }
            if (!ranAny) return;
        }
    }

    @Test
    void independentBuffersDoNotWait() {
        StageGraph p = graph(WRITE_READ);
        StageGraph g = graph(ONE);
        bind(p, TestShapes.of(WRITE_READ, new int[][] {{0}, {0}}, 2), new TestQuantum());
        bind(g, TestShapes.of(ONE, new int[][] {{1}}, 2), new TestQuantum());
        assertTrue(published(g.stage(0)), "G touches only buffer 1");
        retireAll();
    }

    @Test
    void aFailedPredecessorStillReleasesItsBuffers() {
        StageGraph p = graph(WRITE_READ);
        StageGraph g = graph(ONE);
        var pQuantum = new TestQuantum();
        var gQuantum = new TestQuantum();
        stage(p, 1).failure = new IllegalStateException("injected");
        bind(p, TestShapes.of(WRITE_READ, new int[][] {{0}, {0}}, 2), pQuantum);
        bind(g, TestShapes.of(ONE, new int[][] {{0}}, 2), gQuantum);
        retireAll();
        assertEquals("FAILED", pQuantum.outcome.join());
        assertEquals("SUCCESS", gQuantum.outcome.join(), "a failed predecessor does not fail its dependents");
        assertEquals(1, stage(g, 0).launches);
    }

    @Test
    void aCancelledPredecessorThatNeverRanItsStagesReleasesThemAtQuiesce() {
        StageGraph p = graph(WRITE_READ);
        StageGraph g = graph(ONE);
        var pQuantum = new TestQuantum();
        pQuantum.cancelled = true;
        var gQuantum = new TestQuantum();
        bind(p, TestShapes.of(WRITE_READ, new int[][] {{0}, {0}}, 2), pQuantum);
        bind(g, TestShapes.of(ONE, new int[][] {{0}}, 2), gQuantum);
        retireAll();
        assertEquals("CANCELLED", pQuantum.outcome.join());
        assertEquals("SUCCESS", gQuantum.outcome.join());
        RecordingStream lane = (RecordingStream) this.pool.lane(g.stage(0).lane);
        assertTrue(lane.kernels.contains("await:" + p.joinedMarker()), "G awaited P's join: " + lane.kernels);
    }

    @Test
    void aGraphCannotRetireBeforeItsExternalArrivals() {
        StageGraph p = graph(WRITE_READ);
        StageGraph g = graph(ONE);
        var gQuantum = new TestQuantum();
        gQuantum.cancelled = true;
        bind(p, TestShapes.of(WRITE_READ, new int[][] {{0}, {0}}, 2), new TestQuantum());
        runStage(p, 0);
        bind(g, TestShapes.of(ONE, new int[][] {{0}}, 2), gQuantum);
        assertEquals(0, this.home.armed() + this.side.armed() - 0, "nothing of G retired yet");
        assertEquals(0, gQuantum.outcomes.get());
        runStage(p, 1);
        retireAll();
        assertEquals("CANCELLED", gQuantum.outcome.join());
    }

    @Test
    void theOwnerRecordsTheLastGraphOfEachBuffer() {
        StageGraph p = graph(WRITE_READ);
        GraphShape shape = TestShapes.of(WRITE_READ, new int[][] {{0}, {0}}, 2);
        var quantum = new TestQuantum();
        bind(p, shape, quantum);
        assertEquals(shape, this.owner.last(0).shape());
        assertEquals(quantum, this.owner.last(0).quantum());
        assertEquals(List.of(p.stage(1)), List.of(this.owner.last(0).exits()));
        assertEquals(null, this.owner.last(1));
        retireAll();
    }

    @Test
    void aGraphReboundAfterItRetiredDoesNotWaitForItself() {
        StageGraph p = graph(WRITE_READ);
        GraphShape shape = TestShapes.of(WRITE_READ, new int[][] {{0}, {0}}, 2);
        bind(p, shape, new TestQuantum());
        retireAll();
        var second = new TestQuantum();
        bind(p, shape, second);
        retireAll();
        assertTrue(second.outcome.isDone(), "the second binding waited for the first binding of its own graph");
        assertEquals("SUCCESS", second.outcome.join());
    }

    @Test
    void aCancelledGraphBetweenTwoOthersPassesTheEarlierOnesOrderOn() {
        StageGraph a = graph(WRITE_READ);
        StageGraph b = graph(ONE);
        StageGraph c = graph(ONE);
        var bQuantum = new TestQuantum();
        bQuantum.cancelled = true;
        bind(a, TestShapes.of(WRITE_READ, new int[][] {{0}, {0}}, 2), new TestQuantum());
        bind(b, TestShapes.of(ONE, new int[][] {{0}}, 2), bQuantum);
        bind(c, TestShapes.of(ONE, new int[][] {{0}}, 2), new TestQuantum());
        // A's reader submits; B never runs its stage, so C is released onto B's join.
        runStage(a, 0);
        runStage(a, 1);
        retireAll();
        RecordingStream bHome = this.home.kernels.contains("mark:" + b.joinedMarker()) ? this.home : this.side;
        int joined = bHome.kernels.lastIndexOf("mark:" + b.joinedMarker());
        assertTrue(joined >= 0, "B marked its join: " + bHome.kernels);
        assertTrue(
                bHome.kernels.subList(0, joined).contains("await:" + a.stage(1).marker),
                "B's join follows A's reader, so C, ordered behind B's join, follows A too: " + bHome.kernels);
        assertEquals("CANCELLED", bQuantum.outcome.join());
    }
}
