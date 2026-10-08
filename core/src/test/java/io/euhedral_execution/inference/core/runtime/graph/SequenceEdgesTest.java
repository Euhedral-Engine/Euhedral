package io.euhedral_execution.inference.core.runtime.graph;

import static io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.TestQuantum;
import static io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.dependencies;
import static io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.drain;
import static io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.take;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Graphs of one sequence share its carried state (a layer's recurrent state, its KV rows): a later graph's first
/// users of a key follow the earlier graph's last users of it, as for a workspace buffer, but only within the
/// sequence. Other sequences' graphs are not ordered by it.
@Timeout(10)
class SequenceEdgesTest {

    /// Stage 0 carries key 0 (layer 0), stage 1 carries key 1 (layer 1); no workspace buffer.
    private static final StageTopology LAYERS = StageTopology.submitted(dependencies(new int[0], new int[] {0}));

    private final TestLake lake = new TestLake();
    private final StageGraphFixtures.Recycler recycler = new StageGraphFixtures.Recycler();
    private final StageGraphFixtures.RecordingStream home = new StageGraphFixtures.RecordingStream();
    private final StageGraphFixtures.RecordingStream side = new StageGraphFixtures.RecordingStream();
    private final WorkspaceOwner owner = new WorkspaceOwner(0);
    private final LanePool pool;

    SequenceEdgesTest() {
        this.side.base = 100;
        this.pool = new LanePool(new StageGraphFixtures.RecordingStream[] {this.home, this.side});
    }

    private static GraphShape layers() {
        return TestShapes.of(LAYERS, new int[][] {{}, {}}, 0, new int[][] {{0}, {1}}, 2);
    }

    private StageGraph graph() {
        return new StageGraph(LAYERS, StageGraphFixtures.TestStage::new, this.pool, true, this.lake, this.recycler);
    }

    private void bind(StageGraph graph, GraphShape shape, TestQuantum quantum) {
        this.owner.bind(graph, shape, WorkspaceUse.of(shape), CarriedState.of(shape), quantum);
    }

    private boolean published(AbstractFrame frame) {
        List<AbstractFrame> frames = take(this.lake);
        frames.forEach(this.lake::publish);
        return frames.contains(frame);
    }

    private void runStage(StageGraph graph, int index) {
        for (AbstractFrame frame : take(this.lake)) {
            if (frame == graph.stage(index)) StageGraphFixtures.run(frame);
            else this.lake.publish(frame);
        }
    }

    @Test
    void aLaterGraphsCarriedReaderWaitsForTheEarlierGraphsWriter() {
        var sequence = new SequenceOwner();
        GraphShape shape = layers();
        StageGraph earlier = graph(), later = graph();
        var first = new TestQuantum();
        first.sequence = sequence;
        var second = new TestQuantum();
        second.sequence = sequence;
        bind(earlier, shape, first);
        bind(later, shape, second);
        assertFalse(published(later.stage(0)), "layer 0 of the later graph waits for layer 0 of the earlier one");
        runStage(earlier, 0);
        assertTrue(published(later.stage(0)), "and runs once it submitted, ahead of the earlier graph's layer 1");
        drain(this.lake);
    }

    @Test
    void anotherSequencesGraphIsNotOrderedByIt() {
        GraphShape shape = layers();
        StageGraph one = graph(), two = graph();
        var first = new TestQuantum();
        first.sequence = new SequenceOwner();
        var second = new TestQuantum();
        second.sequence = new SequenceOwner();
        bind(one, shape, first);
        bind(two, shape, second);
        assertTrue(published(two.stage(0)), "a different sequence's state is its own");
        drain(this.lake);
    }

    @Test
    void aCancelledEarlierGraphPassesItsWaitsOn() {
        var sequence = new SequenceOwner();
        GraphShape shape = layers();
        StageGraph earliest = graph(), cancelled = graph(), latest = graph();
        var a = new TestQuantum();
        a.sequence = sequence;
        var b = new TestQuantum();
        b.sequence = sequence;
        b.cancelled = true;
        var c = new TestQuantum();
        c.sequence = sequence;
        bind(earliest, shape, a);
        bind(cancelled, shape, b);
        bind(latest, shape, c);
        drain(this.lake);
        while (this.home.armed() > 0 || this.side.armed() > 0) {
            if (this.home.armed() > 0) this.home.retireNext(false);
            if (this.side.armed() > 0) this.side.retireNext(false);
            drain(this.lake);
        }
        assertEquals("CANCELLED", b.outcome.join());
        assertEquals("SUCCESS", c.outcome.join());
        var bHome = this.home.kernels.contains("mark:" + cancelled.joinedMarker()) ? this.home : this.side;
        int joined = bHome.kernels.lastIndexOf("mark:" + cancelled.joinedMarker());
        assertTrue(
                bHome.kernels.subList(0, joined).stream().anyMatch(k -> k.startsWith("await:")),
                "the cancelled graph's join takes over its waits: " + bHome.kernels);
    }
}
