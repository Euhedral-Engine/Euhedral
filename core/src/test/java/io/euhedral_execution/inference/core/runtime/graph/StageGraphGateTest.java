package io.euhedral_execution.inference.core.runtime.graph;

import static io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.dependencies;
import static io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.drain;
import static io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.run;
import static io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.take;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.RecordingStream;
import io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.Recycler;
import io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.TestQuantum;
import java.util.List;
import org.junit.jupiter.api.Test;

/// A stage held by its gate ([StageFrame#passGate]) is published only when it is released, root or not, queues
/// nothing on its lane until then, and keeps its quantum from retiring.
class StageGraphGateTest {

    private static final StageTopology CHAIN =
            StageTopology.submitted(dependencies(new int[0], new int[] {0}, new int[] {1}));

    private final TestLake source = new TestLake();
    private final RecordingStream stream = new RecordingStream();
    private final Recycler recycler = new Recycler();

    /// A stage whose gate is closed until the test releases it.
    private static final class Gated extends StageFrame {
        final boolean gated;
        int gateCalls;

        Gated(StageGraph graph, int stage, boolean gated) {
            super(graph, stage);
            this.gated = gated;
        }

        @Override
        protected boolean passGate() {
            this.gateCalls++;
            return !this.gated;
        }

        void release() {
            releaseGate();
        }

        @Override
        protected void submit() {
            ((RecordingStream) laneStream()).kernel("k" + stage());
        }
    }

    @Test
    void aGatedStageWaitsForItsReleaseAndHoldsTheQuantum() {
        StageGraph graph = new StageGraph(
                CHAIN, (owner, stage) -> new Gated(owner, stage, stage <= 1), this.stream, this.source, this.recycler);
        this.source.admit();
        TestQuantum quantum = new TestQuantum();
        graph.start(quantum);
        assertEquals(List.of(), take(this.source), "the gated root is held");
        assertEquals(1, ((Gated) graph.stage(0)).gateCalls);
        ((Gated) graph.stage(0)).release();
        run(take(this.source).getFirst());
        assertEquals(List.of(), take(this.source), "its gated successor is held in turn");
        assertEquals(List.of("k0"), this.stream.kernels);
        assertFalse(quantum.outcome.isDone(), "a held stage keeps the quantum live");
        ((Gated) graph.stage(1)).release();
        assertEquals(2, drain(this.source), "the released stage ran, then the ungated last one");
        assertEquals(List.of("k0", "k1", "k2"), this.stream.kernels);
        assertEquals(1, ((Gated) graph.stage(1)).gateCalls, "the gate is asked once per quantum");
        this.stream.retireNext(false);
        drain(this.source);
        assertTrue(quantum.outcome.isDone());
        assertEquals("SUCCESS", quantum.outcome.join());
    }
}
