package io.euhedral_execution.inference.core.scheduling.graph;

import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.dependencies;
import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.drain;
import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.run;
import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.take;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.RecordingStream;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.Recycler;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.TestQuantum;
import java.util.List;
import java.util.Set;
import java.util.function.IntFunction;
import org.junit.jupiter.api.Test;

/// Stages that need no lane (host stages), stages with nothing to do (completed in place), and
/// stages that complete later than their own frame (deferred): what each does to the edges, the
/// lanes and the quantum's retirement.
class StageGraphHostStageTest {

    private static final StageTopology CHAIN =
            StageTopology.submitted(dependencies(new int[0], new int[] {0}, new int[] {1}));

    private final QwenExecutionSource source = new QwenExecutionSource();
    private final RecordingStream stream = new RecordingStream();
    private final Recycler recycler = new Recycler();

    /// A stage with a role chosen by the test.
    private static final class Probe extends StageFrame {
        boolean host;
        volatile boolean skip;
        boolean defer;
        int runs;
        Runnable inside;

        Probe(StageGraph graph, int stage) {
            super(graph, stage);
        }

        @Override
        protected boolean host() {
            return this.host;
        }

        @Override
        protected boolean skips() {
            return this.skip;
        }

        @Override
        protected void submit() {
            this.runs++;
            if (this.defer) deferCompletion();
            if (this.inside != null) this.inside.run();
            if (!this.host) ((RecordingStream) laneStream()).kernel("k" + stage());
        }

        void completeNow() {
            completeDeferred();
        }
    }

    private record Role(boolean host, boolean skip, boolean defer) {}

    private StageGraph graph(StageTopology topology, IntFunction<Role> roles) {
        return new StageGraph(
                topology,
                (graph, stage) -> {
                    Probe probe = new Probe(graph, stage);
                    Role role = roles.apply(stage);
                    probe.host = role.host();
                    probe.skip = role.skip();
                    probe.defer = role.defer();
                    return probe;
                },
                this.stream,
                this.source,
                this.recycler);
    }

    private static Role role(boolean host, boolean skip, boolean defer) {
        return new Role(host, skip, defer);
    }

    private TestQuantum start(StageGraph graph) {
        this.source.admit();
        TestQuantum quantum = new TestQuantum();
        graph.start(quantum);
        return quantum;
    }

    private static Probe probe(StageGraph graph, int stage) {
        return (Probe) graph.stage(stage);
    }

    @Test
    void aHostStageSubmitsNothingToAnyLane() {
        StageGraph graph = graph(CHAIN, stage -> role(stage == 1, false, false));
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());
        run(take(this.source).getFirst());
        assertEquals(1, probe(graph, 1).runs, "the host stage ran");
        assertEquals(List.of("k0"), this.stream.kernels, "and queued nothing: its device successor is next");
        run(take(this.source).getFirst());
        assertEquals(List.of("k0", "k2"), this.stream.kernels);
        this.stream.retireNext(false);
        drain(this.source);
        assertEquals("SUCCESS", quantum.outcome.join());
    }

    @Test
    void aStageWithNothingToDoCompletesInPlaceOnTheThreadThatSatisfiedItsLastEdge() {
        StageGraph graph = graph(CHAIN, stage -> role(false, stage == 1, false));
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());
        // Stage 1 was not published: its edge was satisfied and it completed in place, releasing stage 2.
        List<AbstractFrame> next = take(this.source);
        assertEquals(List.of(graph.stage(2)), next);
        assertEquals(0, probe(graph, 1).runs);
        run(next.getFirst());
        assertEquals(List.of("k0", "k2"), this.stream.kernels);
        this.stream.retireNext(false);
        drain(this.source);
        assertEquals("SUCCESS", quantum.outcome.join());
    }

    @Test
    void aChainOfSkippedStagesCompletesInPlaceAndLeavesTheQuantumLive() {
        StageTopology longChain = StageTopology.submitted(
                dependencies(new int[0], new int[] {0}, new int[] {1}, new int[] {2}, new int[] {3}));
        StageGraph graph = graph(longChain, stage -> role(false, stage >= 1 && stage <= 3, false));
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());
        assertEquals(List.of(graph.stage(4)), take(this.source));
        run(graph.stage(4));
        this.stream.retireNext(false);
        drain(this.source);
        assertEquals("SUCCESS", quantum.outcome.join());
    }

    @Test
    void aSkippedStageStillJoinsTheDeviceOrderOfItsPredecessors() {
        RecordingStream home = this.stream, side = new RecordingStream();
        side.base = 100;
        // Stage 1 (skipped) is placed on the side lane; stages 0 and 2 on the home lane.
        LanePool pool = new LanePool(new RecordingStream[] {home, side}, stage -> stage == 1 ? 1 : 0);
        StageGraph graph = new StageGraph(
                CHAIN,
                (owner, stage) -> {
                    Probe probe = new Probe(owner, stage);
                    probe.skip = stage == 1;
                    return probe;
                },
                pool,
                true,
                this.source,
                this.recycler);
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());
        assertEquals(List.of(graph.stage(2)), take(this.source));
        run(graph.stage(2));
        // Markers 1-3: preparation, then stages 0 and 1. The skipped stage waited for stage 0 and recorded its own
        // marker on its lane, so stage 2 on the home lane waits for it: nothing is lost by skipping.
        assertEquals(List.of("await:2", "mark:3"), side.kernels.subList(0, 2));
        assertEquals(
                "await:3",
                home.kernels.stream()
                        .filter(k -> k.equals("await:3"))
                        .findFirst()
                        .orElseThrow());
        assertFalse(home.kernels.contains("k1") || side.kernels.contains("k1"), "the skipped stage launched nothing");
        home.retireNext(false);
        drain(this.source);
        assertEquals("SUCCESS", quantum.outcome.join());
    }

    @Test
    void aDeferredStageReleasesItsSuccessorsOnlyWhenItsWorkCompletes() {
        StageGraph graph = graph(CHAIN, stage -> role(stage == 1, false, stage == 1));
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());
        run(take(this.source).getFirst());
        assertEquals(1, probe(graph, 1).runs);
        assertTrue(take(this.source).isEmpty(), "the frame ended but its work has not completed");
        assertEquals(0, this.stream.armed(), "and the quantum cannot retire while it waits");

        probe(graph, 1).completeNow();
        assertEquals(List.of(graph.stage(2)), take(this.source), "the completion published the successor");
        run(graph.stage(2));
        this.stream.retireNext(false);
        drain(this.source);
        assertEquals("SUCCESS", quantum.outcome.join());
    }

    @Test
    void aCompletionBeforeTheFramesEndIsHeldUntilItAndPublishesOnce() {
        StageGraph graph = graph(CHAIN, stage -> role(stage == 1, false, stage == 1));
        probe(graph, 1).inside = () -> probe(graph, 1).completeNow();
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());
        run(take(this.source).getFirst());
        assertEquals(
                List.of(graph.stage(2)), take(this.source), "published once, after both the end and the completion");
        run(graph.stage(2));
        this.stream.retireNext(false);
        drain(this.source);
        assertEquals("SUCCESS", quantum.outcome.join());
    }

    @Test
    void aFailedQuantumStillWaitsForADeferredStageAndNeverPublishesPastIt() {
        StageGraph graph = graph(CHAIN, stage -> role(stage == 1, false, stage == 1));
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());
        run(take(this.source).getFirst());
        quantum.fail(new IllegalStateException("another stage failed"));
        probe(graph, 1).completeNow();
        assertTrue(take(this.source).isEmpty(), "nothing is published into a stopped quantum");
        this.stream.retireNext(false);
        drain(this.source);
        assertEquals("FAILED", quantum.outcome.join());
        assertEquals(Set.of(graph), Set.copyOf(this.recycler.recycled));
    }
}
