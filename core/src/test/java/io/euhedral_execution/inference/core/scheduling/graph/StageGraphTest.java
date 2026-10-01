package io.euhedral_execution.inference.core.scheduling.graph;

import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.dependencies;
import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.drain;
import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.graph;
import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.run;
import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.stage;
import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.take;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.RecordingStream;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.Recycler;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.TestQuantum;
import io.euhedral_execution.inference.core.scheduling.graph.StageTopology.Boundary;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Structural contract of the reusable frame DAG: root-only admission, readiness created by
/// submissions, publication to the source, one retirement boundary per quantum, and recycling.
class StageGraphTest {

    private static final StageTopology LINEAR =
            StageTopology.submitted(dependencies(new int[0], new int[] {0}, new int[] {1}));
    private static final StageTopology DIAMOND =
            StageTopology.submitted(dependencies(new int[0], new int[] {0}, new int[] {0}, new int[] {1, 2}));

    private final QwenExecutionSource source = new QwenExecutionSource();
    private final RecordingStream stream = new RecordingStream();
    private final Recycler recycler = new Recycler();

    @Test
    void linearGraphExposesOnlyTheRootAndEachSubmissionPublishesOnlyItsSuccessor() {
        StageGraph graph = graph(LINEAR, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);

        List<AbstractFrame> exposed = take(this.source);
        assertEquals(List.of(stage(graph, 0)), exposed, "admission exposes only the root");
        assertTrue(take(this.source).isEmpty(), "nothing else is ready before the root runs");

        run(exposed.getFirst());
        assertEquals(1, stage(graph, 0).launches);
        assertEquals(0, stage(graph, 1).launches, "a stage never runs its successor");
        List<AbstractFrame> second = take(this.source);
        assertEquals(List.of(stage(graph, 1)), second, "the root's submission published B");
        assertTrue(take(this.source).isEmpty(), "C waits for B's edge");

        run(second.getFirst());
        assertEquals(0, stage(graph, 2).launches);
        List<AbstractFrame> third = take(this.source);
        assertEquals(List.of(stage(graph, 2)), third);
        assertEquals(0, this.stream.armed(), "no retirement boundary while stages remain");

        run(third.getFirst());
        assertEquals(List.of("k0", "k1", "k2"), this.stream.kernels);
        assertEquals(1, this.stream.armed(), "the last stage arms the quantum's single boundary");
        assertFalse(quantum.outcome.isDone());
    }

    @Test
    void fanOutPublishesBothBranchesAndFanInPublishesTheJoinExactlyOnce() {
        StageGraph graph = graph(DIAMOND, this.stream, this.source, this.recycler);
        start(graph);
        run(take(this.source).getFirst());

        List<AbstractFrame> branches = take(this.source);
        assertEquals(Set.of(stage(graph, 1), stage(graph, 2)), new HashSet<>(branches), "A released B and C");
        // Euhedral may run the branches in either order; run C first.
        run(stage(graph, 2));
        assertTrue(take(this.source).isEmpty(), "D is not ready after one incoming edge");
        run(stage(graph, 1));
        assertEquals(List.of(stage(graph, 3)), take(this.source), "D is published once, after both edges");
        assertTrue(take(this.source).isEmpty());
    }

    @Test
    void sourceDeliversSuccessorsThroughItsOwnPullWithoutInlineExecution() {
        StageGraph graph = graph(LINEAR, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        AtomicInteger depth = new AtomicInteger();
        List<Integer> depths = new ArrayList<>();
        for (int stage = 0; stage < 3; stage++) {
            stage(graph, stage).beforeLaunch = () -> depths.add(depth.get());
        }
        this.stream.retireOnNotify = true;

        long delivered = this.source.pull(
                frame -> {
                    depth.incrementAndGet();
                    try {
                        run(frame);
                    } finally {
                        depth.decrementAndGet();
                    }
                },
                frame -> false,
                Long.MAX_VALUE);

        // A, B, C and the retirement frame were each handed to Euhedral's consumer; none ran nested
        // inside its predecessor's lifecycle.
        assertEquals(4, delivered);
        assertEquals(List.of(1, 1, 1), depths);
        assertEquals("SUCCESS", quantum.outcome.join());
    }

    @Test
    void crossLaneDependenciesAreOrderedByMarkersAndUsedLanesJoinTheHomeLaneBeforeRetirement() {
        RecordingStream home = this.stream, side = new RecordingStream(), idle = new RecordingStream();
        side.base = 100;
        idle.base = 200;
        // Stage 2 runs on lane 1; every other stage on lane 0 (the graph's home lane). Lane 2 stays unused.
        LanePool pool = new LanePool(
                new RecordingStream[] {home, side, idle}, LanePool.Placement.RANDOM, stage -> stage == 2 ? 1 : 0);
        StageGraph graph =
                new StageGraph(DIAMOND, StageGraphFixtures.TestStage::new, pool, true, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());
        take(this.source);
        run(stage(graph, 2));
        run(stage(graph, 1));
        run(take(this.source).getFirst());
        // Markers 1-4: preparation, then stages 0, 1 and 2; lanes 1 and 2 get tails 101 and 201.
        assertEquals(List.of("await:2", "k2", "mark:4"), side.kernels.subList(0, 3));
        assertEquals(List.of("mark:1", "k0", "mark:2", "k1", "mark:3"), home.kernels.subList(0, 5));
        assertEquals("await:4", home.kernels.get(5), "stage 3 awaits the side lane's producer");
        assertEquals("k3", home.kernels.get(6));
        assertEquals(List.of("mark:101"), side.kernels.subList(3, 4), "quiesce marks the used side lane");
        assertEquals("await:101", home.kernels.get(7));
        assertTrue(idle.kernels.isEmpty(), "an unused lane is not joined");
        assertEquals(1, home.armed());
        home.retireNext(false);
        drain(this.source);
        assertEquals("SUCCESS", quantum.outcome.join());
        graph.close();
        assertFalse(home.closed, "a shared pool outlives its graphs");
        assertEquals(Set.of(1L, 2L, 3L, 4L, 101L, 201L), new HashSet<>(home.closedMarkers));
    }

    @Test
    void pathPlacementKeepsTheLongestChainOnTheHomeLaneWhicheverBranchRunsFirst() {
        RecordingStream home = this.stream, side = new RecordingStream(), other = new RecordingStream();
        side.base = 100;
        other.base = 200;
        // 0 -> 1 -> 3 -> 4 is the longest path; 0 -> 2 -> 4 is the side branch.
        StageTopology topology = StageTopology.submitted(new int[][] {{}, {0}, {0}, {1}, {2, 3}});
        LanePool pool = new LanePool(new RecordingStream[] {home, side, other}, LanePool.Placement.PATH);
        StageGraph graph =
                new StageGraph(topology, StageGraphFixtures.TestStage::new, pool, true, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());
        take(this.source);
        run(stage(graph, 2));
        run(stage(graph, 1));
        run(take(this.source).getFirst());
        run(take(this.source).getFirst());
        // The side branch takes a random lane, which may be the home lane too; the chain never leaves it.
        assertEquals(
                List.of("k0", "k1", "k3", "k4"),
                home.kernels.stream()
                        .filter(name -> name.startsWith("k") && !name.equals("k2"))
                        .toList(),
                "the side branch running first does not take the chain's lane");
        home.retireNext(false);
        drain(this.source);
        assertEquals("SUCCESS", quantum.outcome.join());
        graph.close();
    }

    @Test
    void submissionEdgeReleasesTheSuccessorBeforeTheProducersDeviceWorkRetires() {
        StageGraph graph = graph(LINEAR, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());
        // Kernel A has not retired: the stream holds every boundary, and none is even armed yet.
        assertEquals(0, this.stream.armed());
        run(take(this.source).getFirst());
        run(take(this.source).getFirst());
        assertEquals(List.of("k0", "k1", "k2"), this.stream.kernels, "device order follows stream order");
        assertFalse(quantum.outcome.isDone());
        assertEquals(1, this.stream.armed(), "exactly one device-completion boundary per quantum");
    }

    @Test
    void persistentStateIsCommittedOnlyAfterTheQuantumsDeviceWorkRetires() {
        StageGraph graph = graph(LINEAR, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        drain(this.source);
        assertEquals(List.of("k0", "k1", "k2"), this.stream.kernels);
        assertTrue(quantum.events.isEmpty(), "submitted state is not committed before retirement");
        assertTrue(this.recycler.recycled.isEmpty(), "the graph is not recycled while work is in flight");

        this.stream.retireNext(true);
        assertTrue(quantum.events.isEmpty(), "the driver callback only enqueues the retirement frame");
        drain(this.source);

        assertEquals(List.of("commit:0", "commit:1", "commit:2", "retire", "outcome"), quantum.events);
        assertEquals(List.of(graph), this.recycler.recycled);
        assertEquals("SUCCESS", quantum.outcome.join());
    }

    @Test
    void recycledGraphIsReusableBeforeTheOutcomeIsPublished() {
        StageGraph graph = graph(LINEAR, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        // Recorded, not asserted, inside the hook: the retirement frame never lets a failure escape.
        List<Object> atOutcome = new ArrayList<>();
        quantum.onOutcome = () -> {
            atOutcome.add(List.copyOf(this.recycler.recycled));
            atOutcome.add(this.source.activeGraphs());
        };
        this.stream.retireOnNotify = true;
        drain(this.source);
        assertEquals(1, quantum.outcomes.get());
        assertEquals(List.of(List.of(graph), 0), atOutcome, "recycled and no longer active once the outcome is seen");
        assertNull(graph.quantum(), "no stale quantum remains bound");
    }

    @Test
    void deviceCompletionEdgeWaitsForTheProducersRetirement() {
        StageTopology topology = StageTopology.of(
                dependencies(new int[0], new int[] {0}, new int[] {0, 1}),
                new Boundary[][] {{}, {Boundary.RETIRED}, {Boundary.SUBMITTED, Boundary.SUBMITTED}});
        StageGraph graph = graph(topology, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());

        assertTrue(take(this.source).isEmpty(), "a device-completion edge is not satisfied by submission");
        assertEquals(1, this.stream.armed(), "the producer armed its edge after submitting");
        this.stream.retireNext(true);
        List<AbstractFrame> edge = take(this.source);
        assertEquals(1, edge.size());
        assertFalse(edge.getFirst() instanceof StageFrame, "the edge's retirement is confirmed by its own frame");
        run(edge.getFirst());
        assertEquals(List.of(stage(graph, 1)), take(this.source));
        run(stage(graph, 1));
        run(take(this.source).getFirst());
        assertEquals(List.of("k0", "k1", "k2"), this.stream.kernels);

        this.stream.retireNext(false);
        drain(this.source);
        assertEquals("SUCCESS", quantum.outcome.join());
    }

    @Test
    void failedDeviceCompletionEdgeNeverReleasesItsConsumer() {
        StageTopology topology =
                StageTopology.of(dependencies(new int[0], new int[] {0}), new Boundary[][] {{}, {Boundary.RETIRED}});
        StageGraph graph = graph(topology, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());
        this.stream.failures.put(1L, new IllegalStateException("device fault"));
        this.stream.retireNext(true);
        drain(this.source);

        assertEquals(0, stage(graph, 1).launches);
        assertEquals(1, this.stream.armed(), "the failed quantum still retires through one boundary");
        this.stream.retireNext(true);
        drain(this.source);
        assertEquals("FAILED", quantum.outcome.join());
        assertEquals(List.of("release:0", "retire", "outcome"), quantum.events);
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void stagesOnDifferentWorkersSubmitToTheSameQuantumStreamInDependencyOrder() throws Exception {
        StageGraph graph = graph(LINEAR, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        List<ExecutorService> workers = List.of(worker("worker-a"), worker("worker-b"), worker("worker-c"));
        try {
            for (ExecutorService worker : workers) {
                List<AbstractFrame> ready = take(this.source);
                assertEquals(1, ready.size());
                worker.submit(() -> run(ready.getFirst())).get(10, TimeUnit.SECONDS);
            }
        } finally {
            for (ExecutorService worker : workers) worker.shutdownNow();
        }
        assertEquals(List.of("k0", "k1", "k2"), this.stream.kernels);
        assertEquals(List.of("worker-a", "worker-b", "worker-c"), this.stream.threads);
        assertEquals(1, this.stream.armed());
        this.stream.retireNext(true);
        drain(this.source);
        assertEquals("SUCCESS", quantum.outcome.join());
    }

    @Test
    void failedSubmissionReleasesNoDescendantAndRetiresSubmittedWorkOnce() {
        StageGraph graph = graph(LINEAR, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        IllegalStateException launchFailure = new IllegalStateException("launch failed");
        stage(graph, 1).failure = launchFailure;
        run(take(this.source).getFirst());
        run(take(this.source).getFirst());

        assertTrue(take(this.source).isEmpty(), "C never becomes ready");
        assertEquals(0, stage(graph, 2).launches);
        assertSame(launchFailure, quantum.failure.get());
        assertTrue(this.recycler.recycled.isEmpty(), "submitted work still owns the graph");
        assertEquals(1, this.stream.armed());

        this.stream.retireNext(true);
        drain(this.source);
        assertEquals(List.of("release:0", "release:1", "retire", "outcome"), quantum.events);
        assertEquals(1, quantum.outcomes.get(), "terminal failure is published exactly once");
        assertEquals("FAILED", quantum.outcome.join());
    }

    @Test
    void stageErrorIsRecordedAndNeverEscapesIntoEuhedral() {
        StageGraph graph = graph(LINEAR, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        OutOfMemoryError exhaustion = new OutOfMemoryError("injected host exhaustion");
        stage(graph, 1).error = exhaustion;
        run(take(this.source).getFirst());
        AbstractFrame failing = take(this.source).getFirst();

        // Escaping into Euhedral, the Error would complete this graph's source or end the worker.
        assertDoesNotThrow(failing::execute);
        failing.doFinally();
        assertSame(exhaustion, quantum.failure.get());
        assertTrue(take(this.source).isEmpty(), "C never becomes ready");
        assertEquals(1, this.stream.armed(), "the quantum still retires through its single boundary");

        this.stream.retireNext(true);
        drain(this.source);
        assertEquals(List.of("release:0", "release:1", "retire", "outcome"), quantum.events);
        assertEquals("FAILED", quantum.outcome.join());
    }

    @Test
    void rejectedStageFailsTheQuantumWithoutTouchingEuhedralsSharedRejection() {
        StageGraph graph = graph(LINEAR, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        IllegalStateException shared = new IllegalStateException("Frame was rejected because its cache retired");
        run(take(this.source).getFirst());
        // Euhedral finalizes a frame it could not deliver without running it.
        take(this.source).getFirst().doFinallyWithError(shared);

        assertEquals(0, stage(graph, 1).launches);
        assertTrue(take(this.source).isEmpty(), "C never becomes ready");
        assertNotSame(shared, quantum.failure.get());
        assertSame(shared, quantum.failure.get().getCause());
        quantum.fail(new IllegalStateException("later failure"));
        assertEquals(0, shared.getSuppressed().length, "failures never accumulate on Euhedral's instance");

        this.stream.retireNext(true);
        drain(this.source);
        assertEquals(List.of("release:0", "retire", "outcome"), quantum.events);
        assertEquals("FAILED", quantum.outcome.join());
    }

    @Test
    void rejectedRetirementFrameStillRetiresTheQuantumOnce() {
        StageGraph graph = graph(LINEAR, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        drain(this.source);
        this.stream.retireNext(true);
        List<AbstractFrame> retirement = take(this.source);
        assertEquals(1, retirement.size());

        retirement.getFirst().doFinallyWithError(new IllegalStateException("no routable downstream"));

        assertEquals(List.of("commit:0", "commit:1", "commit:2", "retire", "outcome"), quantum.events);
        assertEquals("SUCCESS", quantum.outcome.join());
        assertEquals(List.of(graph), this.recycler.recycled);
        assertEquals(0, this.source.activeGraphs());
        assertTrue(take(this.source).isEmpty());
    }

    @Test
    void rejectedDeviceCompletionEdgeIsStillConfirmedAndResolvedOnce() {
        StageTopology topology =
                StageTopology.of(dependencies(new int[0], new int[] {0}), new Boundary[][] {{}, {Boundary.RETIRED}});
        StageGraph graph = graph(topology, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());
        this.stream.retireNext(true);
        List<AbstractFrame> edge = take(this.source);
        assertEquals(1, edge.size());

        edge.getFirst().doFinallyWithError(new IllegalStateException("no routable downstream"));

        assertEquals(List.of(1L), this.stream.confirmed, "the producer's boundary is confirmed");
        assertEquals(List.of(stage(graph, 1)), take(this.source), "the consumer is published once");
        run(stage(graph, 1));
        this.stream.retireNext(true);
        drain(this.source);
        assertEquals("SUCCESS", quantum.outcome.join());
    }

    @Test
    void failedBranchKeepsTheJoinUnreadyEvenWhenTheOtherBranchSucceedsLater() {
        StageGraph graph = graph(DIAMOND, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        stage(graph, 1).failure = new IllegalStateException("branch B failed");
        run(take(this.source).getFirst());
        assertEquals(Set.of(stage(graph, 1), stage(graph, 2)), new HashSet<>(take(this.source)));
        run(stage(graph, 1));
        run(stage(graph, 2));

        assertTrue(take(this.source).isEmpty());
        assertEquals(0, stage(graph, 3).launches);
        assertEquals(0, stage(graph, 2).launches, "C observed the failure before launching");
        this.stream.retireNext(true);
        drain(this.source);
        assertEquals("FAILED", quantum.outcome.join());
    }

    @Test
    void cancellationStopsUnsentDescendantsButRetiresSubmittedWorkBeforeRecycling() {
        StageGraph graph = graph(LINEAR, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        run(take(this.source).getFirst());
        quantum.cancelled = true;
        List<AbstractFrame> ready = take(this.source);
        assertEquals(1, ready.size(), "B was ready before cancellation");
        run(ready.getFirst());

        assertEquals(0, stage(graph, 1).launches, "cancelled work is not submitted");
        assertTrue(take(this.source).isEmpty(), "no descendant becomes ready");
        assertTrue(this.recycler.recycled.isEmpty(), "kernel A may still be running");
        assertEquals(1, this.stream.armed());

        this.stream.retireNext(true);
        drain(this.source);
        assertEquals(List.of("release:0", "retire", "outcome"), quantum.events);
        assertEquals(List.of(graph), this.recycler.recycled);
        assertEquals("CANCELLED", quantum.outcome.join());
    }

    @Test
    void failedBoundaryRegistrationRecoversTheStreamAndStillRetiresOnce() {
        StageGraph graph = graph(LINEAR, this.stream, this.source, this.recycler);
        TestQuantum quantum = start(graph);
        IllegalStateException registration = new IllegalStateException("event record failed");
        this.stream.registrationFailure = registration;
        drain(this.source);

        assertEquals(1, this.stream.recoveries.get(), "the stream proves idleness before storage is released");
        assertSame(registration, quantum.failure.get());
        assertEquals(List.of("release:0", "release:1", "release:2", "retire", "outcome"), quantum.events);
        assertEquals("FAILED", quantum.outcome.join());
    }

    @Test
    void repeatedQuantaReuseTheSameGraphFramesAndResetReadiness() {
        StageGraph graph = graph(DIAMOND, this.stream, this.source, this.recycler);
        List<AbstractFrame> frames = List.of(stage(graph, 0), stage(graph, 1), stage(graph, 2), stage(graph, 3));
        this.stream.retireOnNotify = true;
        for (int quantum = 0; quantum < 5; quantum++) {
            TestQuantum bound = start(graph);
            assertSame(bound, graph.quantum());
            drain(this.source);
            assertEquals("SUCCESS", bound.outcome.join());
            assertEquals(List.of("commit:0", "commit:1", "commit:2", "commit:3", "retire", "outcome"), bound.events);
            assertNull(graph.quantum(), "no stale quantum remains after recycling");
        }
        for (int stage = 0; stage < 4; stage++) assertSame(frames.get(stage), stage(graph, stage));
        assertEquals(5, this.recycler.recycled.size());
        for (StageGraph recycled : this.recycler.recycled) assertSame(graph, recycled);
        for (int stage = 0; stage < 4; stage++) assertEquals(5, stage(graph, stage).launches);
        assertEquals(20, this.stream.kernels.size());
    }

    @Test
    void graphRejectsASecondQuantumWhileOneIsBound() {
        StageGraph graph = graph(LINEAR, this.stream, this.source, this.recycler);
        start(graph);
        assertThrows(IllegalStateException.class, () -> graph.start(new TestQuantum()));
        assertThrows(IllegalStateException.class, graph::close);
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void concurrentIncomingEdgesPublishTheJoinExactlyOnce() throws Exception {
        int width = 16;
        int[][] dependencies = new int[width + 2][];
        dependencies[0] = new int[0];
        for (int branch = 1; branch <= width; branch++) dependencies[branch] = new int[] {0};
        dependencies[width + 1] = new int[width];
        for (int branch = 0; branch < width; branch++) dependencies[width + 1][branch] = branch + 1;
        StageGraph graph = graph(StageTopology.submitted(dependencies), this.stream, this.source, this.recycler);
        this.stream.retireOnNotify = true;
        ExecutorService pool = Executors.newFixedThreadPool(width);
        try {
            for (int round = 0; round < 50; round++) {
                TestQuantum quantum = start(graph);
                run(take(this.source).getFirst());
                List<AbstractFrame> branches = take(this.source);
                assertEquals(width, branches.size());
                CyclicBarrier barrier = new CyclicBarrier(width);
                List<Future<?>> finishing = new ArrayList<>();
                for (AbstractFrame branch : branches) {
                    finishing.add(pool.submit(() -> {
                        barrier.await();
                        run(branch);
                        return null;
                    }));
                }
                for (Future<?> future : finishing) future.get(10, TimeUnit.SECONDS);
                List<AbstractFrame> join = take(this.source);
                assertEquals(List.of(stage(graph, width + 1)), join, "the join is published exactly once");
                run(join.getFirst());
                drain(this.source);
                assertEquals("SUCCESS", quantum.outcome.join());
            }
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void topologyRejectsCyclesDuplicatesAndMismatchedBoundaries() {
        assertThrows(IllegalArgumentException.class, () -> StageTopology.submitted(new int[][] {{0}}));
        assertThrows(IllegalArgumentException.class, () -> StageTopology.submitted(new int[][] {{}, {0, 0}}));
        assertThrows(
                IllegalArgumentException.class,
                () -> StageTopology.of(new int[][] {{}, {0}}, new Boundary[][] {{}, {}}));
        StageTopology topology = StageTopology.of(new int[][] {{}, {0}, {0, 1}}, new Boundary[][] {
            {}, {Boundary.RETIRED}, {Boundary.SUBMITTED, Boundary.SUBMITTED}
        });
        assertEquals(List.of(0), List.of(topology.roots()[0]));
        assertEquals(2, topology.inDegree(2));
        assertEquals(List.of(2), List.of(topology.submittedSuccessors(0)[0]));
        assertEquals(List.of(1), List.of(topology.retiredSuccessors(0)[0]));
    }

    private TestQuantum start(StageGraph graph) {
        this.source.admit();
        TestQuantum quantum = new TestQuantum();
        graph.start(quantum);
        return quantum;
    }

    private static ExecutorService worker(String name) {
        ThreadFactory factory = runnable -> {
            Thread thread = new Thread(runnable, name);
            thread.setDaemon(true);
            return thread;
        };
        return Executors.newSingleThreadExecutor(factory);
    }
}
