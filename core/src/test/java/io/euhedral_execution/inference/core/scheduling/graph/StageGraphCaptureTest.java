package io.euhedral_execution.inference.core.scheduling.graph;

import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.dependencies;
import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.drain;
import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.stage;
import static io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.take;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.RecordingStream;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.Recycler;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraphFixtures.TestQuantum;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

/// Captured quanta: the first quantum with a key runs stage by stage, the second also records on shadow
/// streams, and later ones replay the captured graph as one frame whose stages are checked, not run.
class StageGraphCaptureTest {

    private static final StageTopology LINEAR =
            StageTopology.submitted(dependencies(new int[0], new int[] {0}, new int[] {1}));
    private static final StageTopology DIAMOND =
            StageTopology.submitted(dependencies(new int[0], new int[] {0}, new int[] {0}, new int[] {1, 2}));

    private final QwenExecutionSource source = new QwenExecutionSource();
    private final Recycler recycler = new Recycler();
    private final List<CapturingStream> shadows = Collections.synchronizedList(new ArrayList<>());
    private final AtomicLong graphs = new AtomicLong();
    /// Whether a capture is open: a shadow joins it by waiting on a marker, as a CUDA stream does.
    private volatile boolean captureOpen;

    /// A device stream that also records on shadows, checks submissions, and captures and launches graphs.
    final class CapturingStream extends RecordingStream {
        final List<String> captured = Collections.synchronizedList(new ArrayList<>());
        final List<Long> destroyed = Collections.synchronizedList(new ArrayList<>());
        boolean capturing;
        volatile boolean failCapture;
        volatile boolean failShadow;
        private final ThreadLocal<long[]> hash = new ThreadLocal<>();
        private final ThreadLocal<CapturingStream> shadow = new ThreadLocal<>();

        @Override
        void kernel(String name) {
            long[] sum = this.hash.get();
            if (sum != null) sum[0] = sum[0] * 31 + name.hashCode();
            CapturingStream mirror = this.shadow.get();
            if (mirror != null) {
                if (!mirror.capturing) throw new AssertionError("shadow submission outside its capture");
                mirror.captured.add(name);
            }
            if (sum == null || mirror != null) super.kernel(name);
        }

        @Override
        public void await(long marker) {
            super.await(marker);
            if (StageGraphCaptureTest.this.captureOpen) this.capturing = true;
        }

        @Override
        public boolean capturesGraphs() {
            return true;
        }

        @Override
        public long submitRecording(
                Runnable launches, boolean overlapPredecessor, GpuStream shadow, boolean shadowOverlap) {
            long[] sum = {17};
            this.hash.set(sum);
            this.shadow.set((CapturingStream) shadow);
            try {
                submit(launches, overlapPredecessor);
            } finally {
                this.hash.remove();
                this.shadow.remove();
            }
            return this.failShadow ? 0 : sum[0] | 1;
        }

        @Override
        public long beginChecking() {
            return 99;
        }

        @Override
        public long submitChecking(Runnable launches, long sink) {
            long[] sum = {17};
            this.hash.set(sum);
            this.selectedDepth++;
            try {
                launches.run();
            } finally {
                this.selectedDepth--;
                this.hash.remove();
            }
            return sum[0] | 1;
        }

        @Override
        public boolean endChecking() {
            return true;
        }

        @Override
        public void beginCapture() {
            StageGraphCaptureTest.this.captureOpen = true;
            this.capturing = true;
            this.kernels.add("begin-capture");
        }

        @Override
        public long endCapture() {
            StageGraphCaptureTest.this.captureOpen = false;
            for (CapturingStream shadow : StageGraphCaptureTest.this.shadows) shadow.capturing = false;
            this.kernels.add("end-capture");
            return this.failCapture ? 0 : 1000 + StageGraphCaptureTest.this.graphs.incrementAndGet();
        }

        @Override
        public void launchGraph(long graph) {
            this.kernels.add("graph:" + graph);
        }

        @Override
        public void destroyGraph(long graph) {
            this.destroyed.add(graph);
        }
    }

    /// A quantum with capture key `key`.
    private static TestQuantum keyed(Object key) {
        TestQuantum quantum = new TestQuantum();
        quantum.captureKey = key;
        return quantum;
    }

    private StageGraph graph(StageTopology topology, LanePool pool) {
        return new StageGraph(
                topology, StageGraphFixtures.TestStage::new, pool, true, this.source, this.recycler, () -> {
                    CapturingStream shadow = new CapturingStream();
                    shadow.base = 1000 * (this.shadows.size() + 1);
                    this.shadows.add(shadow);
                    return shadow;
                });
    }

    /// Runs one quantum to its outcome: every frame it publishes, then its retirement boundary.
    private String run(StageGraph graph, CapturingStream home, TestQuantum quantum) {
        graph.start(quantum);
        drain(this.source);
        home.retireNext(false);
        drain(this.source);
        String outcome = quantum.outcome.join();
        return outcome;
    }

    private static long count(List<String> kernels, String name) {
        synchronized (kernels) {
            return kernels.stream().filter(name::equals).count();
        }
    }

    @Test
    void theSecondQuantumRecordsAndLaterOnesReplayAsOneFrameWithEveryStageChecked() {
        CapturingStream home = new CapturingStream();
        StageGraph graph = graph(LINEAR, LanePool.single(home));

        assertEquals("SUCCESS", run(graph, home, keyed("decode")));
        assertTrue(this.shadows.isEmpty(), "the first quantum with a key only runs");
        assertEquals("SUCCESS", run(graph, home, keyed("decode")));
        assertEquals(1, this.shadows.size());
        assertEquals(List.of("k0", "k1", "k2"), this.shadows.getFirst().captured, "the second records every launch");
        assertEquals(2, count(home.kernels, "k1"), "and runs it as well");

        TestQuantum replayed = keyed("decode");
        graph.start(replayed);
        List<AbstractFrame> frames = take(this.source);
        assertEquals(1, frames.size(), "a replayed quantum is one frame");
        assertInstanceOf(StageGraph.Replay.class, frames.getFirst());
        StageGraphFixtures.run(frames.getFirst());
        assertEquals("graph:1001", home.kernels.getLast(), "the replay launched the captured graph and nothing else");
        assertEquals(2, count(home.kernels, "k1"));
        assertEquals(3, stage(graph, 1).launches, "every stage's host work still ran");
        home.retireNext(false);
        drain(this.source);
        assertEquals("SUCCESS", replayed.outcome.join());
        assertEquals(List.of("commit:0", "commit:1", "commit:2", "retire", "outcome"), replayed.events);
        graph.close();
        assertEquals(List.of(1001L), home.destroyed);
        assertTrue(this.shadows.getFirst().closed);
    }

    @Test
    void aStageThatWouldSubmitSomethingElseFailsItsReplayAndTheCaptureIsDiscarded() {
        CapturingStream home = new CapturingStream();
        StageGraph graph = graph(LINEAR, LanePool.single(home));
        run(graph, home, keyed("decode"));
        run(graph, home, keyed("decode"));

        stage(graph, 1).kernel = "other";
        TestQuantum diverged = keyed("decode");
        assertEquals("FAILED", run(graph, home, diverged));
        assertTrue(diverged.failure.get().getMessage().contains("stage 1 diverged"));
        assertFalse(diverged.events.contains("commit:1"), "a failed replay commits nothing");

        stage(graph, 1).kernel = null;
        int launches = home.kernels.size();
        assertEquals("SUCCESS", run(graph, home, keyed("decode")));
        assertEquals(List.of(1001L), home.destroyed, "the diverged capture was released");
        assertEquals(
                List.of("k0", "k1", "k2"), home.kernels.subList(launches, launches + 3), "and its key starts over");
    }

    @Test
    void aBrokenRecordingKeepsItsKeyRunningStageByStage() {
        CapturingStream home = new CapturingStream();
        home.failShadow = true;
        StageGraph graph = graph(LINEAR, LanePool.single(home));
        for (int quantum = 0; quantum < 4; quantum++) run(graph, home, keyed("draft"));
        assertEquals(4, count(home.kernels, "k2"), "every quantum ran stage by stage");
        assertEquals(1, count(this.shadows.getFirst().kernels, "begin-capture"), "and only one recording was tried");
        assertEquals(List.of(1001L), home.destroyed, "the broken capture was released at once");
    }

    @Test
    void quantaWithoutAKeyAreNeverCaptured() {
        CapturingStream home = new CapturingStream();
        StageGraph graph = graph(LINEAR, LanePool.single(home));
        for (int quantum = 0; quantum < 3; quantum++) run(graph, home, new TestQuantum());
        assertTrue(this.shadows.isEmpty());
        assertEquals(3, count(home.kernels, "k0"));
    }

    @Test
    void theLeastRecentlyUsedCaptureIsReleasedFirst() {
        CapturingStream home = new CapturingStream();
        StageGraph graph = graph(LINEAR, LanePool.single(home));
        for (int key = 0; key <= StageGraph.MAX_CAPTURES; key++) {
            run(graph, home, keyed(key));
            run(graph, home, keyed(key));
        }
        // Key 0's capture (graph 1001) was evicted when the ninth key arrived; it was never in flight.
        assertEquals(List.of(1001L), home.destroyed);
        graph.close();
        assertEquals(StageGraph.MAX_CAPTURES + 1, home.destroyed.size());
    }

    @Test
    void recordingMirrorsEveryCrossLaneMarkerOnTheShadows() {
        CapturingStream home = new CapturingStream(), side = new CapturingStream();
        side.base = 100;
        // Stage 2 runs on lane 1; the rest on lane 0, the home lane.
        LanePool pool = new LanePool(new RecordingStream[] {home, side}, stage -> stage == 2 ? 1 : 0);
        StageGraph graph = graph(DIAMOND, pool);
        run(graph, home, keyed("verify"));
        assertEquals("SUCCESS", run(graph, home, keyed("verify")));
        CapturingStream homeShadow = this.shadows.get(0), sideShadow = this.shadows.get(1);
        assertEquals(List.of("k0", "k1", "k3"), homeShadow.captured);
        assertEquals(List.of("k2"), sideShadow.captured);
        // Each shadow mirrors its lane: preparation and stage markers, the side lane's tail joining home.
        List<String> mirrored = sideShadow.kernels.stream()
                .filter(op -> op.startsWith("await") || op.startsWith("mark"))
                .toList();
        assertEquals(3, mirrored.size(), "the side shadow awaits stage 0, marks stage 2 and its tail: " + mirrored);
        assertTrue(homeShadow.kernels.getFirst().equals("begin-capture"));
        assertEquals("end-capture", homeShadow.kernels.getLast());
        int joins = (int)
                homeShadow.kernels.stream().filter(op -> op.startsWith("await")).count();
        assertEquals(2, joins, "home's shadow awaits stage 2's marker and the side lane's tail");

        TestQuantum replayed = keyed("verify");
        assertEquals("SUCCESS", run(graph, home, replayed));
        assertEquals(
                "graph:1001",
                home.kernels.stream()
                        .filter(op -> op.startsWith("graph"))
                        .findFirst()
                        .orElseThrow());
        assertEquals(2, count(side.kernels, "k2"), "a replay touches no other lane");
    }
}
