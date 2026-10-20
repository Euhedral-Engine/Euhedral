package io.euhedral_execution.inference.core.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.core.config.LatticeConfig;
import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.control_plane.ControlPlaneShard;
import io.euhedral_execution.core.flow_control.LatticeEdge;
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.AttentionStates;
import io.euhedral_execution.inference.core.model.qwen38.Execution;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionFixtures;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.GdnState;
import io.euhedral_execution.inference.core.model.qwen38.GdnStates;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactHeader;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.WeightLoader;
import io.euhedral_execution.inference.core.model.qwen38.loader.Weights;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.ExecutionMode;

@org.junit.jupiter.api.parallel.Execution(ExecutionMode.SAME_THREAD)
class EuhedralInferenceRuntimeLatticeTest {

    private static final Path DEFAULT_COMPACT_ARTIFACT =
            Path.of(System.getProperty("euhedral.qwen.artifact", "unset-euhedral.qwen.artifact"));

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void realLatticeDeliversARetirementThatOnlyADriverCallbackEnqueued() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(!cpus.isEmpty());
        var lattice = createLattice(cpus);
        var gpu = new HeldGpu();
        var plan = new ExecutionPlan(
                ExecutionFixtures.weights(),
                ExecutionFixtures.norm(),
                List.of(ExecutionFixtures.q3("projection", 64, 201)));
        try {
            lattice.start();
            var runtime = new Execution(lattice, plan, gpu);
            try {
                long sequenceId = 5800;
                for (var kind : List.of(Quantum.ExecutionKind.PREFILL, Quantum.ExecutionKind.DECODE)) {
                    var sequence = new Sequence(++sequenceId);
                    var outcome = runtime.submit(new Quantum(plan, sequence, kind, 0, new int[] {1}));
                    awaitHeld(gpu.stream, 1);
                    assertFalse(outcome.isDone(), "every stage submitted; only the device boundary is pending");
                    // Announced from another thread, as the CUDA driver does; nothing observes the source.
                    Thread driver = Thread.ofPlatform().start(() -> gpu.stream.release(null));
                    driver.join();
                    assertEquals(
                            Quantum.Status.SUCCESS,
                            outcome.get(10, TimeUnit.SECONDS).status());
                    assertEquals(0, gpu.stream.held(), "one device-completion boundary per quantum");
                    sequence.complete();
                }
                assertEquals(List.of("embed", "norm", "linear:201", "embed", "norm", "linear:201"), gpu.operations);
                assertEquals(0, gpu.synchronizations, "ordinary execution uses no device-wide barrier");
            } finally {
                runtime.close();
            }
            assertFalse(runtime.isAttached());
        } finally {
            LatticeTesting.closeAndAwait(lattice);
        }
    }

    /// On the real lattice: once the runtime has been idle its sinks detach and the lattice has nothing left
    /// to poll; a quantum submitted after that attaches fresh sinks and runs as before.
    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void theLakeDetachesWhenIdleAndTheNextQuantumAttachesItAgain() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(!cpus.isEmpty());
        System.setProperty("euhedral.lake.idle-detach-ms", "50");
        var lattice = createLattice(cpus);
        var gpu = new HeldGpu();
        var plan = new ExecutionPlan(
                ExecutionFixtures.weights(),
                ExecutionFixtures.norm(),
                List.of(ExecutionFixtures.q3("projection", 64, 201)));
        try {
            lattice.start();
            var runtime = new Execution(lattice, plan, gpu);
            try {
                long sequenceId = 5900;
                for (int round = 0; round < 3; round++) {
                    var sequence = new Sequence(++sequenceId);
                    var outcome =
                            runtime.submit(new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1}));
                    // The quantum holds a retirement boundary on each lane its stages took, as many as the lane count
                    // (the processor count) and its placement give: a driver releases them until it completes.
                    releaseUntilDone(gpu.stream, outcome);
                    assertEquals(
                            Quantum.Status.SUCCESS,
                            outcome.get(10, TimeUnit.SECONDS).status());
                    sequence.complete();
                    long deadline = System.nanoTime() + 20_000_000_000L;
                    while (runtime.isAttached() && System.nanoTime() < deadline) Thread.sleep(20);
                    assertFalse(runtime.isAttached(), "round " + round + ": the idle runtime let its sinks go");
                }
            } finally {
                runtime.close();
            }
        } finally {
            System.clearProperty("euhedral.lake.idle-detach-ms");
            LatticeTesting.closeAndAwait(lattice);
        }
    }

    /// Two quanta of two runtimes, submitted by concurrent callers, share nothing: once both are admitted,
    /// their stages become runnable together and their embeddings run in the same wave, from frames on
    /// different routes (so Euhedral may place them on different workers). Stepped on the test thread: no
    /// core count is assumed.
    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void independentQuantaFromConcurrentCallersAreRunnableTogetherOnDistinctRoutes() throws Exception {
        var plan = new ExecutionPlan(
                ExecutionFixtures.weights(),
                ExecutionFixtures.norm(),
                List.of(ExecutionFixtures.q3("projection", 64, 201)));
        var gpu = new ConcurrentGpu();
        var lattice = new SteppingLattice();
        var attachments = new AtomicInteger();
        var runtime = new Execution(
                source -> {
                    attachments.incrementAndGet();
                    lattice.addUpstream(source);
                },
                plan,
                gpu);
        // A second runtime on the same lattice: its quanta share nothing with the first's workspace.
        var other = new Execution(lattice, plan, gpu);
        try {
            CompletableFuture<Quantum.Outcome> first;
            CompletableFuture<Quantum.Outcome> second;
            try (var calls = Executors.newFixedThreadPool(2)) {
                var a = calls.submit(() -> runtime.submit(
                        new Quantum(plan, new Sequence(801), Quantum.ExecutionKind.PREFILL, 0, new int[] {1})));
                var b = calls.submit(() -> other.submit(
                        new Quantum(plan, new Sequence(802), Quantum.ExecutionKind.PREFILL, 0, new int[] {2})));
                first = a.get(10, TimeUnit.SECONDS);
                second = b.get(10, TimeUnit.SECONDS);
            }
            var roots = lattice.admitAll();
            assertEquals(0, gpu.embeddings.size(), "a stage ran before both quanta were admitted");
            lattice.run(roots);
            lattice.runUntil(() -> gpu.embeddings.size() >= 2, 256);
            long[] one = gpu.embeddings.get(0);
            long[] two = gpu.embeddings.get(1);
            assertEquals(one[0], two[0], "the independent quanta's embeddings did not run in the same wave");
            assertTrue(one[1] != two[1], "the independent quanta's embeddings share a route");
            lattice.runUntil(() -> first.isDone() && second.isDone(), 4096);
            assertEquals(
                    Quantum.Status.SUCCESS,
                    first.join().status(),
                    () -> String.valueOf(first.join().failure()));
            assertEquals(
                    Quantum.Status.SUCCESS,
                    second.join().status(),
                    () -> String.valueOf(second.join().failure()));
            assertEquals(
                    EuhedralInferenceRuntime.LAKE_SINKS + 1,
                    attachments.get(),
                    "the runtime attaches its lake once: its sinks and the idle watch");
            // Later quanta reuse the idle graphs and the lake: no further source is attached.
            for (int index = 0; index < 4; index++) {
                var decode = runtime.submit(
                        new Quantum(plan, new Sequence(810 + index), Quantum.ExecutionKind.DECODE, 0, new int[] {1}));
                lattice.runUntil(decode::isDone, 4096);
                assertEquals(Quantum.Status.SUCCESS, decode.join().status());
            }
            assertEquals(EuhedralInferenceRuntime.LAKE_SINKS + 1, attachments.get());
            assertEquals(0, runtime.activeQuanta());
        } finally {
            lattice.closeWhileStepping(runtime::close);
            lattice.closeWhileStepping(other::close);
        }
    }

    @Test
    void foreignPlanIsRejectedBeforeAnyGraphIsBuilt() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var attachments = new AtomicInteger();
        var runtime = new Execution(source -> attachments.incrementAndGet(), plan, new ConcurrentGpu());
        var wrongPlan = new ExecutionPlan(ExecutionFixtures.weights());
        var context = new Quantum(wrongPlan, new Sequence(702), Quantum.ExecutionKind.PREFILL, 0, new int[] {1});

        var failure = assertThrows(IllegalArgumentException.class, () -> runtime.execute(List.of(context)));

        assertEquals("quantum belongs to another execution plan", failure.getMessage());
        assertEquals(0, attachments.get());
        assertTrue(runtime.execute(List.of()).isEmpty(), "empty work builds no graph");
        assertEquals(0, attachments.get());
        runtime.close();
    }

    @Test
    void closeReportsADownstreamCompletionFailureAfterTheGraphsRetired() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var completionFailure = new IllegalStateException("injected downstream completion failure");
        var runtime = new Execution(new PullingLattice(completionFailure), plan, new ConcurrentGpu());
        var outcome = runtime.execute(
                List.of(new Quantum(plan, new Sequence(703), Quantum.ExecutionKind.DECODE, 0, new int[] {1})));
        assertEquals(Quantum.Status.SUCCESS, outcome.getFirst().status());

        assertSame(completionFailure, assertThrows(IllegalStateException.class, runtime::close));
        assertFalse(runtime.isAttached(), "the runtime retained a completed source after its callback failed");
        // A closed runtime refuses the admission; the refusal is the quantum's outcome.
        var refused =
                runtime.submit(new Quantum(plan, new Sequence(704), Quantum.ExecutionKind.DECODE, 0, new int[] {1}));
        assertEquals(Quantum.Status.FAILED, refused.get(10, TimeUnit.SECONDS).status());
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aQuantumAdmittedAfterTheRuntimeClosedIsRefusedAndNeverRuns() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new ConcurrentGpu();
        var runtime = new Execution(new PullingLattice(), plan, gpu);
        var sequence = new Sequence(705);
        var first =
                runtime.execute(List.of(new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1})));
        assertEquals(Quantum.Status.SUCCESS, first.getFirst().status());
        // The runtime closed while the lake still takes frames: an admission that runs now finds the decode graph
        // idle, and must not start it, since nothing waits for it any more.
        runtime.runtime().close();
        var late = runtime.submit(new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 1, new int[] {1}));
        var outcome = late.get(10, TimeUnit.SECONDS);
        assertEquals(Quantum.Status.FAILED, outcome.status());
        assertEquals(0, runtime.activeQuanta());
        runtime.close();
    }

    /// Sixteen sequences on two runtimes: the two runtimes' quanta run their embeddings in the same waves,
    /// and a quantum's sixteen independent projections run together, each from its own route. Stepped on the
    /// test thread: no core count is assumed.
    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void independentSequencesAndProjectionsAreRunnableTogetherOnDistinctRoutes() throws Exception {
        var plan = new ExecutionPlan(
                ExecutionFixtures.weights(),
                ExecutionFixtures.norm(),
                java.util.stream.IntStream.range(0, 16)
                        .mapToObj(index -> ExecutionFixtures.q3("projection-" + index, 64, 201 + index))
                        .toList());
        var gpu = new ConcurrentGpu();
        var lattice = new SteppingLattice();
        // Within one runtime, quanta share its one workspace and follow each other buffer by buffer; two runtimes on
        // one lattice share nothing, so their quanta are independent.
        var runtime = new Execution(lattice, plan, gpu);
        var other = new Execution(lattice, plan, gpu);
        List<Sequence> sequences = new ArrayList<>();
        List<CompletableFuture<Quantum.Outcome>> completions = new ArrayList<>();
        try {
            for (int sequenceId = 1; sequenceId <= 16; sequenceId++) {
                Sequence sequence = new Sequence(sequenceId);
                sequences.add(sequence);
                completions.add((sequenceId % 2 == 0 ? runtime : other)
                        .submit(new Quantum(plan, sequence, Quantum.ExecutionKind.PREFILL, 0, new int[] {
                            sequenceId % ExecutionFixtures.VOCABULARY
                        })));
            }
            lattice.runUntil(() -> completions.stream().allMatch(CompletableFuture::isDone), 1 << 16);
            assertTrue(
                    togetherOnDistinctRoutes(gpu.embeddings), "no two embeddings ran in one wave on distinct routes");
            assertTrue(
                    togetherOnDistinctRoutes(gpu.projections), "no two projections ran in one wave on distinct routes");
            for (var completion : completions) {
                var outcome = completion.join();
                assertEquals(Quantum.Status.SUCCESS, outcome.status(), () -> String.valueOf(outcome.failure()));
            }
            for (Sequence sequence : sequences) assertEquals(1, sequence.currentTokenPosition());
            assertEquals(0, runtime.activeQuanta());
            assertEquals(0, other.activeQuanta());
        } finally {
            lattice.closeWhileStepping(runtime::close);
            lattice.closeWhileStepping(other::close);
            for (Sequence sequence : sequences) sequence.complete();
        }
    }

    /// Whether two of `calls` ({wave, route} each) ran in the same wave from different routes.
    private static boolean togetherOnDistinctRoutes(List<long[]> calls) {
        for (int i = 0; i < calls.size(); i++)
            for (int k = i + 1; k < calls.size(); k++)
                if (calls.get(i)[0] == calls.get(k)[0] && calls.get(i)[1] != calls.get(k)[1]) return true;
        return false;
    }

    /// Euhedral's workers, however many the machine has, run sixteen sequences on two runtimes to completion,
    /// and the runtimes let their sinks go once closed.
    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void euhedralWorkersRunIndependentSequencesToCompletion() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(!cpus.isEmpty());
        ControlPlaneLattice lattice = createLattice(cpus);
        try {
            lattice.start();
            var plan = new ExecutionPlan(
                    ExecutionFixtures.weights(),
                    ExecutionFixtures.norm(),
                    java.util.stream.IntStream.range(0, 16)
                            .mapToObj(index -> ExecutionFixtures.q3("projection-" + index, 64, 201 + index))
                            .toList());
            var gpu = new ConcurrentGpu();
            var runtime = new Execution(lattice, plan, gpu);
            var other = new Execution(lattice, plan, gpu);
            List<Sequence> sequences = new ArrayList<>();
            List<CompletableFuture<Quantum.Outcome>> completions = new ArrayList<>();
            try {
                for (int sequenceId = 1; sequenceId <= 16; sequenceId++) {
                    Sequence sequence = new Sequence(sequenceId);
                    sequences.add(sequence);
                    completions.add((sequenceId % 2 == 0 ? runtime : other)
                            .submit(new Quantum(plan, sequence, Quantum.ExecutionKind.PREFILL, 0, new int[] {
                                sequenceId % ExecutionFixtures.VOCABULARY
                            })));
                }
                CompletableFuture.allOf(completions.toArray(CompletableFuture[]::new))
                        .get(60, TimeUnit.SECONDS);
                for (var completion : completions) {
                    var outcome = completion.join();
                    assertEquals(Quantum.Status.SUCCESS, outcome.status(), () -> String.valueOf(outcome.failure()));
                }
                for (Sequence sequence : sequences) assertEquals(1, sequence.currentTokenPosition());
                assertEquals(16 * 16, gpu.projections.size() + gpu.untracedProjections.get());
                assertEquals(0, runtime.activeQuanta());
                assertEquals(0, other.activeQuanta());
            } finally {
                runtime.close();
                other.close();
                for (Sequence sequence : sequences) sequence.complete();
            }
            assertFalse(runtime.isAttached());
            assertFalse(other.isAttached());
            awaitDrained(lattice);
        } finally {
            LatticeTesting.closeAndAwait(lattice);
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void sequenceStateSurvivesAcrossQuantaAndPlanViewsOnTheRealLattice() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(!cpus.isEmpty());
        var lattice = createLattice(cpus);
        try {
            lattice.start();
            var weights = ExecutionFixtures.statefulCompactWeights();
            var plan = ExecutionPlan.prefix(weights, 2);
            var gpu = new SequenceGpu();
            var runtime = new Execution(lattice, plan, gpu);
            var sequence = new Sequence(700);
            long embeddingAddress = weights.tokenEmbedding().deviceAddress();
            try {
                var prefill =
                        runtime.submit(new Quantum(plan, sequence, Quantum.ExecutionKind.PREFILL, 0, new int[] {1, 2}));
                assertEquals(
                        Quantum.Status.SUCCESS,
                        prefill.get(10, TimeUnit.SECONDS).status());
                var recurrent = (GdnStates) sequence.recurrentState();
                var attention = (AttentionStates) sequence.kvCacheState();
                var attentionState = attention.forLayer(1);
                var recurrentState = recurrent.forLayer(0);
                long convolutionAddress = recurrentState.convolutionStateAddress();
                long recurrentAddress = recurrentState.recurrentStateAddress();
                long kvAddress = attentionState.keyCacheAddress();
                assertEquals(2, sequence.currentTokenPosition());
                assertEquals(2, attentionState.length(), "the retired prefill committed its append");
                assertFalse(gpu.freed.contains(convolutionAddress));
                assertFalse(gpu.freed.contains(recurrentAddress));
                assertFalse(gpu.freed.contains(kvAddress));

                var decode =
                        runtime.submit(new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 2, new int[] {3}));
                assertEquals(
                        Quantum.Status.SUCCESS, decode.get(10, TimeUnit.SECONDS).status());

                assertEquals(3, sequence.currentTokenPosition());
                assertSame(recurrent, sequence.recurrentState());
                assertSame(attention, sequence.kvCacheState());
                assertSame(recurrentState, recurrent.forLayer(0));
                assertSame(attentionState, attention.forLayer(1));
                assertEquals(3, attentionState.length());
                assertEquals(convolutionAddress, gpu.lastConvolutionAddress.get());
                assertEquals(recurrentAddress, gpu.lastRecurrentAddress.get());
                assertEquals(2, gpu.convolutionCalls.get());
                assertEquals(2, gpu.recurrenceCalls.get());
                assertEquals(2, gpu.kvAppendCalls.get());
                assertFalse(gpu.freed.contains(embeddingAddress));

                sequence.complete();
                assertTrue(gpu.freed.contains(convolutionAddress));
                assertTrue(gpu.freed.contains(recurrentAddress));
                assertTrue(gpu.freed.contains(kvAddress));
                assertFalse(gpu.freed.contains(embeddingAddress));
            } finally {
                runtime.close();
                if (sequence.terminalState() == Sequence.TerminalState.ACTIVE && !sequence.inFlight())
                    sequence.complete();
            }
        } finally {
            LatticeTesting.closeAndAwait(lattice);
        }
    }

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void actualCompactLayerZeroPrefillsThenDecodesOnTheSameSequence() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(cpus.cardinality() >= 2, "requires two available physical CPUs for lattice workers");
        Path artifactPath = Path.of(System.getProperty("euhedral.qwen.artifact", DEFAULT_COMPACT_ARTIFACT.toString()));
        assumeTrue(Files.isRegularFile(artifactPath), "compact Qwen artifact is missing: " + artifactPath);
        Artifact artifact = ArtifactReader.read(artifactPath);
        assertEquals(ArtifactHeader.COMPACT_VERSION, artifact.header().version());
        assertEquals(64, artifact.config().numHiddenLayers());
        assertEquals(LayerType.GATED_DELTA_NET, artifact.config().layerTypes()[0]);

        var gpu = new SequenceGpu();
        Weights weights = WeightLoader.loadFirstLayer(artifactPath, artifact, gpu);
        var plan = new ExecutionPlan(weights);
        var lattice = createLattice(cpus);
        var sequence = new Sequence(701);
        long embeddingAddress = weights.tokenEmbedding().deviceAddress();
        LatticeEdge registrationProbe = new LatticeEdge(new AtomicBoolean());
        int registeredWorkersBeforeStart = registrationProbe.getThreadCount();
        try {
            lattice.start();
            awaitWorkers(lattice, 2, registrationProbe, registeredWorkersBeforeStart + 2);
            var runtime = new Execution(lattice, plan, gpu);
            try {
                var prefill = runtime.submit(
                        new Quantum(plan, sequence, Quantum.ExecutionKind.PREFILL, 0, new int[] {1814, 1815}));
                var outcome = prefill.get(60, TimeUnit.SECONDS);
                assertEquals(Quantum.Status.SUCCESS, outcome.status(), () -> String.valueOf(outcome.failure()));
                GdnState recurrent = (GdnState) sequence.recurrentState();
                long convolutionAddress = recurrent.convolutionStateAddress();
                long recurrentAddress = recurrent.recurrentStateAddress();
                assertEquals(2, sequence.currentTokenPosition());
                assertFalse(gpu.freed.contains(convolutionAddress));
                assertFalse(gpu.freed.contains(recurrentAddress));
                assertFalse(gpu.freed.contains(embeddingAddress));

                var decode =
                        runtime.submit(new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 2, new int[] {1816}));
                var decoded = decode.get(60, TimeUnit.SECONDS);
                assertEquals(Quantum.Status.SUCCESS, decoded.status(), () -> String.valueOf(decoded.failure()));

                assertSame(recurrent, sequence.recurrentState());
                assertEquals(3, sequence.currentTokenPosition());
                assertEquals(convolutionAddress, gpu.lastConvolutionAddress.get());
                assertEquals(recurrentAddress, gpu.lastRecurrentAddress.get());
                assertEquals(2, gpu.convolutionCalls.get());
                assertEquals(2, gpu.recurrenceCalls.get());
                assertFalse(gpu.freed.contains(embeddingAddress));

                sequence.complete();
                assertTrue(gpu.freed.contains(convolutionAddress));
                assertTrue(gpu.freed.contains(recurrentAddress));
                assertFalse(gpu.freed.contains(embeddingAddress));
            } finally {
                runtime.close();
            }
            awaitDrained(lattice);
        } finally {
            if (sequence.terminalState() == Sequence.TerminalState.ACTIVE && !sequence.inFlight()) {
                sequence.complete();
            }
            LatticeTesting.closeAndAwait(lattice);
        }
    }

    /// Releases `stream`'s held boundaries from a driver thread, oldest first, until `outcome` completes.
    private static void releaseUntilDone(ExecutionFixtures.HoldingStream stream, CompletableFuture<?> outcome)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (!outcome.isDone() && System.nanoTime() < deadline) {
            if (stream.held() == 0) {
                Thread.sleep(1);
                continue;
            }
            Thread driver = Thread.ofPlatform().start(() -> stream.release(null));
            driver.join();
        }
        assertTrue(outcome.isDone(), "the quantum did not complete; " + stream.held() + " boundaries held");
    }

    private static void awaitHeld(ExecutionFixtures.HoldingStream stream, int boundaries) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (stream.held() < boundaries && System.nanoTime() < deadline) Thread.sleep(1);
        assertEquals(boundaries, stream.held(), "the quantum's stages did not all submit");
    }

    /// A synchronous test GPU whose single stream holds each retirement boundary for the test.
    private static final class HeldGpu extends ExecutionFixtures.RecordingGpu {
        final ExecutionFixtures.HoldingStream stream = new ExecutionFixtures.HoldingStream();

        @Override
        public io.euhedral_execution.inference.core.gpu.GpuStream openStream() {
            return this.stream;
        }
    }

    private static ControlPlaneLattice createLattice(BitSet cpus) {
        var workers = new BaseCloneableObject(new DefaultExecutor());
        var baseShard = ControlPlaneShard.createBaseShard("EuhedralRuntimeTestShard", workers);
        return ControlPlaneLattice.getOrCreate(
                new LatticeConfig("EuhedralRuntimeTestLattice", cpus, Duration.ofSeconds(10), baseShard));
    }

    private static BitSet twoWorkerCpus() {
        BitSet cpus = new BitSet();
        BitSet selectedCores = new BitSet();
        var physicalCpus = SystemInfo.getPCpuSet();
        int selected = 0;
        for (int cpu = physicalCpus.nextSetBit(0); cpu >= 0 && selected < 2; cpu = physicalCpus.nextSetBit(cpu + 1)) {
            var info = SystemInfo.getCpuInfo(cpu);
            if (info == null || selectedCores.get(info.core())) continue;
            selectedCores.set(info.core());
            cpus.set(cpu);
            selected++;
        }
        return cpus;
    }

    private static void awaitWorkers(
            ControlPlaneLattice lattice, int expected, LatticeEdge registrationProbe, int expectedRegistrations)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while ((lattice.getActiveWorkers() < expected || registrationProbe.getThreadCount() < expectedRegistrations)
                && System.nanoTime() < deadline) Thread.onSpinWait();
        assertEquals(expected, lattice.getActiveWorkers(), "lattice workers did not register before source attachment");
        assertEquals(
                expectedRegistrations,
                registrationProbe.getThreadCount(),
                "worker source partitions did not register before source attachment");
    }

    private static void awaitDrained(ControlPlaneLattice lattice) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (!lattice.isDrained() && System.nanoTime() < deadline) Thread.sleep(1);
        assertTrue(lattice.isDrained(), "Euhedral workers retained runnable frames after disconnection");
    }

    private static final class ConcurrentGpu extends ExecutionGpu {

        private final AtomicLong nextAddress = new AtomicLong(1_000L);
        /// {wave, route} of each embedding and projection a [SteppingLattice] ran; calls on other lattices
        /// are counted.
        private final List<long[]> embeddings = new CopyOnWriteArrayList<>();
        private final List<long[]> projections = new CopyOnWriteArrayList<>();
        private final AtomicInteger untracedProjections = new AtomicInteger();

        @Override
        public long allocate(long byteSize) {
            return this.nextAddress.getAndIncrement();
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {}

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {}

        @Override
        public void free(long address) {}

        @Override
        public void embedQ3(
                long tokenIdsAddress,
                long embeddingAddress,
                long embeddingByteSize,
                long hiddenStateAddress,
                int tokenCount,
                int vocabularySize,
                int hiddenSize) {
            long[] running = SteppingLattice.running();
            if (running != null) this.embeddings.add(running);
        }

        @Override
        public void rmsNormBf16(
                long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {}

        @Override
        public void linearQ3Bf16(
                long inputAddress,
                long weightsAddress,
                long outputAddress,
                int rows,
                int inFeatures,
                int outFeatures,
                long weightsByteSize) {
            long[] running = SteppingLattice.running();
            if (running != null) this.projections.add(running);
            else this.untracedProjections.incrementAndGet();
        }

        @Override
        public void synchronize() {}
    }

    private static final class SequenceGpu extends ExecutionGpu {

        private final AtomicLong nextAddress = new AtomicLong(100_000L);
        private final Set<Long> freed = ConcurrentHashMap.newKeySet();
        private final AtomicLong lastConvolutionAddress = new AtomicLong();
        private final AtomicLong lastRecurrentAddress = new AtomicLong();
        private final AtomicInteger convolutionCalls = new AtomicInteger();
        private final AtomicInteger recurrenceCalls = new AtomicInteger();
        private final AtomicInteger kvAppendCalls = new AtomicInteger();

        @Override
        public long allocate(long byteSize) {
            return this.nextAddress.getAndAdd(Math.max(1L, byteSize));
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {}

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {}

        @Override
        public void copyDeviceToDevice(long destination, long source, long byteSize) {}

        @Override
        public void free(long address) {
            this.freed.add(address);
        }

        @Override
        public void embedQ3(
                long tokenIdsAddress,
                long embeddingAddress,
                long embeddingByteSize,
                long hiddenStateAddress,
                int tokenCount,
                int vocabularySize,
                int hiddenSize) {}

        @Override
        public void rmsNormBf16(
                long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {}

        @Override
        public void rmsNormUnitOffsetBf16(
                long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {}

        @Override
        public void linearQ3Bf16(
                long inputAddress,
                long weightsAddress,
                long outputAddress,
                int rows,
                int inFeatures,
                int outFeatures,
                long weightsByteSize) {}

        @Override
        public void linearQ4Bf16(
                long inputAddress,
                long weightsAddress,
                long outputAddress,
                int rows,
                int inFeatures,
                int outFeatures,
                long weightsByteSize) {}

        @Override
        public void linearQ5Bf16(
                long inputAddress,
                long weightsAddress,
                long outputAddress,
                int rows,
                int inFeatures,
                int outFeatures,
                long weightsByteSize) {}

        @Override
        public void linearBf16ToFloat(
                long inputAddress,
                long weightsAddress,
                long outputAddress,
                int rows,
                int inFeatures,
                int outFeatures) {}

        @Override
        public void gdnControlFp32(
                long aProjectionAddress,
                long bProjectionAddress,
                long aLogAddress,
                long dtBiasAddress,
                long alphaOutputAddress,
                long betaOutputAddress,
                int rows,
                int heads) {}

        @Override
        public void gdnConvolutionBf16(
                long queryKeyAddress,
                long valueZAddress,
                long convolutionWeightsAddress,
                long convolutionStateAddress,
                long outputAddress,
                int rows,
                int queryKeyWidth,
                int valueWidth,
                int convolutionWidth,
                int kernelSize) {
            this.lastConvolutionAddress.set(convolutionStateAddress);
            this.convolutionCalls.incrementAndGet();
        }

        @Override
        public void gdnRecurrenceBf16(
                long convolvedAddress,
                long alphaAddress,
                long betaAddress,
                long recurrentStateAddress,
                long outputAddress,
                int rows,
                int keyHeads,
                int valueHeads,
                int keyHeadDim,
                int valueHeadDim,
                float outputScale) {
            this.lastRecurrentAddress.set(recurrentStateAddress);
            this.recurrenceCalls.incrementAndGet();
        }

        @Override
        public void gdnGatedRmsNormBf16(
                long recurrentAddress,
                long valueZAddress,
                long normWeightAddress,
                long outputAddress,
                int rows,
                int valueHeads,
                int headDim,
                float epsilon) {}

        @Override
        public void residualAddBf16(long residualAddress, long deltaAddress, long outputAddress, int rows, int width) {}

        @Override
        public void swiGluBf16(long gateUpAddress, long outputAddress, int rows, int intermediateSize) {}

        @Override
        public void zeroDeviceMemory(long address, long byteSize) {}

        @Override
        public void attentionQkNormRopeBf16(
                long queryKeyAddress,
                long queryNormAddress,
                long keyNormAddress,
                long outputAddress,
                int rows,
                int queryHeads,
                int keyValueHeads,
                int headDim,
                int rotaryDim,
                long startPosition,
                long positionAddress,
                float epsilon,
                double ropeTheta) {}

        @Override
        public void attentionKvAppendNvfp4(
                long queryKeyAddress,
                long gateValueAddress,
                long keyCacheAddress,
                long valueCacheAddress,
                int rows,
                int queryWidth,
                int keyValueWidth,
                long startPosition,
                long positionAddress) {
            this.kvAppendCalls.incrementAndGet();
        }

        @Override
        public void attentionCausalNvfp4(
                long queryKeyAddress,
                long gateValueAddress,
                long keyCacheAddress,
                long valueCacheAddress,
                long outputAddress,
                int rows,
                int queryHeads,
                int keyValueHeads,
                int headDim,
                int cacheLength,
                long startPosition,
                long positionAddress,
                long scratchAddress) {}

        @Override
        public void synchronize() {}
    }
}
