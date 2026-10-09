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
import java.util.concurrent.CountDownLatch;
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
            Path.of("/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl");

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
            lattice.close();
        }
    }

    /// On the real lattice: once the runtime has been idle its sinks detach and the lattice has nothing left to poll;
    /// a quantum submitted after that attaches fresh sinks and runs as before.
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
                    awaitHeld(gpu.stream, 1);
                    Thread driver = Thread.ofPlatform().start(() -> gpu.stream.release(null));
                    driver.join();
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
            lattice.close();
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void independentQuantaFromConcurrentCallersRunOnDifferentWorkers() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(cpus.cardinality() == 2);
        var probe = new LatticeEdge(new AtomicBoolean());
        int registrations = probe.getThreadCount();
        var lattice = createLattice(cpus);
        try {
            lattice.start();
            awaitWorkers(lattice, 2, probe, registrations + 2);
            var plan = new ExecutionPlan(
                    ExecutionFixtures.weights(),
                    ExecutionFixtures.norm(),
                    List.of(ExecutionFixtures.q3("projection", 64, 201)));
            var gpu = new ConcurrentGpu();
            gpu.embeddingGate = new WorkGate(0);
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
            try (var calls = java.util.concurrent.Executors.newFixedThreadPool(2)) {
                try {
                    // Euhedral admits every quantum on the workspace owner's worker, which is held until both are
                    // queued (below). Should the gate still hold that worker before the second quantum was admitted,
                    // the second cannot start: such an attempt is released and made again.
                    java.util.concurrent.Future<List<Quantum.Outcome>> first = null;
                    java.util.concurrent.Future<List<Quantum.Outcome>> second = null;
                    boolean concurrent = false;
                    for (int attempt = 0; attempt < 10 && !concurrent; attempt++) {
                        gpu.embeddingGate = new WorkGate(2);
                        int base = 801 + 2 * attempt;
                        var owner = holdOwner(runtime);
                        first = calls.submit(() -> runtime.execute(List.of(new Quantum(
                                plan, new Sequence(base), Quantum.ExecutionKind.PREFILL, 0, new int[] {1}))));
                        second = calls.submit(() -> other.execute(List.of(new Quantum(
                                plan, new Sequence(base + 1), Quantum.ExecutionKind.PREFILL, 0, new int[] {2}))));
                        letAdmissionsQueue();
                        owner.countDown();
                        concurrent = gpu.embeddingGate.awaitEntries(2, TimeUnit.SECONDS);
                        if (!concurrent) {
                            gpu.embeddingGate.release();
                            first.get(10, TimeUnit.SECONDS);
                            second.get(10, TimeUnit.SECONDS);
                        }
                    }
                    assertTrue(concurrent, "independent quanta did not run concurrently");
                    assertEquals(2, gpu.embeddingGate.workerCount(), "independent quanta shared one worker");
                    gpu.embeddingGate.release();
                    assertEquals(
                            Quantum.Status.SUCCESS,
                            first.get(10, TimeUnit.SECONDS).getFirst().status());
                    assertEquals(
                            Quantum.Status.SUCCESS,
                            second.get(10, TimeUnit.SECONDS).getFirst().status());
                    assertEquals(
                            EuhedralInferenceRuntime.LAKE_SINKS + 1,
                            attachments.get(),
                            "the runtime attaches its lake once: its sinks and the idle watch");
                    assertEquals(0, runtime.activeQuanta());
                } finally {
                    gpu.embeddingGate.release();
                }
                // Later quanta reuse the idle graphs and the lake: no further source is attached.
                for (int index = 0; index < 4; index++) {
                    assertEquals(
                            Quantum.Status.SUCCESS,
                            runtime.execute(List.of(new Quantum(
                                            plan,
                                            new Sequence(810 + index),
                                            Quantum.ExecutionKind.DECODE,
                                            0,
                                            new int[] {1})))
                                    .getFirst()
                                    .status());
                }
                assertEquals(EuhedralInferenceRuntime.LAKE_SINKS + 1, attachments.get());
            } finally {
                runtime.close();
                other.close();
            }
            awaitDrained(lattice);
        } finally {
            lattice.close();
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
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void euhedralWorkersRunIndependentSequencesAndProjectionsInParallel() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(cpus.cardinality() >= 2, "requires two available physical CPUs for concurrent workers");
        var registrationProbe = new LatticeEdge(new AtomicBoolean());
        int registeredWorkersBeforeStart = registrationProbe.getThreadCount();
        ControlPlaneLattice lattice = createLattice(cpus);
        try {
            lattice.start();
            awaitWorkers(lattice, 2, registrationProbe, registeredWorkersBeforeStart + 2);
            var plan = new ExecutionPlan(
                    ExecutionFixtures.weights(),
                    ExecutionFixtures.norm(),
                    java.util.stream.IntStream.range(0, 16)
                            .mapToObj(index -> ExecutionFixtures.q3("projection-" + index, 64, 201 + index))
                            .toList());
            var gpu = new ConcurrentGpu();
            // Within one runtime, quanta share its one workspace and follow each other buffer by buffer; two
            // runtimes on one lattice share nothing, so their quanta are independent.
            var runtime = new Execution(lattice, plan, gpu);
            var other = new Execution(lattice, plan, gpu);
            List<Sequence> sequences = new ArrayList<>();
            List<CompletableFuture<Quantum.Outcome>> completions = new ArrayList<>();
            gpu.embeddingGate = new WorkGate(0);
            gpu.projectionGate = new WorkGate(0);
            try {
                // See the test above: an attempt whose gate holds the workspace owner's worker before the other
                // sequences were admitted cannot get two workers into the gate; it is released and made again.
                boolean concurrent = false;
                for (int attempt = 0; attempt < 10 && !concurrent; attempt++) {
                    sequences.clear();
                    completions.clear();
                    gpu.embeddingGate = new WorkGate(2);
                    gpu.projectionGate = new WorkGate(2);
                    var owner = holdOwner(runtime);
                    for (int sequenceId = 1; sequenceId <= 16; sequenceId++) {
                        Sequence sequence = new Sequence(attempt * 100 + sequenceId);
                        sequences.add(sequence);
                        completions.add((sequenceId % 2 == 0 ? runtime : other)
                                .submit(new Quantum(plan, sequence, Quantum.ExecutionKind.PREFILL, 0, new int[] {
                                    sequenceId % ExecutionFixtures.VOCABULARY
                                })));
                    }
                    letAdmissionsQueue();
                    owner.countDown();
                    concurrent = gpu.embeddingGate.awaitEntries(3, TimeUnit.SECONDS);
                    if (!concurrent) {
                        gpu.embeddingGate.release();
                        gpu.projectionGate.release();
                        CompletableFuture.allOf(completions.toArray(CompletableFuture[]::new))
                                .get(30, TimeUnit.SECONDS);
                        for (Sequence sequence : sequences) sequence.complete();
                    }
                }
                try {
                    assertTrue(
                            concurrent,
                            "independent sequences did not execute concurrently; activeWorkers="
                                    + lattice.getActiveWorkers() + ", allowedCpus=" + cpus
                                    + ", embeddingCalls=" + gpu.embeddingCalls.get());
                    assertEquals(2, gpu.embeddingGate.workerCount(), "independent sequences used the same worker");
                } finally {
                    gpu.embeddingGate.release();
                }
                try {
                    assertTrue(
                            gpu.projectionGate.awaitEntries(),
                            "independent projection work did not overlap; activeWorkers="
                                    + lattice.getActiveWorkers() + ", projectionCalls="
                                    + gpu.projectionCalls.get());
                    assertEquals(2, gpu.projectionGate.workerCount(), "projection work used the same worker");
                } finally {
                    gpu.projectionGate.release();
                }
                CompletableFuture.allOf(completions.toArray(CompletableFuture[]::new))
                        .get(30, TimeUnit.SECONDS);
                for (var completion : completions) {
                    var outcome = completion.join();
                    assertEquals(Quantum.Status.SUCCESS, outcome.status(), () -> String.valueOf(outcome.failure()));
                }
                for (Sequence sequence : sequences) assertEquals(1, sequence.currentTokenPosition());
                assertEquals(0, runtime.activeQuanta());
                assertEquals(0, other.activeQuanta());
            } finally {
                gpu.projectionGate.release();
                gpu.embeddingGate.release();
                runtime.close();
                other.close();
                for (Sequence sequence : sequences) sequence.complete();
            }
            assertFalse(runtime.isAttached());
            assertFalse(other.isAttached());
            awaitDrained(lattice);
        } finally {
            lattice.close();
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
            lattice.close();
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
            lattice.close();
        }
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

    /// Holds the workspace owner's worker in a frame of its own until the returned latch opens: the admissions
    /// published meanwhile wait in that worker's cache, in order, and the worker runs them back to back once it is
    /// free, before it takes a stage. A gate that holds workers inside stages (production stages never block) can
    /// then hold the owner's worker without keeping a quantum from being admitted.
    private static CountDownLatch holdOwner(Execution execution) throws InterruptedException {
        var held = new CountDownLatch(1);
        var open = new CountDownLatch(1);
        execution.runtime().publishOnOwner(() -> {
            held.countDown();
            try {
                open.await(30, TimeUnit.SECONDS);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        });
        assertTrue(held.await(10, TimeUnit.SECONDS), "the owner's worker did not take the holding frame");
        return open;
    }

    /// Lets the admissions that were published while the owner's worker was held reach its cache.
    private static void letAdmissionsQueue() throws InterruptedException {
        Thread.sleep(300);
    }

    private static void awaitDrained(ControlPlaneLattice lattice) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (!lattice.isDrained() && System.nanoTime() < deadline) Thread.sleep(1);
        assertTrue(lattice.isDrained(), "Euhedral workers retained runnable frames after disconnection");
    }

    private static final class ConcurrentGpu extends ExecutionGpu {

        private final AtomicLong nextAddress = new AtomicLong(1_000L);
        private final AtomicInteger projectionCalls = new AtomicInteger();
        private final AtomicInteger normCalls = new AtomicInteger();
        private final AtomicInteger embeddingCalls = new AtomicInteger();
        private volatile WorkGate projectionGate = new WorkGate(0);
        private volatile WorkGate embeddingGate = new WorkGate(0);

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
            this.embeddingCalls.incrementAndGet();
            this.embeddingGate.enterIfSelected();
        }

        @Override
        public void rmsNormBf16(
                long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
            this.normCalls.incrementAndGet();
        }

        @Override
        public void linearQ3Bf16(
                long inputAddress,
                long weightsAddress,
                long outputAddress,
                int rows,
                int inFeatures,
                int outFeatures,
                long weightsByteSize) {
            this.projectionCalls.incrementAndGet();
            this.projectionGate.enterIfSelected();
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

    private static final class WorkGate {

        private final CountDownLatch entered;
        private final CountDownLatch released = new CountDownLatch(1);
        private final AtomicInteger selected;
        private final Set<Thread> workers = ConcurrentHashMap.newKeySet();

        private WorkGate(int selectedWorkers) {
            this.entered = new CountDownLatch(selectedWorkers);
            this.selected = new AtomicInteger(selectedWorkers);
            if (selectedWorkers == 0) this.released.countDown();
        }

        private void enterIfSelected() {
            int remaining = this.selected.getAndUpdate(value -> value == 0 ? 0 : value - 1);
            if (remaining == 0) return;
            this.workers.add(Thread.currentThread());
            this.entered.countDown();
            try {
                if (!this.released.await(15, TimeUnit.SECONDS)) {
                    throw new IllegalStateException("timed out waiting for the test work gate");
                }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while waiting for the test work gate", interrupted);
            }
        }

        private boolean awaitEntries() throws InterruptedException {
            return awaitEntries(15, TimeUnit.SECONDS);
        }

        private boolean awaitEntries(long timeout, TimeUnit unit) throws InterruptedException {
            return this.entered.await(timeout, unit);
        }

        private int workerCount() {
            return this.workers.size();
        }

        private void release() {
            this.released.countDown();
        }
    }
}
