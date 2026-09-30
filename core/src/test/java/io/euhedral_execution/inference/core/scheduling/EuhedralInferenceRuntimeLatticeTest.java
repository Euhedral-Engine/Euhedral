package io.euhedral_execution.inference.core.scheduling;

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
import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.data_structures.queues.MpscQueue;
import io.euhedral_execution.hardware_utils.SystemInfo;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.QwenWeightLoader;
import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactHeader;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
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
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class EuhedralInferenceRuntimeLatticeTest {

    private static final Path DEFAULT_COMPACT_ARTIFACT =
            Path.of("/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl");

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void realLatticePollsAsyncGpuCompletionFrameWithoutAnObserver() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(!cpus.isEmpty());
        var lattice = createLattice(cpus);
        var gpu = new CompletionGpu();
        try {
            lattice.start();
            var runtime =
                    new EuhedralInferenceRuntime(lattice, new QwenExecutionPlan(QwenExecutionFixtures.weights()), gpu);
            var completed = new CountDownLatch(1);
            gpu.signal(completed::countDown);
            assertTrue(completed.await(10, TimeUnit.SECONDS), "lattice did not execute the signaled completion frame");
            runtime.disconnectRunner();
        } finally {
            lattice.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void deferredGpuWorkKeepsDependenciesPendingUntilEachCompletionFrameRuns() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(!cpus.isEmpty());
        var lattice = createLattice(cpus);
        var gpu = new DeferredGpu();
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.weights(),
                QwenExecutionFixtures.norm(),
                List.of(QwenExecutionFixtures.q3("projection", 64, 201)));
        lattice.start();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        try (var caller = java.util.concurrent.Executors.newSingleThreadExecutor()) {
            var context = new QwenExecutionContext(
                    plan, new QwenSequenceState(5801), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
            var outcome = caller.submit(() -> runtime.execute(List.of(context)));
            for (int instruction = 0; instruction < plan.instructions().size(); instruction++) {
                assertFalse(outcome.isDone(), "dependencies became terminal before GPU completion");
                gpu.completeNext();
            }
            assertEquals(
                    QwenExecutionContext.Status.SUCCESS,
                    outcome.get(10, TimeUnit.SECONDS).getFirst().status());
            assertEquals(0, gpu.synchronizations, "normal async execution must not use a device-wide barrier");
        } finally {
            runtime.closeCompletionSink();
            lattice.close();
        }
    }

    private static final class DeferredGpu extends QwenExecutionFixtures.RecordingGpu {
        private final MpscQueue<Runnable> pending = new MpscQueue<>(64);
        private final Semaphore available = new Semaphore(0);
        private Consumer<Runnable> completionSink;

        @Override
        public boolean asynchronous() {
            return true;
        }

        @Override
        public void bindCompletionSink(Consumer<Runnable> sink) {
            this.completionSink = sink;
        }

        @Override
        public void deferCompletion(Runnable completed, Consumer<Throwable> failed) {
            pending.offer(completed);
            available.release();
        }

        void completeNext() throws InterruptedException {
            assertTrue(available.tryAcquire(10, TimeUnit.SECONDS), "GPU work was never submitted");
            Runnable completion = pending.poll();
            assertTrue(completion != null);
            completionSink.accept(completion);
        }
    }

    private static final class CompletionGpu extends QwenExecutionFixtures.RecordingGpu {
        private Consumer<Runnable> completionSink;

        @Override
        public boolean asynchronous() {
            return true;
        }

        @Override
        public void bindCompletionSink(Consumer<Runnable> sink) {
            this.completionSink = sink;
        }

        void signal(Runnable completion) {
            this.completionSink.accept(completion);
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void sharedRuntimeAdmitsIndependentCallsConcurrentlyToRealFabric() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(cpus.cardinality() == 2);
        var probe = new LatticeEdge(new AtomicBoolean());
        int registrations = probe.getThreadCount();
        var lattice = createLattice(cpus);
        try {
            lattice.start();
            awaitWorkers(lattice, 2, probe, registrations + 2);
            var plan = new QwenExecutionPlan(
                    QwenExecutionFixtures.weights(),
                    QwenExecutionFixtures.norm(),
                    List.of(QwenExecutionFixtures.q3("projection", 64, 201)));
            var gpu = new ConcurrentGpu();
            var admissionGate = new WorkGate(2);
            var runtime = new EuhedralInferenceRuntime(
                    source -> {
                        admissionGate.enterIfSelected();
                        lattice.addUpstream(source);
                    },
                    plan,
                    gpu);
            try (var calls = java.util.concurrent.Executors.newFixedThreadPool(2)) {
                try {
                    var first = calls.submit(() -> runtime.execute(List.of(new QwenExecutionContext(
                            plan, new QwenSequenceState(801), QwenExecutionContext.ExecutionKind.PREFILL, 0, new int[] {
                                1
                            }))));
                    var second = calls.submit(() -> runtime.execute(List.of(new QwenExecutionContext(
                            plan, new QwenSequenceState(802), QwenExecutionContext.ExecutionKind.PREFILL, 0, new int[] {
                                2
                            }))));
                    assertTrue(admissionGate.awaitEntries(), "one runtime serialized independent input admission");
                    assertEquals(2, admissionGate.workerCount());
                    admissionGate.release();
                    assertEquals(
                            QwenExecutionContext.Status.SUCCESS,
                            first.get(10, TimeUnit.SECONDS).getFirst().status());
                    assertEquals(
                            QwenExecutionContext.Status.SUCCESS,
                            second.get(10, TimeUnit.SECONDS).getFirst().status());
                    assertFalse(runtime.hasAttachedRunner());
                } finally {
                    admissionGate.release();
                }
            }
        } finally {
            lattice.close();
        }
    }

    @Test
    void disconnectReleasesTerminatedRunnerWhenDownstreamCompletionFails() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new ConcurrentGpu();
        var completionFailure = new IllegalStateException("injected downstream completion failure");
        var runtime = new EuhedralInferenceRuntime(throwingCompletionTerminal(completionFailure), plan, gpu);
        var runner = runtime.attachRunner();

        assertSame(completionFailure, assertThrows(IllegalStateException.class, runtime::disconnectRunner));
        assertTrue(runner.isComplete());
        assertFalse(runner.isAttached());
        assertFalse(runtime.hasAttachedRunner(), "runtime retained a terminated runner after callback failure");
    }

    @Test
    void executePreservesSubmissionFailureWhenDisconnectAlsoFails() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new ConcurrentGpu();
        var completionFailure = new IllegalStateException("injected downstream completion failure");
        var runtime = new EuhedralInferenceRuntime(throwingCompletionTerminal(completionFailure), plan, gpu);
        var wrongPlan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var context = new QwenExecutionContext(
                wrongPlan, new QwenSequenceState(702), QwenExecutionContext.ExecutionKind.PREFILL, 0, new int[] {1});

        var submissionFailure = assertThrows(IllegalArgumentException.class, () -> runtime.execute(List.of(context)));

        assertEquals("quantum belongs to another execution plan", submissionFailure.getMessage());
        assertEquals(1, submissionFailure.getSuppressed().length);
        assertSame(completionFailure, submissionFailure.getSuppressed()[0]);
        assertFalse(runtime.hasAttachedRunner());
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void euhedralWorkersRunParallelBranchesAndIndependentSequencesThenDetachRunner() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(cpus.cardinality() >= 2, "requires two available physical CPUs for concurrent workers");
        var registrationProbe = new LatticeEdge(new AtomicBoolean());
        int registeredWorkersBeforeStart = registrationProbe.getThreadCount();
        ControlPlaneLattice lattice = createLattice(cpus);
        try {
            lattice.start();
            awaitWorkers(lattice, 2, registrationProbe, registeredWorkersBeforeStart + 2);
            verifySequenceStateSurvivesRunnerTeardown(lattice);
            var plan = new QwenExecutionPlan(
                    QwenExecutionFixtures.weights(),
                    QwenExecutionFixtures.norm(),
                    java.util.stream.IntStream.range(0, 16)
                            .mapToObj(index -> QwenExecutionFixtures.q3("projection-" + index, 64, 201 + index))
                            .toList());
            var gpu = new ConcurrentGpu();
            var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
            List<EuhedralInferenceRuntime> runtimes = new ArrayList<>();
            runtimes.add(runtime);
            for (int index = 1; index < 16; index++) runtimes.add(new EuhedralInferenceRuntime(lattice, plan, gpu));
            List<QwenExecutionRunner> runners = new ArrayList<>();
            List<QwenSequenceState> sequences = new ArrayList<>();
            List<CompletableFuture<QwenExecutionContext.Outcome>> sequenceCompletions = new ArrayList<>();
            try {
                assertTrue(runtime.execute(List.of()).isEmpty(), "empty work must not attach an idle source");
                assertFalse(runtime.hasAttachedRunner());

                QwenSequenceState executeSequence = new QwenSequenceState(699);
                try {
                    var outcomes = runtime.execute(List.of(new QwenExecutionContext(
                            plan, executeSequence, QwenExecutionContext.ExecutionKind.PREFILL, 0, new int[] {1})));
                    assertEquals(1, outcomes.size());
                    assertEquals(
                            QwenExecutionContext.Status.SUCCESS,
                            outcomes.getFirst().status());
                    assertEquals(1, executeSequence.currentTokenPosition());
                    assertFalse(runtime.hasAttachedRunner(), "execute retained its temporary lattice source");
                } finally {
                    executeSequence.complete();
                }

                gpu.embeddingGate = new WorkGate(2);
                gpu.projectionGate = new WorkGate(2);
                for (int sequenceId = 1; sequenceId <= runtimes.size(); sequenceId++) {
                    EuhedralInferenceRuntime sequenceRuntime = runtimes.get(sequenceId - 1);
                    runners.add(sequenceRuntime.attachRunner());
                    QwenSequenceState sequence = new QwenSequenceState(sequenceId);
                    sequences.add(sequence);
                }
                for (int index = 0; index < sequences.size(); index++) {
                    QwenExecutionContext context = new QwenExecutionContext(
                            plan, sequences.get(index), QwenExecutionContext.ExecutionKind.PREFILL, 0, new int[] {
                                (index + 1) % QwenExecutionFixtures.VOCABULARY
                            });
                    sequenceCompletions.add(runtimes.get(index).submit(context));
                }
                try {
                    assertTrue(
                            gpu.embeddingGate.awaitEntries(),
                            "independent sequences did not execute concurrently; activeWorkers="
                                    + lattice.getActiveWorkers() + ", allowedCpus=" + cpus
                                    + ", projectionCalls=" + gpu.projectionCalls.get()
                                    + ", normCalls=" + gpu.normCalls.get()
                                    + ", embeddingCalls=" + gpu.embeddingCalls.get()
                                    + ", gatedWorkers=" + gpu.embeddingGate.workerCount());
                    assertEquals(2, gpu.embeddingGate.workerCount(), "independent sequences used the same worker");
                } finally {
                    gpu.embeddingGate.release();
                }
                try {
                    assertTrue(
                            gpu.projectionGate.awaitEntries(),
                            "independent projection work did not overlap; activeWorkers="
                                    + lattice.getActiveWorkers() + ", allowedCpus=" + cpus
                                    + ", projectionCalls=" + gpu.projectionCalls.get()
                                    + ", gatedWorkers=" + gpu.projectionGate.workerCount());
                    assertEquals(2, gpu.projectionGate.workerCount(), "projection work used the same worker");
                } finally {
                    gpu.projectionGate.release();
                }
                CompletableFuture.allOf(sequenceCompletions.toArray(CompletableFuture[]::new))
                        .get(30, TimeUnit.SECONDS);
                for (var completion : sequenceCompletions) {
                    var outcome = runtime.await(completion);
                    assertEquals(
                            QwenExecutionContext.Status.SUCCESS,
                            outcome.status(),
                            () -> String.valueOf(outcome.failure()));
                }

                for (int index = 0; index < runtimes.size(); index++) {
                    EuhedralInferenceRuntime sequenceRuntime = runtimes.get(index);
                    sequenceRuntime.disconnectRunner();
                    assertTrue(runners.get(index).isComplete());
                    assertFalse(runners.get(index).isAttached(), "completed runner still has its lattice downstream");
                    assertFalse(sequenceRuntime.hasAttachedRunner(), "runtime retained a detached runner");
                }
                awaitDrained(lattice);
                for (QwenSequenceState sequence : sequences) assertEquals(1, sequence.currentTokenPosition());
            } finally {
                gpu.projectionGate.release();
                gpu.embeddingGate.release();
                for (EuhedralInferenceRuntime sequenceRuntime : runtimes) {
                    if (sequenceRuntime.hasAttachedRunner()) sequenceRuntime.disconnectRunner();
                }
                for (QwenSequenceState sequence : sequences) sequence.complete();
            }
        } finally {
            lattice.close();
        }
    }

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void actualCompactLayerZeroSurvivesPrefillRunnerTeardownAndDecodesOnTheSameSequence() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(cpus.cardinality() >= 2, "requires two available physical CPUs for lattice workers");
        Path artifactPath = Path.of(System.getProperty("euhedral.qwen.artifact", DEFAULT_COMPACT_ARTIFACT.toString()));
        assumeTrue(Files.isRegularFile(artifactPath), "compact Qwen artifact is missing: " + artifactPath);
        QwenArtifact artifact = QwenArtifactReader.read(artifactPath);
        assertEquals(QwenArtifactHeader.COMPACT_VERSION, artifact.header().version());
        assertEquals(64, artifact.config().numHiddenLayers());
        assertEquals(QwenLayerType.GATED_DELTA_NET, artifact.config().layerTypes()[0]);

        var gpu = new SequenceGpu();
        QwenWeights weights = QwenWeightLoader.loadFirstLayer(artifactPath, artifact, gpu);
        var plan = new QwenExecutionPlan(weights);
        var lattice = createLattice(cpus);
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var sequence = new QwenSequenceState(701);
        long embeddingAddress = weights.tokenEmbedding().deviceAddress();
        LatticeEdge registrationProbe = new LatticeEdge(new AtomicBoolean());
        int registeredWorkersBeforeStart = registrationProbe.getThreadCount();
        try {
            lattice.start();
            awaitWorkers(lattice, 2, registrationProbe, registeredWorkersBeforeStart + 2);

            QwenExecutionRunner prefillRunner = runtime.attachRunner();
            QwenGdnSequenceState recurrent;
            long convolutionAddress;
            long recurrentAddress;
            try {
                var prefill = runtime.submit(new QwenExecutionContext(
                        plan, sequence, QwenExecutionContext.ExecutionKind.PREFILL, 0, new int[] {1814, 1815}));
                var outcome = runtime.await(prefill);
                assertEquals(
                        QwenExecutionContext.Status.SUCCESS, outcome.status(), () -> String.valueOf(outcome.failure()));
                recurrent = (QwenGdnSequenceState) sequence.recurrentState();
                convolutionAddress = recurrent.convolutionStateAddress();
                recurrentAddress = recurrent.recurrentStateAddress();
            } finally {
                runtime.disconnectRunner();
            }

            assertTrue(prefillRunner.isComplete());
            assertFalse(prefillRunner.isAttached());
            assertSame(recurrent, sequence.recurrentState());
            assertEquals(2, sequence.currentTokenPosition());
            assertFalse(gpu.freed.contains(convolutionAddress));
            assertFalse(gpu.freed.contains(recurrentAddress));
            assertFalse(gpu.freed.contains(embeddingAddress));

            QwenExecutionRunner decodeRunner = runtime.attachRunner();
            try {
                var decode = runtime.submit(new QwenExecutionContext(
                        plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 2, new int[] {1816}));
                var outcome = runtime.await(decode);
                assertEquals(
                        QwenExecutionContext.Status.SUCCESS, outcome.status(), () -> String.valueOf(outcome.failure()));
            } finally {
                runtime.disconnectRunner();
            }

            assertTrue(decodeRunner.isComplete());
            assertFalse(decodeRunner.isAttached());
            assertSame(recurrent, sequence.recurrentState());
            assertEquals(3, sequence.currentTokenPosition());
            assertEquals(convolutionAddress, gpu.lastConvolutionAddress.get());
            assertEquals(recurrentAddress, gpu.lastRecurrentAddress.get());
            assertEquals(2, gpu.convolutionCalls.get());
            assertEquals(2, gpu.recurrenceCalls.get());
            assertFalse(gpu.freed.contains(embeddingAddress));
            assertTrue(lattice.isDrained());

            sequence.complete();
            assertTrue(gpu.freed.contains(convolutionAddress));
            assertTrue(gpu.freed.contains(recurrentAddress));
            assertFalse(gpu.freed.contains(embeddingAddress));
        } finally {
            if (runtime.hasAttachedRunner()) runtime.disconnectRunner();
            if (sequence.terminalState() == QwenSequenceState.TerminalState.ACTIVE && !sequence.isExecutionClaimed()) {
                sequence.complete();
            }
            lattice.close();
        }
    }

    private static ControlPlaneLattice createLattice(BitSet cpus) {
        var workers = new BaseCloneableObject(new DefaultExecutor());
        var baseShard = ControlPlaneShard.createBaseShard("EuhedralRuntimeTestShard", workers);
        return ControlPlaneLattice.getOrCreate(
                new LatticeConfig("EuhedralRuntimeTestLattice", cpus, Duration.ofSeconds(10), baseShard));
    }

    private static LatticeTerminal throwingCompletionTerminal(RuntimeException failure) {
        return upstream -> upstream.addDownstream(new LatticeReceiver() {
            @Override
            public void addUpstream(LatticeSource source) {}

            @Override
            public void push(AbstractFrame frame) {}

            @Override
            public void onError(Throwable error) {}

            @Override
            public void onComplete() {
                throw failure;
            }
        });
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

    private static void verifySequenceStateSurvivesRunnerTeardown(ControlPlaneLattice lattice) throws Exception {
        var weights = QwenExecutionFixtures.statefulCompactWeights();
        var plan = QwenExecutionPlan.prefix(weights, 2);
        var gpu = new SequenceGpu();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var sequence = new QwenSequenceState(700);
        long embeddingAddress = weights.tokenEmbedding().deviceAddress();
        QwenExecutionRunner prefillRunner = runtime.attachRunner();
        try {
            var prefill = runtime.submit(new QwenExecutionContext(
                    plan, sequence, QwenExecutionContext.ExecutionKind.PREFILL, 0, new int[] {1, 2}));
            assertEquals(
                    QwenExecutionContext.Status.SUCCESS, runtime.await(prefill).status());
            var recurrent = (GdnSequenceStates) sequence.recurrentState();
            var attention = (AttentionSequenceStates) sequence.kvCacheState();
            var attentionState = attention.forLayer(1);
            var recurrentState = recurrent.forLayer(0);
            long convolutionAddress = recurrentState.convolutionStateAddress();
            long recurrentAddress = recurrentState.recurrentStateAddress();
            long kvAddress = attentionState.keyCacheAddress();

            runtime.disconnectRunner();

            assertTrue(prefillRunner.isComplete());
            assertFalse(prefillRunner.isAttached());
            assertSame(recurrent, sequence.recurrentState(), "runner teardown replaced GDN sequence state");
            assertSame(attention, sequence.kvCacheState(), "runner teardown replaced attention KV sequence state");
            assertEquals(2, sequence.currentTokenPosition());
            assertEquals(2, attentionState.length());
            assertFalse(gpu.freed.contains(convolutionAddress));
            assertFalse(gpu.freed.contains(recurrentAddress));
            assertFalse(gpu.freed.contains(kvAddress));
            assertFalse(gpu.freed.contains(embeddingAddress));

            QwenExecutionRunner decodeRunner = runtime.attachRunner();
            try {
                var decode = runtime.submit(new QwenExecutionContext(
                        plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 2, new int[] {3}));
                assertEquals(
                        QwenExecutionContext.Status.SUCCESS,
                        runtime.await(decode).status());
            } finally {
                runtime.disconnectRunner();
            }

            assertTrue(decodeRunner.isComplete());
            assertFalse(decodeRunner.isAttached());
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
            assertFalse(gpu.freed.contains(recurrentAddress));
            assertFalse(gpu.freed.contains(embeddingAddress));

            sequence.complete();
            assertTrue(gpu.freed.contains(convolutionAddress));
            assertTrue(gpu.freed.contains(recurrentAddress));
            assertTrue(gpu.freed.contains(kvAddress));
            assertFalse(gpu.freed.contains(embeddingAddress));
            assertFalse(runtime.hasAttachedRunner());
        } finally {
            if (runtime.hasAttachedRunner()) runtime.disconnectRunner();
            if (sequence.terminalState() == QwenSequenceState.TerminalState.ACTIVE && !sequence.isExecutionClaimed()) {
                sequence.complete();
            }
        }
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
                long startPosition) {
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
            return this.entered.await(15, TimeUnit.SECONDS);
        }

        private int workerCount() {
            return this.workers.size();
        }

        private void release() {
            this.released.countDown();
        }
    }
}
