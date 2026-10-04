package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.ArtifactProfile;
import io.euhedral_execution.inference.core.model_loader.EngineModelFixture;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.EngineExecutionFixture;
import io.euhedral_execution.inference.core.scheduling.GenerationTimingListener;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

@Timeout(60)
class InferenceEngineTest {
    @TempDir
    Path directory;

    @Test
    void successiveGenerationsShareTheEnginesOneLatticeAndRuntime() throws Exception {
        var lattices = new java.util.concurrent.atomic.AtomicInteger();
        var bootstrap = new FakeBootstrap() {
            @Override
            io.euhedral_execution.core.control_plane.ControlPlaneLattice createLattice(InferenceConfig config) {
                lattices.incrementAndGet();
                return super.createLattice(config);
            }
        };
        try (var engine = InferenceEngine.load(config(), bootstrap)) {
            try (var session = engine.createSession(GenerationConfig.greedy(1L))) {
                session.generate("!", 2, ignored -> {});
                session.generate("!", 2, ignored -> {});
            }
            assertEquals(1, lattices.get());
        }
    }

    private static int trackedSessions(InferenceEngine engine) {
        try {
            var field = InferenceEngine.class.getDeclaredField("sessions");
            field.setAccessible(true);
            synchronized (engine) {
                return ((java.util.Collection<?>) field.get(engine)).size();
            }
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }

    @Test
    void completedSessionClosesRemoveTrackingWithoutWaitingForAnotherRequest() throws Exception {
        try (var engine = InferenceEngine.load(config(), new FakeBootstrap())) {
            var live = engine.createSession(GenerationConfig.greedy(1L));
            for (int index = 0; index < 20; index++) {
                var session = engine.createSession(GenerationConfig.greedy(1L));
                session.generate("!", 1, ignored -> {});
                session.close();
                session.close();
                assertEquals(1, trackedSessions(engine));
                assertEquals(java.util.List.of(1), session.generatedTokenIds());
            }
            live.generate("!", 1, ignored -> {});
        }
    }

    @Test
    void callbackCloseRemainsTrackedUntilGenerationUnwinds() throws Exception {
        try (var engine = InferenceEngine.load(config(), new FakeBootstrap())) {
            var session = engine.createSession(GenerationConfig.greedy(1L));
            session.generate("!", 1, text -> {
                session.close();
                assertEquals(1, trackedSessions(engine));
                assertThrows(IllegalStateException.class, engine::close);
            });
            assertEquals(0, trackedSessions(engine));
        }
    }

    @Test
    void failedSessionCloseStaysTrackedUntilSuccessfulRetry() throws Exception {
        var boot = new FakeBootstrap();
        try (var engine = InferenceEngine.load(config(), boot)) {
            var session = engine.createSession(GenerationConfig.greedy(1L));
            session.generate("!", 1, ignored -> {});
            boot.gpu.freeFailures.set(100);
            try {
                assertThrows(IllegalStateException.class, session::close);
                assertEquals(1, trackedSessions(engine));
            } finally {
                boot.gpu.freeFailures.set(0);
            }
            session.close();
            assertEquals(0, trackedSessions(engine));
        }
    }

    @Test
    void ownsSessionsAndClosesModelAfterThem() throws Exception {
        var bootstrap = new FakeBootstrap();
        try (var engine = InferenceEngine.load(config(), bootstrap)) {
            var first = engine.createSession(GenerationConfig.greedy(1L));
            var second = engine.createSession(GenerationConfig.greedy(1L));
            assertNotSame(first, second);
            assertNotEquals(
                    EngineExecutionFixture.sequence(first).sequenceId(),
                    EngineExecutionFixture.sequence(second).sequenceId());
            assertEquals(first.generate("!", 2, ignored -> {}), second.generate("!", 2, ignored -> {}));
            assertNotSame(
                    EngineExecutionFixture.sequence(first).kvCacheState(),
                    EngineExecutionFixture.sequence(second).kvCacheState());
            assertNotSame(
                    EngineExecutionFixture.sequence(first).recurrentState(),
                    EngineExecutionFixture.sequence(second).recurrentState());
            assertEquals(1, bootstrap.gpu.embeddingAddresses.stream().distinct().count());
            assertEquals(1, bootstrap.modelLoads);
            first.close();
            assertFalse(second.isClosed());
            assertFalse(bootstrap.gpuClosed);
            assertFalse(bootstrap.gpu.freed().contains(1000L), "model freed by session close");
            second.generate("!", 1, ignored -> {});
            engine.close();
            assertTrue(first.isClosed());
            assertTrue(second.isClosed());
            assertTrue(engine.isClosed());
            assertTrue(bootstrap.gpuClosed);
            assertTrue(bootstrap.gpu.freed().containsAll(bootstrap.gpu.allocated()));
            assertThrows(IllegalStateException.class, () -> engine.createSession(GenerationConfig.greedy(2L)));
            engine.close();
            assertEquals(1, bootstrap.gpuCloseCount);
        }
    }

    InferenceConfig config() throws Exception {
        return config(0, 2048);
    }

    InferenceConfig config(long prefixCacheBytes, int prefixCacheCheckpointTokens) throws Exception {
        Files.writeString(directory.resolve("tokenizer.json"), """
            {"model":{"type":"BPE","vocab":{"!":0,"A":1,"B":2,"C":3,"D":4,"E":5,"F":6},"merges":[]},
             "normalizer":{"type":"NFC"},
             "pre_tokenizer":{"type":"Sequence","pretokenizers":[
             {"type":"Split","pattern":{"Regex":"."},"behavior":"Isolated","invert":false},
             {"type":"ByteLevel"}]},
             "added_tokens":[{"content":"<eos>","id":7,"special":true}]}
            """);
        Files.writeString(directory.resolve("tokenizer_config.json"), "{\"eos_token\":\"<eos>\"}");
        Files.writeString(directory.resolve("generation_config.json"), "{\"eos_token_id\":7}");
        BitSet cpus = new BitSet();
        cpus.set(SystemInfo.getPCpuSet().nextSetBit(0));
        return new InferenceConfig(
                directory.resolve("model.edrl"),
                directory,
                directory.resolve("lib.so"),
                cpus,
                InferenceConfig.DEFAULT_MAX_CONTEXT_TOKENS,
                Duration.ofSeconds(10),
                prefixCacheBytes,
                prefixCacheCheckpointTokens);
    }

    @Test
    void memoryThatCannotBePinnedLeavesTheEngineServingWithoutACache() throws Exception {
        try (var engine = InferenceEngine.load(config(1 << 20, 512), new FakeBootstrap())) {
            assertNull(engine.prefixCacheStats());
            try (var session = engine.createSession(GenerationConfig.greedy(1L))) {
                assertEquals(java.util.List.of(1), session.generate("!", 1, ignored -> {}));
            }
        }
    }

    @Test
    void anEngineWithACacheReusesAPromptsPrefixAndFreesTheArenaOnClose() throws Exception {
        var bootstrap = new PinningBootstrap();
        String prompt = "!".repeat(700);
        try (var engine = InferenceEngine.load(config(64L << 20, 512), bootstrap)) {
            assertEquals(0, engine.prefixCacheStats().lookups());
            for (int request = 0; request < 2; request++)
                try (var session = engine.createSession(GenerationConfig.greedy(1L))) {
                    session.generate(prompt, 1, ignored -> {});
                }
            var stats = engine.prefixCacheStats();
            assertEquals(2, stats.lookups());
            assertEquals(1, stats.hits());
            assertEquals(512, stats.reusedTokens());
            assertEquals(java.util.List.of(), bootstrap.gpu().pinnedFrees);
        }
        assertEquals(1, bootstrap.gpu().pinnedFrees.size(), "closing the engine unpinned the arena");
    }

    static class PinningGpu extends EngineExecutionFixture.SamplingGpu {
        private final java.util.Map<Long, java.lang.foreign.Arena> arenas =
                new java.util.concurrent.ConcurrentHashMap<>();
        final java.util.List<Long> pinnedFrees = new java.util.concurrent.CopyOnWriteArrayList<>();

        PinningGpu() {
            super(8);
        }

        @Override
        public long allocateHostWeights(long byteSize) {
            var arena = java.lang.foreign.Arena.ofShared();
            long address = arena.allocate(byteSize, 4096).address();
            this.arenas.put(address, arena);
            return address;
        }

        @Override
        public void freeHostWeights(long address) {
            this.pinnedFrees.add(address);
            this.arenas.remove(address).close();
        }
    }

    static class PinningBootstrap extends FakeBootstrap {
        private final PinningGpu pinning = new PinningGpu();

        PinningGpu gpu() {
            return this.pinning;
        }

        @Override
        ExecutionGpu openGpu(Path path) {
            return this.pinning;
        }
    }

    @Test
    void concurrentSessionsShareOneRuntime() throws Exception {
        var bootstrap = new FakeBootstrap();
        var entered = new java.util.concurrent.CountDownLatch(1);
        var release = new java.util.concurrent.CountDownLatch(1);
        bootstrap.gpu.afterEmbedding(() -> {
            entered.countDown();
            await(release);
        });
        try (var engine = InferenceEngine.load(config(), bootstrap);
                var executor = java.util.concurrent.Executors.newFixedThreadPool(2)) {
            var first = engine.createSession(GenerationConfig.greedy(1L));
            var second = engine.createSession(GenerationConfig.greedy(2L));
            var a = executor.submit(() -> first.generate("!", 2, ignored -> {}));
            assertTrue(entered.await(10, java.util.concurrent.TimeUnit.SECONDS));
            var b = executor.submit(() -> second.generate("!", 3, ignored -> {}));
            try {
                // Concurrent quanta run on separate graphs; the second one waits instead of failing.
                assertThrows(
                        java.util.concurrent.TimeoutException.class,
                        () -> b.get(100, java.util.concurrent.TimeUnit.MILLISECONDS));
            } finally {
                release.countDown();
            }
            assertEquals(2, a.get(10, java.util.concurrent.TimeUnit.SECONDS).size());
            assertEquals(3, b.get(10, java.util.concurrent.TimeUnit.SECONDS).size());
            assertEquals(3, first.currentTokenPosition());
            assertEquals(4, second.currentTokenPosition());
        } finally {
            release.countDown();
        }
    }

    @Test
    void startupFailureClosesUploadedModelAndBackend() throws Exception {
        var bootstrap = new FakeBootstrap() {
            @Override
            void startLattice(io.euhedral_execution.core.control_plane.ControlPlaneLattice lattice) {
                super.startLattice(lattice);
                throw new IllegalStateException("after lattice start");
            }
        };
        assertEquals(
                "after lattice start",
                assertThrows(IllegalStateException.class, () -> InferenceEngine.load(config(), bootstrap))
                        .getMessage());
        assertTrue(bootstrap.gpuClosed);
        assertEquals(bootstrap.gpu.allocated(), bootstrap.gpu.freed());
        try (var next = InferenceEngine.load(config(), new FakeBootstrap())) {
            assertFalse(next.isClosed());
        }
    }

    @Test
    void shutdownFromCallbackIsRejectedBeforeClosingBegins() throws Exception {
        try (var engine = InferenceEngine.load(config(), new FakeBootstrap())) {
            var session = engine.createSession(GenerationConfig.greedy(1L));
            session.generate("!", 1, text -> {
                assertThrows(IllegalStateException.class, engine::close);
                assertFalse(engine.isClosed());
            });
            assertEquals(2, session.currentTokenPosition());
        }
    }

    static void await(java.util.concurrent.CountDownLatch latch) {
        try {
            if (!latch.await(10, java.util.concurrent.TimeUnit.SECONDS))
                throw new IllegalStateException("gate timeout");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }

    @Test
    void closingRejectsAdmissionAndQueriesBeforeWaitingForActiveExecution() throws Exception {
        var bootstrap = new FakeBootstrap();
        var entered = new java.util.concurrent.CountDownLatch(1);
        var release = new java.util.concurrent.CountDownLatch(1);
        bootstrap.gpu.afterEmbedding(() -> {
            entered.countDown();
            await(release);
        });
        try (var engine = InferenceEngine.load(config(), bootstrap);
                var executor = java.util.concurrent.Executors.newFixedThreadPool(3)) {
            var session = engine.createSession(GenerationConfig.greedy(1L));
            var generation = executor.submit(() -> session.generate("!", 4, ignored -> {}));
            assertTrue(entered.await(10, java.util.concurrent.TimeUnit.SECONDS));
            var close = executor.submit(engine::close);
            long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(5);
            while (!engine.isClosed() && System.nanoTime() < deadline) Thread.onSpinWait();
            try {
                assertTrue(engine.isClosed());
                assertFalse(bootstrap.gpuClosed);
                assertEquals(1, trackedSessions(engine), "in-flight session retired before execution stopped");
                assertThrows(IllegalStateException.class, () -> engine.createSession(GenerationConfig.greedy(2L)));
                var query = executor.submit(() -> assertThrows(IllegalStateException.class, engine::deviceMemoryInfo));
                query.get(1, java.util.concurrent.TimeUnit.SECONDS);
            } finally {
                release.countDown();
            }
            generation.get(10, java.util.concurrent.TimeUnit.SECONDS);
            close.get(10, java.util.concurrent.TimeUnit.SECONDS);
            assertEquals(0, trackedSessions(engine));
            assertTrue(session.isClosed());
            assertTrue(bootstrap.gpuClosed);
            assertTrue(bootstrap.gpu.freed().containsAll(bootstrap.gpu.allocated()));
        } finally {
            release.countDown();
        }
    }

    @Test
    void failuresBeforeAndAfterGpuCreationReleaseOnlyCreatedResources() throws Exception {
        for (int stage = 0; stage < 5; stage++) {
            final int selected = stage;
            var bootstrap = new FakeBootstrap() {
                @Override
                QwenArtifact readArtifact(Path path) {
                    if (selected == 0) throw new IllegalStateException("artifact");
                    return super.readArtifact(path);
                }

                @Override
                ExecutionGpu openGpu(Path path) {
                    if (selected == 1) throw new IllegalStateException("gpu");
                    return super.openGpu(path);
                }

                @Override
                QwenModel loadModel(
                        Path path, QwenArtifact artifact, ArtifactProfile profile, ExecutionGpu gpu, int maxContext)
                        throws java.io.IOException {
                    if (selected == 2) throw new IllegalStateException("model");
                    if (selected == 3)
                        return EngineModelFixture.load(
                                gpu,
                                new io.euhedral_execution.inference.core.model_loader.QwenWeights(
                                        null, null, null, null, null, null));
                    return super.loadModel(path, artifact, profile, gpu, maxContext);
                }

                @Override
                io.euhedral_execution.core.control_plane.ControlPlaneLattice createLattice(InferenceConfig config) {
                    throw new IllegalStateException("lattice");
                }
            };
            assertThrows(RuntimeException.class, () -> InferenceEngine.load(config(), bootstrap));
            assertEquals(selected >= 2, bootstrap.gpuClosed);
            assertEquals(bootstrap.gpu.allocated(), bootstrap.gpu.freed());
        }
        var config = config();
        Files.delete(directory.resolve("tokenizer.json"));
        var bootstrap = new FakeBootstrap();
        assertThrows(java.io.IOException.class, () -> InferenceEngine.load(config, bootstrap));
        assertFalse(bootstrap.gpuClosed);
        try (var engine = InferenceEngine.load(config(), new FakeBootstrap())) {
            assertFalse(engine.isClosed());
        }
    }

    @Test
    void refusesAContextLongerThanTheModelsPositions() {
        var layers = new io.euhedral_execution.inference.core.model_loader.config.QwenLayerType[4];
        java.util.Arrays.fill(
                layers, io.euhedral_execution.inference.core.model_loader.config.QwenLayerType.FULL_ATTENTION);
        var artifact = new QwenArtifact(
                null,
                InferenceRunSnapshotTest.config(layers),
                new io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor[0]);
        var failure = assertThrows(
                IllegalArgumentException.class,
                () -> new InferenceEngine.Bootstrap().loadModel(Path.of("model.edrl"), artifact, null, null, 262145));
        assertEquals(
                "a context of 262145 tokens is longer than the 262144 positions the model supports; set a smaller max "
                        + "context",
                failure.getMessage());
    }

    @Test
    void rejectsASecondEngineWithoutDisturbingTheFirst() throws Exception {
        var config = config();
        try (var engine = InferenceEngine.load(config, new FakeBootstrap())) {
            var rejected = new FakeBootstrap();
            assertThrows(IllegalStateException.class, () -> InferenceEngine.load(config, rejected));
            assertFalse(rejected.gpuClosed);
            engine.createSession(GenerationConfig.greedy(1L)).generate("!", 1, ignored -> {});
        }
    }

    @Test
    void configurationCopiesCpuSetAndRejectsEmptyWorkers() throws Exception {
        var config = config();
        var returned = config.workerCpus();
        returned.clear();
        assertFalse(config.workerCpus().isEmpty());
        assertThrows(
                IllegalArgumentException.class,
                () -> new InferenceConfig(
                        config.artifactPath(),
                        directory,
                        config.cudaLibraryPath(),
                        new BitSet(),
                        Duration.ofSeconds(1)));
    }

    @Test
    void closeRetainsBackendUntilPersistentCleanupCanBeRetried() throws Exception {
        var boot = new FakeBootstrap();
        var engine = InferenceEngine.load(config(), boot);
        var session = engine.createSession(GenerationConfig.greedy(1L));
        session.generate("!", 1, ignored -> {});
        boot.gpu.freeFailures.set(100);
        try {
            assertThrows(IllegalStateException.class, engine::close);
            assertTrue(engine.isClosed());
            assertEquals(0, boot.gpuCloseCount);
            assertFalse(boot.gpu.freed().contains(1000L));
        } finally {
            boot.gpu.freeFailures.set(0);
            engine.close();
        }
        assertEquals(1, boot.gpuCloseCount);
        assertEquals(new java.util.HashSet<>(boot.gpu.allocated()), new java.util.HashSet<>(boot.gpu.freed()));
        assertEquals(boot.gpu.freed().size(), new java.util.HashSet<>(boot.gpu.freed()).size());
    }

    @Test
    void interruptedCloseDoesNotReuseAStoppedFabricOnReload() throws Exception {
        var configuration = config();
        var first = InferenceEngine.load(configuration, new FakeBootstrap());
        Thread.currentThread().interrupt();
        try {
            first.close();
        } finally {
            Thread.interrupted();
        }
        try (var next = InferenceEngine.load(configuration, new FakeBootstrap());
                var session = next.createSession(GenerationConfig.greedy(1L))) {
            assertEquals(java.util.List.of(1), session.generate("!", 1, ignored -> {}));
        }
    }

    @Test
    void startupRollbackFailureRetainsBackendAndCanBeClosedExplicitly() throws Exception {
        var boot = new FakeBootstrap() {
            @Override
            void startLattice(io.euhedral_execution.core.control_plane.ControlPlaneLattice lattice) {
                super.startLattice(lattice);
                this.gpu.freeFailures.set(100);
                throw new IllegalStateException("startup failed");
            }
        };
        var configuration = config();
        var failed =
                assertThrows(InferenceEngine.StartupFailure.class, () -> InferenceEngine.load(configuration, boot));
        try {
            assertEquals("startup failed", failed.getCause().getMessage());
            assertFalse(boot.gpuClosed);
            assertFalse(boot.gpu.freed().contains(1000L));
            assertThrows(IllegalStateException.class, () -> InferenceEngine.load(configuration, new FakeBootstrap()));
        } finally {
            boot.gpu.freeFailures.set(0);
            failed.close();
        }
        failed.close();
        assertEquals(1, boot.gpuCloseCount);
        assertEquals(boot.gpu.allocated(), boot.gpu.freed());
        try (var next = InferenceEngine.load(configuration, new FakeBootstrap())) {
            assertFalse(next.isClosed());
        }
    }

    private static List<Integer> prefillChunkLengths(EngineExecutionFixture.SamplingGpu gpu, int decodeQuanta) {
        var inputs = gpu.embeddingInputs;
        return inputs.subList(0, inputs.size() - decodeQuanta).stream()
                .map(chunk -> chunk.length)
                .toList();
    }

    @Test
    void prefillRunsInChunksOf512Tokens() throws Exception {
        var bootstrap = new FakeBootstrap();
        var config = config();
        String prompt = "!".repeat(1100);
        try (var engine = InferenceEngine.load(config, bootstrap);
                var session = engine.createSession(GenerationConfig.greedy(1L))) {
            assertEquals(config, engine.config());
            assertEquals(1100, engine.tokenizer().encodeWithModelSpecialTokens(prompt).length);
            session.generate(prompt, 1, ignored -> {});
            assertEquals(List.of(512, 512, 76), prefillChunkLengths(bootstrap.gpu, 1));
        }
    }

    @Test
    void rejectsUnavailableWorkersBeforeLoadingAnyResource() throws Exception {
        var bootstrap = new FakeBootstrap() {
            @Override
            ProcessorTopology processorTopology() {
                return WorkerProcessorSelectionTest.HYBRID;
            }
        };
        Files.delete(config().tokenizerDirectory().resolve("tokenizer.json"));
        for (int unavailable : new int[] {13, 2}) {
            var config = new InferenceConfig(
                    directory.resolve("model.edrl"),
                    directory,
                    directory.resolve("lib.so"),
                    ProcessorTopology.bits(0, unavailable),
                    Duration.ofSeconds(10));
            var failure = assertThrows(IllegalArgumentException.class, () -> InferenceEngine.load(config, bootstrap));
            assertTrue(failure.getMessage().contains("{" + unavailable + "}"), failure.getMessage());
        }
        assertEquals(0, bootstrap.artifactReads);
        assertFalse(bootstrap.gpuClosed);
        try (var engine = InferenceEngine.load(config(), new FakeBootstrap())) {
            assertFalse(engine.isClosed(), "rejection must not retain the process-wide lattice");
        }
    }

    @Test
    void snapshotRecordsTheEngineConfigurationIdentityAndSuppliedGeneration() throws Exception {
        var config = config();
        var generation = new GenerationConfig(0.5f, 3, 0.9f, 7L, false);
        try (var engine = InferenceEngine.load(config, new FakeBootstrap())) {
            var snapshot = engine.snapshot(generation);
            int cpu = config.workerCpus().nextSetBit(0);
            assertEquals(
                    new InferenceRunSnapshot.Configuration(
                            List.of(cpu), InferenceConfig.DEFAULT_MAX_CONTEXT_TOKENS, null, 0, 0),
                    snapshot.configuration());
            assertEquals(List.of(SystemInfo.getCpuInfo(cpu).core()), snapshot.workerCoreIds());
            assertEquals(generation, snapshot.generation());
            assertNull(engine.snapshot().generation());
            assertEquals(
                    directory.resolve("model.edrl").toString(), snapshot.model().artifactPath());
            assertNull(snapshot.model().artifactBytes(), "missing artifact file has no measured size");
            assertEquals(
                    InferenceRunSnapshot.Dimensions.of(engine.modelConfig()),
                    snapshot.model().dimensions());
            var runtime = snapshot.runtime();
            assertEquals(Runtime.version().toString(), runtime.javaVersion());
            assertEquals(InferenceRunSnapshot.UNAVAILABLE, runtime.nativeRuntimeVersion());
            assertEquals(directory.resolve("lib.so").toString(), runtime.nativeLibraryPath());
            assertTrue(runtime.euhedralCoreArtifact().startsWith("euhedral-core"), runtime.euhedralCoreArtifact());
            assertEquals(snapshot.toJson(), engine.snapshot(generation).toJson());
            assertTrue(snapshot.toJson().contains("\"maxContextTokens\":32768"), snapshot.toJson());
        }
    }

    /// Records timing boundaries as strings with their timestamps, in call order, and the output texts with the
    /// threads that received them. Boundaries come from the workers that advance the generation; outputs from
    /// the calling thread, concurrently.
    static final class RecordingTiming implements GenerationTimingListener {
        final List<String> events = java.util.Collections.synchronizedList(new java.util.ArrayList<>());
        final List<Long> times = java.util.Collections.synchronizedList(new java.util.ArrayList<>());
        final List<String> outputs = java.util.Collections.synchronizedList(new java.util.ArrayList<>());
        final java.util.Set<Thread> outputThreads = java.util.concurrent.ConcurrentHashMap.newKeySet();

        private synchronized void add(String event, long... nanos) {
            events.add(event);
            for (long time : nanos) times.add(time);
        }

        @Override
        public void promptEncoded(long nanos, int promptTokens) {
            add("encoded:" + promptTokens, nanos);
        }

        @Override
        public void prefillQuantum(long startNanos, long executedNanos, int tokens) {
            add("prefill:" + tokens, startNanos, executedNanos);
        }

        @Override
        public void firstTokenSelected(long nanos, int tokenId) {
            add("first:" + tokenId, nanos);
        }

        @Override
        public void decodeQuantum(
                long startNanos, long executedNanos, long selectedNanos, boolean sampled, int selectedTokenId) {
            if (!sampled) assertEquals(executedNanos, selectedNanos);
            add(sampled ? "decode:" + selectedTokenId : "commit", startNanos, executedNanos, selectedNanos);
        }

        void output(String text) {
            outputs.add(text);
            outputThreads.add(Thread.currentThread());
        }
    }

    private static void assertNonDecreasing(List<Long> times) {
        for (int index = 1; index < times.size(); index++)
            assertTrue(times.get(index - 1) <= times.get(index), "timing boundary out of order at " + index);
    }

    @Test
    void timingBoundariesFollowPrefillSelectionAndCommitOrderWhileOutputReachesTheCaller() throws Exception {
        var bootstrap = new FakeBootstrap();
        bootstrap.gpu.selectTokens(1, 2, 3);
        try (var engine = InferenceEngine.load(config(), bootstrap);
                var session = engine.createSession(GenerationConfig.greedy(1L))) {
            var timing = new RecordingTiming();
            assertEquals(List.of(1, 2, 3), session.generate("!!!!!", 3, timing::output, null, timing));
            assertEquals(List.of("encoded:5", "prefill:5", "first:1", "decode:2", "decode:3", "commit"), timing.events);
            assertNonDecreasing(timing.times);
            assertEquals(List.of("A", "B", "C"), timing.outputs, "text arrives in generation order");
            assertEquals(java.util.Set.of(Thread.currentThread()), timing.outputThreads, "output runs on the caller");
        }
    }

    @Test
    void prefillOnlyAndImmediateEosReportNoDecodeWork() throws Exception {
        var bootstrap = new FakeBootstrap();
        try (var engine = InferenceEngine.load(config(), bootstrap)) {
            var prefillOnly = new RecordingTiming();
            try (var session = engine.createSession(GenerationConfig.greedy(1L))) {
                assertEquals(List.of(), session.generate("!!!", 0, prefillOnly::output, null, prefillOnly));
            }
            assertEquals(List.of("encoded:3", "prefill:3"), prefillOnly.events, "prefill-only has no first token");

            bootstrap.gpu.selectTokens(7);
            var eos = new RecordingTiming();
            try (var session = engine.createSession(GenerationConfig.greedy(1L))) {
                assertEquals(List.of(7), session.generate("!!!", 8, eos::output, null, eos));
            }
            assertEquals(List.of("encoded:3", "prefill:3", "first:7"), eos.events, "EOS is never committed");
        }
    }

    @Test
    void disabledTimingLeavesGenerationUnchanged() throws Exception {
        var disabled = new FakeBootstrap();
        var enabled = new FakeBootstrap();
        List<Integer> withoutTiming;
        try (var engine = InferenceEngine.load(config(), disabled);
                var session = engine.createSession(GenerationConfig.greedy(1L))) {
            withoutTiming = session.generate("!!!!", 3, ignored -> {}, null, null);
            assertEquals(withoutTiming, session.generatedTokenIds());
        }
        try (var engine = InferenceEngine.load(config(), enabled);
                var session = engine.createSession(GenerationConfig.greedy(1L))) {
            assertEquals(withoutTiming, session.generate("!!!!", 3, ignored -> {}, null, new RecordingTiming()));
        }
        assertEquals(
                disabled.gpu.embeddingInputs.stream()
                        .map(java.util.Arrays::toString)
                        .toList(),
                enabled.gpu.embeddingInputs.stream()
                        .map(java.util.Arrays::toString)
                        .toList());
    }

    static class FakeBootstrap extends InferenceEngine.Bootstrap {
        final EngineExecutionFixture.SamplingGpu gpu = new EngineExecutionFixture.SamplingGpu(8);
        volatile boolean gpuClosed;
        int gpuCloseCount;
        int modelLoads;
        int artifactReads;

        @Override
        QwenArtifact readArtifact(Path path) {
            artifactReads++;
            return null;
        }

        @Override
        ExecutionGpu openGpu(Path path) {
            return gpu;
        }

        @Override
        QwenModel loadModel(Path path, QwenArtifact artifact, ArtifactProfile profile, ExecutionGpu gpu, int maxContext)
                throws java.io.IOException {
            modelLoads++;
            return EngineModelFixture.load(gpu, EngineExecutionFixture.weights());
        }

        @Override
        void closeGpu(ExecutionGpu gpu) {
            gpuClosed = true;
            gpuCloseCount++;
        }
    }
}
