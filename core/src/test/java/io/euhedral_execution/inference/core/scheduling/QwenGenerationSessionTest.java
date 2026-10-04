package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.core.config.LatticeConfig;
import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.control_plane.ControlPlaneShard;
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.hardware_utils.SystemInfo;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.tokenizer.JsonEnvelopeConstraint;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class QwenGenerationSessionTest {

    private static final Path TOKENIZER_DIRECTORY =
            Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
    private static final AtomicLong LATTICE_ID = new AtomicLong();
    private static QwenTokenizer tokenizer;

    @BeforeAll
    static void loadTokenizer() throws Exception {
        assumeTrue(Files.isRegularFile(TOKENIZER_DIRECTORY.resolve("tokenizer.json")));
        tokenizer = QwenTokenizer.load(TOKENIZER_DIRECTORY);
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void continuesOnOneSequenceAndCopiesEachSampledRowInsideItsQuantum() throws Exception {
        int vocabularySize = testVocabularySize();
        var weights = QwenExecutionFixtures.statefulCompactWeights(vocabularySize);
        var plan = new QwenExecutionPlan(weights);
        var gpu = new SamplingGpu(vocabularySize);
        gpu.selectedTokenIds = new int[] {1, 2, 3, 4, 5};
        var lattice = createLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var session = new QwenGenerationSession(tokenizer, plan, runtime, gpu, 810, GenerationConfig.greedy(41L));
        // The final row reaches host memory on the quantum's own stream, before the quantum retires.
        gpu.onRowCopy = () -> assertEquals(1, runtime.activeQuanta(), "the row was copied outside its quantum");
        lattice.start();
        awaitWorker(lattice);
        try {
            // Text reaches the calling thread while later quanta run; quanta advance on the lattice's workers.
            StringBuilder firstOutput = new StringBuilder();
            java.util.Set<Thread> outputThreads = java.util.concurrent.ConcurrentHashMap.newKeySet();
            List<Integer> firstTokens = session.generate("!", 3, text -> {
                firstOutput.append(text);
                outputThreads.add(Thread.currentThread());
                assertTrue(gpu.closedLogits.isEmpty(), "device logits belong to their graph, not to a token");
            });

            int[] firstPromptTokens = tokenizer.encodeWithModelSpecialTokens("!");
            assertArrayEquals(new int[] {0}, firstPromptTokens);
            assertEquals(List.of(1, 2, 3), firstTokens);
            assertEquals(java.util.Set.of(Thread.currentThread()), outputThreads, "output runs on the caller");
            assertEquals(tokenizer.decode(new int[] {1, 2, 3}), firstOutput.toString());
            assertEquals(firstPromptTokens.length + 3L, session.currentTokenPosition());
            assertEquals(List.of(1, 2, 3), session.generatedTokenIds());
            assertEquals(0, runtime.activeQuanta());

            Object recurrentState = session.sequenceState().recurrentState();
            Object kvState = session.sequenceState().kvCacheState();
            long convolutionAddress =
                    ((GdnSequenceStates) recurrentState).forLayer(0).convolutionStateAddress();
            long recurrentAddress =
                    ((GdnSequenceStates) recurrentState).forLayer(0).recurrentStateAddress();
            long keyCacheAddress =
                    ((AttentionSequenceStates) kvState).forLayer(1).keyCacheAddress();

            StringBuilder secondOutput = new StringBuilder();
            List<Integer> secondTokens = session.generate("!", 2, text -> {
                secondOutput.append(text);
                assertTrue(gpu.closedLogits.isEmpty(), "device logits belong to their graph, not to a token");
            });

            int[] continuationPromptTokens = tokenizer.encodeText("!");
            assertEquals(List.of(4, 5), secondTokens);
            assertEquals(tokenizer.decode(new int[] {4, 5}), secondOutput.toString());
            assertEquals(firstPromptTokens.length * 2L + 5L, session.currentTokenPosition());
            assertEquals(List.of(1, 2, 3, 4, 5), session.generatedTokenIds());
            assertSame(recurrentState, session.sequenceState().recurrentState());
            assertSame(kvState, session.sequenceState().kvCacheState());
            keyCacheAddress = ((AttentionSequenceStates) kvState).forLayer(1).keyCacheAddress();
            assertEquals(
                    session.currentTokenPosition(),
                    ((AttentionSequenceStates) kvState).forLayer(1).length());
            assertEquals(List.of(0L, 1L, 2L, 3L, 4L, 5L, 6L), gpu.attentionStartPositions);
            assertEquals(java.util.Collections.nCopies(7, convolutionAddress), gpu.convolutionStateAddresses);
            assertEquals(java.util.Collections.nCopies(7, recurrentAddress), gpu.recurrentStateAddresses);
            assertEquals(keyCacheAddress, gpu.keyCacheAddresses.getLast());
            assertTrue(gpu.keyCacheAddresses.stream().allMatch(address -> address != 0));
            assertEquals(7, gpu.embeddingInputs.size());
            assertArrayEquals(firstPromptTokens, gpu.embeddingInputs.get(0));
            assertArrayEquals(new int[] {1}, gpu.embeddingInputs.get(1));
            assertArrayEquals(new int[] {2}, gpu.embeddingInputs.get(2));
            assertArrayEquals(new int[] {3}, gpu.embeddingInputs.get(3));
            assertArrayEquals(continuationPromptTokens, gpu.embeddingInputs.get(4));
            assertArrayEquals(new int[] {4}, gpu.embeddingInputs.get(5));
            assertArrayEquals(new int[] {5}, gpu.embeddingInputs.get(6));
            assertEquals(5, gpu.sampledLogitRows.size(), "one final-row copy per sampled token");
            assertEquals(2, gpu.allocatedLogits.size(), "one logits buffer each for the prefill and decode graphs");
            assertEquals(0, runtime.activeQuanta());

            session.close();
            assertTrue(session.isClosed());
            assertEquals(
                    QwenSequenceState.TerminalState.COMPLETED,
                    session.sequenceState().terminalState());
            assertTrue(gpu.frees.contains(convolutionAddress));
            assertTrue(gpu.frees.contains(recurrentAddress));
            assertTrue(gpu.frees.contains(keyCacheAddress));
            assertThrows(IllegalStateException.class, () -> ((GdnSequenceStates) recurrentState).forLayer(0));
        } finally {
            session.close();
            runtime.close();
            lattice.close();
        }
        assertLogitsFreedOnceWithTheirGraphs(gpu);
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void aBlockedOutputCallbackNeverStallsTheQuanta() throws Exception {
        int vocabularySize = testVocabularySize();
        var weights = QwenExecutionFixtures.statefulCompactWeights(vocabularySize);
        var plan = new QwenExecutionPlan(weights);
        var gpu = new SamplingGpu(vocabularySize);
        gpu.selectedTokenIds = new int[] {1, 2, 3, 4};
        var lattice = createLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var session = new QwenGenerationSession(tokenizer, plan, runtime, gpu, 815, GenerationConfig.greedy(43L));
        lattice.start();
        awaitWorker(lattice);
        try {
            // The first text is held until every quantum of the call has run: generation advances on the workers
            // whatever the calling thread does with its text.
            long finalPosition = tokenizer.encodeWithModelSpecialTokens("!").length + 4L;
            boolean[] advancedWhileBlocked = {false};
            StringBuilder output = new StringBuilder();
            List<Integer> tokens = session.generate("!", 4, text -> {
                if (output.isEmpty()) {
                    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
                    while (session.currentTokenPosition() < finalPosition && System.nanoTime() < deadline)
                        Thread.onSpinWait();
                    advancedWhileBlocked[0] = session.currentTokenPosition() == finalPosition;
                }
                output.append(text);
            });
            assertTrue(advancedWhileBlocked[0], "every quantum ran while the first output was blocked");
            assertEquals(List.of(1, 2, 3, 4), tokens);
            assertEquals(tokenizer.decode(new int[] {1, 2, 3, 4}), output.toString());
        } finally {
            session.close();
            runtime.close();
            lattice.close();
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void longPromptPrefillsInBoundedQuantaWithoutChangingTheTokenSequence() throws Exception {
        int vocabularySize = testVocabularySize();
        var weights = QwenExecutionFixtures.statefulCompactWeights(vocabularySize);
        var plan = new QwenExecutionPlan(weights);
        var gpu = new SamplingGpu(vocabularySize);
        gpu.selectedTokenId = 1;
        var lattice = createLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var session = new QwenGenerationSession(tokenizer, plan, runtime, gpu, 816, GenerationConfig.greedy(47L));
        String prompt = "! ".repeat(800);
        int[] encoded = tokenizer.encodeWithModelSpecialTokens(prompt);
        assertTrue(encoded.length > 512);
        lattice.start();
        awaitWorker(lattice);
        try {
            assertEquals(List.of(1), session.generate(prompt, 1, ignored -> {}));
            assertTrue(gpu.embeddingInputs.size() > 2, "a long prompt needs more than one prefill quantum");
            assertTrue(
                    gpu.embeddingInputs.size() - 1 <= (encoded.length + 511) / 512,
                    "prefill must not pay a per-quantum overhead for unnecessarily small chunks");
            List<Integer> observed = new ArrayList<>();
            long position = 0;
            for (int index = 0; index < gpu.embeddingInputs.size() - 1; index++) {
                int[] chunk = gpu.embeddingInputs.get(index);
                assertTrue(chunk.length <= 512, "prefill workspace must be bounded");
                assertEquals(position, gpu.attentionStartPositions.get(index));
                for (int token : chunk) observed.add(token);
                position += chunk.length;
            }
            assertEquals(encoded.length, position);
            for (int index = 0; index < encoded.length; index++) assertEquals(encoded[index], observed.get(index));
            assertArrayEquals(new int[] {1}, gpu.embeddingInputs.getLast());
            assertEquals(encoded.length + 1L, session.currentTokenPosition());
            assertEquals(1, gpu.sampledLogitRows.size(), "only the final prefill chunk should sample");
            assertEquals(1, gpu.allocatedLogits.size(), "non-final prefill and final commit need no logits");
            assertEquals(0, runtime.activeQuanta());
        } finally {
            session.close();
            runtime.close();
            lattice.close();
        }
        assertLogitsFreedOnceWithTheirGraphs(gpu);
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void constrainedGenerationMasksXmlBeforeSampling() throws Exception {
        int vocabularySize = testVocabularySize();
        var weights = QwenExecutionFixtures.statefulCompactWeights(vocabularySize);
        var plan = new QwenExecutionPlan(weights);
        var gpu = new SamplingGpu(vocabularySize);
        int xmlToken = tokenizer.encodeText("<")[0];
        gpu.selectedTokenId = xmlToken;
        var lattice = createLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var session = new QwenGenerationSession(tokenizer, plan, runtime, gpu, 817, GenerationConfig.greedy(48L));
        var constraint = new JsonEnvelopeConstraint(tokenizer, List.of("read_file"), true);
        lattice.start();
        awaitWorker(lattice);
        try {
            StringBuilder output = new StringBuilder();
            List<Integer> generated = session.generate("!", 1, output::append, constraint);
            assertEquals(1, generated.size());
            assertTrue(output.toString().startsWith("{"));
            assertFalse(generated.contains(xmlToken));
            assertEquals(1, gpu.sampledLogitRows.size());
            assertEquals(0, runtime.activeQuanta());
        } finally {
            session.close();
            runtime.close();
            lattice.close();
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void rejectsAConcurrentGenerateCallForTheSameSequence() throws Exception {
        int vocabularySize = testVocabularySize();
        var weights = QwenExecutionFixtures.statefulCompactWeights(vocabularySize);
        var plan = new QwenExecutionPlan(weights);
        var gpu = new SamplingGpu(vocabularySize);
        var lattice = createLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var session = new QwenGenerationSession(tokenizer, plan, runtime, gpu, 811, GenerationConfig.greedy(42L));
        var embeddingEntered = new CountDownLatch(1);
        var releaseEmbedding = new CountDownLatch(1);
        gpu.afterEmbedding = () -> {
            embeddingEntered.countDown();
            try {
                if (!releaseEmbedding.await(20, TimeUnit.SECONDS)) {
                    throw new IllegalStateException("test did not release the embedding gate");
                }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("embedding gate was interrupted", interrupted);
            }
        };
        lattice.start();
        awaitWorker(lattice);
        FutureTask<List<Integer>> firstGeneration = new FutureTask<>(() -> session.generate("!", 1, ignored -> {}));
        Thread generationThread = new Thread(firstGeneration, "qwen-generation-session-test");
        generationThread.start();
        try {
            assertTrue(embeddingEntered.await(20, TimeUnit.SECONDS), "generation did not reach the execution path");
            assertThrows(IllegalStateException.class, () -> session.generate("!", 1, ignored -> {}));
        } finally {
            releaseEmbedding.countDown();
        }

        assertEquals(List.of(1), firstGeneration.get(30, TimeUnit.SECONDS));
        assertEquals(0, runtime.activeQuanta());
        session.close();
        runtime.close();
        lattice.close();
    }

    @Test
    void rejectsNegativeTokenLimitsBeforeSubmittingPromptWork() throws Exception {
        int vocabularySize = testVocabularySize();
        var weights = QwenExecutionFixtures.statefulCompactWeights(vocabularySize);
        var plan = new QwenExecutionPlan(weights);
        var gpu = new SamplingGpu(vocabularySize);
        var runtime = new EuhedralInferenceRuntime(ignored -> {}, plan, gpu);
        var session = new QwenGenerationSession(tokenizer, plan, runtime, gpu, 812, GenerationConfig.greedy(43L));

        assertThrows(IllegalArgumentException.class, () -> session.generate("!", -1, ignored -> {}));
        assertTrue(gpu.embeddingInputs.isEmpty());
        session.close();
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void generationEosIsRecordedButNeverSubmittedToAnotherDecodeQuantum() throws Exception {
        int eosToken = tokenizer.generationEosTokenIds().iterator().next();
        int vocabularySize = tokenizer.generationEosTokenIds().stream()
                        .mapToInt(Integer::intValue)
                        .max()
                        .orElseThrow()
                + 1;
        var weights = QwenExecutionFixtures.statefulCompactWeights(vocabularySize);
        var plan = new QwenExecutionPlan(weights);
        var gpu = new SamplingGpu(vocabularySize);
        gpu.selectedTokenId = eosToken;
        var lattice = createLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var session = new QwenGenerationSession(tokenizer, plan, runtime, gpu, 813, GenerationConfig.greedy(44L));
        lattice.start();
        awaitWorker(lattice);
        try {
            StringBuilder output = new StringBuilder();
            assertEquals(List.of(eosToken), session.generate("!", 4, output::append));
            assertEquals("", output.toString());
            assertEquals(1, gpu.embeddingInputs.size());
            assertArrayEquals(tokenizer.encodeWithModelSpecialTokens("!"), gpu.embeddingInputs.getFirst());
            assertEquals(List.of(0L), gpu.attentionStartPositions);
            assertEquals(1, gpu.sampledLogitRows.size());
            assertEquals(1, gpu.allocatedLogits.size());
            assertTrue(gpu.closedLogits.isEmpty(), "the prefill graph keeps its logits buffer");
            assertEquals(1, session.currentTokenPosition());
            assertEquals(0, runtime.activeQuanta());

            gpu.selectedTokenId = 1;
            StringBuilder continuationOutput = new StringBuilder();
            assertEquals(List.of(1), session.generate("!", 1, continuationOutput::append));
            assertEquals(tokenizer.decode(new int[] {1}), continuationOutput.toString());
            assertArrayEquals(tokenizer.encodeText("!"), gpu.embeddingInputs.get(1));
            assertArrayEquals(new int[] {1}, gpu.embeddingInputs.get(2));
            assertEquals(List.of(0L, 1L, 2L), gpu.attentionStartPositions);
            assertEquals(List.of(eosToken, 1), session.generatedTokenIds());
            assertEquals(2, gpu.sampledLogitRows.size());
            assertEquals(1, gpu.allocatedLogits.size(), "the continuation reused its graph's logits buffer");
            assertEquals(0, runtime.activeQuanta());
        } finally {
            session.close();
            runtime.close();
            lattice.close();
        }
        assertLogitsFreedOnceWithTheirGraphs(gpu);
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void samplingThatRejectsTheHostRowFailsWithoutReleasingGraphLogits() throws Exception {
        int vocabularySize = testVocabularySize();
        var weights = QwenExecutionFixtures.statefulCompactWeights(vocabularySize);
        var plan = new QwenExecutionPlan(weights);
        var gpu = new SamplingGpu(vocabularySize);
        gpu.invalidLogits = true;
        var lattice = createLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var session = new QwenGenerationSession(
                tokenizer, plan, runtime, gpu, 814, new GenerationConfig(1.0f, 0, 1.0f, 45L, false));
        lattice.start();
        awaitWorker(lattice);
        try {
            assertThrows(IllegalArgumentException.class, () -> session.generate("!", 2, ignored -> {}));
            assertEquals(1, gpu.sampledLogitRows.size());
            assertTrue(gpu.closedLogits.isEmpty());
            assertEquals(1, gpu.embeddingInputs.size());
            assertEquals(0, runtime.activeQuanta());
            assertTrue(session.isCancelled());
        } finally {
            session.close();
            runtime.close();
            lattice.close();
        }
        assertLogitsFreedOnceWithTheirGraphs(gpu);
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void cancellationStopsAfterTheInFlightQuantumAndPreventsDecodeAdmission() throws Exception {
        int vocabularySize = testVocabularySize();
        var weights = QwenExecutionFixtures.statefulCompactWeights(vocabularySize);
        var plan = new QwenExecutionPlan(weights);
        var gpu = new SamplingGpu(vocabularySize);
        var lattice = createLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var session = new QwenGenerationSession(tokenizer, plan, runtime, gpu, 815, GenerationConfig.greedy(46L));
        var embeddingEntered = new CountDownLatch(1);
        var releaseEmbedding = new CountDownLatch(1);
        gpu.afterEmbedding = () -> {
            embeddingEntered.countDown();
            try {
                if (!releaseEmbedding.await(20, TimeUnit.SECONDS)) {
                    throw new IllegalStateException("test did not release the embedding gate");
                }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("embedding gate was interrupted", interrupted);
            }
        };
        lattice.start();
        awaitWorker(lattice);
        FutureTask<List<Integer>> generation = new FutureTask<>(() -> session.generate("!", 5, ignored -> {}));
        new Thread(generation, "qwen-generation-cancellation-test").start();
        try {
            assertTrue(embeddingEntered.await(20, TimeUnit.SECONDS), "generation did not reach the execution path");
            session.cancel();
        } finally {
            releaseEmbedding.countDown();
        }

        assertEquals(List.of(), generation.get(30, TimeUnit.SECONDS));
        assertTrue(session.isCancelled());
        assertEquals(
                QwenSequenceState.TerminalState.CANCELLED,
                session.sequenceState().terminalState());
        assertEquals(1, gpu.embeddingInputs.size());
        assertTrue(gpu.sampledLogitRows.isEmpty());
        assertEquals(0, runtime.activeQuanta());
        session.close();
        runtime.close();
        lattice.close();
    }

    @Test
    @Timeout(90)
    void doesNotHideExecutionFailureWhenCancellationIsRequestedInFlight() throws Exception {
        int vocabularySize = testVocabularySize();
        var weights = QwenExecutionFixtures.statefulCompactWeights(vocabularySize);
        var plan = new QwenExecutionPlan(weights);
        var gpu = new SamplingGpu(vocabularySize);
        var lattice = createLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var session = new QwenGenerationSession(tokenizer, plan, runtime, gpu, 816, GenerationConfig.greedy(47L));
        var embeddingEntered = new CountDownLatch(1);
        var releaseEmbedding = new CountDownLatch(1);
        gpu.afterEmbedding = () -> {
            embeddingEntered.countDown();
            try {
                if (!releaseEmbedding.await(20, TimeUnit.SECONDS)) {
                    throw new IllegalStateException("test did not release the embedding gate");
                }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("embedding gate was interrupted", interrupted);
            }
        };
        lattice.start();
        awaitWorker(lattice);
        FutureTask<List<Integer>> generation = new FutureTask<>(() -> session.generate("!", 5, ignored -> {}));
        new Thread(generation, "qwen-generation-failure-cancellation-test").start();
        try {
            assertTrue(embeddingEntered.await(20, TimeUnit.SECONDS), "generation did not reach the execution path");
            gpu.embeddingFailure = new IllegalStateException("forced GPU failure");
            session.cancel();
        } finally {
            releaseEmbedding.countDown();
        }

        var executionFailure = assertThrows(ExecutionException.class, () -> generation.get(30, TimeUnit.SECONDS));
        assertTrue(executionFailure.getCause() instanceof IllegalStateException);
        assertEquals(
                "forced GPU failure", executionFailure.getCause().getCause().getMessage());
        assertTrue(session.isCancelled());
        assertEquals(
                QwenSequenceState.TerminalState.FAILED, session.sequenceState().terminalState());
        assertEquals(0, runtime.activeQuanta());
        session.close();
        runtime.close();
        lattice.close();
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void outputCallbackCancellationAndClosePreventAnotherDecode() throws Exception {
        int vocabularySize = testVocabularySize();
        var weights = QwenExecutionFixtures.statefulCompactWeights(vocabularySize);
        var plan = new QwenExecutionPlan(weights);
        var gpu = new SamplingGpu(vocabularySize);
        var lattice = createLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var session = new QwenGenerationSession(tokenizer, plan, runtime, gpu, 817, GenerationConfig.greedy(48L));
        lattice.start();
        awaitWorker(lattice);
        try {
            StringBuilder output = new StringBuilder();
            assertEquals(List.of(1), session.generate("!", 5, text -> {
                output.append(text);
                session.cancel();
                session.close();
            }));
            assertEquals(tokenizer.decode(new int[] {1}), output.toString());
            assertEquals(1, session.currentTokenPosition());
            assertEquals(1, gpu.embeddingInputs.size());
            assertTrue(session.isCancelled());
            assertTrue(session.isClosed());
            assertEquals(
                    QwenSequenceState.TerminalState.CANCELLED,
                    session.sequenceState().terminalState());
            assertEquals(0, runtime.activeQuanta());
        } finally {
            session.close();
            runtime.close();
            lattice.close();
        }
        assertLogitsFreedOnceWithTheirGraphs(gpu);
    }

    /// Device logits live in their graphs' workspace storage; closing the runtime frees each once.
    private static void assertLogitsFreedOnceWithTheirGraphs(SamplingGpu gpu) {
        assertTrue(gpu.liveLogits.isEmpty(), "a graph's logits buffer outlived the runtime");
        assertEquals(gpu.allocatedLogits.size(), gpu.closedLogits.size());
        assertEquals(new java.util.HashSet<>(gpu.allocatedLogits), new java.util.HashSet<>(gpu.closedLogits));
    }

    private static ControlPlaneLattice createLattice() {
        BitSet available = SystemInfo.getPCpuSet();
        int cpu = available.nextSetBit(0);
        assumeTrue(cpu >= 0, "requires one available physical CPU for the lattice worker");
        BitSet cpus = new BitSet();
        cpus.set(cpu);
        var workers = new BaseCloneableObject(new DefaultExecutor());
        var shard = ControlPlaneShard.createBaseShard("QwenGenerationTestShard", workers);
        return ControlPlaneLattice.getOrCreate(new LatticeConfig(
                "QwenGenerationTestLattice-" + LATTICE_ID.incrementAndGet(), cpus, Duration.ofSeconds(10), shard));
    }

    private static void awaitWorker(ControlPlaneLattice lattice) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (lattice.getActiveWorkers() < 1 && System.nanoTime() < deadline) Thread.onSpinWait();
        assertEquals(1, lattice.getActiveWorkers(), "lattice worker did not register");
    }

    private static short bf16(float value) {
        return (short) (Float.floatToIntBits(value) >>> 16);
    }

    private static int testVocabularySize() {
        return tokenizer.generationEosTokenIds().stream()
                        .mapToInt(Integer::intValue)
                        .max()
                        .orElseThrow()
                + 1;
    }

    private static final class SamplingGpu extends QwenExecutionFixtures.RecordingGpu {
        private final int vocabularySize;
        private final Map<Long, int[]> uploadedTokenIds = new java.util.concurrent.ConcurrentHashMap<>();
        private final List<int[]> embeddingInputs = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> allocatedLogits = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> closedLogits = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> sampledLogitRows = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> sampledLogitCloses = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final java.util.Set<Long> pendingLogits = java.util.concurrent.ConcurrentHashMap.newKeySet();
        private final java.util.Set<Long> liveLogits = java.util.concurrent.ConcurrentHashMap.newKeySet();
        private final List<Long> attentionStartPositions = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> convolutionStateAddresses = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> recurrentStateAddresses = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> keyCacheAddresses = new java.util.concurrent.CopyOnWriteArrayList<>();
        private volatile int lastTokenCount;
        private volatile int selectedTokenId = 1;
        private volatile int[] selectedTokenIds = new int[0];
        private int sampleIndex;
        private volatile boolean invalidLogits;
        private volatile RuntimeException embeddingFailure;
        private volatile Runnable onRowCopy = () -> {};

        private SamplingGpu(int vocabularySize) {
            this.vocabularySize = vocabularySize;
        }

        @Override
        public long allocate(long byteSize) {
            long address = super.allocate(byteSize);
            if (byteSize % ((long) this.vocabularySize * Short.BYTES) == 0) {
                this.allocatedLogits.add(address);
                this.liveLogits.add(address);
            }
            return address;
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
            int[] tokenIds = new int[Math.toIntExact(byteSize / Integer.BYTES)];
            for (int index = 0; index < tokenIds.length; index++) {
                tokenIds[index] = source.get(ValueLayout.JAVA_INT, (long) index * Integer.BYTES);
            }
            this.uploadedTokenIds.put(destination, tokenIds);
        }

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            if (byteSize != (long) this.vocabularySize * Short.BYTES) return;
            this.onRowCopy.run();
            long logitsBase = source;
            this.sampledLogitRows.add(logitsBase);
            this.pendingLogits.add(logitsBase);
            int selected = this.selectedTokenIds.length == 0
                    ? this.selectedTokenId
                    : this.selectedTokenIds[Math.min(this.sampleIndex++, this.selectedTokenIds.length - 1)];
            for (int tokenId = 0; tokenId < this.vocabularySize; tokenId++) {
                destination.set(
                        ValueLayout.JAVA_SHORT,
                        (long) tokenId * Short.BYTES,
                        this.invalidLogits
                                ? bf16(Float.NEGATIVE_INFINITY)
                                : tokenId == selected ? bf16(10.0f) : bf16(0.0f));
            }
        }

        @Override
        public synchronized void free(long address) {
            if (this.pendingLogits.remove(address)) this.sampledLogitCloses.add(address);
            if (this.allocatedLogits.contains(address)) {
                assertTrue(this.liveLogits.remove(address), "logits allocation freed more than once");
                this.closedLogits.add(address);
            }
            super.free(address);
        }

        @Override
        public void copyDeviceToDevice(long destination, long source, long byteSize) {}

        @Override
        public void zeroDeviceMemory(long address, long byteSize) {}

        @Override
        public void embedQ3(
                long tokenIdsAddress,
                long embeddingAddress,
                long embeddingByteSize,
                long hiddenStateAddress,
                int tokenCount,
                int vocabularySize,
                int hiddenSize) {
            int[] record = this.uploadedTokenIds.get(tokenIdsAddress);
            if (record == null) throw new IllegalStateException("embedding did not receive uploaded token IDs");
            // The input record carries the token IDs, then the start position.
            int[] tokenIds = java.util.Arrays.copyOf(record, tokenCount);
            this.embeddingInputs.add(tokenIds.clone());
            this.lastTokenCount = tokenCount;
            super.embedQ3(
                    tokenIdsAddress,
                    embeddingAddress,
                    embeddingByteSize,
                    hiddenStateAddress,
                    tokenCount,
                    vocabularySize,
                    hiddenSize);
            if (this.embeddingFailure != null) throw this.embeddingFailure;
        }

        @Override
        public void rmsNormUnitOffsetBf16(
                long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {}

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
            this.convolutionStateAddresses.add(convolutionStateAddress);
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
            this.recurrentStateAddresses.add(recurrentStateAddress);
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
                double ropeTheta) {
            this.attentionStartPositions.add(startPosition);
        }

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
            this.keyCacheAddresses.add(keyCacheAddress);
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
    }
}
