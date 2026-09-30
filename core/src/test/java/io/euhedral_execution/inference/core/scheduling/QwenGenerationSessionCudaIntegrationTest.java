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
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.hardware_utils.SystemInfo;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class QwenGenerationSessionCudaIntegrationTest {

    private static final Path DEFAULT_ARTIFACT =
            Path.of("/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl");
    private static final Path DEFAULT_TOKENIZER = Path.of("/mnt/shared/qwen38-quant/source/qwen");
    private static final long CALLBACK_FREE_MEMORY_TOLERANCE = 1L * 1024L * 1024L;
    private static final long SESSION_RESTORE_TOLERANCE = 16L * 1024L * 1024L;
    // First use JIT-loads process-lifetime CUDA kernel modules outside the model's ownership.
    private static final long MODEL_RESTORE_TOLERANCE = 128L * 1024L * 1024L;
    private static final AtomicLong LATTICE_ID = new AtomicLong();

    @Test
    @Timeout(value = 1800, unit = TimeUnit.SECONDS)
    void plainPromptGeneratesThroughRealLatticeLogitsAndPersistentSequenceState() throws Exception {
        Path artifactPath = Path.of(System.getProperty("euhedral.qwen.artifact", DEFAULT_ARTIFACT.toString()));
        Path tokenizerPath = Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", DEFAULT_TOKENIZER.toString()));
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && !library.isBlank(), "CUDA library property is missing");
        Path libraryPath = Path.of(library);
        assumeTrue(Files.isRegularFile(libraryPath), "CUDA library is missing: " + libraryPath);
        assumeTrue(Files.isRegularFile(artifactPath), "compact Qwen artifact is missing: " + artifactPath);
        assumeTrue(Files.isRegularFile(tokenizerPath.resolve("tokenizer.json")));
        BitSet cpus = twoWorkerCpus();
        assumeTrue(cpus.cardinality() >= 2, "requires two available physical CPUs for the real lattice");

        QwenTokenizer tokenizer = QwenTokenizer.load(tokenizerPath);
        QwenArtifact artifact = QwenArtifactReader.read(artifactPath);
        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath)) {
            ControlPlaneLattice lattice = createLattice(cpus);
            try {
                long freeBeforeWeights = gpu.deviceMemoryInfo().freeBytes();
                QwenModel model = QwenModel.load(artifactPath, artifact, gpu);
                QwenWeights weights = model.weights();
                Throwable failure = null;
                long freeAfterWeights = gpu.deviceMemoryInfo().freeBytes();
                try {
                    QwenExecutionPlan plan = new QwenExecutionPlan(weights);
                    EuhedralInferenceRuntime runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
                    QwenGenerationSession session =
                            new QwenGenerationSession(tokenizer, plan, runtime, gpu, 901, GenerationConfig.greedy(91L));
                    try {
                        LatticeEdge registrationProbe = new LatticeEdge(new AtomicBoolean());
                        int registrationsBeforeStart = registrationProbe.getThreadCount();
                        lattice.start();
                        awaitWorkers(lattice, cpus.cardinality(), registrationProbe, registrationsBeforeStart);
                        String prompt = "The capital of France is";
                        int promptTokenCount = tokenizer.encodeWithModelSpecialTokens(prompt).length;
                        StringBuilder output = new StringBuilder();
                        List<Long> callbackPositions = new ArrayList<>();
                        List<Long> decodeCallbackFreeMemory = new ArrayList<>();
                        AtomicReference<Object> recurrentState = new AtomicReference<>();
                        AtomicReference<Object> kvState = new AtomicReference<>();

                        List<Integer> generated = session.generate(prompt, 5, text -> {
                            output.append(text);
                            callbackPositions.add(session.currentTokenPosition());
                            assertFalse(runtime.hasAttachedRunner(), "runtime retained its source after a token");
                            Object currentRecurrent = session.sequenceState().recurrentState();
                            Object currentKv = session.sequenceState().kvCacheState();
                            if (recurrentState.compareAndSet(null, currentRecurrent)) {
                                assertTrue(currentRecurrent instanceof GdnSequenceStates);
                            } else {
                                assertSame(recurrentState.get(), currentRecurrent);
                            }
                            if (kvState.compareAndSet(null, currentKv)) {
                                assertTrue(currentKv instanceof AttentionSequenceStates);
                            } else {
                                assertSame(kvState.get(), currentKv);
                            }
                            long generatedCount = session.generatedTokenIds().size();
                            assertTrue(session.currentTokenPosition() >= promptTokenCount + generatedCount - 1L);
                            assertTrue(session.currentTokenPosition() <= promptTokenCount + generatedCount);
                            // The first text callback follows prefill, before lazy split-KV
                            // scratch reservation. Compare steady-state decode callbacks only;
                            // scratch address stability and ownership have separate exact tests.
                            if (session.currentTokenPosition() > promptTokenCount) {
                                decodeCallbackFreeMemory.add(
                                        gpu.deviceMemoryInfo().freeBytes());
                            }
                        });

                        assertTrue(generated.size() >= 4, "model stopped before several decode quanta");
                        List<Integer> visibleTokens = generated.stream()
                                .filter(tokenId -> !tokenizer.isGenerationEosToken(tokenId))
                                .toList();
                        int decodeQuanta = visibleTokens.size();
                        assertTrue(decodeQuanta >= 3, "generation did not execute several decode quanta");
                        assertEquals(
                                tokenizer.decode(visibleTokens.stream()
                                        .mapToInt(Integer::intValue)
                                        .toArray()),
                                output.toString());
                        assertEquals(promptTokenCount + visibleTokens.size(), session.currentTokenPosition());
                        assertEquals(
                                session.currentTokenPosition(),
                                ((AttentionSequenceStates) kvState.get())
                                        .forLayer(3)
                                        .length());
                        assertTrue(callbackPositions.size() >= 3, "incremental decoder did not emit token text");
                        assertTrue(decodeCallbackFreeMemory.size() >= 3);
                        long minimumCallbackFree = decodeCallbackFreeMemory.stream()
                                .mapToLong(Long::longValue)
                                .min()
                                .orElseThrow();
                        long maximumCallbackFree = decodeCallbackFreeMemory.stream()
                                .mapToLong(Long::longValue)
                                .max()
                                .orElseThrow();
                        assertTrue(
                                maximumCallbackFree - minimumCallbackFree <= CALLBACK_FREE_MEMORY_TOLERANCE,
                                "retained logits caused free device memory to decline between tokens: min="
                                        + minimumCallbackFree + ", max=" + maximumCallbackFree);
                        assertFalse(runtime.hasAttachedRunner());
                        assertTrue(lattice.isDrained());

                        GdnSequenceStates recurrent = (GdnSequenceStates) recurrentState.get();
                        AttentionSequenceStates attention = (AttentionSequenceStates) kvState.get();
                        session.close();
                        assertEquals(
                                QwenSequenceState.TerminalState.COMPLETED,
                                session.sequenceState().terminalState());
                        assertThrows(IllegalStateException.class, () -> recurrent.forLayer(0));
                        assertThrows(IllegalStateException.class, () -> attention.forLayer(3));
                        long freeAfterSession = gpu.deviceMemoryInfo().freeBytes();
                        assertTrue(
                                freeAfterSession >= freeAfterWeights - SESSION_RESTORE_TOLERANCE,
                                "session close did not release its persistent KV/GDN state and sampled logits");
                    } finally {
                        session.close();
                        if (runtime.hasAttachedRunner()) runtime.disconnectRunner();
                    }
                } catch (Throwable executionFailure) {
                    failure = executionFailure;
                } finally {
                    try {
                        model.close();
                    } catch (Throwable cleanupFailure) {
                        if (failure == null) failure = cleanupFailure;
                        else failure.addSuppressed(cleanupFailure);
                    }
                }
                long freeAfterWeightsRelease = gpu.deviceMemoryInfo().freeBytes();
                if (freeAfterWeightsRelease < freeBeforeWeights - MODEL_RESTORE_TOLERANCE) {
                    IllegalStateException cleanupFailure =
                            new IllegalStateException("generation integration leaked device memory: before="
                                    + freeBeforeWeights + ", after=" + freeAfterWeightsRelease);
                    if (failure == null) failure = cleanupFailure;
                    else failure.addSuppressed(cleanupFailure);
                }
                if (failure instanceof Exception exception) throw exception;
                if (failure instanceof Error error) throw error;
                if (failure != null) throw new IllegalStateException(failure);
            } finally {
                lattice.close();
            }
        }
    }

    private static ControlPlaneLattice createLattice(BitSet cpus) {
        var workers = new BaseCloneableObject(new DefaultExecutor());
        var shard = ControlPlaneShard.createBaseShard("QwenGenerationCudaTestShard", workers);
        return ControlPlaneLattice.getOrCreate(new LatticeConfig(
                "QwenGenerationCudaTestLattice-" + LATTICE_ID.incrementAndGet(), cpus, Duration.ofSeconds(10), shard));
    }

    private static BitSet twoWorkerCpus() {
        BitSet cpus = new BitSet();
        BitSet selectedCores = new BitSet();
        BitSet physicalCpus = SystemInfo.getPCpuSet();
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
            ControlPlaneLattice lattice, int expected, LatticeEdge registrationProbe, int registrationsBeforeStart)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while ((lattice.getActiveWorkers() < expected
                        || registrationProbe.getThreadCount() < registrationsBeforeStart + expected)
                && System.nanoTime() < deadline) Thread.onSpinWait();
        assertEquals(expected, lattice.getActiveWorkers(), "real ControlPlaneLattice workers did not start");
        assertEquals(
                registrationsBeforeStart + expected,
                registrationProbe.getThreadCount(),
                "lattice workers did not register before inference");
    }
}
