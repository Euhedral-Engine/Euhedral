package io.euhedral_execution.inference.core.gpu;

import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.assertBf16Equals;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.bf16ToFloat;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.download;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.AttentionSequenceStates;
import io.euhedral_execution.inference.core.scheduling.GdnSequenceStates;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionRunner;
import io.euhedral_execution.inference.core.scheduling.QwenLogitsRequirement;
import io.euhedral_execution.inference.core.scheduling.QwenLogitsSampler;
import io.euhedral_execution.inference.core.scheduling.QwenSequenceState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class QwenFullModelCudaIntegrationTest {

    private static void reportError(String name, short[] expected, short[] actual) {
        double max = 0, total = 0, squared = 0;
        int different = 0, overOne = 0, maxIndex = 0;
        for (int i = 0; i < expected.length; i++) {
            double error = Math.abs(bf16ToFloat(expected[i]) - bf16ToFloat(actual[i]));
            if (error > max) {
                max = error;
                maxIndex = i;
            }
            total += error;
            squared += error * error;
            if (expected[i] != actual[i]) different++;
            if (error > 1) overOne++;
        }
        System.out.printf(
                java.util.Locale.ROOT,
                "%s count=%d different=%d overOne=%d max=%g mean=%g rms=%g%n",
                name,
                expected.length,
                different,
                overOne,
                max,
                total / expected.length,
                Math.sqrt(squared / expected.length));
        System.out.printf(
                java.util.Locale.ROOT,
                "%s maxIndex=%d expected=%g actual=%g%n",
                name,
                maxIndex,
                bf16ToFloat(expected[maxIndex]),
                bf16ToFloat(actual[maxIndex]));
    }

    private static final int INITIAL_TOKEN = 1814;
    private static final long VRAM_RESTORE_TOLERANCE = 128L * 1024L * 1024L;
    private static final float HIDDEN_TOLERANCE = 1.0f;
    private static final float LOGIT_TOLERANCE = 1.0f;
    private static final List<QwenExecutionPlan.Buffer> FIRST_LAYER_BOUNDARIES = List.of(
            QwenExecutionPlan.Buffer.HIDDEN_STATE,
            QwenExecutionPlan.Buffer.INPUT_NORMALIZED,
            QwenExecutionPlan.Buffer.QK_PROJECTED,
            QwenExecutionPlan.Buffer.VALUE_Z_PROJECTED,
            QwenExecutionPlan.Buffer.GDN_CONVOLVED,
            QwenExecutionPlan.Buffer.GDN_RECURRENT,
            QwenExecutionPlan.Buffer.GDN_NORMALIZED,
            QwenExecutionPlan.Buffer.MIXER_DELTA,
            QwenExecutionPlan.Buffer.MIXER_HIDDEN,
            QwenExecutionPlan.Buffer.POST_MIXER_NORMALIZED,
            QwenExecutionPlan.Buffer.GATE_UP,
            QwenExecutionPlan.Buffer.SWIGLU,
            QwenExecutionPlan.Buffer.FFN_DELTA,
            QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE);

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(Q3DispatchMode.class)
    @Timeout(value = 1200, unit = TimeUnit.SECONDS)
    void realCompactQwenRunsAllLayersAndMatchesCpuReferenceAcrossBoundaries(Q3DispatchMode mode) throws Exception {
        Path artifactPath = Path.of(System.getProperty("euhedral.qwen.artifact"));
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        assertTrue(Files.isRegularFile(artifactPath), "compact Qwen artifact is missing: " + artifactPath);
        QwenArtifact artifact = QwenArtifactReader.read(artifactPath);
        assertEquals(64, artifact.config().numHiddenLayers());
        assertTrue(Arrays.asList(artifact.config().layerTypes()).contains(QwenLayerType.FULL_ATTENTION));
        assertTrue(Arrays.asList(artifact.config().layerTypes()).contains(QwenLayerType.GATED_DELTA_NET));

        try (CudaGpuMemory gpu =
                new CudaGpuMemory(libraryPath, false, mode, Q3DispatchMode.DEFAULT_SMALL_ROW_THRESHOLD)) {
            long freeBefore = gpu.deviceMemoryInfo().freeBytes();
            QwenModel model = QwenModel.load(artifactPath, artifact, gpu);
            QwenWeights weights = model.weights();
            Throwable failure = null;
            QwenSequenceState prefixSequence = new QwenSequenceState(501);
            QwenSequenceState mixedSequence = new QwenSequenceState(502);
            QwenSequenceState stateSequence = new QwenSequenceState(503);
            QwenSequenceState referenceSequence = new QwenSequenceState(504);
            QwenSequenceState isolationSequence = new QwenSequenceState(505);
            List<RunResult> runs = new ArrayList<>();
            try {
                QwenExecutionPlan plan = new QwenExecutionPlan(weights);
                QwenExecutionPlan firstLayerPlan = QwenExecutionPlan.prefix(weights, 1);
                QwenExecutionPlan mixedPlan = QwenExecutionPlan.prefix(weights, 4);
                int fullAttentionLayers = (int) Arrays.stream(weights.config().layerTypes())
                        .filter(type -> type == QwenLayerType.FULL_ATTENTION)
                        .count();
                assertEquals(
                        64,
                        plan.instructions().stream()
                                        .mapToInt(QwenExecutionPlan.Instruction::layerIndex)
                                        .filter(index -> index >= 0)
                                        .max()
                                        .orElseThrow()
                                + 1);
                assertEquals(
                        fullAttentionLayers,
                        plan.instructions().stream()
                                .filter(instruction -> instruction.kind() == QwenExecutionPlan.Kind.ATTENTION_CAUSAL)
                                .count());
                for (int layerIndex = 0; layerIndex < weights.layers().length; layerIndex++) {
                    int selectedLayer = layerIndex;
                    long expectedWeight =
                            weights.layers()[layerIndex].inputNorm().deviceAddress();
                    assertTrue(plan.instructions().stream()
                            .anyMatch(instruction -> instruction.layerIndex() == selectedLayer
                                    && instruction.weightAddress() == expectedWeight));
                }

                QwenFullModelCpuReference.Result reference = QwenFullModelCpuReference.run(weights, gpu, INITIAL_TOKEN);
                QwenFirstLayerCpuReference.Result firstLayerReference =
                        QwenFirstLayerCpuReference.run(weights, gpu, INITIAL_TOKEN);

                RunResult firstLayer = execute(
                        gpu,
                        firstLayerPlan,
                        prefixSequence,
                        QwenExecutionContext.ExecutionKind.DECODE,
                        0,
                        new int[] {INITIAL_TOKEN},
                        FIRST_LAYER_BOUNDARIES);
                runs.add(firstLayer);
                assertSuccessful(firstLayer);
                assertBf16Equals(
                        firstLayerReference.buffers().get(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE),
                        firstLayer.buffers().get(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE),
                        0.08f);
                for (QwenExecutionPlan.Buffer boundary : FIRST_LAYER_BOUNDARIES) {
                    assertBf16Equals(
                            firstLayerReference.buffers().get(boundary),
                            firstLayer.buffers().get(boundary),
                            0.08f);
                }

                RunResult mixed = execute(
                        gpu,
                        mixedPlan,
                        mixedSequence,
                        QwenExecutionContext.ExecutionKind.DECODE,
                        0,
                        new int[] {INITIAL_TOKEN},
                        List.of(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE));
                runs.add(mixed);
                assertSuccessful(mixed);
                assertBf16Equals(
                        reference.layerOutputs().get(3),
                        mixed.buffers().get(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE),
                        HIDDEN_TOLERANCE);

                RunResult prefill = execute(
                        gpu,
                        plan,
                        stateSequence,
                        QwenExecutionContext.ExecutionKind.PREFILL,
                        0,
                        new int[] {INITIAL_TOKEN, 26},
                        List.of(QwenExecutionPlan.Buffer.FINAL_NORMALIZED));
                runs.add(prefill);
                assertSuccessful(prefill);
                assertEquals(2, stateSequence.currentTokenPosition());
                GdnSequenceStates recurrent = (GdnSequenceStates) stateSequence.recurrentState();
                AttentionSequenceStates attention = (AttentionSequenceStates) stateSequence.kvCacheState();
                long firstRecurrentAddress = recurrent.forLayer(0).recurrentStateAddress();
                long firstConvolutionAddress = recurrent.forLayer(0).convolutionStateAddress();
                long recurrentBytes = (long) weights.config().linearNumValueHeads()
                        * weights.config().linearKeyHeadDim()
                        * weights.config().linearValueHeadDim()
                        * Float.BYTES;
                byte[] recurrentAfterPrefill = readDeviceBytes(gpu, firstRecurrentAddress, recurrentBytes);
                int attentionLayers = 0;
                for (int layerIndex = 0; layerIndex < weights.config().layerTypes().length; layerIndex++) {
                    if (weights.config().layerTypes()[layerIndex] != QwenLayerType.FULL_ATTENTION) continue;
                    assertEquals(2, attention.forLayer(layerIndex).length());
                    attentionLayers++;
                }
                assertEquals(fullAttentionLayers, attentionLayers);

                RunResult decode = execute(
                        gpu,
                        plan,
                        stateSequence,
                        QwenExecutionContext.ExecutionKind.DECODE,
                        2,
                        new int[] {13},
                        List.of(QwenExecutionPlan.Buffer.FINAL_NORMALIZED));
                runs.add(decode);
                assertSuccessful(decode);
                assertEquals(3, stateSequence.currentTokenPosition());
                assertTrue(
                        !Arrays.equals(
                                recurrentAfterPrefill, readDeviceBytes(gpu, firstRecurrentAddress, recurrentBytes)),
                        "GDN recurrent state did not advance in the decode quantum");
                assertTrue(firstConvolutionAddress != 0);
                for (int layerIndex = 0; layerIndex < weights.config().layerTypes().length; layerIndex++) {
                    if (weights.config().layerTypes()[layerIndex] == QwenLayerType.FULL_ATTENTION) {
                        assertEquals(3, attention.forLayer(layerIndex).length());
                    }
                }

                RunResult cleanSequence = execute(
                        gpu,
                        plan,
                        referenceSequence,
                        QwenExecutionContext.ExecutionKind.DECODE,
                        0,
                        new int[] {INITIAL_TOKEN},
                        List.of(
                                QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE,
                                QwenExecutionPlan.Buffer.FINAL_NORMALIZED));
                runs.add(cleanSequence);
                assertSuccessful(cleanSequence);
                assertEquals(1, referenceSequence.currentTokenPosition());
                assertNotSame(stateSequence.recurrentState(), referenceSequence.recurrentState());
                assertNotSame(stateSequence.kvCacheState(), referenceSequence.kvCacheState());
                assertEquals(
                        1,
                        ((AttentionSequenceStates) referenceSequence.kvCacheState())
                                .forLayer(3)
                                .length());
                reportError(
                        mode + " hidden",
                        reference.layerOutputs().get(63),
                        cleanSequence.buffers().get(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE));
                reportError(
                        mode + " normalized",
                        reference.finalNormalized(),
                        cleanSequence.buffers().get(QwenExecutionPlan.Buffer.FINAL_NORMALIZED));
                reportError(mode + " logits", reference.logits(), cleanSequence.logits());
                short[] expectedHidden = reference.layerOutputs().get(63);
                short[] actualHidden = cleanSequence.buffers().get(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE);
                // Forced WMMA across 64 layers measured max 1.5, RMS 0.07655;
                // scalar measured max 1.0, RMS 0.08269. Retain the original
                // decode bound and independently constrain tiled aggregate error.
                assertBf16Equals(
                        expectedHidden, actualHidden, mode == Q3DispatchMode.PREFILL ? 2.0f : HIDDEN_TOLERANCE);
                double hiddenSquareError = 0;
                for (int i = 0; i < expectedHidden.length; i++) {
                    double error = bf16ToFloat(expectedHidden[i]) - bf16ToFloat(actualHidden[i]);
                    hiddenSquareError += error * error;
                }
                assertTrue(Math.sqrt(hiddenSquareError / expectedHidden.length) <= 0.1);
                assertBf16Equals(
                        reference.finalNormalized(),
                        cleanSequence.buffers().get(QwenExecutionPlan.Buffer.FINAL_NORMALIZED),
                        HIDDEN_TOLERANCE);
                assertBf16Equals(reference.logits(), cleanSequence.logits(), LOGIT_TOLERANCE);
                for (short value : cleanSequence.logits()) {
                    assertTrue(
                            Float.isFinite(bf16ToFloat(value)),
                            "final vocabulary projection produced a non-finite logit");
                }
                assertEquals(weights.config().vocabSize(), cleanSequence.logits().length);
                int expectedGreedyToken = independentArgmax(cleanSequence.logits());
                QwenLogitsSampler greedySampler = new QwenLogitsSampler(
                        GenerationConfig.greedy(0L), weights.config().vocabSize());
                assertEquals(
                        expectedGreedyToken,
                        greedySampler.selectToken(
                                cleanSequence.context().logitsOutput().orElseThrow(), gpu));

                RunResult isolatedSequence = execute(
                        gpu,
                        plan,
                        isolationSequence,
                        QwenExecutionContext.ExecutionKind.DECODE,
                        0,
                        new int[] {INITIAL_TOKEN},
                        List.of(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE));
                runs.add(isolatedSequence);
                assertSuccessful(isolatedSequence);
                assertArrayEquals(cleanSequence.logits(), isolatedSequence.logits());
                assertNotEquals(
                        ((GdnSequenceStates) stateSequence.recurrentState())
                                .forLayer(0)
                                .recurrentStateAddress(),
                        ((GdnSequenceStates) referenceSequence.recurrentState())
                                .forLayer(0)
                                .recurrentStateAddress());

                GdnSequenceStates completedRecurrent = recurrent;
                AttentionSequenceStates completedAttention = attention;
                stateSequence.complete();
                prefixSequence.complete();
                mixedSequence.complete();
                referenceSequence.complete();
                isolationSequence.complete();
                assertThrows(IllegalStateException.class, () -> completedRecurrent.forLayer(0));
                assertThrows(IllegalStateException.class, () -> completedAttention.forLayer(3));
                for (RunResult run : runs) assertTrue(run.context().workspace().isClosed());
            } catch (Throwable executionFailure) {
                failure = executionFailure;
            } finally {
                for (QwenSequenceState sequence :
                        List.of(isolationSequence, referenceSequence, stateSequence, mixedSequence, prefixSequence)) {
                    try {
                        sequence.complete();
                    } catch (Throwable cleanupFailure) {
                        if (failure == null) failure = cleanupFailure;
                        else failure.addSuppressed(cleanupFailure);
                    }
                }
                for (RunResult run : runs) {
                    try {
                        run.closeLogits();
                    } catch (Throwable cleanupFailure) {
                        if (failure == null) failure = cleanupFailure;
                        else failure.addSuppressed(cleanupFailure);
                    }
                }
                try {
                    model.close();
                } catch (Throwable cleanupFailure) {
                    if (failure == null) failure = cleanupFailure;
                    else failure.addSuppressed(cleanupFailure);
                }
            }
            long freeAfter = gpu.deviceMemoryInfo().freeBytes();
            if (freeAfter < freeBefore - VRAM_RESTORE_TOLERANCE) {
                IllegalStateException cleanupFailure = new IllegalStateException(
                        "full model test leaked device memory: before=" + freeBefore + ", after=" + freeAfter);
                if (failure == null) failure = cleanupFailure;
                else failure.addSuppressed(cleanupFailure);
            }
            if (failure != null) {
                if (failure instanceof Exception exception) throw exception;
                if (failure instanceof Error error) throw error;
                throw new IllegalStateException(failure);
            }
        }
    }

    @Test
    @Timeout(value = 600, unit = TimeUnit.SECONDS)
    void fullQwenGpuGreedySelectionMatchesIndependentArgmaxOfDownloadedLogits() throws Exception {
        Path artifactPath = Path.of(System.getProperty("euhedral.qwen.artifact"));
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        assertTrue(Files.isRegularFile(artifactPath), "compact Qwen artifact is missing: " + artifactPath);
        QwenArtifact artifact = QwenArtifactReader.read(artifactPath);

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath)) {
            QwenModel model = QwenModel.load(artifactPath, artifact, gpu);
            QwenWeights weights = model.weights();
            QwenSequenceState sequence = new QwenSequenceState(601);
            RunResult run = null;
            Throwable failure = null;
            try {
                run = execute(
                        gpu,
                        new QwenExecutionPlan(weights),
                        sequence,
                        QwenExecutionContext.ExecutionKind.DECODE,
                        0,
                        new int[] {INITIAL_TOKEN},
                        List.of());
                assertSuccessful(run);
                assertEquals(weights.config().vocabSize(), run.logits().length);
                int expectedToken = independentArgmax(run.logits());
                QwenLogitsSampler greedySampler = new QwenLogitsSampler(
                        GenerationConfig.greedy(0L), weights.config().vocabSize());

                assertEquals(
                        expectedToken,
                        greedySampler.selectToken(run.context().logitsOutput().orElseThrow(), gpu));
                int[] topTwoTokens = independentTopTwo(run.logits());
                QwenLogitsSampler stochasticSampler = new QwenLogitsSampler(
                        new GenerationConfig(1.0f, 2, 1.0f, 17L, false),
                        weights.config().vocabSize());
                int stochasticToken = stochasticSampler.selectToken(
                        run.context().logitsOutput().orElseThrow(), gpu);
                assertTrue(stochasticToken == topTwoTokens[0] || stochasticToken == topTwoTokens[1]);
            } catch (Throwable testFailure) {
                failure = testFailure;
            } finally {
                try {
                    sequence.complete();
                } catch (Throwable cleanupFailure) {
                    failure = mergeFailure(failure, cleanupFailure);
                }
                if (run != null) {
                    try {
                        run.closeLogits();
                    } catch (Throwable cleanupFailure) {
                        failure = mergeFailure(failure, cleanupFailure);
                    }
                }
                try {
                    model.close();
                } catch (Throwable cleanupFailure) {
                    failure = mergeFailure(failure, cleanupFailure);
                }
            }
            if (failure instanceof Exception exception) throw exception;
            if (failure instanceof Error error) throw error;
            if (failure != null) throw new IllegalStateException(failure);
        }
    }

    @Test
    @Timeout(value = 600, unit = TimeUnit.SECONDS)
    void productionPrefillRouteMatchesReferenceHiddenLogitsAndPersistentStateBitwise() throws Exception {
        Path artifact = Path.of(System.getProperty("euhedral.qwen.artifact"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                QwenModel model = QwenModel.load(artifact, QwenArtifactReader.read(artifact), gpu)) {
            var production = new QwenExecutionPlan(model.weights());
            var reference = QwenExecutionPlan.reference(model.weights());
            for (int rows : new int[] {64, 256, 512, 1024}) {
                var selected = production.forExecution(QwenExecutionContext.ExecutionKind.PREFILL, rows);
                boolean streamed =
                        selected.instructions().stream().anyMatch(i -> i.kind() == QwenExecutionPlan.Kind.FFN_STREAMED);
                assertEquals(rows == 256, streamed, "streamed FFN qualification M=" + rows);
                assertTrue(selected.instructions().stream()
                        .anyMatch(i -> i.kind() == QwenExecutionPlan.Kind.ATTENTION_PRODUCERS));
                int[] tokens = new int[rows];
                for (int i = 0; i < rows; i++) tokens[i] = INITIAL_TOKEN + i % 97;
                RouteResult expected = runRoute(gpu, model, reference, rows, tokens, 900 + rows);
                RouteResult actual = runRoute(gpu, model, production, rows, tokens, 901 + rows);
                assertArrayEquals(expected.hidden(), actual.hidden(), "hidden M=" + rows);
                assertArrayEquals(expected.logits(), actual.logits(), "logits M=" + rows);
                assertEquals(expected.state(), actual.state(), "state M=" + rows);
                assertArrayEquals(expected.decodeLogits(), actual.decodeLogits(), "decode M=" + rows);
                assertEquals(expected.decodeState(), actual.decodeState(), "decode state M=" + rows);
                System.out.println("PREFILL_ROUTE_BITWISE PASS rows=" + rows + " streamed=" + streamed
                        + " workspace_bytes=" + actual.workspaceBytes()
                        + " reference_workspace_bytes=" + expected.workspaceBytes());
            }
        }
    }

    private record RouteResult(
            short[] hidden,
            short[] logits,
            String state,
            short[] decodeLogits,
            String decodeState,
            long workspaceBytes) {}

    private static RouteResult runRoute(
            CudaGpuMemory gpu, QwenModel model, QwenExecutionPlan plan, int rows, int[] tokens, long sequenceId)
            throws Exception {
        var sequence = new QwenSequenceState(sequenceId);
        try {
            RunResult prefill = execute(
                    gpu,
                    plan.forExecution(QwenExecutionContext.ExecutionKind.PREFILL, rows),
                    sequence,
                    QwenExecutionContext.ExecutionKind.PREFILL,
                    0,
                    tokens,
                    List.of(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE),
                    QwenLogitsRequirement.LAST_TOKEN);
            short[] hidden, logits;
            String state;
            try {
                hidden = prefill.buffers().get(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE);
                logits = prefill.logits();
                state = macroStateFingerprint(gpu, model.weights(), sequence);
            } finally {
                prefill.closeLogits();
            }
            RunResult decode = execute(
                    gpu,
                    plan,
                    sequence,
                    QwenExecutionContext.ExecutionKind.DECODE,
                    rows,
                    new int[] {INITIAL_TOKEN},
                    List.of(),
                    QwenLogitsRequirement.LAST_TOKEN);
            try {
                return new RouteResult(
                        hidden,
                        logits,
                        state,
                        decode.logits(),
                        macroStateFingerprint(gpu, model.weights(), sequence),
                        prefill.workspaceBytes());
            } finally {
                decode.closeLogits();
            }
        } finally {
            sequence.complete();
        }
    }

    private static String macroStateFingerprint(CudaGpuMemory gpu, QwenWeights weights, QwenSequenceState sequence)
            throws Exception {
        var hash = java.security.MessageDigest.getInstance("SHA-256");
        var config = weights.config();
        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            if (config.layerTypes()[layer] == QwenLayerType.GATED_DELTA_NET) {
                var gdn = ((GdnSequenceStates) sequence.recurrentState()).forLayer(layer);
                long channels = 2L * config.linearNumKeyHeads() * config.linearKeyHeadDim()
                        + (long) config.linearNumValueHeads() * config.linearValueHeadDim();
                hash.update(readDeviceBytes(
                        gpu,
                        gdn.convolutionStateAddress(),
                        channels * (config.linearConvKernelDim() - 1) * Short.BYTES));
                hash.update(readDeviceBytes(
                        gpu,
                        gdn.recurrentStateAddress(),
                        (long) config.linearNumValueHeads()
                                * config.linearKeyHeadDim()
                                * config.linearValueHeadDim()
                                * Float.BYTES));
            } else {
                var kv = ((AttentionSequenceStates) sequence.kvCacheState()).forLayer(layer);
                assertEquals(sequence.currentTokenPosition(), kv.length());
                long bytes = (long) kv.length() * config.numKeyValueHeads() * config.attentionHeadDim() * Short.BYTES;
                hash.update(readDeviceBytes(gpu, kv.keyCacheAddress(), bytes));
                hash.update(readDeviceBytes(gpu, kv.valueCacheAddress(), bytes));
            }
        }
        return java.util.HexFormat.of().formatHex(hash.digest());
    }

    private static RunResult execute(
            CudaGpuMemory gpu,
            QwenExecutionPlan plan,
            QwenSequenceState sequence,
            QwenExecutionContext.ExecutionKind kind,
            long startPosition,
            int[] tokenIds,
            List<QwenExecutionPlan.Buffer> capturedBuffers)
            throws Exception {
        return execute(
                gpu, plan, sequence, kind, startPosition, tokenIds, capturedBuffers, QwenLogitsRequirement.ALL_TOKENS);
    }

    private static RunResult execute(
            CudaGpuMemory gpu,
            QwenExecutionPlan plan,
            QwenSequenceState sequence,
            QwenExecutionContext.ExecutionKind kind,
            long startPosition,
            int[] tokenIds,
            List<QwenExecutionPlan.Buffer> capturedBuffers,
            QwenLogitsRequirement requirement)
            throws Exception {
        EnumMap<QwenExecutionPlan.Buffer, short[]> buffers = new EnumMap<>(QwenExecutionPlan.Buffer.class);
        AtomicReference<short[]> logits = new AtomicReference<>();
        AtomicReference<QwenExecutionContext> completedContext = new AtomicReference<>();
        var workspaceBytes = new java.util.concurrent.atomic.AtomicLong();
        QwenExecutionRunner runner = new QwenExecutionRunner(plan, gpu, context -> {
            completedContext.set(context);
            var allocations = new java.util.HashMap<Long, Long>();
            for (var spec : context.plan().bufferSpecs()) {
                if (!context.workspace().hasBuffer(spec.buffer())) continue;
                if (spec.buffer() == QwenExecutionPlan.Buffer.LOGITS) {
                    context.logitsOutput()
                            .ifPresent(logitsOutput -> allocations.put(
                                    logitsOutput.deviceAddress(),
                                    context.workspace().bufferByteSize(spec.buffer())));
                } else
                    allocations.merge(
                            context.workspace().address(spec.buffer()),
                            context.workspace().bufferByteSize(spec.buffer()),
                            Math::max);
            }
            workspaceBytes.set(
                    allocations.values().stream().mapToLong(Long::longValue).sum());
            try (Arena arena = Arena.ofShared()) {
                for (QwenExecutionPlan.Buffer buffer : capturedBuffers) {
                    if (!context.workspace().hasBuffer(buffer)) continue;
                    buffers.put(
                            buffer,
                            download(
                                    gpu,
                                    arena,
                                    context.workspace().address(buffer),
                                    Math.toIntExact(context.workspace().bufferByteSize(buffer) / Short.BYTES)));
                }
                context.logitsOutput()
                        .ifPresent(deviceLogits -> logits.set(download(
                                gpu,
                                arena,
                                deviceLogits.deviceAddress(),
                                Math.multiplyExact(deviceLogits.tokenCount(), deviceLogits.vocabularySize()))));
            }
        });
        new DefaultExecutor().input(runner);
        try {
            var outcome =
                    runner.submit(new QwenExecutionContext(plan, sequence, kind, startPosition, tokenIds, requirement));
            runner.request(plan.instructions().size());
            QwenExecutionContext.Outcome result = outcome.get(600, TimeUnit.SECONDS);
            RunResult run = new RunResult(completedContext.get(), buffers, logits.get(), workspaceBytes.get());
            if (result.status() != QwenExecutionContext.Status.SUCCESS) {
                throw new AssertionError("Euhedral graph failed: " + result.failure());
            }
            return run;
        } finally {
            runner.completeGracefully();
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(ints = {3, 33})
    @Timeout(value = 600, unit = TimeUnit.SECONDS)
    void logitsRequirementsPreserveEveryStateAndFinalVocabularyRow(int tokenCount) throws Exception {
        Path artifactPath = Path.of(System.getProperty("euhedral.qwen.artifact"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                QwenModel model = QwenModel.load(artifactPath, QwenArtifactReader.read(artifactPath), gpu)) {
            var plan = new QwenExecutionPlan(model.weights());
            var config = model.weights().config();
            List<byte[]> expectedState = null;
            short[] expectedLastRow = null;
            short[] expectedHidden = null;
            int[] tokens = new int[tokenCount];
            Arrays.fill(tokens, INITIAL_TOKEN);
            for (var requirement : List.of(
                    QwenLogitsRequirement.ALL_TOKENS, QwenLogitsRequirement.LAST_TOKEN, QwenLogitsRequirement.NONE)) {
                var sequence = new QwenSequenceState(701 + requirement.ordinal());
                try {
                    RunResult run = execute(
                            gpu,
                            plan,
                            sequence,
                            QwenExecutionContext.ExecutionKind.PREFILL,
                            0,
                            tokens,
                            List.of(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE),
                            requirement);
                    try {
                        assertEquals(tokenCount, sequence.currentTokenPosition());
                        short[] hidden = run.buffers().get(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE);
                        if (expectedHidden == null) expectedHidden = hidden;
                        else assertArrayEquals(expectedHidden, hidden);
                        List<byte[]> state = new ArrayList<>();
                        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
                            if (config.layerTypes()[layer] == QwenLayerType.GATED_DELTA_NET) {
                                var gdn = ((GdnSequenceStates) sequence.recurrentState()).forLayer(layer);
                                long channels = 2L * config.linearNumKeyHeads() * config.linearKeyHeadDim()
                                        + (long) config.linearNumValueHeads() * config.linearValueHeadDim();
                                state.add(readDeviceBytes(
                                        gpu,
                                        gdn.convolutionStateAddress(),
                                        channels * (config.linearConvKernelDim() - 1) * Short.BYTES));
                                state.add(readDeviceBytes(
                                        gpu,
                                        gdn.recurrentStateAddress(),
                                        (long) config.linearNumValueHeads()
                                                * config.linearKeyHeadDim()
                                                * config.linearValueHeadDim()
                                                * Float.BYTES));
                            } else {
                                var kv = ((AttentionSequenceStates) sequence.kvCacheState()).forLayer(layer);
                                assertEquals(tokenCount, kv.length());
                                long bytes = (long) kv.length()
                                        * config.numKeyValueHeads()
                                        * config.attentionHeadDim()
                                        * Short.BYTES;
                                state.add(readDeviceBytes(gpu, kv.keyCacheAddress(), bytes));
                                state.add(readDeviceBytes(gpu, kv.valueCacheAddress(), bytes));
                            }
                        }
                        if (expectedState == null) expectedState = state;
                        else
                            for (int index = 0; index < state.size(); index++)
                                assertArrayEquals(expectedState.get(index), state.get(index), "state buffer " + index);
                        if (requirement == QwenLogitsRequirement.ALL_TOKENS) {
                            assertEquals(tokenCount * config.vocabSize(), run.logits().length);
                            expectedLastRow = Arrays.copyOfRange(
                                    run.logits(),
                                    (tokenCount - 1) * config.vocabSize(),
                                    tokenCount * config.vocabSize());
                        } else if (requirement == QwenLogitsRequirement.LAST_TOKEN) {
                            assertEquals(config.vocabSize(), run.logits().length);
                            reportError("last-row-" + tokenCount, expectedLastRow, run.logits());
                            if (tokenCount == 3) assertArrayEquals(expectedLastRow, run.logits());
                            else {
                                // WMMA versus decode measured max 0.0625 and RMS 0.000818.
                                // Bound rounding by one BF16 step, with an absolute floor
                                // for near-zero cancellation; transformer state stays exact.
                                for (int i = 0; i < expectedLastRow.length; i++) {
                                    short expected = expectedLastRow[i], actual = run.logits()[i];
                                    float error = Math.abs(bf16ToFloat(expected) - bf16ToFloat(actual));
                                    boolean adjacent = (expected < 0) == (actual < 0)
                                            && Math.abs((expected & 0xffff) - (actual & 0xffff)) <= 1;
                                    assertTrue(
                                            Float.isFinite(error) && (error <= 0.001f || adjacent),
                                            "last row index " + i);
                                }
                            }
                        } else {
                            assertTrue(run.context().logitsOutput().isEmpty());
                        }
                    } finally {
                        run.closeLogits();
                    }
                } finally {
                    sequence.complete();
                }
            }
        }
    }

    private static void assertSuccessful(RunResult run) {
        assertNotNull(run.context());
        assertTrue(run.context().workspace().isClosed());
    }

    private static int independentArgmax(short[] downloadedLogits) {
        int bestTokenId = 0;
        float bestLogit = bf16ToFloat(downloadedLogits[0]);
        for (int tokenId = 1; tokenId < downloadedLogits.length; tokenId++) {
            float logit = bf16ToFloat(downloadedLogits[tokenId]);
            if (logit > bestLogit) {
                bestTokenId = tokenId;
                bestLogit = logit;
            }
        }
        return bestTokenId;
    }

    private static int[] independentTopTwo(short[] downloadedLogits) {
        int[] tokenIds = {-1, -1};
        float[] logits = {Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY};
        for (int tokenId = 0; tokenId < downloadedLogits.length; tokenId++) {
            float logit = bf16ToFloat(downloadedLogits[tokenId]);
            if (logit > logits[0]) {
                tokenIds[1] = tokenIds[0];
                logits[1] = logits[0];
                tokenIds[0] = tokenId;
                logits[0] = logit;
            } else if (logit > logits[1]) {
                tokenIds[1] = tokenId;
                logits[1] = logit;
            }
        }
        return tokenIds;
    }

    private static Throwable mergeFailure(Throwable failure, Throwable nextFailure) {
        if (failure == null) return nextFailure;
        failure.addSuppressed(nextFailure);
        return failure;
    }

    private static byte[] readDeviceBytes(CudaGpuMemory gpu, long address, long byteSize) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(byteSize, Integer.BYTES);
            gpu.copyDeviceToHost(host, address, byteSize);
            return host.toArray(ValueLayout.JAVA_BYTE);
        }
    }

    private record RunResult(
            QwenExecutionContext context,
            EnumMap<QwenExecutionPlan.Buffer, short[]> buffers,
            short[] logits,
            long workspaceBytes) {
        private void closeLogits() {
            context.logitsOutput().ifPresent(deviceLogits -> deviceLogits.close());
        }
    }
}
