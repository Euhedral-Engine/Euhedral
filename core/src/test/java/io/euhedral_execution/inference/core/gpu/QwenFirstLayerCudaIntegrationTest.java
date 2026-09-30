package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactHeader;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenSequenceState;
import io.euhedral_execution.inference.core.scheduling.TestExecution;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class QwenFirstLayerCudaIntegrationTest {

    @Test
    @Timeout(value = 180, unit = TimeUnit.SECONDS)
    void actualLayerZeroRunsThroughEuhedralSourceAndFrames() throws Exception {
        Path artifactPath = Path.of(System.getProperty("euhedral.qwen.artifact"));
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        assertTrue(Files.isRegularFile(artifactPath));
        QwenArtifact artifact = QwenArtifactReader.read(artifactPath);
        assertEquals(QwenArtifactHeader.COMPACT_VERSION, artifact.header().version());
        assertEquals(QwenLayerType.GATED_DELTA_NET, artifact.config().layerTypes()[0]);

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath)) {
            QwenModel model = QwenModel.loadFirstLayer(artifactPath, artifact, gpu);
            QwenWeights weights = model.weights();
            QwenSequenceState sequence = new QwenSequenceState(0x5147454eL);
            try {
                QwenExecutionPlan plan = new QwenExecutionPlan(weights);
                List<String> instructionKinds = plan.instructions().stream()
                        .map(instruction -> instruction.kind().name())
                        .toList();
                assertEquals("RMS_NORM_UNIT_OFFSET", instructionKinds.get(1));
                assertEquals("RMS_NORM_UNIT_OFFSET", instructionKinds.get(12));
                assertEquals(List.of(2, 3, 4, 5), plan.successors(1));
                assertEquals(
                        "MIXER_HIDDEN",
                        plan.instructions().get(11).outputBuffers().getFirst().name());
                assertEquals(
                        "MIXER_HIDDEN",
                        plan.instructions().get(12).inputBuffers().getFirst().name());
                assertEquals(
                        "FINAL_HIDDEN_STATE",
                        plan.instructions().get(16).outputBuffers().getFirst().name());
                assertEquals(
                        "FP32",
                        plan.bufferElementType(QwenExecutionPlan.Buffer.valueOf("A_PROJECTED"))
                                .name());
                assertEquals(
                        "FP32",
                        plan.bufferElementType(QwenExecutionPlan.Buffer.valueOf("B_PROJECTED"))
                                .name());
                int[] tokenIds = {1814}; // Tokenizer vocabulary entry "Ġworld" from the source checkpoint.
                QwenExecutionContext context = new QwenExecutionContext(
                        plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0, tokenIds);
                AtomicReference<short[]> terminalHidden = new AtomicReference<>();
                EnumMap<QwenExecutionPlan.Buffer, short[]> observed = new EnumMap<>(QwenExecutionPlan.Buffer.class);
                QwenExecutionContext.Outcome completed = TestExecution.run(
                        plan,
                        gpu,
                        context,
                        done -> {
                            try (Arena arena = Arena.ofShared()) {
                                for (QwenExecutionPlan.Buffer buffer : List.of(
                                        QwenExecutionPlan.Buffer.HIDDEN_STATE,
                                        QwenExecutionPlan.Buffer.INPUT_NORMALIZED,
                                        QwenExecutionPlan.Buffer.GDN_CONVOLVED,
                                        QwenExecutionPlan.Buffer.GDN_RECURRENT,
                                        QwenExecutionPlan.Buffer.GDN_NORMALIZED,
                                        QwenExecutionPlan.Buffer.MIXER_DELTA,
                                        QwenExecutionPlan.Buffer.POST_MIXER_NORMALIZED,
                                        QwenExecutionPlan.Buffer.FFN_DELTA,
                                        QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE)) {
                                    observed.put(
                                            buffer,
                                            CudaGpuOperationsIntegrationTest.download(
                                                    gpu,
                                                    arena,
                                                    done.workspace().address(buffer),
                                                    Math.toIntExact(
                                                            done.workspace().bufferByteSize(buffer) / Short.BYTES)));
                                }
                                terminalHidden.set(observed.get(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE));
                            }
                        },
                        150);
                assertNotNull(
                        sequence.recurrentState(),
                        "layer execution did not attach its persistent GDN state to the sequence");
                assertEquals(
                        QwenExecutionContext.Status.SUCCESS,
                        completed.status(),
                        "first-layer execution failure: " + completed.failure());
                assertNotNull(terminalHidden.get(), "terminal callback did not observe the layer output");
                assertEquals(weights.config().hiddenSize(), terminalHidden.get().length);
                QwenFirstLayerCpuReference.Result reference = QwenFirstLayerCpuReference.run(weights, gpu, tokenIds[0]);
                for (QwenExecutionPlan.Buffer boundary : List.of(
                        QwenExecutionPlan.Buffer.HIDDEN_STATE,
                        QwenExecutionPlan.Buffer.INPUT_NORMALIZED,
                        QwenExecutionPlan.Buffer.GDN_CONVOLVED,
                        QwenExecutionPlan.Buffer.GDN_RECURRENT,
                        QwenExecutionPlan.Buffer.GDN_NORMALIZED,
                        QwenExecutionPlan.Buffer.MIXER_DELTA,
                        QwenExecutionPlan.Buffer.POST_MIXER_NORMALIZED,
                        QwenExecutionPlan.Buffer.FFN_DELTA,
                        QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE)) {
                    try {
                        CudaGpuOperationsIntegrationTest.assertBf16Equals(
                                reference.buffers().get(boundary), observed.get(boundary), 0.08f);
                    } catch (AssertionError mismatch) {
                        throw new AssertionError(boundary + ": " + mismatch.getMessage(), mismatch);
                    }
                }
                assertEquals(1, sequence.currentTokenPosition());
                assertTrue(context.workspace().isClosed());
                sequence.complete();
            } finally {
                model.close();
            }
        }
    }
}
