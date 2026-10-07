package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Model;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.model.qwen38.TestExecution;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactHeader;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.Weights;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class FirstLayerCudaIntegrationTest {

    @Test
    @Timeout(value = 180, unit = TimeUnit.SECONDS)
    void actualLayerZeroRunsThroughEuhedralSourceAndFrames() throws Exception {
        Path artifactPath = Path.of(System.getProperty("euhedral.qwen.artifact"));
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        assertTrue(Files.isRegularFile(artifactPath));
        Artifact artifact = ArtifactReader.read(artifactPath);
        assertEquals(ArtifactHeader.COMPACT_VERSION, artifact.header().version());
        assertEquals(LayerType.GATED_DELTA_NET, artifact.config().layerTypes()[0]);

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath)) {
            Qwen38Model model = Qwen38Model.loadFirstLayer(artifactPath, artifact, gpu);
            Weights weights = model.weights();
            Sequence sequence = new Sequence(0x5147454eL);
            try {
                ExecutionPlan plan = new ExecutionPlan(weights);
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
                        plan.bufferElementType(ExecutionPlan.Buffer.valueOf("A_PROJECTED"))
                                .name());
                assertEquals(
                        "FP32",
                        plan.bufferElementType(ExecutionPlan.Buffer.valueOf("B_PROJECTED"))
                                .name());
                int[] tokenIds = {1814}; // Tokenizer vocabulary entry "Ġworld" from the source checkpoint.
                Quantum context = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, tokenIds);
                AtomicReference<short[]> terminalHidden = new AtomicReference<>();
                EnumMap<ExecutionPlan.Buffer, short[]> observed = new EnumMap<>(ExecutionPlan.Buffer.class);
                Quantum.Outcome completed = TestExecution.run(
                        plan,
                        gpu,
                        context,
                        done -> {
                            try (Arena arena = Arena.ofShared()) {
                                for (ExecutionPlan.Buffer buffer : List.of(
                                        ExecutionPlan.Buffer.HIDDEN_STATE,
                                        ExecutionPlan.Buffer.INPUT_NORMALIZED,
                                        ExecutionPlan.Buffer.GDN_CONVOLVED,
                                        ExecutionPlan.Buffer.GDN_RECURRENT,
                                        ExecutionPlan.Buffer.GDN_NORMALIZED,
                                        ExecutionPlan.Buffer.MIXER_DELTA,
                                        ExecutionPlan.Buffer.POST_MIXER_NORMALIZED,
                                        ExecutionPlan.Buffer.FFN_DELTA,
                                        ExecutionPlan.Buffer.FINAL_HIDDEN_STATE)) {
                                    observed.put(
                                            buffer,
                                            CudaGpuOperationsIntegrationTest.download(
                                                    gpu,
                                                    arena,
                                                    done.workspace().address(buffer),
                                                    Math.toIntExact(
                                                            done.workspace().bufferByteSize(buffer) / Short.BYTES)));
                                }
                                terminalHidden.set(observed.get(ExecutionPlan.Buffer.FINAL_HIDDEN_STATE));
                            }
                        },
                        150);
                assertNotNull(
                        sequence.recurrentState(),
                        "layer execution did not attach its persistent GDN state to the sequence");
                assertEquals(
                        Quantum.Status.SUCCESS,
                        completed.status(),
                        "first-layer execution failure: " + completed.failure());
                assertNotNull(terminalHidden.get(), "terminal callback did not observe the layer output");
                assertEquals(weights.config().hiddenSize(), terminalHidden.get().length);
                FirstLayerCpuReference.Result reference = FirstLayerCpuReference.run(weights, gpu, tokenIds[0]);
                for (ExecutionPlan.Buffer boundary : List.of(
                        ExecutionPlan.Buffer.HIDDEN_STATE,
                        ExecutionPlan.Buffer.INPUT_NORMALIZED,
                        ExecutionPlan.Buffer.GDN_CONVOLVED,
                        ExecutionPlan.Buffer.GDN_RECURRENT,
                        ExecutionPlan.Buffer.GDN_NORMALIZED,
                        ExecutionPlan.Buffer.MIXER_DELTA,
                        ExecutionPlan.Buffer.POST_MIXER_NORMALIZED,
                        ExecutionPlan.Buffer.FFN_DELTA,
                        ExecutionPlan.Buffer.FINAL_HIDDEN_STATE)) {
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
