package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactDenseFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The NVFP4 artifact (tools/euhedral_artifacts/nvfp4.py): the text and
/// MTP inventory without the vision tower, NVFP4 projections and a Q3 embedding, loads onto the GPU
/// with exactly its payload resident. Skipped unless -Peuhedral.qwen.nvfp4-artifact names an artifact.
class QwenNvfp4CudaLoadIntegrationTest {

    @Test
    @Timeout(value = 900, unit = TimeUnit.SECONDS)
    void nvfp4ArtifactLoadsWithExactlyItsPayloadResident() throws Exception {
        Path path = Path.of(System.getProperty("euhedral.qwen.nvfp4-artifact", ""));
        assumeTrue(Files.isRegularFile(path), "no NVFP4 artifact: " + path);
        QwenArtifact artifact = QwenArtifactReader.read(path);
        assertEquals(785, artifact.tensors().length);
        assertEquals(
                0,
                Arrays.stream(artifact.tensors())
                        .filter(t -> t.name().startsWith("vision/"))
                        .count());
        // The MTP layer and draft head are speculative decoding's, loaded with it; this load is the base model.
        long payload = Arrays.stream(artifact.tensors())
                .filter(t -> !t.name().startsWith("mtp/") && !t.name().startsWith("text/draft_head"))
                .mapToLong(TensorDescriptor::byteSize)
                .sum();
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                QwenModel model = QwenModel.load(path, artifact, gpu)) {
            var weights = model.weights();
            assertEquals(payload, gpu.allocatedBytes());
            assertEquals(WeightFormat.Q3_G64_FP16, weights.tokenEmbedding().format());
            assertEquals(WeightFormat.NVFP4, weights.lmHead().format());
            for (var layer : weights.layers()) {
                var ffn = assertInstanceOf(QwenCompactDenseFfnWeights.class, layer.ffn());
                assertEquals(WeightFormat.NVFP4, ffn.gateUp().format());
                assertEquals(WeightFormat.NVFP4, ffn.down().format());
            }
            var memory = gpu.deviceMemoryInfo();
            System.out.println("NVFP4_LOAD PASS objects=785 resident_bytes=" + gpu.allocatedBytes()
                    + " device_free_bytes=" + memory.freeBytes() + " device_total_bytes=" + memory.totalBytes());
        }
    }
}
