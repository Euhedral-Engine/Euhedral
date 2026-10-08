package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.loader.DenseFfnWeights;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.testing.SharedQwen38;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The NVFP4 artifact (tools/euhedral_artifacts/nvfp4.py): the text and
/// MTP inventory without the vision tower, NVFP4 projections and a Q3 embedding, loads onto the GPU
/// with exactly its payload resident. Skipped unless -Peuhedral.qwen.nvfp4-artifact names an artifact.
@ModelGroup.Nvfp4
class Nvfp4CudaLoadIntegrationTest {

    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void nvfp4ArtifactLoadsWithExactlyItsPayloadResident() throws Exception {
        var loaded = SharedQwen38.nvfp4();
        Artifact artifact = loaded.artifact();
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
        CudaGpuMemory gpu = loaded.gpu();
        var weights = loaded.model().weights();
        // The shared device also holds whatever earlier classes left (retained scratch), so the payload is the
        // sum over the model's own resident objects.
        assertEquals(
                payload,
                weights.runtimeObjects().values().stream()
                        .mapToLong(TensorHandle::byteSize)
                        .sum());
        assertTrue(gpu.allocatedBytes() >= payload);
        assertEquals(WeightFormat.Q3_G64_FP16, weights.tokenEmbedding().format());
        assertEquals(WeightFormat.NVFP4, weights.lmHead().format());
        for (var layer : weights.layers()) {
            var ffn = assertInstanceOf(DenseFfnWeights.class, layer.ffn());
            assertEquals(WeightFormat.NVFP4, ffn.gateUp().format());
            assertEquals(WeightFormat.NVFP4, ffn.down().format());
        }
        var memory = gpu.deviceMemoryInfo();
        System.out.println("NVFP4_LOAD PASS objects=785 resident_bytes=" + gpu.allocatedBytes() + " device_free_bytes="
                + memory.freeBytes() + " device_total_bytes=" + memory.totalBytes());
    }
}
