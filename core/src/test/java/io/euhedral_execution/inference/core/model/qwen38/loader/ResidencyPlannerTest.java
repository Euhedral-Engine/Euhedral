package io.euhedral_execution.inference.core.model.qwen38.loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile.Quantization;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class ResidencyPlannerTest {
    private static final long MIB = 1L << 20;
    private static final ArtifactProfile PROFILE = new ArtifactProfile(Quantization.NVFP4, false, true);

    private static TensorDescriptor tensor(String name, long bytes) {
        return new TensorDescriptor(
                name,
                new long[] {1, 1},
                TensorDataType.BF16,
                WeightFormat.NVFP4,
                WeightLayout.ROW_SPLIT_K128_V1,
                0,
                bytes);
    }

    private static Qwen38Config config(int attentionLayers) {
        LayerType[] layers = new LayerType[attentionLayers + 3];
        java.util.Arrays.fill(layers, LayerType.GATED_DELTA_NET);
        for (int index = 0; index < attentionLayers; index++)
            layers[index * 4 % layers.length] = LayerType.FULL_ATTENTION;
        return new Qwen38Config(
                8,
                128,
                layers.length,
                4,
                4,
                256,
                128,
                4,
                4,
                128,
                128,
                4,
                1e-6,
                1e7,
                0.25,
                4096,
                "silu",
                layers,
                0,
                0,
                0,
                0,
                false,
                true,
                1);
    }

    private static Artifact artifact() {
        List<TensorDescriptor> tensors = new ArrayList<>();
        tensors.add(tensor("text/token_embedding", 400 * MIB));
        tensors.add(tensor("text/output_head", 600 * MIB));
        for (int layer = 0; layer < 8; layer++) {
            tensors.add(tensor("text/layers/" + layer + "/gdn/query_key", 10 * MIB));
            tensors.add(tensor("text/layers/" + layer + "/mlp/down", 100 * MIB));
        }
        tensors.add(tensor("mtp/input_projection", 50 * MIB));
        tensors.add(tensor("text/draft_head", 70 * MIB));
        return new Artifact(null, config(2), tensors.toArray(TensorDescriptor[]::new));
    }

    @Test
    void kvBytesCountWholePagesOfTheAttentionLayersOnly() {
        // 2 attention layers x 4 heads x 144-byte rows x 2 planes = 2304 bytes per token.
        assertEquals(2304L * 256, ResidencyPlanner.kvBytes(config(2), 1));
        assertEquals(2304L * 256, ResidencyPlanner.kvBytes(config(2), 256));
        assertEquals(2304L * 512, ResidencyPlanner.kvBytes(config(2), 257));
    }

    @Test
    void nothingLeavesTheDeviceWhenEverythingFits() {
        var plan = ResidencyPlanner.plan(artifact(), PROFILE, 64L << 30, 4096);
        assertTrue(plan.fits());
        assertTrue(plan.hostBacked().isEmpty());
        // Weights without the vision tower, the KV of one sequence and the reserve.
        long weights = (400 + 600 + 8 * 110 + 50 + 70) * MIB;
        assertEquals(
                weights + ResidencyPlanner.kvBytes(config(2), 4096) + ResidencyPlanner.RESERVE_BYTES,
                plan.deviceBytes());
    }

    @Test
    void theEmbeddingAndSmallProjectionsLeaveFirstWhenDeviceMemoryIsShort() {
        long resident = (400 + 600 + 8 * 110 + 50 + 70) * MIB
                + ResidencyPlanner.kvBytes(config(2), 4096)
                + ResidencyPlanner.RESERVE_BYTES;
        var plan = ResidencyPlanner.plan(artifact(), PROFILE, resident - 100 * MIB, 4096);
        assertTrue(plan.fits());
        assertTrue(plan.hostBacked().contains(HostWeightSelection.EMBEDDING));
        assertFalse(plan.hostBacked().contains("text/output_head"));
        assertTrue(plan.deviceBytes() <= resident - 100 * MIB);
    }

    @Test
    void aContextThatCannotFitIsReported() {
        var plan = ResidencyPlanner.plan(artifact(), PROFILE, 1L << 30, 4096);
        assertFalse(plan.fits());
    }

    @Test
    void aNonSpeculativeProfileLeavesTheDraftObjectsOffTheDevice() {
        var plain = new ArtifactProfile(Quantization.NVFP4, false, false);
        var speculative = ResidencyPlanner.plan(artifact(), PROFILE, 64L << 30, 4096);
        var without = ResidencyPlanner.plan(artifact(), plain, 64L << 30, 4096);
        assertEquals(120 * MIB, speculative.deviceBytes() - without.deviceBytes());
    }
}
