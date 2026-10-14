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
import io.euhedral_execution.inference.core.model.qwen38.offload.DiskWindow;
import io.euhedral_execution.inference.core.state.KvFormat;
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

    @Test
    void aBf16CacheTakesItsRowsBytesPerToken() {
        Qwen38Config config = config(2);
        long nvfp4 =
                ResidencyPlanner.kvBytes(config, 131072, io.euhedral_execution.inference.core.state.KvFormat.NVFP4);
        long bf16 = ResidencyPlanner.kvBytes(config, 131072, io.euhedral_execution.inference.core.state.KvFormat.BF16);
        org.junit.jupiter.api.Assertions.assertEquals(nvfp4, ResidencyPlanner.kvBytes(config, 131072));
        org.junit.jupiter.api.Assertions.assertEquals(nvfp4 / 144 * 512, bf16);
    }

    private static final List<String> FAMILIES = List.of(
            "gdn/query_key",
            "gdn/output",
            "attention/output",
            "attention/query_key",
            "attention/gate_value",
            "gdn/value_z",
            "mlp/down",
            "mlp/gate_up");

    /// The embedding, the output head and every family of each of `config`'s layers, each `familyBytes`.
    private static Artifact artifact(Qwen38Config config, long familyBytes) {
        List<TensorDescriptor> tensors = new ArrayList<>();
        tensors.add(tensor("text/token_embedding", familyBytes));
        tensors.add(tensor("text/output_head", familyBytes));
        for (int layer = 0; layer < config.numHiddenLayers(); layer++)
            for (String family : FAMILIES) tensors.add(tensor("text/layers/" + layer + "/" + family, familyBytes));
        return new Artifact(null, config, tensors.toArray(TensorDescriptor[]::new));
    }

    /// Free device memory that holds the KV and the reserve and `weights` bytes more.
    private static long free(Qwen38Config config, long weights) {
        return ResidencyPlanner.kvBytes(config, 4096) + ResidencyPlanner.RESERVE_BYTES + weights;
    }

    @Test
    void weightsBeyondTheHostBudgetGoToDiskInSelectionOrder() {
        Artifact artifact = artifact(config(1), 16 * MIB);
        var plan =
                ResidencyPlanner.plan(artifact, PROFILE, free(config(1), 200 * MIB), 4096, 256 * MIB, KvFormat.NVFP4);
        assertTrue(plan.fits());
        assertFalse(plan.diskBacked().isEmpty());
        assertTrue(plan.hostBacked().contains(HostWeightSelection.EMBEDDING));
        assertTrue(plan.hostArenaBytes() + plan.windowBytes() <= 256 * MIB);
        assertEquals(DiskWindow.SLOTS * DiskWindow.slotBytesFor(16 * MIB), plan.windowBytes());
        long host = 0;
        for (TensorDescriptor tensor : artifact.tensors())
            if (plan.hostBacked().contains(tensor.name())) host += tensor.byteSize();
        assertEquals(host, plan.hostArenaBytes());
        List<String> order = List.copyOf(HostWeightSelection.select(artifact, Long.MAX_VALUE));
        int lastHost = plan.hostBacked().stream().mapToInt(order::indexOf).max().orElse(-1);
        int firstDisk =
                plan.diskBacked().stream().mapToInt(order::indexOf).min().orElseThrow();
        assertTrue(lastHost < firstDisk);
        for (String name : plan.diskBacked()) assertFalse(plan.hostBacked().contains(name));
        assertTrue(plan.deviceBytes() <= free(config(1), 200 * MIB));
    }

    @Test
    void theDeviceRingHoldsTheLargestHostOrDiskTensor() {
        Artifact artifact = artifact(config(1), 16 * MIB);
        var disk =
                ResidencyPlanner.plan(artifact, PROFILE, free(config(1), 200 * MIB), 4096, 256 * MIB, KvFormat.NVFP4);
        var host = ResidencyPlanner.plan(
                artifact, PROFILE, free(config(1), 200 * MIB), 4096, Long.MAX_VALUE, KvFormat.NVFP4);
        // The same device need: every offloaded tensor is staged through the same ring, whichever tier holds it.
        assertEquals(host.deviceBytes(), disk.deviceBytes());
        java.util.Set<String> offloaded = new java.util.HashSet<>(disk.hostBacked());
        offloaded.addAll(disk.diskBacked());
        assertEquals(host.hostBacked(), offloaded);
    }

    @Test
    void aWindowThatDoesNotFitTheHostBudgetIsRefused() {
        var plan = ResidencyPlanner.plan(
                artifact(config(1), 16 * MIB), PROFILE, free(config(1), 200 * MIB), 4096, 32 * MIB, KvFormat.NVFP4);
        assertFalse(plan.fits());
        assertTrue(plan.hostArenaBytes() + plan.windowBytes() > 32 * MIB);
    }

    @Test
    void anEmbeddingBeyondTheHostBudgetIsRefused() {
        var plan = ResidencyPlanner.plan(
                artifact(config(1), 16 * MIB), PROFILE, free(config(1), 200 * MIB), 4096, 8 * MIB, KvFormat.NVFP4);
        assertFalse(plan.fits());
        assertFalse(plan.diskBacked().contains(HostWeightSelection.EMBEDDING));
    }

    @Test
    void unlimitedHostMemoryNeedsNoDisk() {
        var plan = ResidencyPlanner.plan(
                artifact(config(1), 16 * MIB),
                PROFILE,
                free(config(1), 200 * MIB),
                4096,
                Long.MAX_VALUE,
                KvFormat.NVFP4);
        assertTrue(plan.fits());
        assertFalse(plan.hostBacked().isEmpty());
        assertTrue(plan.diskBacked().isEmpty());
        assertEquals(0, plan.windowBytes());
    }

    @Test
    void aHostBudgetThatHoldsTheSelectionNeedsNoWindow() {
        Artifact artifact = artifact(config(1), 16 * MIB);
        var unlimited = ResidencyPlanner.plan(
                artifact, PROFILE, free(config(1), 200 * MIB), 4096, Long.MAX_VALUE, KvFormat.NVFP4);
        var exact = ResidencyPlanner.plan(
                artifact, PROFILE, free(config(1), 200 * MIB), 4096, unlimited.hostArenaBytes(), KvFormat.NVFP4);
        assertTrue(exact.fits());
        assertEquals(unlimited.hostBacked(), exact.hostBacked());
        assertTrue(exact.diskBacked().isEmpty());
        assertEquals(0, exact.windowBytes());
    }

    @Test
    void onlyRowSplitAndBf16ProjectionsAreStageable() {
        assertTrue(HostWeightSelection.stageable(tensor("text/layers/0/mlp/down", MIB)));
        assertFalse(HostWeightSelection.stageable(tensor("text/output_head", MIB)));
        assertFalse(HostWeightSelection.stageable(new TensorDescriptor(
                "text/layers/0/mlp/down",
                new long[] {1, 1},
                TensorDataType.BF16,
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_P2E2_V1,
                0,
                MIB)));
        assertTrue(HostWeightSelection.stageable(new TensorDescriptor(
                "text/layers/0/mlp/down",
                new long[] {1, 1},
                TensorDataType.BF16,
                WeightFormat.BF16,
                WeightLayout.CONTIGUOUS_LE_V1,
                0,
                MIB)));
    }
}
