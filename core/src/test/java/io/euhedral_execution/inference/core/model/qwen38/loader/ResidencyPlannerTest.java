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
    void theOutputHeadIsMappedOnlyWhenEverythingElseOffloadableIsNotEnough() {
        long resident = (400 + 600 + 8 * 110 + 50 + 70) * MIB
                + ResidencyPlanner.kvBytes(config(2), 4096)
                + ResidencyPlanner.RESERVE_BYTES;
        // Embedding and every projection leave 880 MiB net of the staging ring: 1000 MiB needs the head too.
        var plan = ResidencyPlanner.plan(
                artifact(), PROFILE, resident - 1000 * MIB, 4096, Long.MAX_VALUE, KvFormat.NVFP4, 1, false);
        assertTrue(plan.fits());
        assertTrue(plan.hostBacked().contains(HostWeightSelection.OUTPUT_HEAD));
        assertFalse(plan.hostBacked().contains("text/draft_head"));
        assertTrue(plan.diskBacked().isEmpty());
        assertEquals(new java.util.ArrayList<>(plan.hostBacked()).getLast(), HostWeightSelection.OUTPUT_HEAD);
        // The head is in the host arena and not in the staging ring.
        assertEquals((400 + 600 + 8 * 110) * MIB, plan.hostArenaBytes());
        assertTrue(plan.deviceBytes() <= resident - 1000 * MIB);
        var tooLittle = ResidencyPlanner.plan(
                artifact(), PROFILE, resident - 2000 * MIB, 4096, Long.MAX_VALUE, KvFormat.NVFP4, 1, false);
        assertFalse(tooLittle.fits());
    }

    @Test
    void theOutputHeadStaysInHostMemoryWhenTheWindowSendsEarlierObjectsToDisk() {
        long resident = (400 + 600 + 8 * 110 + 50 + 70) * MIB
                + ResidencyPlanner.kvBytes(config(2), 4096)
                + ResidencyPlanner.RESERVE_BYTES;
        var plan = ResidencyPlanner.plan(
                artifact(), PROFILE, resident - 1000 * MIB, 4096, 1100 * MIB, KvFormat.NVFP4, 1, false);
        assertTrue(plan.hostBacked().contains(HostWeightSelection.OUTPUT_HEAD));
        assertFalse(plan.diskBacked().contains(HostWeightSelection.OUTPUT_HEAD));
        assertTrue(plan.hostBacked().contains(HostWeightSelection.EMBEDDING));
    }

    @Test
    void theOutputHeadPropertyMapsItWithTheFirstOffload() {
        long resident = (400 + 600 + 8 * 110 + 50 + 70) * MIB
                + ResidencyPlanner.kvBytes(config(2), 4096)
                + ResidencyPlanner.RESERVE_BYTES;
        var early = ResidencyPlanner.plan(
                artifact(), PROFILE, resident - 100 * MIB, 4096, Long.MAX_VALUE, KvFormat.NVFP4, 1, true);
        assertTrue(early.fits());
        assertTrue(early.hostBacked().contains(HostWeightSelection.OUTPUT_HEAD));
        // Plans that fit resident are unchanged.
        var fits = ResidencyPlanner.plan(artifact(), PROFILE, 64L << 30, 4096, Long.MAX_VALUE, KvFormat.NVFP4, 1, true);
        assertTrue(fits.hostBacked().isEmpty());
        assertEquals(ResidencyPlanner.plan(artifact(), PROFILE, 64L << 30, 4096).deviceBytes(), fits.deviceBytes());
    }

    @Test
    void anNvfp4HeadIsNotMappedWhenVerificationOutgrowsTheDecodeKernels() {
        long resident = (400 + 600 + 8 * 110 + 50 + 70) * MIB
                + ResidencyPlanner.kvBytes(config(2), 4096)
                + ResidencyPlanner.RESERVE_BYTES;
        long free = resident - 1000 * MIB;
        // Rows per verification are sequences x (1 + draft depth).
        int depth = PROFILE.speculativeDepth();
        int most = ResidencyPlanner.NVFP4_ROW_EXACT_MAX_ROWS / (1 + depth);
        var within = ResidencyPlanner.plan(artifact(), PROFILE, free, 4096, Long.MAX_VALUE, KvFormat.NVFP4, most, true);
        assertTrue(within.hostBacked().contains(HostWeightSelection.OUTPUT_HEAD));
        var beyond =
                ResidencyPlanner.plan(artifact(), PROFILE, free, 4096, Long.MAX_VALUE, KvFormat.NVFP4, most + 1, true);
        assertFalse(beyond.hostBacked().contains(HostWeightSelection.OUTPUT_HEAD));
        var late =
                ResidencyPlanner.plan(artifact(), PROFILE, free, 4096, Long.MAX_VALUE, KvFormat.NVFP4, most + 1, false);
        assertFalse(late.hostBacked().contains(HostWeightSelection.OUTPUT_HEAD));
        assertFalse(late.fits());
    }

    @Test
    void theEarlyHeadStaysLastAndTheArenaSplitKeepsItInHostMemory() {
        long resident = (400 + 600 + 8 * 110 + 50 + 70) * MIB
                + ResidencyPlanner.kvBytes(config(2), 4096)
                + ResidencyPlanner.RESERVE_BYTES;
        var plan = ResidencyPlanner.plan(
                artifact(), PROFILE, resident - 100 * MIB, 4096, 1100 * MIB, KvFormat.NVFP4, 1, true);
        var order = new ArrayList<>(plan.hostBacked());
        order.addAll(plan.diskBacked());
        assertTrue(plan.hostBacked().contains(HostWeightSelection.OUTPUT_HEAD));
        assertFalse(plan.diskBacked().contains(HostWeightSelection.OUTPUT_HEAD));
        assertEquals(HostWeightSelection.EMBEDDING, order.getFirst());
        // The head's bytes are in the arena before earlier objects are placed in the window.
        assertTrue(plan.hostArenaBytes() + plan.windowBytes() <= 1100 * MIB || !plan.fits());
        assertTrue(plan.hostArenaBytes() >= 1000 * MIB);
    }

    @Test
    void aPartiallyOffloadedDefaultPlanKeepsTheHeadAndIsTheSelection() {
        long resident = (400 + 600 + 8 * 110 + 50 + 70) * MIB
                + ResidencyPlanner.kvBytes(config(2), 4096)
                + ResidencyPlanner.RESERVE_BYTES;
        var plan = ResidencyPlanner.plan(
                artifact(), PROFILE, resident - 100 * MIB, 4096, Long.MAX_VALUE, KvFormat.NVFP4, 1, false);
        assertTrue(plan.fits());
        assertFalse(plan.hostBacked().contains(HostWeightSelection.OUTPUT_HEAD));
        assertTrue(HostWeightSelection.select(artifact(), 1L << 40).containsAll(plan.hostBacked()));
    }

    @Test
    void moreSequencesKeepFewerWeightsOnTheDevice() {
        long free = 3000L * MIB;
        var one = ResidencyPlanner.plan(artifact(), PROFILE, free, 4096, Long.MAX_VALUE, KvFormat.NVFP4, 1);
        var four = ResidencyPlanner.plan(artifact(), PROFILE, free, 4096, Long.MAX_VALUE, KvFormat.NVFP4, 4);
        assertTrue(one.fits());
        assertTrue(four.fits());
        assertTrue(four.hostBacked().size() >= one.hostBacked().size());
        long kv = ResidencyPlanner.kvBytes(config(2), 4096);
        assertTrue(four.deviceBytes() >= one.deviceBytes() + 3 * kv);
        assertEquals(
                one.deviceBytes(),
                ResidencyPlanner.plan(artifact(), PROFILE, free, 4096, Long.MAX_VALUE, KvFormat.NVFP4)
                        .deviceBytes());
    }

    @Test
    void autoSequencesNeverOffloadMoreThanOneSequenceDoes() {
        long free = 3000L * MIB;
        var artifact = artifact();
        var one = ResidencyPlanner.plan(artifact, PROFILE, free, 4096, Long.MAX_VALUE, KvFormat.NVFP4, 1);
        assertFalse(one.hostBacked().isEmpty());
        int chosen =
                ResidencyPlanner.sequencesFor(artifact, PROFILE, free, 4096, Long.MAX_VALUE, KvFormat.NVFP4, 1_000_000);
        assertTrue(chosen > 1);
        var atChosen = ResidencyPlanner.plan(artifact, PROFILE, free, 4096, Long.MAX_VALUE, KvFormat.NVFP4, chosen);
        var beyond = ResidencyPlanner.plan(artifact, PROFILE, free, 4096, Long.MAX_VALUE, KvFormat.NVFP4, chosen + 1);
        assertTrue(atChosen.fits());
        assertEquals(one.hostBacked(), atChosen.hostBacked());
        assertTrue(!beyond.fits() || !beyond.hostBacked().equals(one.hostBacked()));
        assertEquals(
                7,
                ResidencyPlanner.sequencesFor(artifact, PROFILE, 64L << 30, 4096, Long.MAX_VALUE, KvFormat.NVFP4, 7));
        assertEquals(
                1, ResidencyPlanner.sequencesFor(artifact, PROFILE, 1L << 30, 4096, Long.MAX_VALUE, KvFormat.NVFP4, 7));
    }

    private static final long WEIGHTS = (400 + 600 + 8 * 110 + 50 + 70) * MIB;
    private static final int LONG_CONTEXT = 262144;
    private static final int DEVICE_TOKENS = 4096;

    private static ResidencyPlanner.Plan spillPlan(long free, long host, int sequences) {
        return ResidencyPlanner.plan(
                artifact(), PROFILE, free, LONG_CONTEXT, host, KvFormat.NVFP4, sequences, false, DEVICE_TOKENS);
    }

    /// Device bytes of a plan whose caches keep `DEVICE_TOKENS` on the device and one region of host pages.
    private static long deviceShare(int sequences) {
        long region = (LONG_CONTEXT - DEVICE_TOKENS) / 256L * ResidencyPlanner.pageBytes(config(2), KvFormat.NVFP4);
        return sequences * (ResidencyPlanner.kvBytes(config(2), DEVICE_TOKENS) + region);
    }

    @Test
    void noCachePageLeavesTheDeviceWhenEverythingFits() {
        var plan = spillPlan(64L << 30, Long.MAX_VALUE, 1);
        assertTrue(plan.fits());
        assertEquals(0, plan.kvHostPages());
        assertEquals(0, plan.kvHostBytes());
        assertTrue(plan.hostBacked().isEmpty());
    }

    @Test
    void cachePagesPastTheDeviceMinimumLeaveBeforeAnyWeight() {
        long free = WEIGHTS + ResidencyPlanner.RESERVE_BYTES + deviceShare(1) + MIB;
        assertTrue(free < WEIGHTS + ResidencyPlanner.RESERVE_BYTES + ResidencyPlanner.kvBytes(config(2), LONG_CONTEXT));
        var plan = spillPlan(free, Long.MAX_VALUE, 1);
        assertTrue(plan.fits());
        assertTrue(plan.hostBacked().isEmpty());
        int hostPages = (LONG_CONTEXT - DEVICE_TOKENS) / 256;
        assertEquals(hostPages, plan.kvHostPages());
        // Each of the 2 attention layers keeps its host pages; one device region holds one layer's.
        assertEquals(2L * hostPages * ResidencyPlanner.pageBytes(config(2), KvFormat.NVFP4), plan.kvHostBytes());
        assertEquals(WEIGHTS + ResidencyPlanner.RESERVE_BYTES + deviceShare(1), plan.deviceBytes());
    }

    @Test
    void weightsLeaveOnlyWhenTheDeviceShareStillDoesNotFit() {
        long free = WEIGHTS + ResidencyPlanner.RESERVE_BYTES + deviceShare(1) - 200 * MIB;
        var plan = spillPlan(free, Long.MAX_VALUE, 1);
        assertTrue(plan.fits());
        assertTrue(plan.kvHostPages() > 0);
        assertFalse(plan.hostBacked().isEmpty());
        assertTrue(plan.deviceBytes() <= free);
    }

    @Test
    void aHostBudgetThatCannotHoldThePagesKeepsTheCacheOnTheDevice() {
        long free = WEIGHTS + ResidencyPlanner.RESERVE_BYTES + deviceShare(1) + MIB;
        var plan = spillPlan(free, 64 * MIB, 1);
        assertEquals(0, plan.kvHostPages());
        assertFalse(plan.hostBacked().isEmpty());
    }

    @Test
    void moreSequencesFitBesideResidentWeightsWithHostPages() {
        long free = WEIGHTS + ResidencyPlanner.RESERVE_BYTES + 3 * deviceShare(1) + 64 * MIB;
        var artifact = artifact();
        int allDevice = ResidencyPlanner.sequencesFor(
                artifact, PROFILE, free, LONG_CONTEXT, Long.MAX_VALUE, KvFormat.NVFP4, 16);
        int spilled = ResidencyPlanner.sequencesFor(
                artifact, PROFILE, free, LONG_CONTEXT, Long.MAX_VALUE, KvFormat.NVFP4, 16, DEVICE_TOKENS);
        assertTrue(spilled > allDevice, spilled + " sequences with host pages, " + allDevice + " without");
        assertTrue(spillPlan(free, Long.MAX_VALUE, spilled).hostBacked().isEmpty());
    }

    @Test
    void sequencesBeyondWhatFitAreReported() {
        assertFalse(ResidencyPlanner.plan(artifact(), PROFILE, 3000L * MIB, 4096, Long.MAX_VALUE, KvFormat.NVFP4, 4096)
                .fits());
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
