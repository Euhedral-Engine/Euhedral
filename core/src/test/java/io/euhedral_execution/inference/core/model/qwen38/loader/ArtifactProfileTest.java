package io.euhedral_execution.inference.core.model.qwen38.loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
import io.euhedral_execution.inference.core.model.qwen38.speculative.DraftLength;
import io.euhedral_execution.inference.core.state.KvFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;

class ArtifactProfileTest {
    private static TensorDescriptor tensor(String name, WeightFormat format, WeightLayout layout) {
        return new TensorDescriptor(name, new long[] {1, 1}, TensorDataType.BF16, format, layout, 0, 1);
    }

    private static Artifact artifact(WeightFormat format, WeightLayout layout, boolean speculative) {
        List<TensorDescriptor> tensors = new ArrayList<>();
        tensors.add(tensor("text/token_embedding", WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_K128_V1));
        tensors.add(tensor("text/layers/0/mlp/gate_up", format, layout));
        if (speculative) {
            tensors.add(tensor("mtp/input_projection", WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1));
            tensors.add(tensor("text/draft_head", WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1));
        }
        return new Artifact(null, null, tensors.toArray(TensorDescriptor[]::new));
    }

    @Test
    void theFourArtifactsAreRecognizedFromTheirProjections() {
        var q3 = ArtifactProfile.of(artifact(WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_K128_V1, true));
        assertEquals(new ArtifactProfile(Quantization.Q3, false, true), q3);
        assertEquals("q3", q3.artifactName());
        var q3c = ArtifactProfile.of(artifact(WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_P2E2_V1, true));
        assertEquals("q3-compressed", q3c.artifactName());
        var nvfp4 = ArtifactProfile.of(artifact(WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1, true));
        assertEquals("nvfp4", nvfp4.artifactName());
        var sd4 = ArtifactProfile.of(artifact(WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_SD4_V1, true));
        assertEquals("nvfp4-compressed", sd4.artifactName());
        assertTrue(sd4.compressed());
    }

    @Test
    void theUnquantizedModelIsRecognizedAndKeepsABf16Cache() {
        var bf16 = ArtifactProfile.of(artifact(WeightFormat.BF16, WeightLayout.CONTIGUOUS_LE_V1, true));
        assertEquals(new ArtifactProfile(Quantization.BF16, false, true), bf16);
        assertEquals("bf16", bf16.artifactName());
        assertEquals(io.euhedral_execution.inference.core.state.KvFormat.BF16, bf16.kvFormat());
        assertEquals(DraftLength.fixed(4), bf16.draftLength());
        assertEquals(
                io.euhedral_execution.inference.core.state.KvFormat.NVFP4,
                ArtifactProfile.of(artifact(WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1, true))
                        .kvFormat());
        org.junit.jupiter.api.Assertions.assertThrows(
                IllegalArgumentException.class,
                () -> ArtifactProfile.of(artifact(WeightFormat.BF16, WeightLayout.ROW_SPLIT_K128_V1, true)));
    }

    @Test
    void speculativeDepthFollowsTheQuantization() {
        assertEquals(2, new ArtifactProfile(Quantization.Q3, false, true).speculativeDepth());
        assertEquals(2, new ArtifactProfile(Quantization.Q3, true, true).speculativeDepth());
        assertEquals(4, new ArtifactProfile(Quantization.NVFP4, false, true).speculativeDepth());
        assertEquals(4, new ArtifactProfile(Quantization.NVFP4, true, true).speculativeDepth());
        assertEquals(0, new ArtifactProfile(Quantization.NVFP4, false, false).speculativeDepth());
        assertEquals(DraftLength.fixed(2), new ArtifactProfile(Quantization.Q3, true, true).draftLength());
        assertEquals(DraftLength.fixed(4), new ArtifactProfile(Quantization.NVFP4, true, true).draftLength());
    }

    @Test
    void anArtifactWithoutTheMtpLayerAndDraftHeadDoesNotSpeculate() {
        var profile = ArtifactProfile.of(artifact(WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_K128_V1, false));
        assertFalse(profile.speculative());
    }

    @Test
    void anArtifactWithTheDFlash2DrafterDraftsWithDFlash2() {
        List<TensorDescriptor> tensors =
                new ArrayList<>(List.of(artifact(WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_SD4_V1, true)
                        .tensors()));
        tensors.add(tensor("dflash2/config", WeightFormat.I32, WeightLayout.CONTIGUOUS_LE_V1));
        var profile = ArtifactProfile.of(new Artifact(null, null, tensors.toArray(TensorDescriptor[]::new)));
        assertEquals(ArtifactProfile.Speculation.DFLASH2, profile.speculation());
        assertEquals(6, profile.speculativeDepth());
        assertEquals("nvfp4-compressed-dflash2", profile.artifactName());
        assertTrue(CompactWeightLoader.uploads("dflash2/fc", ArtifactProfile.Speculation.DFLASH2));
        assertFalse(CompactWeightLoader.uploads("dflash2/config", ArtifactProfile.Speculation.DFLASH2));
        assertFalse(CompactWeightLoader.uploads("mtp/input_projection", ArtifactProfile.Speculation.DFLASH2));
        assertFalse(CompactWeightLoader.uploads("dflash2/fc", ArtifactProfile.Speculation.MTP));
        assertTrue(CompactWeightLoader.uploads("text/draft_head", ArtifactProfile.Speculation.MTP));
        assertTrue(CompactWeightLoader.uploads("text/output_head", ArtifactProfile.Speculation.NONE));
    }

    @Test
    void unsupportedArtifactsAreRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> ArtifactProfile.of(artifact(WeightFormat.FP16, WeightLayout.CONTIGUOUS_LE_V1, false)));
        assertThrows(
                IllegalArgumentException.class,
                () -> ArtifactProfile.of(artifact(WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_P2E2_V1, false)));
        assertThrows(
                IllegalArgumentException.class,
                () -> ArtifactProfile.of(new Artifact(null, null, new TensorDescriptor[0])));
    }

    @Test
    void theCacheFormatIsTheArtifactsUnlessSelected() {
        var profile = new ArtifactProfile(Quantization.Q3, false, ArtifactProfile.Speculation.MTP);
        assertEquals(KvFormat.NVFP4, profile.kvFormat());
        assertEquals(KvFormat.NVFP4, ArtifactProfile.kvFormat(null, profile));
        assertEquals(KvFormat.NVFP4, ArtifactProfile.kvFormat(null, null));
        assertEquals(KvFormat.BF16, ArtifactProfile.kvFormat(KvFormat.BF16, profile));
        assertEquals(KvFormat.BF16, ArtifactProfile.kvFormat(KvFormat.BF16, null));
    }

    private static final WeightFormat TRELLIS = WeightFormat.TRELLIS;
    private static final WeightLayout TILE = WeightLayout.TRELLIS_TILE16_V1;

    /// A text layer's projections in the given formats, the way the stored artifacts hold them: the norms and the
    /// small GDN objects stay BF16, the embedding and heads have their own formats, and the MTP layer is NVFP4
    /// whatever the text layers are.
    private static Artifact layers(
            WeightFormat gateUp,
            WeightLayout gateUpLayout,
            WeightFormat down,
            WeightLayout downLayout,
            OptionalInt draftDepth) {
        List<TensorDescriptor> tensors = new ArrayList<>();
        tensors.add(tensor("text/token_embedding", WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_K128_V1));
        tensors.add(tensor("text/output_head", WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1));
        tensors.add(tensor("text/draft_head", WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1));
        for (int layer = 0; layer < 2; layer++) {
            String prefix = "text/layers/" + layer;
            tensors.add(tensor(prefix + "/input_norm", WeightFormat.BF16, WeightLayout.CONTIGUOUS_LE_V1));
            tensors.add(tensor(prefix + "/gdn/a_projection", WeightFormat.BF16, WeightLayout.CONTIGUOUS_LE_V1));
            tensors.add(tensor(prefix + "/gdn/a_log", WeightFormat.FP32, WeightLayout.CONTIGUOUS_LE_V1));
            tensors.add(tensor(prefix + "/mlp/gate_up", gateUp, gateUpLayout));
            tensors.add(tensor(prefix + "/mlp/down", down, downLayout));
            if (gateUp == TRELLIS || down == TRELLIS) {
                tensors.add(tensor(prefix + "/mlp/down.suh", WeightFormat.FP16, WeightLayout.CONTIGUOUS_LE_V1));
            }
        }
        tensors.add(tensor("mtp/input_projection", WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1));
        tensors.add(tensor("mtp/layer/mlp/gate_up", WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1));
        Qwen38Config config = draftDepth.isPresent() ? config().withDraftDepth(draftDepth.getAsInt()) : config();
        return new Artifact(null, config, tensors.toArray(TensorDescriptor[]::new));
    }

    private static Qwen38Config config() {
        return new Qwen38Config(
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1.0e-6,
                1.0,
                1.0,
                1,
                "silu",
                new LayerType[] {LayerType.FULL_ATTENTION},
                0,
                0,
                0,
                0,
                false,
                false,
                1);
    }

    private static Artifact pure(WeightFormat format, WeightLayout layout) {
        return layers(format, layout, format, layout, OptionalInt.empty());
    }

    @Test
    void oneQuantizationAcrossTheProjectionsIsTheProfileItWasBefore() {
        assertEquals(
                new ArtifactProfile(Quantization.NVFP4, false, true),
                ArtifactProfile.of(pure(WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1)));
        assertEquals(new ArtifactProfile(Quantization.TRELLIS, false, true), ArtifactProfile.of(pure(TRELLIS, TILE)));
        var sd4 = ArtifactProfile.of(pure(WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_SD4_V1));
        assertEquals(Quantization.NVFP4, sd4.quantization());
        assertTrue(sd4.compressed());
        assertEquals("nvfp4-compressed", sd4.artifactName());
        // Compressed Q3 keeps its down projections row-split; Q4 and Q5 widths are still Q3.
        var q3c = ArtifactProfile.of(layers(
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_P2E2_V1,
                WeightFormat.Q4_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1,
                OptionalInt.empty()));
        assertEquals(new ArtifactProfile(Quantization.Q3, true, true), q3c);
    }

    @Test
    void theHeadsTheEmbeddingTheSmallObjectsAndTheMtpLayerDoNotDecideTheProfile() {
        // Trellis projections beside NVFP4 heads, a Q3 embedding, BF16 small projections and an NVFP4 MTP layer.
        var profile = ArtifactProfile.of(pure(TRELLIS, TILE));
        assertEquals(Quantization.TRELLIS, profile.quantization());
        assertFalse(profile.compressed());
        assertEquals(ArtifactProfile.Speculation.MTP, profile.speculation());
    }

    @Test
    void trellisBesideNvfp4ProjectionsIsMixed() {
        var profile = ArtifactProfile.of(
                layers(TRELLIS, TILE, WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1, OptionalInt.empty()));
        assertEquals(Quantization.MIXED, profile.quantization());
        assertFalse(profile.compressed());
        assertEquals("mixed", profile.artifactName());
        assertEquals(KvFormat.NVFP4, profile.kvFormat());
        assertEquals(ArtifactProfile.Speculation.MTP, profile.speculation());
    }

    @Test
    void aMixedArtifactWithAnSd4ProjectionIsNotCompressed() {
        var profile = ArtifactProfile.of(
                layers(WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_SD4_V1, TRELLIS, TILE, OptionalInt.empty()));
        assertEquals(Quantization.MIXED, profile.quantization());
        assertFalse(profile.compressed());
    }

    @Test
    void aStoredDraftDepthIsTheDraftLength() {
        var mixed = ArtifactProfile.of(
                layers(TRELLIS, TILE, WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1, OptionalInt.of(3)));
        assertEquals(DraftLength.fixed(3), mixed.draftLength());
        assertEquals(3, mixed.speculativeDepth());
        // A pure artifact that stores one drafts that deep, whatever the policy would have chosen.
        var q3 = ArtifactProfile.of(layers(
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1,
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1,
                OptionalInt.of(5)));
        assertEquals(DraftLength.fixed(5), q3.draftLength());
    }

    @Test
    void aMixedArtifactWithoutADraftDepthIsRefusedForMtp() {
        var mixed = ArtifactProfile.of(
                layers(TRELLIS, TILE, WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1, OptionalInt.empty()));
        var failure = assertThrows(IllegalStateException.class, mixed::draftLength);
        assertEquals("mixed artifact stores no MTP draft depth", failure.getMessage());
        assertThrows(IllegalStateException.class, mixed::speculativeDepth);
    }

    @Test
    void aPureArtifactWithoutADraftDepthKeepsThePolicy() {
        assertEquals(
                DraftLength.fixed(2),
                ArtifactProfile.of(pure(WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_K128_V1))
                        .draftLength());
        assertEquals(
                DraftLength.fixed(4),
                ArtifactProfile.of(pure(WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1))
                        .draftLength());
        assertEquals(
                DraftLength.fixed(4), ArtifactProfile.of(pure(TRELLIS, TILE)).draftLength());
    }
}
