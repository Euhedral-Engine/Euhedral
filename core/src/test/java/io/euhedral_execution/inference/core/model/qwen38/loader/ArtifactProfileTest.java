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
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class ArtifactProfileTest {
    private static TensorDescriptor tensor(String name, WeightFormat format, WeightLayout layout) {
        return new TensorDescriptor(name, new long[] {1, 1}, TensorDataType.BF16, format, layout, 0, 1);
    }

    private static Artifact artifact(WeightFormat format, WeightLayout layout, boolean speculative) {
        List<TensorDescriptor> tensors = new ArrayList<>();
        tensors.add(tensor("text/token_embedding", WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_K128_V1));
        tensors.add(tensor("text/layers/0/mlp/down", format, layout));
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
    void speculativeDepthFollowsTheQuantization() {
        assertEquals(2, new ArtifactProfile(Quantization.Q3, false, true).speculativeDepth());
        assertEquals(2, new ArtifactProfile(Quantization.Q3, true, true).speculativeDepth());
        assertEquals(4, new ArtifactProfile(Quantization.NVFP4, false, true).speculativeDepth());
        assertEquals(4, new ArtifactProfile(Quantization.NVFP4, true, true).speculativeDepth());
        assertEquals(0, new ArtifactProfile(Quantization.NVFP4, false, false).speculativeDepth());
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
                () -> ArtifactProfile.of(artifact(WeightFormat.BF16, WeightLayout.CONTIGUOUS_LE_V1, false)));
        assertThrows(
                IllegalArgumentException.class,
                () -> ArtifactProfile.of(artifact(WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_P2E2_V1, false)));
        assertThrows(
                IllegalArgumentException.class,
                () -> ArtifactProfile.of(new Artifact(null, null, new TensorDescriptor[0])));
    }
}
