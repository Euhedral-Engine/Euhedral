package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.util.Locale;

/// What an artifact is: its quantization, whether its projections are stored compressed, and whether it
/// carries the MTP layer and draft head. The engine reads this from the artifact's tensors and derives
/// its whole execution policy from it, so the user's only choices are the artifact files themselves.
///
/// - Q3: Q3G64_F16S projections. Compressed: P2E2 (lossless).
/// - NVFP4: block-scaled NVFP4 projections. Compressed: SD4 scale tables.
public record ArtifactProfile(Quantization quantization, boolean compressed, boolean speculative) {

    public enum Quantization {
        Q3,
        NVFP4
    }

    /// Drafts per verification: the depth that measured fastest for the artifact's quantization (a Q3
    /// verifier step costs more relative to a draft, so its break-even depth is lower).
    public int speculativeDepth() {
        if (!this.speculative) return 0;
        return this.quantization == Quantization.Q3 ? 2 : 3;
    }

    /// Reads the profile from a text layer's FFN down projection, which every artifact stores in its
    /// quantization, and from the presence of the MTP layer and draft head.
    public static ArtifactProfile of(QwenArtifact artifact) {
        TensorDescriptor probe = null;
        boolean mtp = false;
        boolean draftHead = false;
        for (TensorDescriptor tensor : artifact.tensors()) {
            String name = tensor.name();
            if (probe == null && name.startsWith("text/layers/") && name.endsWith("/mlp/down")) probe = tensor;
            if (name.startsWith("mtp/")) mtp = true;
            if (name.startsWith("text/draft_head")) draftHead = true;
        }
        if (probe == null) throw new IllegalArgumentException("artifact has no text layer FFN projection");
        Quantization quantization =
                switch (probe.format()) {
                    case Q3_G64_FP16 -> Quantization.Q3;
                    case NVFP4 -> Quantization.NVFP4;
                    default ->
                        throw new IllegalArgumentException(
                                "unsupported projection format " + probe.format() + "; expected Q3 or NVFP4");
                };
        WeightLayout layout = probe.layout();
        if (quantization == Quantization.Q3 && layout == WeightLayout.ROW_SPLIT_K128_SD4_V1
                || quantization == Quantization.NVFP4 && layout == WeightLayout.ROW_SPLIT_P2E2_V1)
            throw new IllegalArgumentException("layout " + layout + " does not belong to " + quantization);
        boolean compressed = layout == WeightLayout.ROW_SPLIT_P2E2_V1 || layout == WeightLayout.ROW_SPLIT_K128_SD4_V1;
        return new ArtifactProfile(quantization, compressed, mtp && draftHead);
    }

    /// The artifact's name in logs and run records: `q3`, `q3-compressed`, `nvfp4`, `nvfp4-compressed`.
    public String artifactName() {
        return this.quantization.name().toLowerCase(Locale.ROOT) + (this.compressed ? "-compressed" : "");
    }
}
