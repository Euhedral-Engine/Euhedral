package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.util.Locale;

/// What an artifact is: its quantization, whether its projections are stored compressed, and which drafter it
/// carries for speculative decoding. The engine reads this from the artifact's tensors and derives its whole
/// execution policy from it, so the user's only choices are the artifact files themselves.
///
/// - Q3: Q3G64_F16S projections. Compressed: P2E2 (lossless).
/// - NVFP4: block-scaled NVFP4 projections. Compressed: SD4 scale tables.
/// - Speculation: DFlash2 when the artifact holds the DFlash2 drafter (`dflash2/`), else MTP when it holds the MTP
///   layer and draft head, else none.
public record ArtifactProfile(Quantization quantization, boolean compressed, Speculation speculation) {

    public enum Quantization {
        Q3,
        NVFP4
    }

    /// The speculative decoding strategy an artifact's drafter selects.
    public enum Speculation {
        NONE,
        MTP,
        DFLASH2
    }

    public ArtifactProfile(Quantization quantization, boolean compressed, boolean speculative) {
        this(quantization, compressed, speculative ? Speculation.MTP : Speculation.NONE);
    }

    public boolean speculative() {
        return this.speculation != Speculation.NONE;
    }

    /// Drafts per verification. MTP: the depth that measured fastest for the artifact's quantization
    /// (a Q3 verifier step costs more relative to a draft, so its break-even depth is lower). DFlash2:
    /// the first 5 of each block's 7 drafts, because the exact verifier's 7- and 8-row NVFP4 twins
    /// cost 15-30% more than its 6-row twin, which costs no more than 4 rows (docs/DFLASH2.md).
    public int speculativeDepth() {
        return switch (this.speculation) {
            case NONE -> 0;
            case MTP -> this.quantization == Quantization.Q3 ? 2 : 3;
            case DFLASH2 -> 5;
        };
    }

    /// Reads the profile from a text layer's FFN down projection, which every artifact stores in its
    /// quantization, and from the presence of the MTP layer and draft head.
    public static ArtifactProfile of(QwenArtifact artifact) {
        TensorDescriptor probe = null;
        boolean mtp = false;
        boolean draftHead = false;
        boolean dflash2 = false;
        for (TensorDescriptor tensor : artifact.tensors()) {
            String name = tensor.name();
            if (probe == null && name.startsWith("text/layers/") && name.endsWith("/mlp/down")) probe = tensor;
            if (name.startsWith("mtp/")) mtp = true;
            if (name.startsWith("text/draft_head")) draftHead = true;
            if (name.startsWith(DFlash2Inventory.PREFIX)) dflash2 = true;
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
        Speculation speculation = dflash2 ? Speculation.DFLASH2 : mtp && draftHead ? Speculation.MTP : Speculation.NONE;
        return new ArtifactProfile(quantization, compressed, speculation);
    }

    /// The artifact's name in logs and run records: `q3`, `q3-compressed`, `nvfp4`, `nvfp4-compressed`, with
    /// `-dflash2` when it drafts with DFlash2.
    public String artifactName() {
        return this.quantization.name().toLowerCase(Locale.ROOT)
                + (this.compressed ? "-compressed" : "")
                + (this.speculation == Speculation.DFLASH2 ? "-dflash2" : "");
    }
}
