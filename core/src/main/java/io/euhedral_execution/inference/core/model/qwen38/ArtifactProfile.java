package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.artifact.format.LinearFormat;
import io.euhedral_execution.inference.core.artifact.format.LinearFormats;
import io.euhedral_execution.inference.core.artifact.format.TrellisFormat;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Inventory;
import io.euhedral_execution.inference.core.model.qwen38.speculative.DraftLength;
import io.euhedral_execution.inference.core.state.KvFormat;
import java.util.EnumSet;
import java.util.Locale;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.Set;

/// What an artifact is: its quantization, whether its projections are stored compressed, and which drafter it
/// carries for speculative decoding. The engine reads this from the artifact's tensors and derives its whole
/// execution policy from it, so the user's only choices are the artifact files themselves.
///
/// - Q3: Q3G64_F16S projections. Compressed: P2E2 (lossless).
/// - NVFP4: block-scaled NVFP4 projections. Compressed: SD4 scale tables.
/// - BF16: the unquantized model, every object in BF16; its attention caches are BF16 too.
/// - TRELLIS: trellis-coded projections at 1 to 8 bits per weight; its attention caches are NVFP4.
/// - MIXED: text-layer projections of more than one of the above, each object in its own format; its attention
///   caches are NVFP4. It stores the MTP draft depth it was built for.
/// - Speculation: DFlash2 when the artifact holds the DFlash2 drafter (`dflash2/`), else MTP when it holds the MTP
///   layer and draft head, else none.
public record ArtifactProfile(
        Quantization quantization, boolean compressed, Speculation speculation, OptionalInt draftDepth) {

    public enum Quantization {
        Q3,
        NVFP4,
        /// The unquantized model: every object in BF16.
        BF16,
        /// Trellis-coded (EXL3) projections with their scale vectors.
        TRELLIS,
        /// Projections of more than one quantization, each object routed by its own format.
        MIXED
    }

    /// The speculative decoding strategy an artifact's drafter selects.
    public enum Speculation {
        NONE,
        MTP,
        DFLASH2
    }

    public ArtifactProfile {
        Objects.requireNonNull(draftDepth, "draftDepth");
    }

    /// A profile whose artifact stores no draft depth.
    public ArtifactProfile(Quantization quantization, boolean compressed, Speculation speculation) {
        this(quantization, compressed, speculation, OptionalInt.empty());
    }

    public ArtifactProfile(Quantization quantization, boolean compressed, boolean speculative) {
        this(quantization, compressed, speculative ? Speculation.MTP : Speculation.NONE);
    }

    public boolean speculative() {
        return this.speculation != Speculation.NONE;
    }

    /// Drafts per verification: [#draftLength]'s most for MTP; DFlash2 verifies the first 6 of each block's 7 drafts
    /// (docs/speculation/DFLASH2.md).
    public int speculativeDepth() {
        return switch (this.speculation) {
            case NONE -> 0;
            case MTP -> draftLength().most();
            case DFLASH2 -> 6;
        };
    }

    /// How many tokens an MTP step drafts: the depth the artifact stores when it stores one; else the depth that
    /// measured fastest for a pure artifact (docs/speculation/MTP_VERIFIER.md). A mixed artifact has no such
    /// measurement and
    /// must store its depth. A Q3 verifier row
    /// costs more than an NVFP4 one, whose decode kernels take 8 rows for little more than one, so Q3 drafts less
    /// deep. Lengths gated on the drafts' confidence measured no faster (docs/speculation/MTP_DRAFT_CONFIDENCE.md).
    /// BF16 drafts
    /// as deep as NVFP4: its tensor-core verifier rows cost about what one row does.
    public DraftLength draftLength() {
        if (this.speculation != Speculation.MTP)
            throw new IllegalStateException("the artifact does not draft with MTP");
        if (this.draftDepth.isPresent()) return DraftLength.fixed(this.draftDepth.getAsInt());
        if (this.quantization == Quantization.MIXED)
            throw new IllegalStateException("mixed artifact stores no MTP draft depth");
        return DraftLength.fixed(this.quantization == Quantization.Q3 ? 2 : 4);
    }

    /// The attention cache format the artifact's sequences take unless the engine selects one: BF16 for the
    /// unquantized model, NVFP4 for every quantized one.
    public KvFormat kvFormat() {
        return this.quantization == Quantization.BF16 ? KvFormat.BF16 : KvFormat.NVFP4;
    }

    /// `selected`, or when it is null the cache format of an artifact with `profile` (null: an unprofiled artifact,
    /// NVFP4).
    public static KvFormat kvFormat(KvFormat selected, ArtifactProfile profile) {
        if (selected != null) return selected;
        return profile == null ? KvFormat.NVFP4 : profile.kvFormat();
    }

    /// Reads the profile from the text layers' projections and from the presence of the MTP layer and draft head.
    ///
    /// The projections are every object under `text/layers/` with a registered linear format, but for the norms and
    /// small vectors outside the registry and the trellis companions. The BF16 ones are ignored when any other
    /// quantization is present (the small GDN projections stay BF16 in every quantized artifact). The embedding,
    /// the heads and the MTP layer do not decide the profile: a Q3 artifact keeps its MTP layer in NVFP4. When the
    /// projections are of one quantization, a text layer's FFN gate-up projection gives its layout (compressed q3
    /// keeps its down projections row-split); when they are of several, the artifact is [Quantization#MIXED],
    /// uncompressed. The artifact's draft depth, when it stores one, is the profile's.
    public static ArtifactProfile of(Artifact artifact) {
        Set<Quantization> projections = EnumSet.noneOf(Quantization.class);
        for (TensorDescriptor tensor : artifact.tensors()) {
            String name = tensor.name();
            if (!name.startsWith("text/layers/") || isCompanion(name)) continue;
            LinearFormats.find(tensor.format(), tensor.layout()).ifPresent(f -> projections.add(f.quantization()));
        }
        if (projections.size() > 1) projections.remove(Quantization.BF16);
        OptionalInt draftDepth = artifact.config() == null
                ? OptionalInt.empty()
                : artifact.config().draftDepth();
        TensorDescriptor probe = null;
        boolean mtp = false;
        boolean draftHead = false;
        boolean dflash2 = false;
        for (TensorDescriptor tensor : artifact.tensors()) {
            String name = tensor.name();
            if (probe == null && name.startsWith("text/layers/") && name.endsWith("/mlp/gate_up")) probe = tensor;
            if (name.startsWith("mtp/")) mtp = true;
            if (name.startsWith("text/draft_head")) draftHead = true;
            if (name.startsWith(DFlash2Inventory.PREFIX)) dflash2 = true;
        }
        if (probe == null) throw new IllegalArgumentException("artifact has no text layer FFN projection");
        Speculation speculation = dflash2 ? Speculation.DFLASH2 : mtp && draftHead ? Speculation.MTP : Speculation.NONE;
        if (projections.size() > 1) return new ArtifactProfile(Quantization.MIXED, false, speculation, draftDepth);
        Quantization family =
                switch (probe.format()) {
                    case Q3_G64_FP16 -> Quantization.Q3;
                    case NVFP4 -> Quantization.NVFP4;
                    case BF16 -> Quantization.BF16;
                    case TRELLIS -> Quantization.TRELLIS;
                    default ->
                        throw new IllegalArgumentException("unsupported projection format " + probe.format()
                                + "; expected Q3, NVFP4, BF16 or TRELLIS");
                };
        WeightLayout layout = probe.layout();
        LinearFormat format = LinearFormats.find(probe.format(), layout)
                .orElseThrow(() -> new IllegalArgumentException("layout " + layout + " does not belong to " + family));
        Quantization quantization = format.quantization();
        boolean compressed = format.compressed();
        return new ArtifactProfile(quantization, compressed, speculation, draftDepth);
    }

    private static boolean isCompanion(String name) {
        return name.endsWith(TrellisFormat.SUH)
                || name.endsWith(TrellisFormat.SVH)
                || name.endsWith(TrellisFormat.SEGMENTS);
    }

    /// The artifact's name in logs and run records: `q3`, `q3-compressed`, `nvfp4`, `nvfp4-compressed`, `bf16`,
    /// `trellis`, `mixed`, with `-dflash2` when it drafts with DFlash2.
    public String artifactName() {
        return this.quantization.name().toLowerCase(Locale.ROOT)
                + (this.compressed ? "-compressed" : "")
                + (this.speculation == Speculation.DFLASH2 ? "-dflash2" : "");
    }
}
