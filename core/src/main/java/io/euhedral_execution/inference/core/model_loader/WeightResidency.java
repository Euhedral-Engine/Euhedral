package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;

/// Which artifact objects a load places on the GPU.
public enum WeightResidency {
    /// Every object of the artifact.
    ALL,
    /// Only the objects text generation executes: the vision tower, the MTP layer and the draft head
    /// stay in the artifact until a route uses them.
    EXECUTED,
    /// The executed objects plus the MTP layer and draft head, for speculative decoding: everything but
    /// the vision tower.
    SPECULATIVE;

    /// Whether a compact runtime object is uploaded under this residency.
    public boolean uploads(String name) {
        return switch (this) {
            case ALL -> true;
            case EXECUTED ->
                !(name.startsWith("vision/") || name.startsWith("mtp/") || name.startsWith("text/draft_head"));
            case SPECULATIVE -> !name.startsWith("vision/");
        };
    }

    /// Device bytes the artifact's uploaded objects occupy under this residency.
    public long residentBytes(QwenArtifact artifact) {
        long bytes = 0;
        for (TensorDescriptor tensor : artifact.tensors()) if (uploads(tensor.name())) bytes += tensor.byteSize();
        return bytes;
    }
}
