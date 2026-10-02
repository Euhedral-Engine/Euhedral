package io.euhedral_execution.inference.core.model_loader;

/// Which artifact objects a load places on the GPU.
public enum WeightResidency {
    /// Every object of the artifact.
    ALL,
    /// Only the objects text generation executes: the vision tower, the MTP layer and the draft head
    /// stay in the artifact until a route uses them.
    EXECUTED;

    /// Whether a compact runtime object is uploaded under this residency.
    public boolean uploads(String name) {
        return this == ALL
                || !(name.startsWith("vision/") || name.startsWith("mtp/") || name.startsWith("text/draft_head"));
    }
}
