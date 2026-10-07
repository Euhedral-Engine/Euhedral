package io.euhedral_execution.inference.core;

/// What an engine reports about its model, whatever the model: the artifact's name, and the speculative strategy it
/// selects (`none`, `mtp`, `dflash2`) drafting `speculativeDepth` tokens per verification. `artifactName` is null
/// when the artifact was not profiled.
public record ModelDescription(String artifactName, String speculation, int speculativeDepth) {
    public static final ModelDescription NONE = new ModelDescription(null, "none", 0);

    public boolean speculative() {
        return !"none".equals(this.speculation);
    }
}
