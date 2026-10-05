package io.euhedral_execution.inference.core.model_loader.qwen4;

/// What a load prepares for execution beyond the text model. The engine derives it from the generation strategy; it
/// is not a user setting. The MTP layer and the vision tower stay in the artifact, validated but unloaded, unless the
/// mode selects them.
public record Qwen4Mode(boolean mtp, boolean vision) {

    public static final Qwen4Mode TEXT = new Qwen4Mode(false, false);
}
