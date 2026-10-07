package io.euhedral_execution.inference.core.model.qwen4.loader;

/// What a load prepares for execution beyond the text model. The engine derives it from the generation strategy; it
/// is not a user setting. The MTP layer and the vision tower stay in the artifact, validated but unloaded, unless the
/// mode selects them.
public record Mode(boolean mtp, boolean vision) {

    public static final Mode TEXT = new Mode(false, false);
}
