package io.euhedral_execution.inference.core.settings;

import java.util.Objects;

/// Kernel thresholds.
///
/// @param trellis the trellis linear kernels
public record KernelSettings(TrellisKernelSettings trellis) {

    public KernelSettings {
        Objects.requireNonNull(trellis, "trellis");
    }

    public static KernelSettings defaults() {
        return new KernelSettings(TrellisKernelSettings.defaults());
    }
}
