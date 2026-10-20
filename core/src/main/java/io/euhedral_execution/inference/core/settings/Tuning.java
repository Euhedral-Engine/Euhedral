package io.euhedral_execution.inference.core.settings;

import java.util.Objects;

/// The settings that shape how the engine runs rather than what it serves.
///
/// @param batching how sequences share passes
/// @param memory memory limits
/// @param host CPUs and native files
/// @param kernels kernel thresholds
public record Tuning(BatchingSettings batching, MemorySettings memory, HostSettings host, KernelSettings kernels) {

    public Tuning {
        Objects.requireNonNull(batching, "batching");
        Objects.requireNonNull(memory, "memory");
        Objects.requireNonNull(host, "host");
        Objects.requireNonNull(kernels, "kernels");
    }

    public static Tuning defaults() {
        return new Tuning(
                BatchingSettings.defaults(),
                MemorySettings.defaults(),
                HostSettings.defaults(),
                KernelSettings.defaults());
    }
}
