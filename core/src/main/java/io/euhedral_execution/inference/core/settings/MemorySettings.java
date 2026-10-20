package io.euhedral_execution.inference.core.settings;

import java.util.Objects;
import java.util.OptionalLong;

/// Memory the engine may use. Sizes are bytes; an empty limit is resolved at startup from the machine.
///
/// @param hostLimit host memory the engine may use
/// @param hostWeightsLimit pinned host memory for host-backed weights
/// @param gpuLimit device memory the residency plan may use
/// @param gpuReserve device memory left for other programs
/// @param pinnedBufferPool released pinned buffers kept for reuse
/// @param weightStreaming streaming of weights from disk
public record MemorySettings(
        OptionalLong hostLimit,
        OptionalLong hostWeightsLimit,
        OptionalLong gpuLimit,
        long gpuReserve,
        OptionalLong pinnedBufferPool,
        WeightStreamingSettings weightStreaming) {

    public MemorySettings {
        Objects.requireNonNull(hostLimit, "host-limit");
        if (hostLimit.isPresent() && hostLimit.getAsLong() < 0)
            throw new IllegalArgumentException("host-limit must not be negative");
        positive(hostWeightsLimit, "host-weights-limit");
        positive(gpuLimit, "gpu-limit");
        if (gpuReserve < 0) throw new IllegalArgumentException("gpu-reserve must not be negative");
        Objects.requireNonNull(pinnedBufferPool, "pinnedBufferPool");
        if (pinnedBufferPool.isPresent() && pinnedBufferPool.getAsLong() < 0)
            throw new IllegalArgumentException("pinned-buffer-pool must not be negative");
        Objects.requireNonNull(weightStreaming, "weightStreaming");
    }

    private static void positive(OptionalLong value, String key) {
        Objects.requireNonNull(value, key);
        if (value.isPresent() && value.getAsLong() < 1) throw new IllegalArgumentException(key + " must be positive");
    }

    public static MemorySettings defaults() {
        return new MemorySettings(
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                700L << 20,
                OptionalLong.empty(),
                WeightStreamingSettings.defaults());
    }
}
