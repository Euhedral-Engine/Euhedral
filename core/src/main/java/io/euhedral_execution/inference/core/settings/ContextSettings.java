package io.euhedral_execution.inference.core.settings;

import io.euhedral_execution.inference.core.state.KvFormat;
import java.util.Objects;
import java.util.Optional;

/// The context a sequence may hold.
///
/// @param maxTokens longest context of one sequence
/// @param kvCacheFormat the attention cache format, or empty for the artifact's
/// @param kvCacheOnGpuTokens tokens of a sequence whose attention cache lives on the GPU before it spills to the host
public record ContextSettings(int maxTokens, Optional<KvFormat> kvCacheFormat, int kvCacheOnGpuTokens) {

    public ContextSettings {
        if (maxTokens < 1) throw new IllegalArgumentException("max-tokens must be positive");
        Objects.requireNonNull(kvCacheFormat, "kvCacheFormat");
        if (kvCacheOnGpuTokens < 0) throw new IllegalArgumentException("kv-cache-on-gpu-tokens must not be negative");
    }

    public static ContextSettings defaults() {
        return new ContextSettings(32768, Optional.empty(), 65536);
    }
}
