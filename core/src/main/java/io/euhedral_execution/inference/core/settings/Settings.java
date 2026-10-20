package io.euhedral_execution.inference.core.settings;

import java.util.Objects;

/// Every deployer setting, read once at startup by {@link SettingsLoader}.
///
/// @param model the model to serve
/// @param server the HTTP server
/// @param context the context a sequence may hold
/// @param prefixCache the prompt prefix cache
/// @param batching how sequences share passes
/// @param memory memory limits
/// @param host CPUs and native files
/// @param kernels kernel thresholds
public record Settings(
        ModelSettings model,
        ServerSettings server,
        ContextSettings context,
        PrefixCacheSettings prefixCache,
        BatchingSettings batching,
        MemorySettings memory,
        HostSettings host,
        KernelSettings kernels) {

    public Settings {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(server, "server");
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(prefixCache, "prefixCache");
        Objects.requireNonNull(batching, "batching");
        Objects.requireNonNull(memory, "memory");
        Objects.requireNonNull(host, "host");
        Objects.requireNonNull(kernels, "kernels");
    }

    /// The engine tuning part of the settings.
    public Tuning tuning() {
        return new Tuning(batching, memory, host, kernels);
    }
}
