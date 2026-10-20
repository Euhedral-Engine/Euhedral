package io.euhedral_execution.inference.core.settings;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Objects;
import java.util.Optional;

/// The host CPUs and files the engine uses.
///
/// @param cpus the CPUs workers run on, such as `2-5,8`, or empty for all
/// @param taskParallelism workers taking host work at once
/// @param idleRelease idle time before workers are released; zero never releases them
/// @param nativeCache the directory the jar unpacks its native files into, or empty for the user's cache directory
public record HostSettings(
        Optional<String> cpus, int taskParallelism, Duration idleRelease, Optional<Path> nativeCache) {

    public HostSettings {
        Objects.requireNonNull(cpus, "cpus");
        if (cpus.isPresent() && cpus.get().isBlank()) throw new IllegalArgumentException("cpus must not be blank");
        if (taskParallelism < 1) throw new IllegalArgumentException("task-parallelism must be positive");
        Objects.requireNonNull(idleRelease, "idleRelease");
        if (idleRelease.isNegative()) throw new IllegalArgumentException("idle-release must not be negative");
        Objects.requireNonNull(nativeCache, "nativeCache");
    }

    public static HostSettings defaults() {
        return new HostSettings(Optional.empty(), 8, Duration.ofSeconds(1), Optional.empty());
    }
}
