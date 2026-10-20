package io.euhedral_execution.inference.core.settings;

import io.euhedral_execution.inference.core.ProcessorTopology;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
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

    /// The processors workers run on: the configured ones, or, when none are configured, every available one the
    /// process's CPU quota pays for ([ProcessorTopology#withinQuota]).
    public BitSet selectedCpus() {
        return cpus.isEmpty() ? ProcessorTopology.system().withinQuota().processorIds() : parseCpus(cpus.get());
    }

    /// Comma-separated processor IDs and inclusive ranges, such as `2,3,8-11`.
    static BitSet parseCpus(String specification) {
        BitSet cpus = new BitSet();
        for (String part : specification.split(",")) {
            String item = part.strip();
            if (item.isEmpty()) continue;
            try {
                int dash = item.indexOf('-');
                int first = Integer.parseInt((dash < 0 ? item : item.substring(0, dash)).strip());
                int last = dash < 0
                        ? first
                        : Integer.parseInt(item.substring(dash + 1).strip());
                if (first < 0 || last < first) throw new NumberFormatException();
                cpus.set(first, last + 1);
            } catch (NumberFormatException invalid) {
                throw new IllegalArgumentException("cpus has an invalid processor ID or range: " + item);
            }
        }
        if (cpus.isEmpty()) throw new IllegalArgumentException("cpus selects no processors");
        return cpus;
    }

    public static HostSettings defaults() {
        return new HostSettings(Optional.empty(), 8, Duration.ofSeconds(1), Optional.empty());
    }
}
