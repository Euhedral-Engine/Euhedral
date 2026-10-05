package io.euhedral_execution.inference.core.model_loader.qwen4;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/// The host memory a load may pin. Pinned (page-locked) memory is taken from system memory that is available
/// without evicting anything the machine needs, less a margin.
public record HostBudget(long pinnableBytes) {

    private static final long MARGIN_FLOOR = 4L << 30;

    public HostBudget {
        if (pinnableBytes < 0) throw new IllegalArgumentException("pinnableBytes must not be negative");
    }

    /// A budget from the system's available memory: 90% of it, but never closer than 4 GiB to exhaustion.
    public static HostBudget ofAvailable(long availableBytes) {
        long margin = Math.max(MARGIN_FLOOR, availableBytes / 10);
        return new HostBudget(Math.max(0, availableBytes - margin));
    }

    /// The budget of this machine, from `MemAvailable` in `/proc/meminfo`; unlimited when it cannot be read.
    public static HostBudget system() {
        try {
            for (String line : Files.readAllLines(Path.of("/proc/meminfo"))) {
                if (!line.startsWith("MemAvailable:")) continue;
                String[] fields = line.trim().split("\\s+");
                return ofAvailable(Long.parseLong(fields[1]) * 1024L);
            }
        } catch (IOException | RuntimeException unreadable) {
            // fall through: not Linux, or no /proc
        }
        return new HostBudget(Long.MAX_VALUE / 4);
    }
}
