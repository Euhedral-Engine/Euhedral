package io.euhedral_execution.inference.core.model_loader.qwen4;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/// The host memory a load may use, in the two forms the expert hierarchy needs.
///
/// - `pinnableBytes`: memory the runtime is willing to page-lock. Pinned memory is a small transfer tier: the staging
///   slots that DMA reads, host-mapped fixed objects, the gathered embedding.
/// - `residentBytes`: ordinary (pageable) memory the runtime may hold for itself, which is where the routed experts
///   live on the host. It is the memory the machine can spare, less room for the operating system, and it includes
///   what pinning takes: pinned memory is carved out of it by the plan, not added to it.
///
/// Both come from the same physical memory; the plan spends the pinned part first and gives the rest to the
/// expert tier.
///
/// The pinned budget is derived from the machine (`MemAvailable`, less room for the operating system): a
/// small tier does not need configuring. The ordinary-memory budget is never assumed: the engine does not
/// take memory the user did not offer. `EUHEDRAL_HOST_MEMORY_MIB` (or the system property
/// `euhedral.host.memory-mib`) states how much ordinary memory the engine may use for the expert tier; the
/// margin for the operating system is still taken from it. Unset, the budget is zero and the experts are read
/// from the artifact on every device miss.
public record HostBudget(long pinnableBytes, long residentBytes) {

    private static final long MARGIN_FLOOR = 4L << 30;

    /// The variable that states the memory available to the engine, in MiB.
    public static final String ENVIRONMENT = "EUHEDRAL_HOST_MEMORY_MIB";

    public HostBudget {
        if (pinnableBytes < 0) throw new IllegalArgumentException("pinnableBytes must not be negative");
        if (residentBytes < 0) throw new IllegalArgumentException("residentBytes must not be negative");
    }

    /// A budget whose pageable memory equals its pinnable memory.
    public HostBudget(long pinnableBytes) {
        this(pinnableBytes, pinnableBytes);
    }

    /// A budget from the memory the machine can spare: 90% of it, but never closer than 4 GiB to exhaustion.
    public static HostBudget ofAvailable(long availableBytes) {
        long margin = Math.max(MARGIN_FLOOR, availableBytes / 10);
        long usable = Math.max(0, availableBytes - margin);
        return new HostBudget(usable, usable);
    }

    /// The budget of this machine: pinnable memory from `MemAvailable`, ordinary memory only as stated by
    /// [#ENVIRONMENT] (none when unstated). Without `/proc/meminfo` the pinnable budget is unlimited.
    public static HostBudget system() {
        return of(System.getenv(ENVIRONMENT), System.getProperty("euhedral.host.memory-mib"), meminfo());
    }

    /// As [#system] over explicit sources: `stated` MiB (null: not stated), the property, and `MemAvailable`
    /// bytes (negative: unknown).
    static HostBudget of(String stated, String property, long meminfoBytes) {
        long pinnable = meminfoBytes >= 0 ? ofAvailable(meminfoBytes).pinnableBytes() : Long.MAX_VALUE / 4;
        String value = stated != null && !stated.isBlank() ? stated : property;
        long resident = 0;
        if (value != null && !value.isBlank()) {
            long mib = Long.parseLong(value.trim());
            if (mib < 0) throw new IllegalArgumentException(ENVIRONMENT + " must not be negative: " + mib);
            resident = ofAvailable(mib << 20).residentBytes();
        }
        return new HostBudget(pinnable, resident);
    }

    private static long meminfo() {
        try {
            for (String line : Files.readAllLines(Path.of("/proc/meminfo"))) {
                if (!line.startsWith("MemAvailable:")) continue;
                String[] fields = line.trim().split("\\s+");
                return Long.parseLong(fields[1]) * 1024L;
            }
        } catch (IOException | RuntimeException unreadable) {
            // fall through: not Linux, or no /proc
        }
        return -1;
    }
}
