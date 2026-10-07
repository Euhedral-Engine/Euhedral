package io.euhedral_execution.inference.core.model.qwen4.loader;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The host memory a load may use, in the two forms the expert hierarchy needs.
///
/// - `pinnableBytes`: memory the runtime is willing to page-lock. Pinned memory is a small transfer tier: the staging
///   buffers that DMA reads, host-mapped fixed objects, the gathered embedding.
/// - `residentBytes`: ordinary memory the runtime may devote to the model's residency and caches: the pinned part
///   and the routed experts' RAM tier. Pinned memory is carved out of it by the plan, not added to it.
///
/// Both come from the same physical memory; the plan spends the pinned part first and gives the rest to the
/// expert tier. The budget controls memory only: the tier's records, the staging buffers, the copy streams and the
/// cache's shards are derived by the planner within it.
///
/// By default the budget is automatic: what the machine can spare now (`MemAvailable`, or less when the process's
/// cgroup allows less), less room for the operating system (10%, never closer than 4 GiB to exhaustion) and for the
/// runtime itself (the JVM heap's room to grow and a native reserve). `EUHEDRAL_HOST_MEMORY_MIB` (or the system
/// property `euhedral.host.memory-mib`) states the budget instead, in MiB, and is respected as stated: it may be
/// smaller than the automatic one (0 means no RAM tier) or larger (a warning says so, as the memory is not free now).
public record HostBudget(long pinnableBytes, long residentBytes, Source source) {

    private static final Logger LOG = LoggerFactory.getLogger(HostBudget.class);

    private static final long MARGIN_FLOOR = 4L << 30;

    /// Native memory the runtime keeps for itself beyond the JVM heap: the driver, the native library, thread
    /// stacks, the lattice.
    static final long NATIVE_RESERVE = 1L << 30;

    /// The variable that states the memory available to the engine, in MiB.
    public static final String ENVIRONMENT = "EUHEDRAL_HOST_MEMORY_MIB";

    /// Where the resident budget came from.
    public enum Source {
        /// Derived from the memory the machine can spare.
        AUTOMATIC,
        /// Stated by the user.
        EXPLICIT
    }

    public HostBudget {
        if (pinnableBytes < 0) throw new IllegalArgumentException("pinnableBytes must not be negative");
        if (residentBytes < 0) throw new IllegalArgumentException("residentBytes must not be negative");
        java.util.Objects.requireNonNull(source, "source");
    }

    /// A stated budget.
    public HostBudget(long pinnableBytes, long residentBytes) {
        this(pinnableBytes, residentBytes, Source.EXPLICIT);
    }

    /// A stated budget whose pageable memory equals its pinnable memory.
    public HostBudget(long pinnableBytes) {
        this(pinnableBytes, pinnableBytes);
    }

    /// A budget from the memory the machine can spare: 90% of it, but never closer than 4 GiB to exhaustion.
    public static HostBudget ofAvailable(long availableBytes) {
        long margin = Math.max(MARGIN_FLOOR, availableBytes / 10);
        long usable = Math.max(0, availableBytes - margin);
        return new HostBudget(usable, usable, Source.AUTOMATIC);
    }

    /// The budget of this machine and process: automatic, or as stated by [#ENVIRONMENT] (or the property).
    public static HostBudget system() {
        return of(
                System.getenv(ENVIRONMENT),
                System.getProperty("euhedral.host.memory-mib"),
                availableBytes(),
                runtimeReserve());
    }

    /// As [#system] over explicit sources, with no runtime reserve.
    static HostBudget of(String stated, String property, long availableBytes) {
        return of(stated, property, availableBytes, 0);
    }

    /// As [#system] over explicit sources: `stated` MiB (null: not stated), the property, the bytes the machine can
    /// spare now (negative: unknown), and the runtime's own reserve.
    static HostBudget of(String stated, String property, long availableBytes, long runtimeReserve) {
        long machine = availableBytes >= 0
                ? Math.max(0, ofAvailable(availableBytes).residentBytes() - runtimeReserve)
                : Long.MAX_VALUE / 4;
        String value = stated != null && !stated.isBlank() ? stated : property;
        if (value == null || value.isBlank()) return new HostBudget(machine, machine, Source.AUTOMATIC);
        long mib = Long.parseLong(value.trim());
        if (mib < 0) throw new IllegalArgumentException(ENVIRONMENT + " must not be negative: " + mib);
        long bytes = mib << 20;
        if (bytes > machine)
            LOG.warn(
                    "{}={} MiB is more than this machine can spare now ({} MiB); it is used as stated",
                    ENVIRONMENT,
                    mib,
                    machine >> 20);
        return new HostBudget(machine, bytes, Source.EXPLICIT);
    }

    /// What the runtime keeps for itself: the JVM heap's room to grow beyond what it holds now, and the native
    /// reserve.
    static long runtimeReserve() {
        Runtime runtime = Runtime.getRuntime();
        long heapGrowth = Math.max(0, runtime.maxMemory() - runtime.totalMemory());
        return heapGrowth + NATIVE_RESERVE;
    }

    /// The memory this process can take now: `MemAvailable`, or the headroom of the process's cgroup and its
    /// ancestors when that is less. Negative when unknown (not Linux, or no `/proc`).
    static long availableBytes() {
        long available = meminfo();
        long cgroup = cgroupHeadroom(Path.of("/proc/self/cgroup"), Path.of("/sys/fs/cgroup"));
        if (available < 0) return cgroup;
        return cgroup < 0 ? available : Math.min(available, cgroup);
    }

    /// The least headroom (`memory.max` less `memory.current`) of the process's cgroup (v2) and its ancestors, or -1
    /// when none of them has a limit or it cannot be read.
    static long cgroupHeadroom(Path selfCgroup, Path root) {
        try {
            String path = null;
            for (String line : Files.readAllLines(selfCgroup))
                if (line.startsWith("0::")) path = line.substring(3).trim();
            if (path == null) return -1;
            long least = -1;
            Path group = root.resolve(path.startsWith("/") ? path.substring(1) : path)
                    .normalize();
            while (group.startsWith(root)) {
                long headroom = headroom(group);
                if (headroom >= 0 && (least < 0 || headroom < least)) least = headroom;
                if (group.equals(root)) break;
                group = group.getParent();
            }
            return least;
        } catch (IOException | RuntimeException unreadable) {
            return -1;
        }
    }

    private static long headroom(Path group) {
        try {
            List<String> max = Files.readAllLines(group.resolve("memory.max"));
            if (max.isEmpty() || max.get(0).trim().equals("max")) return -1;
            long limit = Long.parseLong(max.get(0).trim());
            long current = Long.parseLong(
                    Files.readAllLines(group.resolve("memory.current")).get(0).trim());
            return Math.max(0, limit - current);
        } catch (IOException | RuntimeException unreadable) {
            return -1;
        }
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

    /// One line for the startup report: the budget and where it came from.
    public String describe() {
        return (this.residentBytes >> 20) + " MiB, "
                + (this.source == Source.AUTOMATIC
                        ? "automatic (available memory less the system's and the runtime's reserves)"
                        : "stated by " + ENVIRONMENT);
    }
}
