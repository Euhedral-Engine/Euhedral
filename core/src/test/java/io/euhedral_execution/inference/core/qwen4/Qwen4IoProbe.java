package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Model;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/// Where a run's storage traffic comes from, for the performance record: the artifact's whole drive in
/// `/proc/diskstats` (bytes, busy time, time summed over the requests in flight), this process's reads and major
/// faults (`/proc/self/io`, `/proc/self/stat`), the machine's swap-ins (`/proc/vmstat`), the expert reads the engine
/// counted, and the n-gram tables' resident bytes. A sampler reads the drive's bytes every 10 ms, so a phase can tell
/// the time the drive ran at its ceiling from the time it idled.
final class Qwen4IoProbe implements AutoCloseable {

    /// The drive's sustained read rate measured with whole records (`docs/FLASH_NEXT_DISK.md`): the floor of a
    /// phase's SSD time is its bytes over this rate.
    static final double DRIVE_BYTES_PER_SECOND = 7.0e9;

    /// A 10 ms window counts as saturated at 85% of [#DRIVE_BYTES_PER_SECOND] and idle below 5%.
    private static final long WINDOW_NANOS = 10_000_000;

    private final String device;
    private final Thread sampler;
    private volatile boolean stopped;
    /// Windows by class, summed by the sampler: idle, partial, saturated.
    private final long[] windows = new long[3];

    record Snapshot(
            long nanos,
            long deviceReadBytes,
            long deviceWriteBytes,
            long busyMillis,
            long queueMillis,
            long processReadBytes,
            long majorFaults,
            long swapIns,
            long[] windows) {}

    Qwen4IoProbe(Path artifact) throws IOException {
        this.device = wholeDevice(artifact);
        this.sampler = Thread.ofPlatform().daemon().name("io-probe").start(this::sample);
    }

    String device() {
        return this.device;
    }

    /// The whole drive that holds `file` (a partition's parent), by name in `/proc/diskstats`.
    private static String wholeDevice(Path file) throws IOException {
        long dev = ((Number) Files.getAttribute(file, "unix:dev")).longValue();
        long major = ((dev >>> 8) & 0xfff) | ((dev >>> 32) & ~0xfffL);
        long minor = (dev & 0xff) | ((dev >>> 12) & ~0xffL);
        Path node = Path.of("/sys/dev/block", major + ":" + minor).toRealPath();
        if (Files.exists(node.resolve("partition"))) node = node.getParent();
        return node.getFileName().toString();
    }

    private long[] diskstats() throws IOException {
        for (String line : Files.readAllLines(Path.of("/proc/diskstats"))) {
            String[] f = line.trim().split("\\s+");
            if (f[2].equals(this.device))
                return new long[] {
                    Long.parseLong(f[5]) * 512, Long.parseLong(f[9]) * 512, Long.parseLong(f[12]), Long.parseLong(f[13])
                };
        }
        throw new IOException("no " + this.device + " in /proc/diskstats");
    }

    private void sample() {
        try {
            long last = diskstats()[0], lastAt = System.nanoTime();
            while (!this.stopped) {
                Thread.sleep(WINDOW_NANOS / 1_000_000);
                long now = System.nanoTime(), bytes = diskstats()[0];
                double rate = (bytes - last) * 1e9 / Math.max(1, now - lastAt);
                int kind = rate < 0.05 * DRIVE_BYTES_PER_SECOND ? 0 : rate >= 0.85 * DRIVE_BYTES_PER_SECOND ? 2 : 1;
                synchronized (this.windows) {
                    this.windows[kind] += now - lastAt;
                }
                last = bytes;
                lastAt = now;
            }
        } catch (IOException | InterruptedException ignored) {
            // The probe stops with the run.
        }
    }

    Snapshot snapshot() throws IOException {
        long[] disk = diskstats();
        long processRead = 0;
        for (String line : Files.readAllLines(Path.of("/proc/self/io")))
            if (line.startsWith("read_bytes:"))
                processRead = Long.parseLong(line.substring(11).trim());
        String stat = Files.readString(Path.of("/proc/self/stat"));
        // Fields after the command's closing parenthesis start at the state, field 3; majflt is field 12.
        String[] fields = stat.substring(stat.lastIndexOf(')') + 2).split(" ");
        long majflt = Long.parseLong(fields[12 - 3]);
        long swapIns = 0;
        for (String line : Files.readAllLines(Path.of("/proc/vmstat")))
            if (line.startsWith("pswpin "))
                swapIns = Long.parseLong(line.substring(7).trim());
        long[] copy;
        synchronized (this.windows) {
            copy = this.windows.clone();
        }
        return new Snapshot(System.nanoTime(), disk[0], disk[1], disk[2], disk[3], processRead, majflt, swapIns, copy);
    }

    /// The phase from `before` to now, per token: SSD bytes by source, the drive's use, the expert reads, the
    /// major faults and the n-gram tables' resident bytes before and after. `expertReadsBefore` and
    /// `expertBytesBefore` are the engine's artifact counters, `prefetchedBefore` its prefetched records, at
    /// `before`.
    List<String> report(
            Snapshot before,
            Qwen4Model model,
            long expertReadsBefore,
            long expertBytesBefore,
            long prefetchedBefore,
            long prefetchedAfter,
            long ngramResidentBefore,
            int tokens)
            throws IOException {
        Snapshot after = snapshot();
        var artifact = model.hierarchyStats().artifact();
        double seconds = (after.nanos() - before.nanos()) / 1e9;
        long total = after.deviceReadBytes() - before.deviceReadBytes();
        long expert = artifact.bytesRead() - expertBytesBefore;
        long reads = artifact.recordReads() - expertReadsBefore;
        double recordBytes = artifact.recordReads() == 0 ? 0 : (double) artifact.bytesRead() / artifact.recordReads();
        long prefetched = Math.min(expert, (long) ((prefetchedAfter - prefetchedBefore) * recordBytes));
        long process = after.processReadBytes() - before.processReadBytes();
        long swap = (after.swapIns() - before.swapIns()) * 4096;
        long processOther = Math.max(0, process - expert);
        long rest = Math.max(0, total - process - swap);
        long ngramAfter = model.ngram().residentBytes();
        double busy = (after.busyMillis() - before.busyMillis()) / 1e3;
        double queue = (after.queueMillis() - before.queueMillis()) / 1e3;
        long[] w = new long[3];
        long windowTotal = 0;
        for (int i = 0; i < 3; i++) windowTotal += w[i] = after.windows()[i] - before.windows()[i];
        double perToken = 1e6 * tokens;
        double floor = total / DRIVE_BYTES_PER_SECOND;
        return List.of(
                String.format(
                        "  io: SSD %.1f MB/token = expert %.1f (demand %.1f, prefetch %.1f) + other reads of this process"
                                + " %.2f + swap-in %.2f + rest of the machine %.2f; written %.2f MB/token",
                        total / perToken,
                        expert / perToken,
                        (expert - prefetched) / perToken,
                        prefetched / perToken,
                        processOther / perToken,
                        swap / perToken,
                        rest / perToken,
                        (after.deviceWriteBytes() - before.deviceWriteBytes()) / perToken),
                String.format(
                        "  io: %.2f GB/s over %.2f s; drive busy %.0f%% (10 ms windows: saturated %.0f%%, partial"
                                + " %.0f%%, idle %.0f%%); requests in flight %.2f on average, %.2f while busy;"
                                + " %.2f GB/s while busy",
                        total / 1e9 / seconds,
                        seconds,
                        100 * busy / seconds,
                        100.0 * w[2] / Math.max(1, windowTotal),
                        100.0 * w[1] / Math.max(1, windowTotal),
                        100.0 * w[0] / Math.max(1, windowTotal),
                        queue / seconds,
                        queue / Math.max(busy, 1e-9),
                        total / 1e9 / Math.max(busy, 1e-9)),
                String.format(
                        "  io: expert reads %.2f/token (%.3f/layer); the bytes at %.1f GB/s take %.2f ms/token, %.0f%%"
                                + " of the wall time; major faults %.1f/token; n-gram resident %.2f -> %.2f GiB of"
                                + " %.2f",
                        (double) reads / tokens,
                        (double) reads / tokens / 48,
                        DRIVE_BYTES_PER_SECOND / 1e9,
                        floor * 1e3 / tokens,
                        100 * floor / seconds,
                        (double) (after.majorFaults() - before.majorFaults()) / tokens,
                        ngramResidentBefore / (double) (1L << 30),
                        ngramAfter / (double) (1L << 30),
                        model.ngram().hostBytes() / (double) (1L << 30)));
    }

    @Override
    public void close() throws InterruptedException {
        this.stopped = true;
        this.sampler.join();
    }
}
