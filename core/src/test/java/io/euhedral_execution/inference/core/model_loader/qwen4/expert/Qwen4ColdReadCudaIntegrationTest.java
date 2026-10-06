package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Artifact;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4ArtifactReader;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

/// How concurrent cold reads of expert records scale on this machine's storage, and how fast a resident RAM
/// tier fills at startup. Opt-in (`EUHEDRAL_QWEN4_BENCH_COLD=1`): it evicts the artifact's pages from the OS
/// page cache before each measurement (`posix_fadvise(DONTNEED)`, which an unprivileged process may do for
/// clean pages), so every read goes to the device; the artifact's pages are cold afterwards.
///
/// Part 1 reads random records with N plain threads (the device's ceiling, independent of any scheduler).
/// Part 2 fills a resident [RamTier] of the first banks with N readers, as the startup load does.
class Qwen4ColdReadCudaIntegrationTest {

    private static final int[] CONCURRENCY = {1, 2, 4, 8, 16, 32};
    private static final long RANDOM_BUDGET_NANOS = 8_000_000_000L;
    private static final long PRELOAD_TARGET_BYTES = 12L << 30;

    private static void dropCache(Path file) throws Throwable {
        Linker linker = Linker.nativeLinker();
        var open = linker.downcallHandle(
                linker.defaultLookup().find("open").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
        var advise = linker.downcallHandle(
                linker.defaultLookup().find("posix_fadvise").orElseThrow(),
                FunctionDescriptor.of(
                        ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_INT));
        var close = linker.downcallHandle(
                linker.defaultLookup().find("close").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        try (var arena = java.lang.foreign.Arena.ofConfined()) {
            int fd = (int) open.invokeExact(arena.allocateFrom(file.toString()), 0);
            int result = (int) advise.invokeExact(fd, 0L, 0L, 4 /* POSIX_FADV_DONTNEED */);
            int ignored = (int) close.invokeExact(fd);
            if (result != 0) throw new IllegalStateException("posix_fadvise failed: " + result);
        }
    }

    @Test
    void coldReadsAndPreloadByConcurrency() throws Throwable {
        assumeTrue(System.getenv("EUHEDRAL_QWEN4_BENCH_COLD") != null, "set EUHEDRAL_QWEN4_BENCH_COLD=1");
        Path path = Path.of(System.getProperty(
                "euhedral.qwen4.artifact", "/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl"));
        assumeTrue(Files.isRegularFile(path), "no artifact");
        Qwen4Artifact artifact = Qwen4ArtifactReader.read(path);
        ExpertBank[] banks = artifact.banks();
        List<String> report = new ArrayList<>();
        report.add("artifact " + path + ", " + banks.length + " banks");

        // Part 1: random record reads with N threads.
        long recordBytes = banks[0].maxRecordBytes();
        report.add(String.format(
                "part 1: random record reads (%.2f MiB records), %d s each",
                recordBytes / 1048576.0, RANDOM_BUDGET_NANOS / 1_000_000_000L));
        for (int threads : CONCURRENCY) {
            dropCache(path);
            FileRecordSource source = new FileRecordSource(path, banks);
            AtomicLong done = new AtomicLong();
            long begin = System.nanoTime();
            List<Thread> workers = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                int seed = t;
                Thread thread = new Thread(() -> {
                    SplittableRandom random = new SplittableRandom(1000L * threads + seed);
                    var arena = java.lang.foreign.Arena.ofConfined();
                    MemorySegment destination = arena.allocate(recordBytes, 4096);
                    try {
                        while (System.nanoTime() - begin < RANDOM_BUDGET_NANOS) {
                            ExpertBank bank = banks[random.nextInt(banks.length)];
                            int expert = random.nextInt(bank.expertCount());
                            source.read(bank, expert, destination.asSlice(0, bank.recordBytes(expert)));
                            done.incrementAndGet();
                        }
                    } catch (Exception failure) {
                        throw new IllegalStateException(failure);
                    } finally {
                        arena.close();
                    }
                });
                workers.add(thread);
                thread.start();
            }
            for (Thread thread : workers) thread.join();
            double seconds = (System.nanoTime() - begin) / 1e9;
            report.add(String.format(
                    "  %2d threads: %7.0f records/s, %5.2f GB/s",
                    threads, done.get() / seconds, done.get() * recordBytes / 1e9 / seconds));
            source.close();
        }

        // Part 2: the startup fill of a resident tier, by N readers.
        int bankCount = 0;
        long bytes = 0;
        while (bankCount < banks.length && bytes < PRELOAD_TARGET_BYTES) bytes += banks[bankCount++].totalBytes();
        ExpertBank[] subset = java.util.Arrays.copyOf(banks, bankCount);
        int experts = 0;
        for (ExpertBank bank : subset) experts += bank.expertCount();
        report.add(String.format(
                "part 2: preload of %d banks (%.1f GiB, %d records), by readers",
                bankCount, bytes / 1073741824.0, experts));
        for (int readers : CONCURRENCY) {
            dropCache(path);
            try (RamTier tier = new RamTier(subset, experts, 8, ReplacementPolicy.BANK_PARTITIONED);
                    FileRecordSource source = new FileRecordSource(path, subset)) {
                long begin = System.nanoTime();
                tier.preload(source, readers);
                double seconds = (System.nanoTime() - begin) / 1e9;
                report.add(
                        String.format("  %2d readers: %6.2f s, %5.2f GB/s", readers, seconds, bytes / 1e9 / seconds));
            }
        }
        String text = String.join("\n", report) + "\n";
        System.out.println(text);
        Files.writeString(Path.of("build", "qwen4-cold-read.txt"), text);
    }
}
