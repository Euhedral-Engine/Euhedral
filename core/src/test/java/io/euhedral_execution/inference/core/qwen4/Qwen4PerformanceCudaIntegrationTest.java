package io.euhedral_execution.inference.core.qwen4;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.qwen4.HostBudget;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Mode;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Model;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/// First performance record of ordinary Flash-Next generation: load time, prefill of 512, 4K and
/// 16K tokens, decode at short, 4K, 16K and 32K context, expert-cache behavior and the time per
/// component. Measurements, not assertions; opt-in because it takes minutes:
/// `EUHEDRAL_QWEN4_PERF=1`; `EUHEDRAL_QWEN4_PERF_MAX` bounds the contexts (default 4096). The
/// prompt is the repository's own documentation, tokenized with the model's tokenizer.
class Qwen4PerformanceCudaIntegrationTest {

    private static final int MAX = Integer.parseInt(System.getenv().getOrDefault("EUHEDRAL_QWEN4_PERF_MAX", "4096"));
    private static final int CONTEXT = Math.max(MAX, 4096) + 1024;

    static int[] corpus(int tokens) throws IOException {
        Path tokenizerDirectory =
                Path.of(System.getProperty("euhedral.qwen4.tokenizer-dir", "/mnt/shared/qwen38-flash-next/nvfp4"));
        assumeTrue(Files.isRegularFile(tokenizerDirectory.resolve("tokenizer.json")), "no tokenizer");
        QwenTokenizer tokenizer = QwenTokenizer.load(tokenizerDirectory);
        StringBuilder text = new StringBuilder();
        try (Stream<Path> docs = Files.list(SpeculativeRoot.docs())) {
            for (Path doc :
                    docs.filter(p -> p.toString().endsWith(".md")).sorted().toList())
                text.append(Files.readString(doc)).append("\n\n");
        }
        int[] ids = tokenizer.encodeText(text.toString());
        int[] out = new int[tokens];
        for (int i = 0; i < tokens; i++) out[i] = ids[i % ids.length];
        return out;
    }

    /// The repository's docs directory, found from the working directory of the test JVM.
    private static final class SpeculativeRoot {
        static Path docs() {
            Path at = Path.of("").toAbsolutePath();
            while (at != null && !Files.isDirectory(at.resolve("docs"))) at = at.getParent();
            return at.resolve("docs");
        }
    }

    private final List<String> report = new ArrayList<>();

    /// Per-layer use of the expert cache, accumulated over the MoE blocks a run executed.
    private static final class LayerStats implements Qwen4ExecutionPlan.ExpertTrace {
        final long[] blocks = new long[48],
                unique = new long[48],
                hits = new long[48],
                misses = new long[48],
                evictions = new long[48],
                bytes = new long[48],
                waitNanos = new long[48];
        final long[] uniqueMax = new long[48];

        @Override
        public void layer(
                int layer,
                int rows,
                int uniqueExperts,
                io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCacheStats.Snapshot before,
                io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCacheStats.Snapshot after) {
            blocks[layer]++;
            unique[layer] += uniqueExperts;
            uniqueMax[layer] = Math.max(uniqueMax[layer], uniqueExperts);
            hits[layer] += after.hits() - before.hits();
            misses[layer] += after.misses() - before.misses();
            evictions[layer] += after.evictions() - before.evictions();
            bytes[layer] += after.transferBytes() - before.transferBytes();
            waitNanos[layer] += after.loadWaitNanos() - before.loadWaitNanos();
        }

        String summary(String title) {
            StringBuilder text =
                    new StringBuilder(title + ": layer unique-experts/block (max) hit-rate misses/block MB/block\n");
            long totalHits = 0, totalMisses = 0, totalUnique = 0, totalBlocks = 0;
            for (int l = 0; l < 48; l++) {
                if (blocks[l] == 0) continue;
                long requests = hits[l] + misses[l];
                text.append(String.format(
                        "  L%02d %6.1f (%d) %5.1f%% %6.1f %7.1f%n",
                        l,
                        (double) unique[l] / blocks[l],
                        uniqueMax[l],
                        100.0 * hits[l] / Math.max(1, requests),
                        (double) misses[l] / blocks[l],
                        bytes[l] / 1e6 / blocks[l]));
                totalHits += hits[l];
                totalMisses += misses[l];
                totalUnique += unique[l];
                totalBlocks += blocks[l];
            }
            text.append(String.format(
                    "  all: %.1f unique experts/block, hit rate %.1f%%%n",
                    (double) totalUnique / Math.max(1, totalBlocks),
                    100.0 * totalHits / Math.max(1, totalHits + totalMisses)));
            return text.toString();
        }
    }

    private static int intEnv(String name, int fallback) {
        String value = System.getenv(name);
        return value == null ? fallback : Integer.parseInt(value);
    }

    private void line(String text) {
        System.out.println(text);
        this.report.add(text);
    }

    private static String cache(
            Qwen4Model model, long hits, long misses, long evictions, long bytes, long waitNanos, int tokens) {
        var now = model.expertCache().stats().snapshot();
        long dh = now.hits() - hits, dm = now.misses() - misses;
        return String.format(
                "hit rate %.1f%%, %.2f misses/token, %.1f MB H2D/token, expert wait %.2f ms/token",
                100.0 * dh / Math.max(1, dh + dm),
                (double) dm / tokens,
                (now.transferBytes() - bytes) / 1e6 / tokens,
                (now.loadWaitNanos() - waitNanos) / 1e6 / tokens);
    }

    /// Where the expert loads between `before` and `after` spent their time, per load and per token.
    private static String loads(
            io.euhedral_execution.inference.core.qwen4.ExpertCacheOwner.Timings before,
            io.euhedral_execution.inference.core.qwen4.ExpertCacheOwner.Timings after,
            long fullBefore,
            long fullAfter,
            int tokens) {
        long n = Math.max(1, after.loads() - before.loads());
        return String.format(
                "%.1f loads/token; per load: dispatch %.0f us, read %.0f us, submit hop %.0f us, copy %.0f us,"
                        + " retire hop %.0f us; %.2f fetches/token found the cache full",
                (double) n / tokens,
                (after.dispatch() - before.dispatch()) / 1e3 / n,
                (after.read() - before.read()) / 1e3 / n,
                (after.submitHop() - before.submitHop()) / 1e3 / n,
                (after.copy() - before.copy()) / 1e3 / n,
                (after.retireHop() - before.retireHop()) / 1e3 / n,
                (double) (fullAfter - fullBefore) / tokens);
    }

    /// The tiers' work between `before` and now, per token.
    private static String tiers(
            Qwen4Model model,
            io.euhedral_execution.inference.core.model_loader.qwen4.ExpertHierarchyStats before,
            int tokens) {
        var after = model.hierarchyStats();
        long gpuHits = after.gpu().hits() - before.gpu().hits();
        long gpuMisses = after.gpu().misses() - before.gpu().misses();
        StringBuilder out =
                new StringBuilder(String.format("gpu hit %.1f%%", 100.0 * gpuHits / Math.max(1, gpuHits + gpuMisses)));
        if (after.ram() != null) {
            long ramHits = after.ram().totalHits() - before.ram().totalHits();
            long ramOther = after.ram().totalMisses()
                    + after.ram().totalBypasses()
                    - before.ram().totalMisses()
                    - before.ram().totalBypasses();
            out.append(String.format(
                    ", ram hit %.1f%% (%d resident)",
                    100.0 * ramHits / Math.max(1, ramHits + ramOther),
                    after.ram().residentExperts()));
        }
        long reads = after.artifact().recordReads() - before.artifact().recordReads();
        long readNanos = after.artifact().readNanos() - before.artifact().readNanos();
        long copyBytes = after.staging().ramCopyBytes() - before.staging().ramCopyBytes();
        long copyNanos = after.staging().ramCopyNanos() - before.staging().ramCopyNanos();
        out.append(String.format(
                ", artifact read %.0f us/record, ram copy %.1f GB/s,",
                readNanos / 1e3 / Math.max(1, reads), (double) copyBytes / Math.max(1, copyNanos)));
        out.append(String.format(
                " disk %.1f MB/token (reads in flight <= %d), ram copy %.1f MB/token, H2D %.1f MB/token at %.1f GB/s",
                (after.artifact().bytesRead() - before.artifact().bytesRead()) / 1e6 / tokens,
                after.artifact().concurrentReadsHighWater(),
                (after.staging().ramCopyBytes() - before.staging().ramCopyBytes()) / 1e6 / tokens,
                (after.h2d().bytes() - before.h2d().bytes()) / 1e6 / tokens,
                (double) (after.h2d().bytes() - before.h2d().bytes())
                        / Math.max(1, after.h2d().nanos() - before.h2d().nanos())));
        return out.toString();
    }

    /// A window for a profiler: warm decode at 64 context, then two idle seconds that mark the window's start in a
    /// timeline, then 32 decode steps, and nothing after them.
    private void profileWindow(Qwen4ExecutionPlan executor, int[] prompt, Qwen4ExecutionPlan.LogitsSink sink)
            throws Exception {
        try (Qwen4Sequence sequence = executor.newSequence()) {
            Qwen4Blocking.step(executor, sequence, prompt, 0, 64, null);
            int[] token = new int[1];
            for (int i = 0; i < 64; i++) {
                token[0] = prompt[(64 + i * 97) % prompt.length];
                Qwen4Blocking.step(executor, sequence, token, 0, 1, sink);
            }
            Thread.sleep(2000);
            long start = System.nanoTime();
            int steps = 32;
            for (int i = 0; i < steps; i++) {
                token[0] = prompt[(64 + i * 89) % prompt.length];
                Qwen4Blocking.step(executor, sequence, token, 0, 1, sink);
            }
            line(String.format(
                    "profile window: %d decode steps at 64 context, %.1f ms/token",
                    steps, (System.nanoTime() - start) / 1e6 / steps));
        }
    }

    @Test
    void firstPerformanceRecord() throws Exception {
        assumeTrue(
                System.getenv("EUHEDRAL_QWEN4_PERF") != null || Boolean.getBoolean("euhedral.qwen4.perf"),
                "set EUHEDRAL_QWEN4_PERF=1 to run the performance record");
        assumeTrue(Qwen4TestSupport.hasArtifact(), "no artifact");
        int[] prompt = corpus(CONTEXT);
        try (Qwen4TestLattice lattice = Qwen4TestLattice.start(4);
                CudaGpuMemory gpu = Qwen4TestSupport.openGpu()) {
            long loadStart = System.nanoTime();
            // A profiler that replays kernels needs device memory of its own: EUHEDRAL_QWEN4_PERF_DEVICE_MIB caps
            // what the plan may use.
            long free = gpu.deviceMemoryInfo().freeBytes();
            long cap = intEnv("EUHEDRAL_QWEN4_PERF_DEVICE_MIB", 0);
            try (Qwen4Model model = Qwen4Model.open(
                            Qwen4TestSupport.artifactPath(),
                            gpu,
                            cap > 0 ? Math.min(free, cap << 20) : free,
                            HostBudget.system(),
                            Qwen4Mode.TEXT,
                            CONTEXT);
                    Qwen4TestLattice.Run run = lattice.run(gpu, model, CONTEXT);
                    Qwen4ExecutionPlan executor = run.plan()) {
                line(String.format("load: %.1f s (model %.1f s)", (System.nanoTime() - loadStart) / 1e9, 0.0));
                line("workers: " + lattice.cpus().cardinality() + " on processors " + lattice.cpus());
                var startTier = model.hierarchyStats().ram();
                if (startTier != null)
                    line(String.format(
                            "tier at startup: %d records, %.1f GiB read at %.2f GB/s",
                            startTier.residentExperts(),
                            startTier.preloadBytes() / (double) (1L << 30),
                            startTier.preloadBytesPerSecond() / 1e9));
                line("expert cache: " + model.expertCache().slotCount() + " slots; plan:\n"
                        + model.plan().report());
                int chunk = Math.min(executor.maxRows(), intEnv("EUHEDRAL_QWEN4_CHUNK", executor.maxRows()));
                line("prefill chunk: " + chunk + " tokens (plan " + executor.maxRows() + ")");
                int vocabulary = executor.vocabularySize();
                var readback = gpu.allocateReadbackBuffer((long) vocabulary * 2);
                Qwen4ExecutionPlan.LogitsSink sink =
                        address -> gpu.copyDeviceToReadback(readback, address, vocabulary * 2L);
                // Warm the kernels and the cache with a short run.
                try (Qwen4Sequence warm = executor.newSequence()) {
                    Qwen4Blocking.step(executor, warm, prompt, 0, 64, sink);
                }
                if (System.getenv("EUHEDRAL_QWEN4_PERF_PROFILE") != null) {
                    profileWindow(executor, prompt, sink);
                    return;
                }
                for (int target : Arrays.stream(new int[] {512, 4096, 16384})
                        .filter(t -> t <= MAX)
                        .toArray()) {
                    try (Qwen4Sequence sequence = executor.newSequence()) {
                        var before = model.expertCache().stats().snapshot();
                        var tiersBefore = model.hierarchyStats();
                        var loadsBefore = executor.expertTimings();
                        long fullBefore = executor.expertFullFetches();
                        long busyBefore = model.asyncReads() == null
                                ? 0
                                : model.asyncReads().busyNanos();
                        long diskBefore = model.hierarchyStats().artifact().bytesRead();
                        LayerStats layers = new LayerStats();
                        if (target == 4096) executor.trace(layers);
                        long start = System.nanoTime();
                        int at = 0;
                        while (at < target) {
                            int rows = Math.min(chunk, target - at);
                            Qwen4Blocking.step(executor, sequence, prompt, at, rows, at + rows == target ? sink : null);
                            at += rows;
                        }
                        double seconds = (System.nanoTime() - start) / 1e9;
                        executor.trace(null);
                        if (target == 4096)
                            line(layers.summary("expert cache, prefill of 4096 tokens in 512-token chunks"));
                        line(String.format(
                                "prefill %d tokens: %.2f s, %.0f tokens/s; %s",
                                target,
                                seconds,
                                target / seconds,
                                cache(
                                        model,
                                        before.hits(),
                                        before.misses(),
                                        before.evictions(),
                                        before.transferBytes(),
                                        before.loadWaitNanos(),
                                        target)));
                        line("  tiers: " + tiers(model, tiersBefore, target));
                        line("  loads: "
                                + loads(
                                        loadsBefore,
                                        executor.expertTimings(),
                                        fullBefore,
                                        executor.expertFullFetches(),
                                        target));
                        if (model.asyncReads() != null) {
                            double busy = (model.asyncReads().busyNanos() - busyBefore) / 1e9;
                            long disk = model.hierarchyStats().artifact().bytesRead() - diskBefore;
                            line(String.format(
                                    "  disk: reads in flight %.0f%% of the time, %.2f GB/s while they were, %.2f GB/s"
                                            + " overall",
                                    100 * busy / seconds, disk / 1e9 / Math.max(busy, 1e-9), disk / 1e9 / seconds));
                        }
                    }
                }
                // `EUHEDRAL_QWEN4_PERF_PREFILL_ONLY=1` stops after the prefills, for screening the expert path.
                boolean decode = !"1".equals(System.getenv("EUHEDRAL_QWEN4_PERF_PREFILL_ONLY"));
                for (int context : Arrays.stream(new int[] {64, 4096, 16384, 32768})
                        .filter(t -> decode && t <= MAX)
                        .toArray()) {
                    try (Qwen4Sequence sequence = executor.newSequence()) {
                        int at = 0;
                        while (at < context) {
                            int rows = Math.min(chunk, context - at);
                            Qwen4Blocking.step(executor, sequence, prompt, at, rows, null);
                            at += rows;
                        }
                        // Twice over the same tokens: the first finds the tiers as the prefill left them, the second
                        // warm.
                        for (int round = 0; round < 2; round++) {
                            int[] token = new int[1];
                            var before = model.expertCache().stats().snapshot();
                            var tiersBefore = model.hierarchyStats();
                            var loadsBefore = executor.expertTimings();
                            long fullBefore = executor.expertFullFetches();
                            var counters = executor.moeCounters();
                            LayerStats decodeLayers = new LayerStats();
                            if (context == 4096) executor.trace(decodeLayers);
                            int steps = 32;
                            long start = System.nanoTime();
                            for (int i = 0; i < steps; i++) {
                                token[0] = prompt[(context + i * 97) % prompt.length];
                                Qwen4Blocking.step(executor, sequence, token, 0, 1, sink);
                                short[] logits = new short[vocabulary];
                                MemorySegment.copy(readback.segment(), ValueLayout.JAVA_SHORT, 0, logits, 0, 16);
                            }
                            double seconds = (System.nanoTime() - start) / 1e9;
                            executor.trace(null);
                            if (context == 4096) line(decodeLayers.summary("expert cache, decode at 4096 context"));
                            var after = executor.moeCounters();
                            line(String.format(
                                    "decode (%s) at %d context: %.1f ms/token, %.1f tokens/s; %s; route wait %.2f ms + acquire %.2f ms per token",
                                    round == 0 ? "cold" : "warm",
                                    context,
                                    seconds * 1e3 / steps,
                                    steps / seconds,
                                    cache(
                                            model,
                                            before.hits(),
                                            before.misses(),
                                            before.evictions(),
                                            before.transferBytes(),
                                            before.loadWaitNanos(),
                                            steps),
                                    (after.routeWaitNanos() - counters.routeWaitNanos()) / 1e6 / steps,
                                    (after.expertWaitNanos() - counters.expertWaitNanos()) / 1e6 / steps));
                            line("  tiers: " + tiers(model, tiersBefore, steps));
                            line("  loads: "
                                    + loads(
                                            loadsBefore,
                                            executor.expertTimings(),
                                            fullBefore,
                                            executor.expertFullFetches(),
                                            steps));
                        }
                    }
                }
                // Component times: decode at 4K and a 512-token prefill chunk, each step followed by a device wait.
                for (int[] shape : decode ? new int[][] {{4096, 1}, {4096, 512}} : new int[0][]) {
                    try (Qwen4Sequence sequence = executor.newSequence()) {
                        int at = 0;
                        while (at < shape[0]) {
                            int rows = Math.min(chunk, shape[0] - at);
                            Qwen4Blocking.step(executor, sequence, prompt, at, rows, null);
                            at += rows;
                        }
                        var timings = new Qwen4ExecutionPlan.Timings();
                        executor.timings(timings);
                        int steps = shape[1] == 1 ? 16 : 1;
                        for (int i = 0; i < steps; i++) {
                            int rows = shape[1];
                            int[] tokens = Arrays.copyOfRange(prompt, at, at + rows);
                            Qwen4Blocking.step(executor, sequence, tokens, 0, rows, sink);
                            at += rows;
                        }
                        executor.timings(null);
                        StringBuilder text = new StringBuilder(
                                String.format("components, %d-row steps at %d context (ms/step):", shape[1], shape[0]));
                        for (int i = 0; i < Qwen4ExecutionPlan.Timings.NAMES.length; i++)
                            text.append(String.format(
                                    " %s %.2f;", Qwen4ExecutionPlan.Timings.NAMES[i], timings.nanos[i] / 1e6 / steps));
                        line(text.toString());
                    }
                }
                var telemetry = model.telemetry();
                line("device allocated " + gpu.allocatedBytes() / (1 << 20) + " MiB, peak "
                        + gpu.peakAllocatedBytes() / (1 << 20) + " MiB; " + telemetry);
                readback.close();
            }
        }
        Files.writeString(Path.of("build", "qwen4-performance.txt"), String.join("\n", this.report) + "\n");
    }
}
