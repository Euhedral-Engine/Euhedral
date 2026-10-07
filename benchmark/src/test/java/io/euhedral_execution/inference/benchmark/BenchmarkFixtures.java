package io.euhedral_execution.inference.benchmark;

import io.euhedral_execution.inference.benchmark.config.BenchmarkOptions;
import io.euhedral_execution.inference.benchmark.config.Scenario;
import io.euhedral_execution.inference.benchmark.measure.IterationTiming;
import io.euhedral_execution.inference.benchmark.prompt.PromptMaterial;
import io.euhedral_execution.inference.benchmark.result.BenchmarkResult;
import io.euhedral_execution.inference.benchmark.run.BenchmarkRunner;
import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.InferenceRunSnapshot;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.nio.file.Path;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/// Shared fakes for benchmark tests: a target that follows the session timing contract without CUDA.
public final class BenchmarkFixtures {
    private BenchmarkFixtures() {}

    public static final int EOS = 7;
    /// The prefill chunk the real session uses.
    public static final int CHUNK = InferenceConfig.PREFILL_CHUNK_TOKENS;

    public static java.util.BitSet bits(int... ids) {
        var set = new java.util.BitSet();
        for (int id : ids) set.set(id);
        return set;
    }

    public static int words(String text) {
        return text.split(" ").length;
    }

    public static InferenceRunSnapshot snapshot(GenerationConfig generation) {
        return new InferenceRunSnapshot(
                InferenceRunSnapshot.SCHEMA_VERSION,
                new InferenceRunSnapshot.Configuration(List.of(0), 32768, "fake.edrl", 0, 0),
                List.of(0),
                new InferenceRunSnapshot.Model(
                        "/fake.edrl",
                        null,
                        2,
                        new InferenceRunSnapshot.Dimensions(8, 4, 1, 1, 0, 1, 1, 4, 8, 0, 0, 0, 0, 64)),
                generation,
                new InferenceRunSnapshot.RuntimeIdentity(
                        "25", "vendor", "vm", "Linux", "amd64", null, "euhedral-core-0.0.7.jar", "/lib.so", null));
    }

    /// Simulates the session contract: fresh session per call, hook events per 512-token prefill chunk and decode
    /// quantum.
    public static final class FakeTarget implements BenchmarkRunner.Target {
        public final List<IterationTiming> timings = new ArrayList<>();
        public int sessions;
        public boolean closed;
        public int failOnCall = -1;
        public int eosAfter = Integer.MAX_VALUE;
        long clock;

        @Override
        public InferenceRunSnapshot snapshot(GenerationConfig generation) {
            return BenchmarkFixtures.snapshot(generation);
        }

        @Override
        public List<Integer> generate(
                String prompt, int maxNewTokens, GenerationConfig generation, IterationTiming timing) {
            int call = this.sessions++;
            this.timings.add(timing);
            // One synthetic clock for every boundary, as the real session uses one nanoTime source.
            timing.markEntry(tick());
            int promptTokens = words(prompt);
            timing.promptEncoded(tick(), promptTokens);
            for (int offset = 0; offset < promptTokens; offset += CHUNK) {
                if (call == this.failOnCall) throw new IllegalStateException("injected quantum failure");
                timing.prefillQuantum(tick(), tick(), Math.min(CHUNK, promptTokens - offset));
            }
            List<Integer> tokens = new ArrayList<>();
            if (maxNewTokens > 0) {
                int first = this.eosAfter == 1 ? EOS : 1;
                timing.firstTokenSelected(tick(), first);
                tokens.add(first);
                while (tokens.getLast() != EOS && tokens.size() < maxNewTokens) {
                    int next = tokens.size() + 1 >= this.eosAfter ? EOS : 1;
                    timing.decodeQuantum(tick(), tick(), tick(), true, next);
                    tokens.add(next);
                }
                if (tokens.getLast() != EOS) {
                    long start = tick();
                    long executed = tick();
                    timing.decodeQuantum(start, executed, executed, false, -1);
                }
            }
            timing.markReturn(tick());
            return tokens;
        }

        private long tick() {
            return this.clock += 10;
        }

        @Override
        public boolean isEos(int tokenId) {
            return tokenId == EOS;
        }

        @Override
        public long[] memory() {
            return new long[] {100L - this.sessions, 1_000L};
        }

        @Override
        public void close() {
            this.closed = true;
        }
    }

    public static BenchmarkRunner.Context context() {
        return new BenchmarkRunner.Context(
                "run-1",
                new BenchmarkResult.Fork(null, "unspecified", 42L, "2026-09-25T00:00:00Z"),
                new BenchmarkResult.Provenance("measured", "euhedral-inference-benchmark", "run ...", null, null, null),
                Clock.fixed(Instant.parse("2026-09-25T12:00:00Z"), ZoneOffset.UTC));
    }

    public static Map<Scenario, List<PromptMaterial>> prompts(List<Scenario> scenarios) {
        Map<Scenario, List<PromptMaterial>> prompts = new LinkedHashMap<>();
        for (Scenario scenario : scenarios)
            prompts.put(
                    scenario,
                    List.of(PromptMaterial.build(scenario.targetPromptTokens(), 1L, BenchmarkFixtures::words)));
        return prompts;
    }

    public static BenchmarkOptions options(List<Scenario> scenarios, int warmup, int iterations) {
        return new BenchmarkOptions(
                Path.of("/a"),
                Path.of("/t"),
                Path.of("/l"),
                null,
                null,
                null,
                null,
                null,
                scenarios,
                warmup,
                iterations,
                null,
                null,
                Path.of("r.jsonl"),
                null,
                null,
                true,
                null,
                null,
                null);
    }
}
