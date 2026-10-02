package io.euhedral_execution.inference.benchmark.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.benchmark.prompt.PromptMaterial;
import io.euhedral_execution.inference.core.InferenceTuning;
import io.euhedral_execution.inference.core.gpu.Q3DispatchMode;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.WeightResidency;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/// Validated `run` configuration, read from a JSON file. Only the three input paths are required;
/// every other field has the default documented in `docs/BENCHMARKING.md`. Unknown fields are
/// rejected. Relative paths resolve against the working directory.
///
/// - `cpus`: `all`, `one-per-core`, `performance`, `performance-one-per-core`, or IDs/ranges such as `2-5,8`.
/// - `prefillChunks`: one engine load per value; each runs every scenario.
/// - `output`: a `.json` path writes one document; any other path writes JSONL. Null selects a
///   timestamped JSONL file under `benchmark-results/`.
/// - `gpuMemory`: record device free/total memory before and after each iteration, outside timing.
/// - `gpuHeadroomMiB`: free device memory required beyond the artifact size before loading.
@JsonIgnoreProperties(ignoreUnknown = false)
public record BenchmarkOptions(
        @JsonProperty(value = "artifact", required = true) Path artifact,
        @JsonProperty(value = "tokenizer", required = true) Path tokenizer,

        @JsonProperty(value = "cudaLibrary", required = true)
        Path cudaLibrary,

        @JsonProperty("cpus") String cpus,
        @JsonProperty("excludeCpus") List<Integer> excludeCpus,
        @JsonProperty("excludeCores") List<Integer> excludeCores,
        @JsonProperty("prefillChunks") List<Integer> prefillChunks,
        @JsonProperty("scenarios") List<Scenario> scenarios,
        @JsonProperty("warmup") Integer warmup,
        @JsonProperty("iterations") Integer iterations,
        @JsonProperty("generation") Generation generation,
        @JsonProperty("promptSeed") Long promptSeed,
        @JsonProperty("output") Path output,
        @JsonProperty("overwrite") Boolean overwrite,
        @JsonProperty("append") Boolean append,
        @JsonProperty("gpuMemory") Boolean gpuMemory,
        @JsonProperty("gpuHeadroomMiB") Long gpuHeadroomMiB,
        @JsonProperty("shutdownTimeoutSeconds") Long shutdownTimeoutSeconds,
        @JsonProperty("q3DispatchMode") Q3DispatchMode q3DispatchMode,
        @JsonProperty("q3SmallRowThreshold") Integer q3SmallRowThreshold,
        @JsonProperty("weightResidency") WeightResidency weightResidency,
        @JsonProperty("hostWeightMiB") Long hostWeightMiB,
        @JsonProperty("stagingSlots") Integer stagingSlots) {

    static final ObjectMapper JSON = new ObjectMapper();

    public BenchmarkOptions {
        Objects.requireNonNull(artifact, "artifact");
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(cudaLibrary, "cudaLibrary");
        cpus = cpus == null ? "all" : cpus.strip();
        excludeCpus = excludeCpus == null ? List.of() : List.copyOf(excludeCpus);
        excludeCores = excludeCores == null ? List.of() : List.copyOf(excludeCores);
        prefillChunks = prefillChunks == null
                ? List.of(InferenceTuning.DEFAULT_PREFILL_CHUNK_TOKENS)
                : List.copyOf(prefillChunks);
        scenarios = scenarios == null ? Scenario.DEFAULT_SUITE : List.copyOf(scenarios);
        warmup = warmup == null ? 1 : warmup;
        iterations = iterations == null ? 3 : iterations;
        generation = generation == null ? new Generation(null, null, null, null, null) : generation;
        promptSeed = promptSeed == null ? PromptMaterial.DEFAULT_SEED : promptSeed;
        overwrite = overwrite != null && overwrite;
        append = append != null && append;
        gpuMemory = gpuMemory != null && gpuMemory;
        gpuHeadroomMiB = gpuHeadroomMiB == null ? 1024L : gpuHeadroomMiB;
        shutdownTimeoutSeconds = shutdownTimeoutSeconds == null ? 10L : shutdownTimeoutSeconds;
        q3DispatchMode = q3DispatchMode == null ? Q3DispatchMode.AUTO : q3DispatchMode;
        q3SmallRowThreshold =
                q3SmallRowThreshold == null ? Q3DispatchMode.DEFAULT_SMALL_ROW_THRESHOLD : q3SmallRowThreshold;
        if (q3SmallRowThreshold < 0) throw new IllegalArgumentException("Q3 threshold must not be negative");
        weightResidency = weightResidency == null ? WeightResidency.ALL : weightResidency;
        hostWeightMiB = hostWeightMiB == null ? 0L : hostWeightMiB;
        if (hostWeightMiB < 0) throw new IllegalArgumentException("hostWeightMiB must not be negative");
        stagingSlots = stagingSlots == null ? QwenModel.DEFAULT_STAGING_SLOTS : stagingSlots;
        if (stagingSlots < 2) throw new IllegalArgumentException("stagingSlots must be at least 2");

        if (cpus.isEmpty()) throw new IllegalArgumentException("cpus must not be blank");
        for (int id : excludeCpus) if (id < 0) throw new IllegalArgumentException("excludeCpus must not be negative");
        for (int id : excludeCores) if (id < 0) throw new IllegalArgumentException("excludeCores must not be negative");
        if (prefillChunks.isEmpty()) throw new IllegalArgumentException("prefillChunks selects no values");
        for (int chunk : prefillChunks)
            if (chunk <= 0) throw new IllegalArgumentException("prefillChunks values must be positive");
        if (prefillChunks.stream().distinct().count() != prefillChunks.size())
            throw new IllegalArgumentException("prefillChunks contains duplicates");
        if (scenarios.isEmpty()) throw new IllegalArgumentException("no scenarios selected");
        if (warmup < 0) throw new IllegalArgumentException("warmup must not be negative");
        if (iterations <= 0) throw new IllegalArgumentException("iterations must be positive");
        if (overwrite && append) throw new IllegalArgumentException("choose either overwrite or append");
        if (append && isJson(output)) throw new IllegalArgumentException("append requires JSONL output");
        if (gpuHeadroomMiB < 0) throw new IllegalArgumentException("gpuHeadroomMiB must not be negative");
        if (shutdownTimeoutSeconds <= 0) throw new IllegalArgumentException("shutdownTimeoutSeconds must be positive");
    }

    /// Compatibility constructor for callers that do not select experimental GPU execution.
    public BenchmarkOptions(
            Path artifact,
            Path tokenizer,
            Path cudaLibrary,
            String cpus,
            List<Integer> excludeCpus,
            List<Integer> excludeCores,
            List<Integer> prefillChunks,
            List<Scenario> scenarios,
            Integer warmup,
            Integer iterations,
            Generation generation,
            Long promptSeed,
            Path output,
            Boolean overwrite,
            Boolean append,
            Boolean gpuMemory,
            Long gpuHeadroomMiB,
            Long shutdownTimeoutSeconds) {
        this(
                artifact,
                tokenizer,
                cudaLibrary,
                cpus,
                excludeCpus,
                excludeCores,
                prefillChunks,
                scenarios,
                warmup,
                iterations,
                generation,
                promptSeed,
                output,
                overwrite,
                append,
                gpuMemory,
                gpuHeadroomMiB,
                shutdownTimeoutSeconds,
                null,
                null,
                null,
                null,
                null);
    }

    /// Sampling settings. `greedy` (default) selects argmax, so temperature/topK/topP are rejected;
    /// `sample` uses them (defaults 0.7, 20, 0.8) with the given seed.
    @JsonIgnoreProperties(ignoreUnknown = false)
    public record Generation(
            @JsonProperty("mode") String mode,
            @JsonProperty("seed") Long seed,
            @JsonProperty("temperature") Float temperature,
            @JsonProperty("topK") Integer topK,
            @JsonProperty("topP") Float topP) {
        public Generation {
            mode = mode == null ? "greedy" : mode;
            seed = seed == null ? 1L : seed;
            switch (mode) {
                case "greedy" -> {
                    if (temperature != null || topK != null || topP != null)
                        throw new IllegalArgumentException("temperature, topK, and topP require mode \"sample\"");
                }
                case "sample" -> {
                    temperature = temperature == null ? 0.7f : temperature;
                    topK = topK == null ? 20 : topK;
                    topP = topP == null ? 0.8f : topP;
                }
                default -> throw new IllegalArgumentException("generation.mode must be greedy or sample");
            }
        }

        public GenerationConfig toConfig() {
            return this.mode.equals("greedy")
                    ? GenerationConfig.greedy(this.seed)
                    : new GenerationConfig(this.temperature, this.topK, this.topP, this.seed, false);
        }
    }

    /// Reads a configuration file, supplying a timestamped JSONL output path when none is set.
    public static BenchmarkOptions load(Path file, Instant now) throws IOException {
        return withDefaultOutput(JSON.readValue(file.toFile(), BenchmarkOptions.class), now);
    }

    static BenchmarkOptions withDefaultOutput(BenchmarkOptions options, Instant now) {
        if (options.output() != null) return options;
        Path output = Path.of("benchmark-results/euhedral-"
                + DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'")
                        .withZone(ZoneOffset.UTC)
                        .format(now)
                + ".jsonl");
        return new BenchmarkOptions(
                options.artifact(),
                options.tokenizer(),
                options.cudaLibrary(),
                options.cpus(),
                options.excludeCpus(),
                options.excludeCores(),
                options.prefillChunks(),
                options.scenarios(),
                options.warmup(),
                options.iterations(),
                options.generation(),
                options.promptSeed(),
                output,
                options.overwrite(),
                options.append(),
                options.gpuMemory(),
                options.gpuHeadroomMiB(),
                options.shutdownTimeoutSeconds(),
                options.q3DispatchMode(),
                options.q3SmallRowThreshold(),
                options.weightResidency(),
                options.hostWeightMiB(),
                options.stagingSlots());
    }

    public boolean json() {
        return isJson(this.output);
    }

    public Duration shutdownTimeout() {
        return Duration.ofSeconds(this.shutdownTimeoutSeconds);
    }

    /// One engine load per prefill-chunk value; each runs every scenario.
    public List<InferenceTuning> sweep(InferenceTuning base) {
        List<InferenceTuning> tunings = new ArrayList<>();
        for (int chunk : this.prefillChunks)
            tunings.add(base.withPrefillChunkTokens(chunk)
                    .withQ3Dispatch(this.q3DispatchMode, this.q3SmallRowThreshold)
                    .withWeightResidency(this.weightResidency)
                    .withHostWeights(this.hostWeightMiB * 1024L * 1024L, this.stagingSlots));
        return tunings;
    }

    private static boolean isJson(Path output) {
        return output != null && output.toString().endsWith(".json");
    }
}
