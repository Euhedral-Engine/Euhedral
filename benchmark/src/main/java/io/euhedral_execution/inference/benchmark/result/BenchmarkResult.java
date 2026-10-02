package io.euhedral_execution.inference.benchmark.result;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import java.util.Objects;

/// One benchmark row, schema `euhedral-inference.benchmark-result` version 1.
///
/// Rows are self-describing: `implementation` and `provenance` say what produced them, `runId` and
/// `fork` identify the JVM, and `engine` holds the engine's `InferenceRunSnapshot` (tuning, model,
/// runtime identity, and generation settings) for measured rows. Timings are integer nanoseconds and
/// throughputs are tokens per second; either is null when the work it describes did not complete.
/// See `docs/BENCHMARKING.md` for every metric's boundaries and denominator.
@JsonInclude(JsonInclude.Include.ALWAYS)
public record BenchmarkResult(
        String schema,
        int schemaVersion,
        String implementation,
        Provenance provenance,
        String runId,
        Fork fork,
        String recordedAt,
        ScenarioRecord scenario,
        boolean warmup,
        int iteration,
        String status,
        String statusReason,
        Work work,
        Timings timings,
        Throughput throughput,
        JsonNode engine,
        GpuMemory gpuMemory) {
    public static final String SCHEMA = "euhedral-inference.benchmark-result";
    public static final int SCHEMA_VERSION = 1;
    public static final String IMPLEMENTATION = "euhedral-inference";
    public static final String SUCCESS = "success";
    public static final String INELIGIBLE = "ineligible";
    public static final String FAILED = "failed";

    public static final ObjectMapper JSON =
            new ObjectMapper().enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);

    public BenchmarkResult {
        Objects.requireNonNull(schema, "schema");
        Objects.requireNonNull(implementation, "implementation");
        Objects.requireNonNull(provenance, "provenance");
        Objects.requireNonNull(scenario, "scenario");
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(work, "work");
    }

    public BenchmarkResult withSource(String implementation, Provenance provenance) {
        return new BenchmarkResult(
                this.schema,
                this.schemaVersion,
                implementation,
                provenance,
                this.runId,
                this.fork,
                this.recordedAt,
                this.scenario,
                this.warmup,
                this.iteration,
                this.status,
                this.statusReason,
                this.work,
                this.timings,
                this.throughput,
                this.engine,
                this.gpuMemory);
    }

    public String toJson() {
        try {
            return JSON.writeValueAsString(this);
        } catch (com.fasterxml.jackson.core.JsonProcessingException failure) {
            throw new IllegalStateException(failure);
        }
    }

    /// `kind` is `measured` for harness rows and `imported` for external rows. Imported rows name
    /// their `source` file and its `sourceSha256`; `note` is free text supplied by the importer.
    public record Provenance(
            String kind, String tool, String commandLine, String source, String sourceSha256, String note) {}

    /// `id` is an externally supplied fork identifier (`source = external`), or null when this JVM was
    /// not declared as a separate fork (`source = unspecified`). Iterations inside one JVM share
    /// `pid` and `runId` and are never distinct forks.
    public record Fork(String id, String source, Long pid, String jvmStartedAt) {}

    /// `promptSha256` identifies the exact prompt text; see [PromptMaterial] for regeneration.
    public record ScenarioRecord(
            String name,
            String kind,
            int targetPromptTokens,
            int requestedNewTokens,
            String promptGenerator,
            Long promptSeed,
            String promptSha256) {}

    /// Actual completed work. `generatedTokens` counts every returned token ID, including a sampled
    /// terminator. `decodeSampledTokens` counts tokens sampled by decode quanta (tokens 2..N);
    /// `finalCommitQuanta` is 1 when the last token was committed by a decode quantum that sampled nothing.
    public record Work(
            Integer promptTokens,
            Integer prefillQuanta,
            Integer prefillTokens,
            Integer generatedTokens,
            Integer decodeSampledTokens,
            Integer finalCommitQuanta,
            Boolean eosObserved,
            Speculative speculative,
            String outputSha256) {
        public Work(
                Integer promptTokens,
                Integer prefillQuanta,
                Integer prefillTokens,
                Integer generatedTokens,
                Integer decodeSampledTokens,
                Integer finalCommitQuanta,
                Boolean eosObserved,
                Speculative speculative) {
            this(
                    promptTokens,
                    prefillQuanta,
                    prefillTokens,
                    generatedTokens,
                    decodeSampledTokens,
                    finalCommitQuanta,
                    eosObserved,
                    speculative,
                    null);
        }

        public Work(
                Integer promptTokens,
                Integer prefillQuanta,
                Integer prefillTokens,
                Integer generatedTokens,
                Integer decodeSampledTokens,
                Integer finalCommitQuanta,
                Boolean eosObserved) {
            this(
                    promptTokens,
                    prefillQuanta,
                    prefillTokens,
                    generatedTokens,
                    decodeSampledTokens,
                    finalCommitQuanta,
                    eosObserved,
                    null,
                    null);
        }

        /// SHA-256 of the returned token IDs (decimal, comma-separated): equal across arms that must
        /// generate the same tokens, such as greedy decode with and without speculative decoding.
        public static String outputSha256(List<Integer> tokens) {
            StringBuilder text = new StringBuilder();
            for (int token : tokens) text.append(text.isEmpty() ? "" : ",").append(token);
            return io.euhedral_execution.inference.benchmark.prompt.PromptMaterial.sha256(text.toString());
        }
    }

    /// Speculative decoding: verifications, `acceptedDrafts[a]` = verifications that accepted a drafts,
    /// output tokens they committed, and the time spent verifying versus drafting between them (ns).
    public record Speculative(
            int verifications, long[] acceptedDrafts, int committedTokens, long verifyNanos, long draftNanos) {}

    /// Nanosecond durations. Null means the boundary was not reached.
    public record Timings(
            Long tokenization,
            Long prefill,
            Long firstTokenSample,
            Long timeToFirstToken,
            Long decode,
            Long decodeQuantaSum,
            Long finalCommit,
            Long timeToLastToken,
            Long endToEnd) {}

    public record Throughput(
            Double prefillTokensPerSecond, Double decodeTokensPerSecond, Double endToEndOutputTokensPerSecond) {}

    /// Device-wide free/total bytes before and after the iteration, observed outside timing.
    /// Device free/total around the iteration, and the engine's own live allocations: before, the peak
    /// during the iteration (sequence state, workspaces and scratch included), after, and the graphs'
    /// retained workspace after.
    public record GpuMemory(
            Long beforeFreeBytes,
            Long afterFreeBytes,
            Long totalBytes,
            Long allocatedBeforeBytes,
            Long peakAllocatedBytes,
            Long allocatedAfterBytes,
            Long retainedWorkspaceBytes) {}
}
