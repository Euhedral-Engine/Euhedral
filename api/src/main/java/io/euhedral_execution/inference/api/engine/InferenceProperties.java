package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.ProcessorTopology;
import io.euhedral_execution.inference.core.state.KvFormat;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/// Engine inputs bound from `euhedral.inference.*` (or `EUHEDRAL_INFERENCE_*` environment variables).
///
/// `workerCpus` lists processor IDs accepted by Euhedral, not a worker count: comma-separated IDs and
/// inclusive ranges such as `2,3,8-11`; unset, the processors of as many cores as the process has CPU time for (all
/// of them without a cgroup CPU quota), and Euhedral pins one worker per core. `modelId` is the public name clients
/// send as `model`.
/// `maxContextTokens` is the longest prompt plus completion a request may use (default 32768); the
/// engine keeps device memory for that much context and holds back weights in pinned host memory when
/// the GPU cannot fit both. `prefixCacheBytes` is the pinned host memory that keeps the state of earlier prompts,
/// so a request sharing a prefix with one of them prefills only what follows (default 4 GiB, 0 turns it off);
/// `prefixCacheCheckpointTokens` is the prompt tokens between stored checkpoints, a multiple of 512. `kvCache` is the
/// format of the attention cache, `nvfp4` or `bf16`; unset, the artifact's (NVFP4). `maxSequences` is the
/// generations that run at once, each with its own sequence state in device memory (default 1; 0 takes as many as
/// fit without holding back more weights than one sequence does).
@ConfigurationProperties("euhedral.inference")
public record InferenceProperties(
        Path artifactPath,
        Path tokenizerDirectory,
        Path cudaLibraryPath,
        String workerCpus,
        @DefaultValue("32768") int maxContextTokens,
        @DefaultValue("10s") Duration shutdownTimeout,
        String modelId,
        @DefaultValue("4294967296") long prefixCacheBytes,
        @DefaultValue("2048") int prefixCacheCheckpointTokens,
        KvFormat kvCache,
        @DefaultValue("1") int maxSequences) {

    public InferenceProperties {
        require(artifactPath, "artifact-path");
        require(tokenizerDirectory, "tokenizer-directory");
        require(cudaLibraryPath, "cuda-library-path");
        require(shutdownTimeout, "shutdown-timeout");
        require(modelId, "model-id");
        if (modelId.isBlank()) throw new IllegalArgumentException("euhedral.inference.model-id must not be blank");
        if (workerCpus != null) parseCpus(workerCpus);
        if (maxContextTokens <= 0)
            throw new IllegalArgumentException("euhedral.inference.max-context-tokens must be positive");
        if (prefixCacheBytes < 0)
            throw new IllegalArgumentException("euhedral.inference.prefix-cache-bytes must not be negative");
        if (prefixCacheCheckpointTokens <= 0 || prefixCacheCheckpointTokens % 512 != 0)
            throw new IllegalArgumentException(
                    "euhedral.inference.prefix-cache-checkpoint-tokens must be a positive multiple of 512");
        if (maxSequences < 0)
            throw new IllegalArgumentException("euhedral.inference.max-sequences must not be negative");
    }

    /// Converts to the core record, which performs its own lifetime and CPU-set validation.
    public InferenceConfig toInferenceConfig() {
        return new InferenceConfig(
                this.artifactPath,
                this.tokenizerDirectory,
                this.cudaLibraryPath,
                selectedCpus(),
                this.maxContextTokens,
                this.shutdownTimeout,
                this.prefixCacheBytes,
                this.prefixCacheCheckpointTokens,
                this.kvCache,
                this.maxSequences);
    }

    /// The selected processors: the configured ones, or, when none are configured, the available processors of as
    /// many cores as the CPU time the process may use (its cgroup quota, which the JVM reports as available
    /// processors); without a quota that is every available processor.
    BitSet selectedCpus() {
        if (workerCpus != null && !workerCpus.isBlank()) return parseCpus(workerCpus);
        return ProcessorTopology.system().processorsOfCores(Runtime.getRuntime().availableProcessors());
    }

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
                throw new IllegalArgumentException(
                        "euhedral.inference.worker-cpus has an invalid processor ID or range: " + item);
            }
        }
        if (cpus.isEmpty()) throw new IllegalArgumentException("euhedral.inference.worker-cpus selects no processors");
        return cpus;
    }

    private static void require(Object value, String property) {
        if (value == null) throw new IllegalArgumentException("euhedral.inference." + property + " must be set");
    }
}
