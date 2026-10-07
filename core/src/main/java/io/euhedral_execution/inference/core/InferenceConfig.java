package io.euhedral_execution.inference.core;

import io.euhedral_execution.inference.core.model.qwen38.prefix.PrefixCache;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.Objects;

/// Explicit runtime inputs. `workerCpus` are logical processor IDs accepted by Euhedral, not a worker
/// count; the engine validates them against [ProcessorTopology] before loading any model resource, and
/// the set is defensively copied. `maxContextTokens` is the longest sequence (prompt plus generation)
/// the engine keeps device memory for: weights that do not fit beside that much KV cache stay in pinned
/// host memory, chosen automatically. `prefixCacheBytes` is the pinned host memory that keeps the state of
/// earlier prompts for reuse (0 turns the cache off) and `prefixCacheCheckpointTokens` the prompt tokens between
/// stored checkpoints, a multiple of the prefill chunk.
public record InferenceConfig(
        Path artifactPath,
        Path tokenizerDirectory,
        Path cudaLibraryPath,
        BitSet workerCpus,
        int maxContextTokens,
        Duration shutdownTimeout,
        long prefixCacheBytes,
        int prefixCacheCheckpointTokens) {
    public static final int DEFAULT_MAX_CONTEXT_TOKENS = 32768;
    public static final long DEFAULT_PREFIX_CACHE_BYTES = 4L << 30;
    public static final int DEFAULT_PREFIX_CACHE_CHECKPOINT_TOKENS = 2048;

    public InferenceConfig {
        Objects.requireNonNull(artifactPath, "artifactPath");
        Objects.requireNonNull(tokenizerDirectory, "tokenizerDirectory");
        Objects.requireNonNull(cudaLibraryPath, "cudaLibraryPath");
        workerCpus = (BitSet) Objects.requireNonNull(workerCpus, "workerCpus").clone();
        if (workerCpus.isEmpty()) throw new IllegalArgumentException("workerCpus must not be empty");
        if (maxContextTokens <= 0) throw new IllegalArgumentException("maxContextTokens must be positive");
        Objects.requireNonNull(shutdownTimeout, "shutdownTimeout");
        if (shutdownTimeout.isNegative() || shutdownTimeout.isZero())
            throw new IllegalArgumentException("shutdownTimeout must be positive");
        shutdownTimeout.toNanos();
        if (prefixCacheBytes < 0) throw new IllegalArgumentException("prefixCacheBytes must not be negative");
        if (prefixCacheCheckpointTokens <= 0 || prefixCacheCheckpointTokens % PrefixCache.CHUNK_TOKENS != 0)
            throw new IllegalArgumentException(
                    "prefixCacheCheckpointTokens must be a positive multiple of " + PrefixCache.CHUNK_TOKENS);
    }

    /// Without a prefix cache: programmatic callers (tests, the benchmark) opt in explicitly.
    public InferenceConfig(
            Path artifactPath,
            Path tokenizerDirectory,
            Path cudaLibraryPath,
            BitSet workerCpus,
            int maxContextTokens,
            Duration shutdownTimeout) {
        this(
                artifactPath,
                tokenizerDirectory,
                cudaLibraryPath,
                workerCpus,
                maxContextTokens,
                shutdownTimeout,
                0L,
                DEFAULT_PREFIX_CACHE_CHECKPOINT_TOKENS);
    }

    /// The default context capacity.
    public InferenceConfig(
            Path artifactPath,
            Path tokenizerDirectory,
            Path cudaLibraryPath,
            BitSet workerCpus,
            Duration shutdownTimeout) {
        this(
                artifactPath,
                tokenizerDirectory,
                cudaLibraryPath,
                workerCpus,
                DEFAULT_MAX_CONTEXT_TOKENS,
                shutdownTimeout);
    }

    @Override
    public BitSet workerCpus() {
        return (BitSet) this.workerCpus.clone();
    }
}
