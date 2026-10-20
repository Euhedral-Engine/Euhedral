package io.euhedral_execution.inference.core;

import io.euhedral_execution.inference.core.nativefiles.NativeFiles;
import io.euhedral_execution.inference.core.settings.Settings;
import io.euhedral_execution.inference.core.state.KvFormat;
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
/// stored checkpoints, a multiple of the prefill chunk. `kvCache` is the format sequences keep their attention
/// caches in; null takes the artifact's. `maxSequences` is the number of concurrent sequences device memory is
/// sized for: at least 1, planned exactly as asked (weights may leave the device to make room), or 0 for as many
/// as fit without offloading more weights than one sequence does. `kvDeviceMinimumTokens` is the tokens of each
/// sequence's attention cache that stay on the device: when the weights and every sequence's whole cache do not fit,
/// the
/// pages past it live in pinned host memory before any weight leaves the device.
public record InferenceConfig(
        Path artifactPath,
        Path tokenizerDirectory,
        Path cudaLibraryPath,
        BitSet workerCpus,
        int maxContextTokens,
        Duration shutdownTimeout,
        long prefixCacheBytes,
        int prefixCacheCheckpointTokens,
        KvFormat kvCache,
        int maxSequences,
        int kvDeviceMinimumTokens) {
    /// Prompt tokens per prefill quantum, and the grain the prefix cache's checkpoints are aligned to.
    public static final int PREFILL_CHUNK_TOKENS = 512;

    /// The rows a qwen38 workspace holds per quantum, the longest prefill chunk it runs:
    /// `euhedral.workspace.rows`, by default one prefill chunk.
    public static int workspaceRows() {
        return Integer.getInteger("euhedral.workspace.rows", PREFILL_CHUNK_TOKENS);
    }

    public static final int DEFAULT_MAX_CONTEXT_TOKENS = 32768;
    public static final long DEFAULT_PREFIX_CACHE_BYTES = 4L << 30;
    public static final int DEFAULT_PREFIX_CACHE_CHECKPOINT_TOKENS = 2048;
    public static final int DEFAULT_KV_DEVICE_MINIMUM_TOKENS = 65536;

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
        if (prefixCacheCheckpointTokens <= 0 || prefixCacheCheckpointTokens % PREFILL_CHUNK_TOKENS != 0)
            throw new IllegalArgumentException(
                    "prefixCacheCheckpointTokens must be a positive multiple of " + PREFILL_CHUNK_TOKENS);
        if (maxSequences < 0) throw new IllegalArgumentException("maxSequences must not be negative");
        if (kvDeviceMinimumTokens <= 0) throw new IllegalArgumentException("kvDeviceMinimumTokens must be positive");
    }

    /// The engine input the settings describe, with the native library `nativeFiles` holds.
    public static InferenceConfig of(Settings settings, NativeFiles nativeFiles) {
        return new InferenceConfig(
                settings.model().artifact(),
                settings.model().tokenizer(),
                nativeFiles.library(),
                settings.host().selectedCpus(),
                settings.context().maxTokens(),
                settings.server().shutdownTimeout(),
                settings.prefixCache().size(),
                settings.prefixCache().checkpointTokens(),
                settings.context().kvCacheFormat().orElse(null),
                settings.batching().maxSequences(),
                settings.context().kvCacheOnGpuTokens());
    }

    /// The default device share of each sequence's attention cache.
    public InferenceConfig(
            Path artifactPath,
            Path tokenizerDirectory,
            Path cudaLibraryPath,
            BitSet workerCpus,
            int maxContextTokens,
            Duration shutdownTimeout,
            long prefixCacheBytes,
            int prefixCacheCheckpointTokens,
            KvFormat kvCache,
            int maxSequences) {
        this(
                artifactPath,
                tokenizerDirectory,
                cudaLibraryPath,
                workerCpus,
                maxContextTokens,
                shutdownTimeout,
                prefixCacheBytes,
                prefixCacheCheckpointTokens,
                kvCache,
                maxSequences,
                DEFAULT_KV_DEVICE_MINIMUM_TOKENS);
    }

    /// One sequence.
    public InferenceConfig(
            Path artifactPath,
            Path tokenizerDirectory,
            Path cudaLibraryPath,
            BitSet workerCpus,
            int maxContextTokens,
            Duration shutdownTimeout,
            long prefixCacheBytes,
            int prefixCacheCheckpointTokens,
            KvFormat kvCache) {
        this(
                artifactPath,
                tokenizerDirectory,
                cudaLibraryPath,
                workerCpus,
                maxContextTokens,
                shutdownTimeout,
                prefixCacheBytes,
                prefixCacheCheckpointTokens,
                kvCache,
                1);
    }

    /// The artifact's attention cache format.
    public InferenceConfig(
            Path artifactPath,
            Path tokenizerDirectory,
            Path cudaLibraryPath,
            BitSet workerCpus,
            int maxContextTokens,
            Duration shutdownTimeout,
            long prefixCacheBytes,
            int prefixCacheCheckpointTokens) {
        this(
                artifactPath,
                tokenizerDirectory,
                cudaLibraryPath,
                workerCpus,
                maxContextTokens,
                shutdownTimeout,
                prefixCacheBytes,
                prefixCacheCheckpointTokens,
                null);
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
