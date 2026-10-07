package io.euhedral_execution.inference.core.prefix;

/// Counters of a prefix cache, for metrics and tests.
public record PrefixCacheStats(
        long lookups,
        long hits,
        long reusedTokens,
        long captured,
        long skipped,
        long failed,
        long evictions,
        long usedBytes,
        long totalBytes,
        int nodes,
        long captureNanos,
        long restores,
        long restoreNanos) {}
