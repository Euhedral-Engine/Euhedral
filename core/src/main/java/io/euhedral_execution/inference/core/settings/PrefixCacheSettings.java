package io.euhedral_execution.inference.core.settings;

/// The cache of prompt prefixes shared between requests.
///
/// @param size bytes of host memory the cache may hold; 0 disables it
/// @param checkpointTokens tokens between stored checkpoints, a positive multiple of 512
public record PrefixCacheSettings(long size, int checkpointTokens) {

    public PrefixCacheSettings {
        if (size < 0) throw new IllegalArgumentException("size must not be negative");
        if (checkpointTokens < 512 || checkpointTokens % 512 != 0)
            throw new IllegalArgumentException("checkpoint-tokens must be a positive multiple of 512");
    }

    public static PrefixCacheSettings defaults() {
        return new PrefixCacheSettings(4L << 30, 2048);
    }
}
