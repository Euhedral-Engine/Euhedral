package io.euhedral_execution.inference.core.state;

/// How an attention cache stores each token's key and value rows of one KV head (256 values): NVFP4 codes with one
/// E4M3 scale per 16 values, or BF16 as computed.
public enum KvFormat {
    /// 128 code bytes and 16 scale bytes per 256-value head row.
    NVFP4(144),
    /// 256 BF16 values per head row.
    BF16(512);

    /// Bytes of one 256-value head row.
    public final int headRowBytes;

    KvFormat(int headRowBytes) {
        this.headRowBytes = headRowBytes;
    }
}
