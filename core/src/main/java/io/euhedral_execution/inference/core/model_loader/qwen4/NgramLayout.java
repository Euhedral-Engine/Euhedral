package io.euhedral_execution.inference.core.model_loader.qwen4;

import io.euhedral_execution.inference.core.model_loader.artifact.Nvfp4Layout;

/// Geometry of [io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout#ROW_INTERLEAVED_NVFP4_V1]
/// tables (the n-gram embedding shards): `rows` rows of `rowBytes(k)` bytes with no padding, each the row's `k / 2`
/// bytes
/// of E2M1 codes (even column in the low nibble) followed by its `k / 16` E4M3 block scales; then, at the next
/// 256-byte boundary after the last row, the table's little-endian FP32 global scale. A value decodes to
/// `e2m1(code) * e4m3(scale) * global`.
///
/// Unlike the row-split layout, one row is one contiguous range, so a lookup reads `rowBytes(k)` bytes from one place.
public final class NgramLayout {

    private static final long TRAILER_ALIGNMENT = 256;

    private NgramLayout() {}

    public static boolean supports(long k) {
        return k > 0 && k % Nvfp4Layout.BLOCK == 0;
    }

    public static long rowBytes(long k) {
        if (!supports(k)) throw new IllegalArgumentException("row width must be a positive multiple of 16: " + k);
        return k / 2 + k / Nvfp4Layout.BLOCK;
    }

    /// Offset of row `row`.
    public static long rowOffset(long row, long k) {
        return Math.multiplyExact(row, rowBytes(k));
    }

    public static long trailerOffset(long rows, long k) {
        long body = Math.multiplyExact(rows, rowBytes(k));
        return Math.addExact(body, TRAILER_ALIGNMENT - 1) / TRAILER_ALIGNMENT * TRAILER_ALIGNMENT;
    }

    public static long byteSize(long rows, long k) {
        return Math.addExact(trailerOffset(rows, k), Float.BYTES);
    }
}
