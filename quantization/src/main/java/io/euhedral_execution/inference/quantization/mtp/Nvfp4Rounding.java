package io.euhedral_execution.inference.quantization.mtp;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/// NVFP4 round to nearest as the converter writes it (tools/euhedral_artifacts/nvfp4.py, layout
/// row-split-k128-v1), in float32: the global scale is the largest magnitude over 6 x 448; each block of 16 along a
/// row takes the E4M3 code nearest to its largest magnitude / 6 / global, each value the E2M1 code nearest to
/// |value| / (scale x global); ties go to the even code, values past a table's end to its last entry. The training
/// kernel (`drafter.cu` nvfp4_round) computes the same bytes.
final class Nvfp4Rounding {
    static final float[] E2M1 = {0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f};
    static final float[] E4M3 = new float[127];

    static {
        for (int code = 0; code < 127; code++)
            E4M3[code] = code < 8
                    ? (code & 7) / 8f * 0.015625f
                    : (1 + (code & 7) / 8f) * (float) Math.pow(2, (code >> 3) - 7);
    }

    private Nvfp4Rounding() {}

    /// The nearest entry of an ascending table (numpy searchsorted side left, then the nearer neighbour).
    static int roundToTable(float value, float[] table) {
        int low = 0, high = table.length;
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (table[middle] < value) low = middle + 1;
            else high = middle;
        }
        int upper = Math.min(Math.max(low, 1), table.length - 1), lower = upper - 1;
        float below = value - table[lower], above = table[upper] - value;
        int index = above < below || (above == below && upper % 2 == 0) ? upper : lower;
        if (value >= table[table.length - 1]) index = table.length - 1;
        if (value <= table[0]) index = 0;
        return index;
    }

    /// Bytes of an NVFP4 object [rows, k] (k a multiple of 128) of row-major float32 `w`: packed codes, the scale
    /// plane and the global scale, each plane on a 256-byte boundary.
    static long bytes(long rows, int k) {
        long scaleOffset = align((long) rows * k / 2);
        return align(scaleOffset + (long) rows * k / 16) + 4;
    }

    /// Writes the NVFP4 object of `w` [rows, k] into `out` ([#bytes] long, zeroed).
    static void round(MemorySegment w, long rows, int k, MemorySegment out) {
        if (k % 128 != 0) throw new IllegalArgumentException("row length " + k + " is not a multiple of 128");
        long count = rows * k;
        float amax = 0;
        for (long i = 0; i < count; i++) amax = Math.max(amax, Math.abs(w.getAtIndex(ValueLayout.JAVA_FLOAT, i)));
        float global = amax / 2688f;
        long scaleOffset = align(count / 2), globalOffset = align(scaleOffset + count / 16);
        int[] codes = new int[16];
        for (long block = 0; block < count / 16; block++) {
            float largest = 0;
            for (int i = 0; i < 16; i++)
                largest = Math.max(largest, Math.abs(w.getAtIndex(ValueLayout.JAVA_FLOAT, block * 16 + i)));
            int scale = global > 0 ? roundToTable(largest / 6f / global, E4M3) : 0;
            float decode = E4M3[scale] * global;
            for (int i = 0; i < 16; i++) {
                float v = w.getAtIndex(ValueLayout.JAVA_FLOAT, block * 16 + i);
                int magnitude = roundToTable(decode > 0 ? Math.abs(v) / decode : 0f, E2M1);
                codes[i] = magnitude | (v < 0 && magnitude > 0 ? 8 : 0);
            }
            for (int i = 0; i < 8; i++)
                out.set(ValueLayout.JAVA_BYTE, block * 8 + i, (byte) (codes[2 * i] | (codes[2 * i + 1] << 4)));
            out.set(ValueLayout.JAVA_BYTE, scaleOffset + block, (byte) scale);
        }
        out.set(ValueLayout.JAVA_FLOAT_UNALIGNED, globalOffset, global);
    }

    private static long align(long value) {
        return (value + 255) & ~255L;
    }
}
