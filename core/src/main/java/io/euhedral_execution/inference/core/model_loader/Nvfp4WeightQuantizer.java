package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.model_loader.artifact.Nvfp4Layout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/// Round-to-nearest NVFP4 quantization of an FP32 weight matrix into the artifact's NVFP4 layout
/// (Nvfp4Layout): a global FP32 scale amax / (6 · 448), one E4M3 scale per 16 values, E2M1 codes. Used only
/// where a weight has no NVFP4 copy in the artifact (the compact Q3 artifact's W8 MTP attention, which
/// drafts only and never decides an output token).
public final class Nvfp4WeightQuantizer {
    private static final float[] E2M1 = {0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f};
    private static final float[] E4M3 = new float[127];

    static {
        for (int bits = 0; bits < 127; bits++) {
            int exponent = (bits >> 3) & 15, mantissa = bits & 7;
            E4M3[bits] = exponent == 0
                    ? mantissa / 8f * (float) Math.pow(2, -6)
                    : (1 + mantissa / 8f) * (float) Math.pow(2, exponent - 7);
        }
    }

    private Nvfp4WeightQuantizer() {}

    /// Quantizes `rows` × `k` values (row-major) into a new NVFP4 payload.
    public static byte[] quantize(float[] values, int rows, int k) {
        if ((long) rows * k != values.length) throw new IllegalArgumentException("shape mismatch");
        long padded = (k + 127L) / 128 * 128, rowBytes = padded / 2, rowScales = padded / 16;
        long scaleOffset = Nvfp4Layout.scaleOffset(rows, k);
        long globalOffset = Nvfp4Layout.globalScaleOffset(rows, k);
        byte[] out = new byte[Math.toIntExact(globalOffset + Float.BYTES)];
        float amax = 0;
        for (float v : values) amax = Math.max(amax, Math.abs(v));
        float global = amax > 0 ? amax / (6f * 448f) : 1f;
        for (int row = 0; row < rows; row++) {
            for (int block = 0; block < padded / 16; block++) {
                float blockMax = 0;
                for (int i = 0; i < 16; i++) {
                    int col = block * 16 + i;
                    if (col < k) blockMax = Math.max(blockMax, Math.abs(values[row * k + col]));
                }
                int scale = nearest(E4M3, blockMax / 6f / global);
                float step = E4M3[scale] * global;
                out[Math.toIntExact(scaleOffset + row * rowScales + block)] = (byte) scale;
                for (int i = 0; i < 16; i += 2) {
                    int col = block * 16 + i;
                    int low = code(col < k ? values[row * k + col] : 0f, step);
                    int high = code(col + 1 < k ? values[row * k + col + 1] : 0f, step);
                    out[Math.toIntExact(row * rowBytes + col / 2)] = (byte) (low | (high << 4));
                }
            }
        }
        MemorySegment.ofArray(out)
                .set(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), globalOffset, global);
        return out;
    }

    private static int code(float value, float step) {
        if (step == 0) return 0;
        int magnitude = nearest(E2M1, Math.abs(value) / step);
        return value < 0 && magnitude != 0 ? magnitude | 8 : magnitude;
    }

    /// Index of the table value nearest to `x` (tables are ascending); saturates at the largest.
    private static int nearest(float[] table, float x) {
        int best = 0;
        float bestDistance = Float.POSITIVE_INFINITY;
        for (int i = 0; i < table.length; i++) {
            float distance = Math.abs(table[i] - x);
            if (distance < bestDistance) {
                best = i;
                bestDistance = distance;
            }
        }
        return best;
    }
}
