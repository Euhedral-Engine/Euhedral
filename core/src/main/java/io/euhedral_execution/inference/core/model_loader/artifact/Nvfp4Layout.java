package io.euhedral_execution.inference.core.model_loader.artifact;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/// Row-split geometry of [io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat#NVFP4]
/// tensors (tools/convert_qwen_safetensors_to_compact_edrl.py, `--profile nvfp4`). Rows of K values, K
/// padded to a multiple of 128:
///
/// - code plane at 0: E2M1 codes, two per byte, the even K in the low nibble (K/2 bytes per row);
/// - scale plane at align256(code plane): one E4M3 scale per 16 values (K/16 bytes per row);
/// - global scale at align256(scale plane end): one little-endian FP32.
///
/// A weight is e2m1(code) * e4m3(scale) * global.
public final class Nvfp4Layout {

    public static final int BLOCK = 16;
    private static final long PLANE_ALIGNMENT = 256;
    private static final long K_ALIGNMENT = 128;
    private static final ValueLayout.OfFloat GLOBAL =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private Nvfp4Layout() {}

    public static long paddedK(long k) {
        return alignUp(k, K_ALIGNMENT);
    }

    public static long scaleOffset(long rows, long k) {
        return alignUp(Math.multiplyExact(rows, paddedK(k)) / 2, PLANE_ALIGNMENT);
    }

    public static long globalScaleOffset(long rows, long k) {
        return alignUp(
                Math.addExact(scaleOffset(rows, k), Math.multiplyExact(rows, paddedK(k)) / BLOCK), PLANE_ALIGNMENT);
    }

    public static long byteSize(long rows, long k) {
        return Math.addExact(globalScaleOffset(rows, k), Float.BYTES);
    }

    /// Rejects NaN block scales (E4M3 codes 0x7F and 0xFF) and a global scale that is not a finite,
    /// non-negative number; every code is a valid E2M1 value.
    public static void validate(MemorySegment tensor, long rows, long k) {
        if (tensor.byteSize() != byteSize(rows, k)) {
            throw new IllegalArgumentException("NVFP4 tensor size does not match its shape");
        }
        float global = tensor.get(GLOBAL, globalScaleOffset(rows, k));
        if (!Float.isFinite(global) || global < 0) {
            throw new IllegalArgumentException("NVFP4 global scale is not a finite non-negative number: " + global);
        }
        long scales = scaleOffset(rows, k);
        long end = scales + rows * paddedK(k) / BLOCK;
        for (long offset = scales; offset < end; offset++) {
            if ((tensor.get(ValueLayout.JAVA_BYTE, offset) & 0x7f) == 0x7f) {
                throw new IllegalArgumentException("NVFP4 block scale at byte " + (offset - scales) + " is NaN");
            }
        }
    }

    private static long alignUp(long value, long alignment) {
        return Math.addExact(value, alignment - 1) / alignment * alignment;
    }
}
