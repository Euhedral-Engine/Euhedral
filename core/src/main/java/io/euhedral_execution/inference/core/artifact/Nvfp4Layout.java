package io.euhedral_execution.inference.core.artifact;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/// Row-split geometry of [io.euhedral_execution.inference.core.artifact.WeightFormat#NVFP4]
/// tensors (tools/euhedral_artifacts/nvfp4.py). Rows of K values, K
/// padded to a multiple of 128:
///
/// - code plane at 0: E2M1 codes, two per byte, the even K in the low nibble (K/2 bytes per row);
/// - scale plane at align256(code plane): one E4M3 scale per 16 values (K/16 bytes per row);
/// - global scale at align256(scale plane end): one little-endian FP32.
///
/// A weight is e2m1(code) * e4m3(scale) * global.
///
/// [WeightLayout#ROW_SPLIT_K128_SD4_V1] (`--profile nvfp4-sd4`) stores a 4-bit index per block instead,
/// two per byte with the even block in the low nibble (K/32 bytes per row), and at align256(scale plane
/// end) the table of 16 E4M3 codes the indices select, followed by the global scale.
public final class Nvfp4Layout {

    public static final int BLOCK = 16;
    /// Entries of an SD4 tensor's scale table.
    public static final int TABLE = 16;
    private static final long PLANE_ALIGNMENT = 256;
    private static final long K_ALIGNMENT = 128;
    private static final ValueLayout.OfFloat GLOBAL =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private Nvfp4Layout() {}

    public static long paddedK(long k) {
        return alignUp(k, K_ALIGNMENT);
    }

    /// Bytes of the scale plane per row: one per block, or (SD4) one per two blocks.
    public static long rowScaleBytes(long k, WeightLayout layout) {
        return paddedK(k) / (sd4(layout) ? 2 * BLOCK : BLOCK);
    }

    public static long scaleOffset(long rows, long k) {
        return alignUp(Math.multiplyExact(rows, paddedK(k)) / 2, PLANE_ALIGNMENT);
    }

    /// Offset of the trailer: the global scale, or (SD4) the scale table followed by the global scale.
    public static long trailerOffset(long rows, long k, WeightLayout layout) {
        return alignUp(
                Math.addExact(scaleOffset(rows, k), Math.multiplyExact(rows, rowScaleBytes(k, layout))),
                PLANE_ALIGNMENT);
    }

    public static long trailerBytes(WeightLayout layout) {
        return (sd4(layout) ? TABLE : 0) + Float.BYTES;
    }

    public static long globalScaleOffset(long rows, long k) {
        return trailerOffset(rows, k, WeightLayout.ROW_SPLIT_K128_V1);
    }

    public static long byteSize(long rows, long k) {
        return byteSize(rows, k, WeightLayout.ROW_SPLIT_K128_V1);
    }

    public static long byteSize(long rows, long k, WeightLayout layout) {
        return Math.addExact(trailerOffset(rows, k, layout), trailerBytes(layout));
    }

    public static void validate(MemorySegment tensor, long rows, long k) {
        validate(tensor, rows, k, WeightLayout.ROW_SPLIT_K128_V1);
    }

    /// Rejects NaN block scales (E4M3 codes 0x7F and 0xFF; for SD4, in the table, which every index can
    /// select) and a global scale that is not a finite, non-negative number; every code is a valid E2M1
    /// value.
    public static void validate(MemorySegment tensor, long rows, long k, WeightLayout layout) {
        if (tensor.byteSize() != byteSize(rows, k, layout)) {
            throw new IllegalArgumentException("NVFP4 tensor size does not match its shape");
        }
        long trailer = trailerOffset(rows, k, layout);
        float global = tensor.get(GLOBAL, trailer + trailerBytes(layout) - Float.BYTES);
        if (!Float.isFinite(global) || global < 0) {
            throw new IllegalArgumentException("NVFP4 global scale is not a finite non-negative number: " + global);
        }
        long scales = sd4(layout) ? trailer : scaleOffset(rows, k);
        long end = sd4(layout) ? trailer + TABLE : scales + rows * rowScaleBytes(k, layout);
        for (long offset = scales; offset < end; offset++) {
            if ((tensor.get(ValueLayout.JAVA_BYTE, offset) & 0x7f) == 0x7f) {
                throw new IllegalArgumentException("NVFP4 block scale at byte " + (offset - scales) + " is NaN");
            }
        }
    }

    public static boolean supports(WeightLayout layout) {
        return layout == WeightLayout.ROW_SPLIT_K128_V1 || layout == WeightLayout.ROW_SPLIT_K128_SD4_V1;
    }

    private static boolean sd4(WeightLayout layout) {
        if (!supports(layout)) throw new IllegalArgumentException("not an NVFP4 layout: " + layout);
        return layout == WeightLayout.ROW_SPLIT_K128_SD4_V1;
    }

    private static long alignUp(long value, long alignment) {
        return Math.addExact(value, alignment - 1) / alignment * alignment;
    }
}
