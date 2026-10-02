package io.euhedral_execution.inference.core.model_loader.artifact;

import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/// Geometry of [WeightLayout#ROW_SPLIT_P2E2_V1], a lossless and smaller layout of Q3G64_F16S tensors
/// (docs/COMPRESSED_Q3.md, native/src/q3/p2e2.cuh). Rows of `k` codes, `k` a multiple of [#SLICE]:
///
/// - primary plane at 0: 2 bits per code, 16 codes per little-endian word;
/// - row-base plane: one word per row, the payload unit of the row's first BIG code;
/// - scale plane: the row-split FP16 scales, byte for byte;
/// - payload plane: 2-bit units of the BIG codes in row-major order, then [#PAD_WORDS] zero words.
///
/// Each plane starts on a 256-byte boundary. The payload length depends on the codes, so a byte
/// size is valid between an empty and a full payload.
public final class P2e2Layout {

    public static final int SLICE = 1024;
    public static final int PAD_WORDS = 80;
    private static final long PLANE_ALIGNMENT = 256;
    private static final ValueLayout.OfInt WORD = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private P2e2Layout() {}

    public static boolean supports(long[] shape, TensorDataType dataType, WeightFormat format) {
        return shape != null
                && shape.length == 2
                && shape[0] > 0
                && shape[1] > 0
                && shape[1] % SLICE == 0
                && dataType == TensorDataType.BF16
                && format == WeightFormat.Q3_G64_FP16;
    }

    public static long rowBaseOffset(long rows, long k) {
        return alignUp(Math.multiplyExact(rows, k) / 4);
    }

    public static long scaleOffset(long rows, long k) {
        return alignUp(Math.addExact(rowBaseOffset(rows, k), Math.multiplyExact(rows, 4L)));
    }

    public static long payloadOffset(long rows, long k) {
        return alignUp(Math.addExact(scaleOffset(rows, k), Math.multiplyExact(rows, k) / 32));
    }

    public static boolean acceptsByteSize(long rows, long k, long byteSize) {
        long payload = payloadOffset(rows, k);
        long minimum = payload + 4L * PAD_WORDS;
        long maximum = minimum + 4L * ((Math.multiplyExact(rows, k) + 15) / 16);
        return byteSize >= minimum && byteSize <= maximum && (byteSize - payload) % 4 == 0;
    }

    /// Bytes of the row-split Q3 tensor this layout encodes.
    public static long expandedByteSize(long rows, long k) {
        return CompactTensorLayout.expectedByteSize(
                new long[] {rows, k}, TensorDataType.BF16, WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_K128_V1);
    }

    /// Checks the row-base plane against the BIG codes of the primary plane and the payload length,
    /// so that kernels never read past the tensor.
    public static void validate(MemorySegment tensor, long rows, long k) {
        if (!acceptsByteSize(rows, k, tensor.byteSize())) {
            throw new IllegalArgumentException("P2E2 tensor size does not match its shape");
        }
        long words = k / 16;
        long rowBase = rowBaseOffset(rows, k);
        long payload = payloadOffset(rows, k);
        long capacity = ((tensor.byteSize() - payload) / 4 - PAD_WORDS) * 16;
        long units = 0;
        for (long row = 0; row < rows; row++) {
            if (Integer.toUnsignedLong(tensor.get(WORD, rowBase + 4 * row)) != units) {
                throw new IllegalArgumentException("P2E2 row base of row " + row + " does not match its primary plane");
            }
            long offset = 4 * row * words;
            for (long word = 0; word < words; word++) {
                int value = tensor.get(WORD, offset + 4 * word);
                units += Integer.bitCount(value & (value >>> 1) & 0x55555555);
            }
        }
        if (units > capacity) {
            throw new IllegalArgumentException("P2E2 payload plane is shorter than its primary plane requires");
        }
    }

    private static long alignUp(long value) {
        return Math.addExact(value, PLANE_ALIGNMENT - 1) / PLANE_ALIGNMENT * PLANE_ALIGNMENT;
    }
}
