package io.euhedral_execution.inference.core.artifact;

import io.euhedral_execution.inference.core.artifact.format.LinearFormat;
import io.euhedral_execution.inference.core.artifact.format.LinearFormats;
import io.euhedral_execution.inference.core.artifact.format.PayloadSizes;
import java.util.Optional;

/// Validates and sizes the compact EDRL layouts used by the offline converter.
public final class CompactTensorLayout {

    private CompactTensorLayout() {}

    /// Whether `byteSize` is a valid payload size: what the registered [LinearFormat] accepts for `format` and
    /// `layout` (exactly its size for the fixed-size layouts, P2E2's bounds for P2E2, whose size depends on the
    /// codes).
    public static boolean acceptsByteSize(
            long[] shape, TensorDataType dataType, WeightFormat format, WeightLayout layout, long byteSize) {
        Optional<LinearFormat> registered = LinearFormats.find(format, layout);
        if (registered.isPresent()) {
            if (layout != WeightLayout.ROW_SPLIT_P2E2_V1) requireMetadata(shape, dataType, format, layout);
            return registered.get().acceptsByteSize(shape, dataType, byteSize);
        }
        if (layout == WeightLayout.ROW_SPLIT_P2E2_V1) {
            throw new IllegalArgumentException(
                    "P2E2 layout requires Q3 BF16 metadata and a rank-2 shape whose K is a multiple of "
                            + P2e2Layout.SLICE);
        }
        return expectedByteSize(shape, dataType, format, layout) == byteSize;
    }

    public static long expectedByteSize(
            long[] shape, TensorDataType dataType, WeightFormat format, WeightLayout layout) {
        requireMetadata(shape, dataType, format, layout);
        Optional<LinearFormat> registered = LinearFormats.find(format, layout);
        if (registered.isPresent()) return registered.get().byteSize(shape, dataType);
        return switch (layout) {
            case CONTIGUOUS_LE_V1 -> PayloadSizes.directSize(shape, dataType, format);
            case ROW_SPLIT_K128_V1 -> {
                if (dataType != TensorDataType.BF16) {
                    throw new IllegalArgumentException("grouped compact storage requires BF16 source metadata");
                }
                if (shape.length != 2 || shape[0] <= 0 || shape[1] <= 0) {
                    throw new IllegalArgumentException("row-split layout requires a positive rank-2 shape");
                }
                throw new IllegalArgumentException("unsupported row-split format " + format);
            }
            case ROW_SPLIT_P2E2_V1 -> throw new IllegalArgumentException("P2E2 tensor sizes depend on their codes");
            case ROW_INTERLEAVED_NVFP4_V1 ->
                throw new IllegalArgumentException("interleaved NVFP4 rows belong to version 3 artifacts");
            case TRELLIS_TILE16_V1 ->
                throw new IllegalArgumentException("trellis tensors are sized by the registered trellis format");
            case ROW_SPLIT_K128_SD4_V1 ->
                throw new IllegalArgumentException("SD4 layout requires NVFP4 storage of BF16 source metadata");
        };
    }

    private static void requireMetadata(
            long[] shape, TensorDataType dataType, WeightFormat format, WeightLayout layout) {
        if (shape == null || dataType == null || format == null || layout == null) {
            throw new IllegalArgumentException("compact descriptor metadata is incomplete");
        }
    }
}
