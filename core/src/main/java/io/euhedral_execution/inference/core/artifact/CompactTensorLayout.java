package io.euhedral_execution.inference.core.artifact;

import io.euhedral_execution.inference.core.artifact.format.LinearFormat;
import io.euhedral_execution.inference.core.artifact.format.LinearFormats;
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
            case CONTIGUOUS_LE_V1 -> directSize(shape, dataType, format);
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

    /// The size of a direct (non-quantized) payload of `shape` in `format`.
    public static long directSize(long[] shape, TensorDataType dataType, WeightFormat format) {
        if (!isDirectFormat(format)) {
            throw new IllegalArgumentException("contiguous layout does not support " + format);
        }
        long elements = 1;
        for (long dimension : shape) {
            if (dimension <= 0 || elements > Long.MAX_VALUE / dimension) {
                throw new IllegalArgumentException("compact shape is invalid or overflows");
            }
            elements *= dimension;
        }
        int wordBytes =
                switch (format) {
                    case BF16, FP16 -> 2;
                    case FP32, I32 -> 4;
                    default -> throw new IllegalArgumentException("unsupported direct format " + format);
                };
        if (dataType == TensorDataType.INT32 && format != WeightFormat.I32) {
            throw new IllegalArgumentException("INT32 source dtype requires I32 storage metadata");
        }
        if (format == WeightFormat.BF16 && dataType != TensorDataType.BF16) {
            throw new IllegalArgumentException("BF16 storage requires BF16 source metadata");
        }
        if (format == WeightFormat.FP16 && dataType != TensorDataType.FP16) {
            throw new IllegalArgumentException("FP16 storage requires FP16 source metadata");
        }
        if (format == WeightFormat.I32 && dataType != TensorDataType.INT32) {
            throw new IllegalArgumentException("I32 storage requires INT32 source metadata");
        }
        if (elements > Long.MAX_VALUE / wordBytes) {
            throw new IllegalArgumentException("compact payload size overflows");
        }
        return elements * wordBytes;
    }

    private static boolean isDirectFormat(WeightFormat format) {
        return format == WeightFormat.BF16
                || format == WeightFormat.FP16
                || format == WeightFormat.FP32
                || format == WeightFormat.I32;
    }

    public static long alignUp(long value, long alignment) {
        if (value < 0 || value > Long.MAX_VALUE - alignment + 1) {
            throw new IllegalArgumentException("compact size overflows");
        }
        return (value + alignment - 1) / alignment * alignment;
    }

    public static long multiply(long left, long right) {
        if (left < 0 || right < 0 || (right != 0 && left > Long.MAX_VALUE / right)) {
            throw new IllegalArgumentException("compact size overflows");
        }
        return left * right;
    }

    public static long add(long left, long right) {
        if (left < 0 || right < 0 || left > Long.MAX_VALUE - right) {
            throw new IllegalArgumentException("compact size overflows");
        }
        return left + right;
    }
}
