package io.euhedral_execution.inference.core.model_loader.artifact;

import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;

/// Validates and sizes the compact EDRL layouts used by the offline converter.
public final class CompactTensorLayout {

    private static final long PLANE_ALIGNMENT = 256;
    private static final long K_ALIGNMENT = 128;

    private CompactTensorLayout() {}

    /// Whether `byteSize` is a valid payload size: exactly [#expectedByteSize] for the fixed-size
    /// layouts, and within [P2e2Layout]'s bounds for P2E2, whose size depends on the codes.
    public static boolean acceptsByteSize(
            long[] shape, TensorDataType dataType, WeightFormat format, WeightLayout layout, long byteSize) {
        if (layout == WeightLayout.ROW_SPLIT_P2E2_V1) {
            if (!P2e2Layout.supports(shape, dataType, format)) {
                throw new IllegalArgumentException(
                        "P2E2 layout requires Q3 BF16 metadata and a rank-2 shape whose K is a multiple of "
                                + P2e2Layout.SLICE);
            }
            return P2e2Layout.acceptsByteSize(shape[0], shape[1], byteSize);
        }
        return expectedByteSize(shape, dataType, format, layout) == byteSize;
    }

    public static long expectedByteSize(
            long[] shape, TensorDataType dataType, WeightFormat format, WeightLayout layout) {
        if (shape == null || dataType == null || format == null || layout == null) {
            throw new IllegalArgumentException("compact descriptor metadata is incomplete");
        }
        return switch (layout) {
            case CONTIGUOUS_LE_V1 -> directSize(shape, dataType, format);
            case ROW_SPLIT_K128_V1 -> {
                if (dataType != TensorDataType.BF16) {
                    throw new IllegalArgumentException("grouped compact storage requires BF16 source metadata");
                }
                yield rowSplitSize(shape, format);
            }
            case ROW_SPLIT_P2E2_V1 -> throw new IllegalArgumentException("P2E2 tensor sizes depend on their codes");
            case ROW_INTERLEAVED_NVFP4_V1 ->
                throw new IllegalArgumentException("interleaved NVFP4 rows belong to version 3 artifacts");
            case ROW_SPLIT_K128_SD4_V1 -> {
                if (dataType != TensorDataType.BF16 || format != WeightFormat.NVFP4) {
                    throw new IllegalArgumentException("SD4 layout requires NVFP4 storage of BF16 source metadata");
                }
                if (shape.length != 2 || shape[0] <= 0 || shape[1] <= 0) {
                    throw new IllegalArgumentException("row-split layout requires a positive rank-2 shape");
                }
                yield Nvfp4Layout.byteSize(shape[0], shape[1], layout);
            }
        };
    }

    private static long directSize(long[] shape, TensorDataType dataType, WeightFormat format) {
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

    private static long rowSplitSize(long[] shape, WeightFormat format) {
        if (shape.length != 2 || shape[0] <= 0 || shape[1] <= 0) {
            throw new IllegalArgumentException("row-split layout requires a positive rank-2 shape");
        }
        if (format == WeightFormat.NVFP4) {
            return Nvfp4Layout.byteSize(shape[0], shape[1]);
        }
        int bits =
                switch (format) {
                    case Q3_G64_FP16 -> 3;
                    case Q4_G64_FP16 -> 4;
                    case Q5_G64_FP16 -> 5;
                    default -> throw new IllegalArgumentException("unsupported row-split format " + format);
                };
        int groupSize = 64;
        long n = shape[0];
        long kPad = alignUp(shape[1], K_ALIGNMENT);
        long groups = kPad / groupSize;
        long basePerGroup = bits == 3 ? 24 : 32;
        long highPerGroup = bits <= 4 ? 0 : 8;
        long baseRow = multiply(groups, basePerGroup);
        long highRow = multiply(groups, highPerGroup);
        long baseBytes = multiply(n, baseRow);
        long highBytes = multiply(n, highRow);
        long highOffset = alignUp(baseBytes, PLANE_ALIGNMENT);
        long scaleOffset = add(highOffset, alignUp(highBytes, PLANE_ALIGNMENT));
        return add(scaleOffset, multiply(multiply(n, groups), 2));
    }

    private static boolean isDirectFormat(WeightFormat format) {
        return format == WeightFormat.BF16
                || format == WeightFormat.FP16
                || format == WeightFormat.FP32
                || format == WeightFormat.I32;
    }

    private static long alignUp(long value, long alignment) {
        if (value < 0 || value > Long.MAX_VALUE - alignment + 1) {
            throw new IllegalArgumentException("compact size overflows");
        }
        return (value + alignment - 1) / alignment * alignment;
    }

    private static long multiply(long left, long right) {
        if (left < 0 || right < 0 || (right != 0 && left > Long.MAX_VALUE / right)) {
            throw new IllegalArgumentException("compact size overflows");
        }
        return left * right;
    }

    private static long add(long left, long right) {
        if (left < 0 || right < 0 || left > Long.MAX_VALUE - right) {
            throw new IllegalArgumentException("compact size overflows");
        }
        return left + right;
    }
}
