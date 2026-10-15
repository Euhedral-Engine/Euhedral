package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.WeightFormat;

/// Overflow-checked payload size arithmetic shared by the compact layouts.
public final class PayloadSizes {

    private PayloadSizes() {}

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
