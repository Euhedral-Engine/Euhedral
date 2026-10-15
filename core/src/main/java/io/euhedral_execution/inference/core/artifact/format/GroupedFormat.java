package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import java.util.Arrays;

/// A grouped projection (Q3, Q4 or Q5 at 64-value groups) in `ROW_SPLIT_K128_V1`: the artifact chooses each
/// projection's width and the plan runs the matching kernel.
final class GroupedFormat extends GroupedLinearFormat {

    private static final int GROUP_SIZE = 64;
    private static final long PLANE_ALIGNMENT = 256;
    private static final long K_ALIGNMENT = 128;

    private final WeightFormat format;

    GroupedFormat(WeightFormat format) {
        if (!grouped(format)) throw new IllegalArgumentException("not a grouped format: " + format);
        this.format = format;
    }

    @Override
    public WeightFormat format() {
        return this.format;
    }

    @Override
    public WeightLayout layout() {
        return WeightLayout.ROW_SPLIT_K128_V1;
    }

    @Override
    public long byteSize(long[] shape, TensorDataType dataType) {
        if (dataType != TensorDataType.BF16) {
            throw new IllegalArgumentException("grouped compact storage requires BF16 source metadata");
        }
        if (shape.length != 2 || shape[0] <= 0 || shape[1] <= 0) {
            throw new IllegalArgumentException("row-split layout requires a positive rank-2 shape");
        }
        int bits =
                switch (this.format) {
                    case Q3_G64_FP16 -> 3;
                    case Q4_G64_FP16 -> 4;
                    default -> 5;
                };
        long n = shape[0];
        long kPad = CompactTensorLayout.alignUp(shape[1], K_ALIGNMENT);
        long groups = kPad / GROUP_SIZE;
        long basePerGroup = bits == 3 ? 24 : 32;
        long highPerGroup = bits <= 4 ? 0 : 8;
        long baseRow = CompactTensorLayout.multiply(groups, basePerGroup);
        long highRow = CompactTensorLayout.multiply(groups, highPerGroup);
        long baseBytes = CompactTensorLayout.multiply(n, baseRow);
        long highBytes = CompactTensorLayout.multiply(n, highRow);
        long highOffset = CompactTensorLayout.alignUp(baseBytes, PLANE_ALIGNMENT);
        long scaleOffset = CompactTensorLayout.add(highOffset, CompactTensorLayout.alignUp(highBytes, PLANE_ALIGNMENT));
        return CompactTensorLayout.add(
                scaleOffset, CompactTensorLayout.multiply(CompactTensorLayout.multiply(n, groups), 2));
    }

    @Override
    public boolean acceptsByteSize(long[] shape, TensorDataType dataType, long byteSize) {
        return byteSize(shape, dataType) == byteSize;
    }

    @Override
    public boolean accepts(TensorDescriptor descriptor, ExpectedTensor expected) {
        return Arrays.equals(expected.shape(), descriptor.shape())
                && expected.dataType() == descriptor.dataType()
                && grouped(expected.format())
                && expected.layout() == WeightLayout.ROW_SPLIT_K128_V1;
    }

    @Override
    public boolean stageable() {
        return true;
    }

    @Override
    public ExecutionPlan.Kind kind(ExecutionPlan.Kind role) {
        return switch (this.format) {
            case Q3_G64_FP16 -> ExecutionPlan.Kind.Q3_LINEAR;
            case Q4_G64_FP16 -> ExecutionPlan.Kind.Q4_LINEAR;
            default -> ExecutionPlan.Kind.Q5_LINEAR;
        };
    }

    @Override
    public TensorHandle validateHandle(TensorHandle handle, long rows, int width, WeightFormat roleFormat) {
        return validateQuantized(this, handle, rows, width, this.format == roleFormat || grouped(roleFormat));
    }

    @Override
    public boolean gateUpRegion() {
        return this.format == WeightFormat.Q3_G64_FP16;
    }
}
