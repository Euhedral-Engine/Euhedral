package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.artifact.P2e2Layout;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.artifact.WeightLoadException;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/// Q3 entropy coded as P2E2 ([P2e2Layout]): lossless with respect to the row-split layout, sized by its codes. Its
/// payload is never staged: its row offsets index the payload where it was loaded.
final class P2e2Format extends GroupedLinearFormat {

    @Override
    public WeightFormat format() {
        return WeightFormat.Q3_G64_FP16;
    }

    @Override
    public WeightLayout layout() {
        return WeightLayout.ROW_SPLIT_P2E2_V1;
    }

    @Override
    public long byteSize(long[] shape, TensorDataType dataType) {
        throw new IllegalArgumentException("P2E2 tensor sizes depend on their codes");
    }

    @Override
    public boolean acceptsByteSize(long[] shape, TensorDataType dataType, long byteSize) {
        if (!P2e2Layout.supports(shape, dataType, WeightFormat.Q3_G64_FP16)) {
            throw new IllegalArgumentException(
                    "P2E2 layout requires Q3 BF16 metadata and a rank-2 shape whose K is a multiple of "
                            + P2e2Layout.SLICE);
        }
        return P2e2Layout.acceptsByteSize(shape[0], shape[1], byteSize);
    }

    @Override
    public boolean accepts(TensorDescriptor descriptor, ExpectedTensor expected) {
        return Arrays.equals(expected.shape(), descriptor.shape())
                && expected.dataType() == descriptor.dataType()
                && grouped(expected.format())
                && (expected.layout() == WeightLayout.ROW_SPLIT_P2E2_V1
                        || expected.layout() == WeightLayout.ROW_SPLIT_K128_V1);
    }

    @Override
    public boolean stageable() {
        return false;
    }

    @Override
    public boolean compressed() {
        return true;
    }

    @Override
    public void validatePayload(TensorDescriptor descriptor, MemorySegment payload) throws WeightLoadException {
        try {
            P2e2Layout.validate(payload, descriptor.shape()[0], descriptor.shape()[1]);
        } catch (IllegalArgumentException exception) {
            throw new WeightLoadException("invalid P2E2 tensor '" + descriptor.name() + "': " + exception.getMessage());
        }
    }

    @Override
    public ExecutionPlan.Kind kind(ExecutionPlan.Kind role) {
        return ExecutionPlan.Kind.Q3_LINEAR;
    }

    @Override
    public TensorHandle validateHandle(TensorHandle handle, long rows, int width, WeightFormat roleFormat) {
        return validateQuantized(this, handle, rows, width, grouped(roleFormat));
    }

    @Override
    public boolean gateUpRegion() {
        return true;
    }
}
