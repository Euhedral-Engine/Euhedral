package io.euhedral_execution.inference.core.model.qwen4.loader;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;

/// A fixed object of a version 3 artifact: a tensor in one contiguous file range, its component group and the
/// CRC-32 (IEEE) of its payload.
public record Tensor(
        String name,
        long[] shape,
        TensorDataType dataType,
        WeightFormat format,
        WeightLayout layout,
        ComponentGroup group,
        long dataOffset,
        long byteSize,
        int crc32) {

    /// The descriptor the version 2 tensor loaders take.
    public TensorDescriptor descriptor() {
        return new TensorDescriptor(
                this.name, this.shape, this.dataType, this.format, this.layout, this.dataOffset, this.byteSize);
    }
}
