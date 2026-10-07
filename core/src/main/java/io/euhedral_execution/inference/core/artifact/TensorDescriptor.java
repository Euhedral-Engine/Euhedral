package io.euhedral_execution.inference.core.artifact;

public record TensorDescriptor(
        String name,
        long[] shape,
        TensorDataType dataType,
        WeightFormat format,
        WeightLayout layout,
        long dataOffset,
        long byteSize) {}
