package io.euhedral_execution.inference.core.model_loader.layer_weights;

/// A loaded tensor. A device-resident tensor has a `deviceAddress`; a host-backed tensor instead has
/// a `hostAddress` in pinned host memory, and execution stages it into device memory before use.
public record TensorHandle(
        String name,
        long[] shape,
        TensorDataType dataType,
        WeightFormat format,
        WeightLayout layout,
        long deviceAddress,
        long byteSize,
        long hostAddress) {

    public TensorHandle(
            String name,
            long[] shape,
            TensorDataType dataType,
            WeightFormat format,
            WeightLayout layout,
            long deviceAddress,
            long byteSize) {
        this(name, shape, dataType, format, layout, deviceAddress, byteSize, 0L);
    }

    public TensorHandle(
            String name,
            long[] shape,
            TensorDataType dataType,
            WeightFormat format,
            long deviceAddress,
            long byteSize) {
        this(name, shape, dataType, format, WeightLayout.CONTIGUOUS_LE_V1, deviceAddress, byteSize);
    }

    /// Whether the payload lives in pinned host memory rather than on the device.
    public boolean hostBacked() {
        return this.hostAddress != 0;
    }
}
