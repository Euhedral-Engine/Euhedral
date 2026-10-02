package io.euhedral_execution.inference.core.model_loader.layer_weights;

/// A loaded tensor. A device-resident tensor has a `deviceAddress`; a host-backed tensor instead has
/// a `hostAddress` in pinned host memory, and execution stages it into device memory before use (a
/// staging transfer's handle carries both: its slot and its source). A `hostMapped` tensor lives in
/// pinned host memory at `hostAddress` and kernels read it in place at `deviceAddress`, so execution
/// treats it as resident; its memory belongs to the host arena.
public record TensorHandle(
        String name,
        long[] shape,
        TensorDataType dataType,
        WeightFormat format,
        WeightLayout layout,
        long deviceAddress,
        long byteSize,
        long hostAddress,
        boolean hostMapped) {

    public TensorHandle(
            String name,
            long[] shape,
            TensorDataType dataType,
            WeightFormat format,
            WeightLayout layout,
            long deviceAddress,
            long byteSize,
            long hostAddress) {
        this(name, shape, dataType, format, layout, deviceAddress, byteSize, hostAddress, false);
    }

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

    /// Whether the payload lives in pinned host memory and is staged to the device on use.
    public boolean hostBacked() {
        return this.hostAddress != 0 && !this.hostMapped;
    }
}
