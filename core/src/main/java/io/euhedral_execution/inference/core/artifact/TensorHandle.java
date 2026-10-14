package io.euhedral_execution.inference.core.artifact;

/// A loaded tensor. A device-resident tensor has a `deviceAddress`; a host-backed tensor instead has
/// a `hostAddress` in pinned host memory, and execution stages it into device memory before use (a
/// staging transfer's handle carries both: its slot and its source). A `hostMapped` tensor lives in
/// pinned host memory at `hostAddress` and kernels read it in place at `deviceAddress`, so execution
/// treats it as resident; its memory belongs to the host arena. A disk-backed tensor has neither: its
/// payload stays in the artifact file at `fileOffset` and execution reads it into a pinned window, then stages it
/// like a host-backed tensor (`fileOffset` is -1 for every other tensor).
public record TensorHandle(
        String name,
        long[] shape,
        TensorDataType dataType,
        WeightFormat format,
        WeightLayout layout,
        long deviceAddress,
        long byteSize,
        long hostAddress,
        boolean hostMapped,
        long fileOffset) {

    public TensorHandle(
            String name,
            long[] shape,
            TensorDataType dataType,
            WeightFormat format,
            WeightLayout layout,
            long deviceAddress,
            long byteSize,
            long hostAddress,
            boolean hostMapped) {
        this(name, shape, dataType, format, layout, deviceAddress, byteSize, hostAddress, hostMapped, -1L);
    }

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

    /// A disk-backed handle for `descriptor`'s payload, read from the artifact file at its data offset.
    public static TensorHandle disk(TensorDescriptor descriptor) {
        return new TensorHandle(
                descriptor.name(),
                descriptor.shape(),
                descriptor.dataType(),
                descriptor.format(),
                descriptor.layout(),
                0L,
                descriptor.byteSize(),
                0L,
                false,
                descriptor.dataOffset());
    }

    /// Whether the payload lives in pinned host memory and is staged to the device on use.
    public boolean hostBacked() {
        return this.hostAddress != 0 && !this.hostMapped;
    }

    /// Whether the payload stays in the artifact file and is read, then staged, on use.
    public boolean diskBacked() {
        return this.fileOffset >= 0;
    }
}
