package io.euhedral_execution.inference.core.gpu;

import java.lang.foreign.MemorySegment;

/// The minimal CPU-facing contract for GPU memory ownership and copies.
public interface GpuMemory {

    long allocate(long byteSize);

    void copyHostToDevice(long destination, MemorySegment source, long byteSize);

    void copyDeviceToHost(MemorySegment destination, long source, long byteSize);

    default void copyDeviceToDevice(long destination, long source, long byteSize) {
        throw new UnsupportedOperationException("device-to-device copy is not implemented by this GPU memory provider");
    }

    void free(long address);

    /// Pins `byteSize` bytes of host memory that the device's copy engines read directly, for weights
    /// staged to the device on use. Returns its host address.
    default long allocateHostWeights(long byteSize) {
        throw new UnsupportedOperationException("host-backed weights are not implemented by this GPU memory provider");
    }

    default void freeHostWeights(long address) {
        throw new UnsupportedOperationException("host-backed weights are not implemented by this GPU memory provider");
    }
}
