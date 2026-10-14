package io.euhedral_execution.inference.quantization.gpu;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/// Copies Java arrays to the device: the engine's copies take native memory, so each array goes through a native
/// buffer first (the copy completes before the buffer is released).
public final class Upload {

    private Upload() {}

    public static void ints(CudaGpuMemory gpu, long device, int[] values) {
        try (Arena arena = Arena.ofConfined()) {
            gpu.copyHostToDevice(device, arena.allocateFrom(ValueLayout.JAVA_INT, values), 4L * values.length);
        }
    }

    public static void floats(CudaGpuMemory gpu, long device, float[] values) {
        try (Arena arena = Arena.ofConfined()) {
            gpu.copyHostToDevice(device, arena.allocateFrom(ValueLayout.JAVA_FLOAT, values), 4L * values.length);
        }
    }

    public static void shorts(CudaGpuMemory gpu, long device, short[] values) {
        try (Arena arena = Arena.ofConfined()) {
            gpu.copyHostToDevice(device, arena.allocateFrom(ValueLayout.JAVA_SHORT, values), 2L * values.length);
        }
    }

    public static void longs(CudaGpuMemory gpu, long device, long[] values) {
        try (Arena arena = Arena.ofConfined()) {
            gpu.copyHostToDevice(device, arena.allocateFrom(ValueLayout.JAVA_LONG, values), 8L * values.length);
        }
    }

    public static void bytes(CudaGpuMemory gpu, long device, byte[] values) {
        try (Arena arena = Arena.ofConfined()) {
            gpu.copyHostToDevice(device, arena.allocateFrom(ValueLayout.JAVA_BYTE, values), values.length);
        }
    }

    /// Any segment, heap or native, of `bytes` bytes.
    public static void segment(CudaGpuMemory gpu, long device, MemorySegment source, long bytes) {
        if (source.isNative()) {
            gpu.copyHostToDevice(device, source, bytes);
            return;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment copy = arena.allocate(bytes, 16);
            MemorySegment.copy(source, 0, copy, 0, bytes);
            gpu.copyHostToDevice(device, copy, bytes);
        }
    }
}
