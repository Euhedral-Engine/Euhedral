package io.euhedral_execution.inference.core.artifact;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;

/// Loads one unmodified tensor payload into GPU memory. P2E2 payloads are checked for internal
/// consistency first, so that the kernels that decode them stay within the tensor, and NVFP4 payloads
/// for NaN scales.
public final class TensorLoader {

    private TensorLoader() {}

    public static TensorHandle load(Path artifactPath, TensorDescriptor descriptor, GpuMemory gpuMemory)
            throws IOException {
        Objects.requireNonNull(artifactPath, "artifactPath");
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(gpuMemory, "gpuMemory");

        long deviceAddress = 0;
        boolean allocated = false;
        try (Arena hostArena = Arena.ofConfined()) {
            MemorySegment payload = TensorDataReader.read(artifactPath, descriptor, hostArena);
            validate(descriptor, payload);
            deviceAddress = gpuMemory.allocate(descriptor.byteSize());
            allocated = true;
            gpuMemory.copyHostToDevice(deviceAddress, payload, descriptor.byteSize());
            return new TensorHandle(
                    descriptor.name(),
                    descriptor.shape(),
                    descriptor.dataType(),
                    descriptor.format(),
                    descriptor.layout(),
                    deviceAddress,
                    descriptor.byteSize());
        } catch (Throwable failure) {
            if (allocated) {
                try {
                    gpuMemory.free(deviceAddress);
                } catch (Throwable cleanupFailure) {
                    failure.addSuppressed(cleanupFailure);
                }
            }
            return propagate(failure);
        }
    }

    /// Copies the payload into pinned host memory at `hostAddress`, which holds at least its byte size;
    /// execution stages it to the device before each use.
    public static TensorHandle loadToHost(Path artifactPath, TensorDescriptor descriptor, long hostAddress)
            throws IOException {
        Objects.requireNonNull(artifactPath, "artifactPath");
        Objects.requireNonNull(descriptor, "descriptor");
        if (hostAddress == 0) throw new IllegalArgumentException("hostAddress must be set");
        try (Arena hostArena = Arena.ofConfined()) {
            MemorySegment payload = TensorDataReader.read(artifactPath, descriptor, hostArena);
            validate(descriptor, payload);
            MemorySegment.ofAddress(hostAddress)
                    .reinterpret(descriptor.byteSize())
                    .copyFrom(payload.asSlice(0, descriptor.byteSize()));
            return new TensorHandle(
                    descriptor.name(),
                    descriptor.shape(),
                    descriptor.dataType(),
                    descriptor.format(),
                    descriptor.layout(),
                    0L,
                    descriptor.byteSize(),
                    hostAddress);
        }
    }

    private static void validate(TensorDescriptor descriptor, MemorySegment payload) throws WeightLoadException {
        if (descriptor.format() == WeightFormat.NVFP4 && Nvfp4Layout.supports(descriptor.layout())) {
            try {
                Nvfp4Layout.validate(payload, descriptor.shape()[0], descriptor.shape()[1], descriptor.layout());
            } catch (IllegalArgumentException exception) {
                throw new WeightLoadException(
                        "invalid NVFP4 tensor '" + descriptor.name() + "': " + exception.getMessage());
            }
        }
        if (descriptor.layout() == WeightLayout.ROW_SPLIT_P2E2_V1) {
            try {
                P2e2Layout.validate(payload, descriptor.shape()[0], descriptor.shape()[1]);
            } catch (IllegalArgumentException exception) {
                throw new WeightLoadException(
                        "invalid P2E2 tensor '" + descriptor.name() + "': " + exception.getMessage());
            }
        }
    }

    private static TensorHandle propagate(Throwable failure) throws IOException {
        if (failure instanceof IOException exception) {
            throw exception;
        }
        if (failure instanceof RuntimeException exception) {
            throw exception;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        throw new AssertionError(failure);
    }
}
