package io.euhedral_execution.inference.core.generation;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/// Caller-owned BF16 logits, detached from a quantum's workspace for callers that read logits on the GPU.
public final class DeviceLogits implements AutoCloseable {

    private final ExecutionGpu gpu;
    private final int tokenCount;
    private final int vocabularySize;
    private long deviceAddress;

    public DeviceLogits(ExecutionGpu gpu, long deviceAddress, int tokenCount, int vocabularySize) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (deviceAddress == 0) throw new IllegalArgumentException("deviceAddress must not be null");
        if (tokenCount <= 0 || vocabularySize <= 0)
            throw new IllegalArgumentException("logit dimensions must be positive");
        this.deviceAddress = deviceAddress;
        this.tokenCount = tokenCount;
        this.vocabularySize = vocabularySize;
    }

    public long deviceAddress() {
        ensureOpen();
        return this.deviceAddress;
    }

    public int tokenCount() {
        return this.tokenCount;
    }

    public int vocabularySize() {
        return this.vocabularySize;
    }

    public long byteSize() {
        return Math.multiplyExact(Math.multiplyExact((long) this.tokenCount, this.vocabularySize), Short.BYTES);
    }

    /// Copies logits to caller-provided host storage; the device allocation remains owned by this object.
    public void copyToHost(MemorySegment destination) {
        Objects.requireNonNull(destination, "destination");
        ensureOpen();
        if (destination.byteSize() < byteSize())
            throw new IllegalArgumentException("destination is too small for logits");
        this.gpu.copyDeviceToHost(destination, this.deviceAddress, byteSize());
    }

    /// Copies the final vocabulary row; ownership of the retained device allocation stays here.
    public void copyFinalTokenRowToHost(ExecutionGpu gpu, MemorySegment destination) {
        Objects.requireNonNull(gpu, "gpu");
        Objects.requireNonNull(destination, "destination");
        ensureOpen();
        if (gpu != this.gpu) throw new IllegalArgumentException("logits belong to a different GPU");

        long rowByteSize = Math.multiplyExact((long) this.vocabularySize, Short.BYTES);
        if (destination.byteSize() < rowByteSize)
            throw new IllegalArgumentException("destination is too small for the final logits row");
        long rowOffset = Math.multiplyExact((long) (this.tokenCount - 1), rowByteSize);
        long rowAddress = Math.addExact(this.deviceAddress, rowOffset);
        this.gpu.copyDeviceToHost(destination.asSlice(0, rowByteSize), rowAddress, rowByteSize);
    }

    @Override
    public void close() {
        if (this.deviceAddress == 0) return;
        this.gpu.free(this.deviceAddress);
        this.deviceAddress = 0;
    }

    private void ensureOpen() {
        if (this.deviceAddress == 0) throw new IllegalStateException("Qwen logits have been released");
    }
}
