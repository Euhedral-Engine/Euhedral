package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.util.List;
import java.util.Objects;

/// GPU storage scoped to one Qwen submission; hidden states are BF16 values in token-major order.
public final class QwenExecutionWorkspace implements AutoCloseable {

    private final GpuMemory gpuMemory;
    private final int tokenCount;
    private final int hiddenSize;
    private final long byteSize;
    private final long[] projectionAddresses;
    private final long[] projectionByteSizes;
    private final long[] firstLayerAddresses;
    private final long[] firstLayerByteSizes;
    private final long[] storageByteSizes;
    private final int[] storageOwners;
    private long normalizedAddress;
    private long hiddenStateAddress;
    private boolean closed;

    QwenExecutionWorkspace(GpuMemory gpuMemory, int tokenCount, int hiddenSize) {
        this(gpuMemory, tokenCount, hiddenSize, List.of());
    }

    QwenExecutionWorkspace(GpuMemory gpuMemory, int tokenCount, int hiddenSize, List<Integer> projectionWidths) {
        this(gpuMemory, tokenCount, hiddenSize, projectionWidths, List.of(), QwenLogitsRequirement.ALL_TOKENS);
    }

    QwenExecutionWorkspace(GpuMemory gpuMemory, int tokenCount, QwenExecutionPlan plan) {
        this(gpuMemory, tokenCount, plan, QwenLogitsRequirement.ALL_TOKENS);
    }

    QwenExecutionWorkspace(
            GpuMemory gpuMemory, int tokenCount, QwenExecutionPlan plan, QwenLogitsRequirement logitsRequirement) {
        this(
                gpuMemory,
                tokenCount,
                plan.weights().config().hiddenSize(),
                List.of(),
                plan.bufferSpecs(),
                logitsRequirement);
        if (plan.reusePrefillStorage()) configureRegionStorage();
        if (!plan.hasFirstLayer()) {
            throw new IllegalArgumentException("first-layer buffers require a first-layer plan");
        }
    }

    private QwenExecutionWorkspace(
            GpuMemory gpuMemory,
            int tokenCount,
            int hiddenSize,
            List<Integer> projectionWidths,
            List<QwenExecutionPlan.BufferSpec> firstLayerBuffers,
            QwenLogitsRequirement logitsRequirement) {
        this.gpuMemory = Objects.requireNonNull(gpuMemory, "gpuMemory");
        Objects.requireNonNull(projectionWidths, "projectionWidths");
        Objects.requireNonNull(firstLayerBuffers, "firstLayerBuffers");
        Objects.requireNonNull(logitsRequirement, "logitsRequirement");
        if (tokenCount <= 0) {
            throw new IllegalArgumentException("tokenCount must be positive");
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive");
        }
        this.tokenCount = tokenCount;
        this.hiddenSize = hiddenSize;
        this.projectionAddresses = new long[projectionWidths.size()];
        this.projectionByteSizes = new long[projectionWidths.size()];
        this.firstLayerAddresses = new long[QwenExecutionPlan.Buffer.values().length];
        this.firstLayerByteSizes = new long[QwenExecutionPlan.Buffer.values().length];
        this.storageOwners = new int[this.firstLayerByteSizes.length];
        for (int index = 0; index < this.storageOwners.length; index++) this.storageOwners[index] = index;
        try {
            this.byteSize = Math.multiplyExact(Math.multiplyExact((long) tokenCount, hiddenSize), Short.BYTES);
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("hidden-state workspace size overflows", overflow);
        }
        for (int index = 0; index < projectionWidths.size(); index++) {
            int width = projectionWidths.get(index);
            if (width <= 0) {
                throw new IllegalArgumentException("projection width must be positive");
            }
            this.projectionByteSizes[index] =
                    Math.multiplyExact(Math.multiplyExact((long) tokenCount, width), Short.BYTES);
        }
        for (QwenExecutionPlan.BufferSpec spec : firstLayerBuffers) {
            int index = spec.buffer().ordinal();
            if (this.firstLayerByteSizes[index] != 0) {
                throw new IllegalArgumentException("duplicate first-layer buffer: " + spec.buffer());
            }
            int elementBytes = spec.elementType() == QwenExecutionPlan.ElementType.BF16 ? Short.BYTES : Float.BYTES;
            int rows = spec.buffer() == QwenExecutionPlan.Buffer.LOGITS
                            || spec.buffer() == QwenExecutionPlan.Buffer.FINAL_NORMALIZED
                    ? logitsRequirement.outputRows(tokenCount)
                    : tokenCount;
            this.firstLayerByteSizes[index] =
                    Math.multiplyExact(Math.multiplyExact((long) rows, spec.width()), elementBytes);
        }
        this.storageByteSizes = this.firstLayerByteSizes.clone();
    }

    /// Fixed lifetime pairs in the combined prefill graph, not a general lifetime allocator.
    private void configureRegionStorage() {
        // Each previous value's final consumer precedes the next producer through the layer DAG.
        reuse(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE, QwenExecutionPlan.Buffer.HIDDEN_STATE);
        reuse(QwenExecutionPlan.Buffer.POST_MIXER_NORMALIZED, QwenExecutionPlan.Buffer.INPUT_NORMALIZED);
        reuse(QwenExecutionPlan.Buffer.FFN_DELTA, QwenExecutionPlan.Buffer.MIXER_DELTA);
        reuse(QwenExecutionPlan.Buffer.SWIGLU, QwenExecutionPlan.Buffer.VALUE_Z_PROJECTED);
        reuse(QwenExecutionPlan.Buffer.FFN_STAGING, QwenExecutionPlan.Buffer.VALUE_Z_PROJECTED);
        reuse(QwenExecutionPlan.Buffer.FFN_ACCUMULATORS, QwenExecutionPlan.Buffer.QK_PROJECTED);
    }

    private void reuse(QwenExecutionPlan.Buffer value, QwenExecutionPlan.Buffer owner) {
        int from = value.ordinal(), to = owner.ordinal();
        if (this.firstLayerByteSizes[from] == 0 || this.firstLayerByteSizes[to] == 0) return;
        this.storageOwners[from] = to;
        this.storageByteSizes[to] = Math.max(this.storageByteSizes[to], this.firstLayerByteSizes[from]);
        this.storageByteSizes[from] = 0;
    }

    /// Allocate only after the submission owns this object, so partial failure remains reclaimable.
    void allocateBuffers() {
        if (this.closed || this.hiddenStateAddress != 0) {
            throw new IllegalStateException("workspace has already been allocated or closed");
        }
        if (hasFirstLayerBuffers()) {
            for (int index = 0; index < this.firstLayerByteSizes.length; index++) {
                if (this.storageByteSizes[index] != 0) {
                    this.firstLayerAddresses[index] = allocate(this.storageByteSizes[index]);
                }
            }
            this.hiddenStateAddress = this.firstLayerAddresses[QwenExecutionPlan.Buffer.HIDDEN_STATE.ordinal()];
            this.normalizedAddress = this.firstLayerAddresses[QwenExecutionPlan.Buffer.INPUT_NORMALIZED.ordinal()];
            return;
        }
        this.hiddenStateAddress = allocate(this.byteSize);
        if (this.projectionByteSizes.length != 0) {
            this.normalizedAddress = allocate(this.byteSize);
            for (int index = 0; index < this.projectionByteSizes.length; index++) {
                this.projectionAddresses[index] = allocate(this.projectionByteSizes[index]);
            }
        }
    }

    private long allocate(long bytes) {
        long address = this.gpuMemory.allocate(bytes);
        if (address == 0) {
            throw new IllegalStateException("GPU returned a null workspace address");
        }
        return address;
    }

    public int tokenCount() {
        return this.tokenCount;
    }

    public int hiddenSize() {
        return this.hiddenSize;
    }

    public long byteSize() {
        return this.byteSize;
    }

    /// Returns the opaque device address while this submission workspace is live.
    public long hiddenStateAddress() {
        if (this.closed) {
            throw new IllegalStateException("Qwen execution workspace is closed");
        }
        return this.hiddenStateAddress;
    }

    public long normalizedStateAddress() {
        if (this.closed || this.normalizedAddress == 0) {
            throw new IllegalStateException("normalized buffer is unavailable");
        }
        return this.normalizedAddress;
    }

    public long projectionAddress(int index) {
        if (this.closed || this.projectionAddresses[index] == 0) {
            throw new IllegalStateException("projection buffer is unavailable");
        }
        return this.projectionAddresses[index];
    }

    /// Returns a named instruction-graph buffer while this workspace is live.
    public long address(QwenExecutionPlan.Buffer buffer) {
        Objects.requireNonNull(buffer, "buffer");
        long address = this.firstLayerAddresses[this.storageOwners[buffer.ordinal()]];
        if (this.closed || address == 0) {
            throw new IllegalStateException("instruction buffer is unavailable: " + buffer);
        }
        return address;
    }

    /// Transfers a named buffer to a caller that will release it independently of the workspace.
    public long detachAddress(QwenExecutionPlan.Buffer buffer) {
        Objects.requireNonNull(buffer, "buffer");
        int index = buffer.ordinal();
        for (int value = 0; value < this.storageOwners.length; value++) {
            if (value != index && this.firstLayerByteSizes[value] > 0 && this.storageOwners[value] == index)
                throw new IllegalStateException("shared region storage cannot be detached: " + buffer);
        }
        long address = this.firstLayerAddresses[index];
        if (this.closed || address == 0) {
            throw new IllegalStateException("instruction buffer cannot be detached: " + buffer);
        }
        this.firstLayerAddresses[index] = 0;
        return address;
    }

    public boolean hasBuffer(QwenExecutionPlan.Buffer buffer) {
        Objects.requireNonNull(buffer, "buffer");
        return !this.closed && this.firstLayerByteSizes[buffer.ordinal()] > 0;
    }

    public long address(QwenExecutionPlan.Buffer buffer, int index) {
        if (buffer == QwenExecutionPlan.Buffer.SLICE_PROJECTION) {
            return projectionAddress(index);
        }
        if (index != 0) {
            throw new IllegalStateException("named instruction buffers are not indexed");
        }
        return address(buffer);
    }

    public long bufferByteSize(QwenExecutionPlan.Buffer buffer) {
        Objects.requireNonNull(buffer, "buffer");
        long byteSize = this.firstLayerByteSizes[buffer.ordinal()];
        if (this.closed || byteSize == 0) {
            throw new IllegalStateException("instruction buffer is unavailable: " + buffer);
        }
        return byteSize;
    }

    public boolean isClosed() {
        return this.closed;
    }

    @Override
    public void close() {
        if (this.closed) {
            return;
        }
        Throwable failure = null;
        if (hasFirstLayerBuffers()) {
            for (int index = 0; index < this.firstLayerAddresses.length; index++) {
                try {
                    if (this.firstLayerAddresses[index] != 0) {
                        this.gpuMemory.free(this.firstLayerAddresses[index]);
                        this.firstLayerAddresses[index] = 0;
                    }
                } catch (RuntimeException | Error cleanupFailure) {
                    failure = combine(failure, cleanupFailure);
                }
            }
            this.hiddenStateAddress = 0;
            this.normalizedAddress = 0;
            this.closed = allFirstLayerAddressesReleased();
            if (failure instanceof Error error) {
                throw error;
            }
            if (failure != null) {
                throw (RuntimeException) failure;
            }
            return;
        }
        for (int index = 0; index < this.projectionAddresses.length; index++) {
            try {
                if (this.projectionAddresses[index] != 0) {
                    this.gpuMemory.free(this.projectionAddresses[index]);
                    this.projectionAddresses[index] = 0;
                }
            } catch (RuntimeException | Error cleanupFailure) {
                failure = combine(failure, cleanupFailure);
            }
        }
        try {
            if (this.normalizedAddress != 0) {
                this.gpuMemory.free(this.normalizedAddress);
                this.normalizedAddress = 0;
            }
        } catch (RuntimeException | Error cleanupFailure) {
            failure = combine(failure, cleanupFailure);
        }
        try {
            if (this.hiddenStateAddress != 0) {
                this.gpuMemory.free(this.hiddenStateAddress);
                this.hiddenStateAddress = 0;
            }
        } catch (RuntimeException | Error cleanupFailure) {
            failure = combine(failure, cleanupFailure);
        }
        this.closed = this.hiddenStateAddress == 0 && this.normalizedAddress == 0;
        for (long address : this.projectionAddresses) {
            this.closed &= address == 0;
        }
        if (failure != null) {
            if (failure instanceof Error error) {
                throw error;
            }
            throw (RuntimeException) failure;
        }
    }

    private static Throwable combine(Throwable first, Throwable next) {
        if (first == null) {
            return next;
        }
        first.addSuppressed(next);
        return first;
    }

    private boolean hasFirstLayerBuffers() {
        for (long byteSize : this.firstLayerByteSizes) {
            if (byteSize != 0) return true;
        }
        return false;
    }

    private boolean allFirstLayerAddressesReleased() {
        for (long address : this.firstLayerAddresses) {
            if (address != 0) return false;
        }
        return true;
    }
}
