package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.runtime.graph.CaptureFingerprint;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/// GPU storage bound to one Qwen submission; hidden states are BF16 values in token-major order.
///
/// A workspace is the quantum's view of its buffers. Production workspaces borrow their allocations
/// from the executing graph's [WorkspaceStorage]: [#allocateBuffers] acquires them at admission and
/// [#close] releases the binding at retirement, leaving the allocations for the graph's next quantum.
/// A workspace constructed from a [GpuMemory] owns private storage and frees it on close.
public final class Workspace implements AutoCloseable {

    static final int BUFFER_SLOTS = ExecutionPlan.Buffer.values().length;
    static final int TOKEN_IDS_SLOT = BUFFER_SLOTS;
    static final int HIDDEN_SLOT = BUFFER_SLOTS + 1;
    static final int NORMALIZED_SLOT = BUFFER_SLOTS + 2;
    static final int PROJECTION_SLOTS = BUFFER_SLOTS + 3;
    private static final int LOGITS_SLOT = ExecutionPlan.Buffer.LOGITS.ordinal();

    private final WorkspaceStorage storage;
    /// The runtime's workspace, sized at load: every slot but the input record and the logits; null when the
    /// quantum's graph storage holds every slot (a workspace that owns its storage).
    private SharedWorkspace shared;
    private final boolean ownsStorage;
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
    private long inputAddress;
    private boolean allocated;
    private boolean closed;

    Workspace(GpuMemory gpuMemory, int tokenCount, int hiddenSize) {
        this(gpuMemory, tokenCount, hiddenSize, List.of());
    }

    Workspace(GpuMemory gpuMemory, int tokenCount, int hiddenSize, List<Integer> projectionWidths) {
        this(ownedStorage(gpuMemory), true, tokenCount, hiddenSize, projectionWidths);
    }

    /// Borrows `storage` for a plan without first-layer buffers.
    Workspace(WorkspaceStorage storage, int tokenCount, int hiddenSize, List<Integer> projectionWidths) {
        this(storage, false, tokenCount, hiddenSize, projectionWidths);
    }

    private Workspace(
            WorkspaceStorage storage,
            boolean ownsStorage,
            int tokenCount,
            int hiddenSize,
            List<Integer> projectionWidths) {
        this(storage, ownsStorage, tokenCount, hiddenSize, projectionWidths, List.of(), LogitsRequirement.ALL_TOKENS);
    }

    Workspace(GpuMemory gpuMemory, int tokenCount, ExecutionPlan plan) {
        this(gpuMemory, tokenCount, plan.shape(), LogitsRequirement.ALL_TOKENS);
    }

    Workspace(GpuMemory gpuMemory, int tokenCount, ExecutionPlan plan, LogitsRequirement logitsRequirement) {
        this(gpuMemory, tokenCount, plan.shape(), logitsRequirement);
    }

    Workspace(GpuMemory gpuMemory, int tokenCount, Shape plan, LogitsRequirement logitsRequirement) {
        this(ownedStorage(gpuMemory), true, tokenCount, plan, logitsRequirement);
    }

    /// Borrows `storage` for a first-layer view.
    Workspace(WorkspaceStorage storage, int tokenCount, Shape plan, LogitsRequirement logitsRequirement) {
        this(storage, false, tokenCount, plan, logitsRequirement);
    }

    private Workspace(
            WorkspaceStorage storage,
            boolean ownsStorage,
            int tokenCount,
            Shape plan,
            LogitsRequirement logitsRequirement) {
        this(
                storage,
                ownsStorage,
                tokenCount,
                plan.plan().weights().config().hiddenSize(),
                List.of(),
                plan.bufferSpecs(),
                logitsRequirement);
        if (plan.reusePrefillStorage()) configureRegionStorage();
        if (!plan.hasFirstLayer()) {
            throw new IllegalArgumentException("first-layer buffers require a first-layer plan");
        }
    }

    private Workspace(
            WorkspaceStorage storage,
            boolean ownsStorage,
            int tokenCount,
            int hiddenSize,
            List<Integer> projectionWidths,
            List<ExecutionPlan.BufferSpec> firstLayerBuffers,
            LogitsRequirement logitsRequirement) {
        this.storage = Objects.requireNonNull(storage, "storage");
        this.ownsStorage = ownsStorage;
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
        this.firstLayerAddresses = new long[BUFFER_SLOTS];
        this.firstLayerByteSizes = new long[BUFFER_SLOTS];
        this.storageOwners = new int[BUFFER_SLOTS];
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
        for (ExecutionPlan.BufferSpec spec : firstLayerBuffers) {
            int index = spec.buffer().ordinal();
            if (this.firstLayerByteSizes[index] != 0) {
                throw new IllegalArgumentException("duplicate first-layer buffer: " + spec.buffer());
            }
            int elementBytes = spec.elementType() == ExecutionPlan.ElementType.BF16 ? Short.BYTES : Float.BYTES;
            int rows = spec.buffer() == ExecutionPlan.Buffer.LOGITS
                            || spec.buffer() == ExecutionPlan.Buffer.FINAL_NORMALIZED
                    ? logitsRequirement.outputRows(tokenCount)
                    : tokenCount;
            this.firstLayerByteSizes[index] =
                    Math.multiplyExact(Math.multiplyExact((long) rows, spec.width()), elementBytes);
        }
        this.storageByteSizes = this.firstLayerByteSizes.clone();
    }

    private static WorkspaceStorage ownedStorage(GpuMemory gpuMemory) {
        return new WorkspaceStorage(Objects.requireNonNull(gpuMemory, "gpuMemory"));
    }

    /// Fixed lifetime pairs in the combined prefill graph, not a general lifetime allocator.
    private void configureRegionStorage() {
        // Each previous value's final consumer precedes the next producer through the stage topology,
        // which orders every pair of stages touching one storage (ExecutionPlan.withStorageHazards).
        for (var pair : ExecutionPlan.REGION_STORAGE) reuse(pair.getKey(), pair.getValue());
    }

    private void reuse(ExecutionPlan.Buffer value, ExecutionPlan.Buffer owner) {
        int from = value.ordinal(), to = owner.ordinal();
        if (this.firstLayerByteSizes[from] == 0 || this.firstLayerByteSizes[to] == 0) return;
        this.storageOwners[from] = to;
        this.storageByteSizes[to] = Math.max(this.storageByteSizes[to], this.firstLayerByteSizes[from]);
        this.storageByteSizes[from] = 0;
    }

    /// A first-layer workspace whose buffers are the runtime's `shared` ones, sized at load, except the input record
    /// and the logits, which stay in the graph's `storage`.
    static Workspace bound(
            SharedWorkspace shared,
            WorkspaceStorage storage,
            int tokenCount,
            Shape shape,
            LogitsRequirement logitsRequirement) {
        Workspace workspace = new Workspace(storage, tokenCount, shape, logitsRequirement);
        workspace.shared = Objects.requireNonNull(shared, "shared");
        return workspace;
    }

    /// As [#bound] for a view without a first layer.
    static Workspace bound(
            SharedWorkspace shared,
            WorkspaceStorage storage,
            int tokenCount,
            int hiddenSize,
            List<Integer> projectionWidths) {
        Workspace workspace = new Workspace(storage, tokenCount, hiddenSize, projectionWidths);
        workspace.shared = Objects.requireNonNull(shared, "shared");
        return workspace;
    }

    /// The bytes each slot needs for `rows` rows of `shape`, every logits row included: what a workspace sized at
    /// load for `rows` holds.
    static long[] slotBytes(Shape shape, int rows, int slots) {
        long[] bytes = new long[slots];
        Workspace sizing = shape.hasFirstLayer()
                ? new Workspace(SIZING, rows, shape, LogitsRequirement.ALL_TOKENS)
                : new Workspace(SIZING, rows, shape.plan().weights().config().hiddenSize(), shape.projectionWidths());
        if (shape.hasFirstLayer()) {
            System.arraycopy(sizing.storageByteSizes, 0, bytes, 0, sizing.storageByteSizes.length);
        } else {
            bytes[HIDDEN_SLOT] = sizing.byteSize;
            if (sizing.projectionByteSizes.length != 0) bytes[NORMALIZED_SLOT] = sizing.byteSize;
            for (int index = 0; index < sizing.projectionByteSizes.length; index++)
                bytes[PROJECTION_SLOTS + index] = sizing.projectionByteSizes[index];
        }
        bytes[TOKEN_IDS_SLOT] = 0;
        bytes[LOGITS_SLOT] = 0;
        return bytes;
    }

    /// Storage that only sizes: it is never asked for memory.
    private static final WorkspaceStorage SIZING = new WorkspaceStorage(new SizingMemory());

    private static final class SizingMemory implements GpuMemory {
        @Override
        public long allocate(long byteSize) {
            throw new IllegalStateException("sizing storage allocates nothing");
        }

        @Override
        public void free(long address) {}

        @Override
        public void copyHostToDevice(long destination, java.lang.foreign.MemorySegment source, long byteSize) {}

        @Override
        public void copyDeviceToHost(java.lang.foreign.MemorySegment destination, long source, long byteSize) {}
    }

    /// The runtime workspace's expansion scratch, or 0 when this workspace has none (it owns its storage, or no
    /// declared stage takes scratch on this GPU).
    public long scratchAddress() {
        return this.shared == null ? 0 : this.shared.scratchAddress();
    }

    public long scratchBytes() {
        return this.shared == null ? 0 : this.shared.scratchBytes();
    }

    private long acquire(int slot, long bytes) {
        if (this.shared != null && slot != TOKEN_IDS_SLOT && slot != LOGITS_SLOT)
            return this.shared.acquire(slot, bytes);
        return this.storage.acquire(slot, bytes);
    }

    /// Acquires this submission's buffers once it owns the workspace. Storage that a partial failure
    /// already acquired stays with its storage owner, which reclaims it.
    void allocateBuffers() {
        if (this.closed || this.allocated) {
            throw new IllegalStateException("workspace has already been allocated or closed");
        }
        this.allocated = true;
        this.inputAddress = this.storage.acquire(TOKEN_IDS_SLOT, inputByteSize());
        if (hasFirstLayerBuffers()) {
            for (int index = 0; index < this.firstLayerByteSizes.length; index++) {
                if (this.storageByteSizes[index] != 0) {
                    this.firstLayerAddresses[index] = acquire(index, this.storageByteSizes[index]);
                }
            }
            this.hiddenStateAddress = this.firstLayerAddresses[ExecutionPlan.Buffer.HIDDEN_STATE.ordinal()];
            this.normalizedAddress = this.firstLayerAddresses[ExecutionPlan.Buffer.INPUT_NORMALIZED.ordinal()];
            return;
        }
        this.hiddenStateAddress = acquire(HIDDEN_SLOT, this.byteSize);
        if (this.projectionByteSizes.length != 0) {
            this.normalizedAddress = acquire(NORMALIZED_SLOT, this.byteSize);
            for (int index = 0; index < this.projectionByteSizes.length; index++) {
                this.projectionAddresses[index] = acquire(PROJECTION_SLOTS + index, this.projectionByteSizes[index]);
            }
        }
    }

    /// Bytes of the quantum's input record: its token IDs (32-bit), then its start position (64-bit, 8-byte
    /// aligned). The quantum uploads the record once at admission, before its first launch, so kernels read
    /// the position from device memory instead of taking it as a launch parameter.
    public long inputByteSize() {
        return positionOffset() + Long.BYTES;
    }

    private long positionOffset() {
        return ((long) this.tokenCount * Integer.BYTES + Long.BYTES - 1) / Long.BYTES * Long.BYTES;
    }

    void fingerprint(CaptureFingerprint fingerprint) {
        this.storage.fingerprint(fingerprint);
        if (this.shared != null) this.shared.fingerprint(fingerprint);
    }

    /// The input record's device address: its token IDs.
    public long tokenIdsAddress() {
        if (this.closed || this.inputAddress == 0) throw new IllegalStateException("input record is unavailable");
        return this.inputAddress;
    }

    /// The device address of the quantum's start position in its input record.
    public long positionAddress() {
        return tokenIdsAddress() + positionOffset();
    }

    /// Fills `record` (at least [#inputByteSize] bytes) with the input record for `tokenIds` at `startPosition`.
    public void writeInput(MemorySegment record, int[] tokenIds, long startPosition) {
        if (tokenIds.length != this.tokenCount) throw new IllegalArgumentException("token count mismatch");
        for (int index = 0; index < tokenIds.length; index++)
            record.set(ValueLayout.JAVA_INT, (long) index * Integer.BYTES, tokenIds[index]);
        for (long offset = (long) tokenIds.length * Integer.BYTES; offset < positionOffset(); offset += Integer.BYTES)
            record.set(ValueLayout.JAVA_INT, offset, 0);
        record.set(ValueLayout.JAVA_LONG_UNALIGNED, positionOffset(), startPosition);
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
    public long address(ExecutionPlan.Buffer buffer) {
        Objects.requireNonNull(buffer, "buffer");
        long address = this.firstLayerAddresses[this.storageOwners[buffer.ordinal()]];
        if (this.closed || address == 0) {
            throw new IllegalStateException("instruction buffer is unavailable: " + buffer);
        }
        return address;
    }

    /// Transfers a named buffer to a caller that will release it independently of the workspace.
    public long detachAddress(ExecutionPlan.Buffer buffer) {
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
        this.storage.detach(index);
        this.firstLayerAddresses[index] = 0;
        return address;
    }

    public boolean hasBuffer(ExecutionPlan.Buffer buffer) {
        Objects.requireNonNull(buffer, "buffer");
        return !this.closed && this.firstLayerByteSizes[buffer.ordinal()] > 0;
    }

    public long address(ExecutionPlan.Buffer buffer, int index) {
        if (buffer == ExecutionPlan.Buffer.SLICE_PROJECTION) {
            return projectionAddress(index);
        }
        if (index != 0) {
            throw new IllegalStateException("named instruction buffers are not indexed");
        }
        return address(buffer);
    }

    public long bufferByteSize(ExecutionPlan.Buffer buffer) {
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

    /// Ends this submission's binding. Borrowed storage stays allocated for its owner's next quantum;
    /// owned storage is freed, and a failed free leaves the workspace open so close can be retried.
    @Override
    public void close() {
        if (this.closed) {
            return;
        }
        if (this.ownsStorage) this.storage.close();
        this.closed = true;
        Arrays.fill(this.firstLayerAddresses, 0);
        Arrays.fill(this.projectionAddresses, 0);
        this.hiddenStateAddress = 0;
        this.normalizedAddress = 0;
        this.inputAddress = 0;
    }

    private boolean hasFirstLayerBuffers() {
        for (long byteSize : this.firstLayerByteSizes) {
            if (byteSize != 0) return true;
        }
        return false;
    }
}
