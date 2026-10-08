package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.CaptureFingerprint;
import java.util.Objects;

/// The dense runtime's one workspace: one allocation per [Workspace] slot, sized at load for `maxRows` rows of every
/// view of the plan, and bound by every graph. Nothing grows while quanta run: a quantum that needs more than a slot
/// holds is refused at admission. Reuse of a slot across graphs is ordered by edges ([Shape#workspaceBuffers] and
/// the runtime's workspace owner). The input record and the logits stay in each graph's own storage.
///
/// Its buffer indexes are the [Workspace] slots.
public final class SharedWorkspace implements AutoCloseable {

    private final ExecutionGpu gpu;
    private final int maxRows;
    private final long[] addresses;
    private final long[] capacities;
    private boolean closed;

    public SharedWorkspace(ExecutionGpu gpu, ExecutionPlan plan, int maxRows) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        Objects.requireNonNull(plan, "plan");
        if (maxRows <= 0) throw new IllegalArgumentException("maxRows must be positive");
        this.maxRows = maxRows;
        int slots = slotCount(plan);
        this.addresses = new long[slots];
        this.capacities = new long[slots];
        for (Shape shape : plan.shapes()) {
            long[] bytes = Workspace.slotBytes(shape, maxRows, slots);
            for (int slot = 0; slot < slots; slot++)
                this.capacities[slot] = Math.max(this.capacities[slot], bytes[slot]);
        }
        try {
            for (int slot = 0; slot < slots; slot++) {
                if (this.capacities[slot] == 0) continue;
                long address = gpu.allocate(this.capacities[slot]);
                if (address == 0) throw new IllegalStateException("GPU returned a null workspace address");
                this.addresses[slot] = address;
            }
        } catch (RuntimeException | Error failure) {
            close();
            throw failure;
        }
    }

    /// The [Workspace] slots of `plan`: every named buffer, the input record, the hidden and normalized state, and
    /// one per projection of its widest view.
    static int slotCount(ExecutionPlan plan) {
        int projections = 0;
        for (Shape shape : plan.shapes())
            projections = Math.max(projections, shape.projectionWidths().size());
        return Workspace.PROJECTION_SLOTS + projections;
    }

    /// The workspace buffers a shape of `plan` may name.
    static int bufferCount(ExecutionPlan plan) {
        return slotCount(plan);
    }

    /// Slot `slot`'s allocation; `bytes` must fit the size it was given at load.
    long acquire(int slot, long bytes) {
        if (this.closed) throw new IllegalStateException("the workspace is closed");
        if (bytes > this.capacities[slot])
            throw new IllegalArgumentException("the quantum exceeds the workspace sized at load for " + this.maxRows
                    + " rows (slot " + slot + " needs " + bytes + " bytes, holds " + this.capacities[slot] + ")");
        return this.addresses[slot];
    }

    /// The rows the workspace was sized for.
    public int maxRows() {
        return this.maxRows;
    }

    void fingerprint(CaptureFingerprint fingerprint) {
        for (int slot = 0; slot < this.addresses.length; slot++)
            fingerprint.add(this.gpu, this.addresses[slot]).add(this.capacities[slot]);
    }

    /// Device bytes the workspace holds.
    public long retainedBytes() {
        if (this.closed) return 0;
        long total = 0;
        for (int slot = 0; slot < this.addresses.length; slot++)
            if (this.addresses[slot] != 0) total += this.capacities[slot];
        return total;
    }

    /// Frees the workspace. Lifecycle: the runtime closes it after every quantum retired.
    @Override
    public void close() {
        RuntimeException failure = null;
        for (int slot = 0; slot < this.addresses.length; slot++) {
            if (this.addresses[slot] == 0) continue;
            try {
                this.gpu.free(this.addresses[slot]);
                this.addresses[slot] = 0;
            } catch (RuntimeException freeFailure) {
                if (failure == null) failure = freeFailure;
                else failure.addSuppressed(freeFailure);
            }
        }
        this.closed = true;
        if (failure != null) throw failure;
    }
}
