package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.artifact.format.LinearFormat;
import io.euhedral_execution.inference.core.artifact.format.LinearFormats;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.ScratchUse;
import io.euhedral_execution.inference.core.runtime.graph.CaptureFingerprint;
import java.util.Objects;

/// The dense runtime's one workspace: one allocation per [Workspace] slot, sized at load for `maxRows` rows of every
/// view of the plan, and bound by every graph. Nothing grows while quanta run: a quantum that needs more than a slot
/// holds is refused at admission. Reuse of a slot across graphs is ordered by edges ([Shape#workspaceBuffers] and
/// the runtime's workspace owner). The input record and the logits stay in each graph's own storage.
///
/// Its buffer indexes are the [Workspace] slots, then the expansion scratch, then the plan's staging slots (allocated
/// with the model; ordered here like any buffer).
public final class SharedWorkspace implements AutoCloseable {

    private final ExecutionGpu gpu;
    private final int maxRows;
    private final long[] addresses;
    private final long[] capacities;
    /// The expansion scratch buffers: the largest region a declared stage of any view takes at `maxRows`, per buffer;
    /// 0 when none does.
    private final long[] scratchAddresses;
    private final long[] scratchSizes;
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
        long[] scratch = new long[plan.scratchRegions()];
        for (Shape shape : plan.shapes()) {
            for (int stage = 0; stage < shape.topology().size(); stage++) {
                ScratchUse use = shape.scratchUse(stage);
                if (use == null) continue;
                ExecutionPlan.Instruction instruction = shape.instructions().get(stage);
                int region = shape.scratchRegion(stage);
                LinearFormat format = scratchFormat(instruction);
                // A second buffer exists only for stages that never take a prefill route.
                long bytes = region == 0
                        ? format.scratchBytes(
                                gpu,
                                use,
                                maxRows,
                                instruction.inputWidth(),
                                instruction.outputWidth(),
                                instruction.weights())
                        : format.decodeScratchBytes(
                                gpu,
                                use,
                                maxRows,
                                instruction.inputWidth(),
                                instruction.outputWidth(),
                                instruction.weights());
                scratch[region] = Math.max(scratch[region], bytes);
            }
        }
        this.scratchSizes = scratch;
        this.scratchAddresses = new long[scratch.length];
        try {
            for (int region = 0; region < scratch.length; region++)
                if (scratch[region] > 0) this.scratchAddresses[region] = gpu.allocate(scratch[region]);
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

    /// The format that sizes `instruction`'s scratch: the one its weights are in ([LinearFormats#routing]).
    private static LinearFormat scratchFormat(ExecutionPlan.Instruction instruction) {
        return instruction.weights().isEmpty()
                ? LinearFormats.routing(null, null)
                : LinearFormats.routing(instruction.weightFormat(), instruction.weightLayout());
    }

    /// The [Workspace] slots of `plan`: every named buffer, the input record, the hidden and normalized state, and
    /// one per projection of its widest view.
    static int slotCount(ExecutionPlan plan) {
        int projections = 0;
        for (Shape shape : plan.shapes())
            projections = Math.max(projections, shape.projectionWidths().size());
        return Workspace.PROJECTION_SLOTS + projections;
    }

    /// The workspace buffers a shape of `plan` may name: its slots, the expansion scratch, then the staging slots.
    static int bufferCount(ExecutionPlan plan) {
        return slotCount(plan)
                + plan.scratchRegions()
                + (plan.staging() == null ? 0 : plan.staging().slots());
    }

    /// Staging slot `slot`'s buffer index: a transfer writes it, the staged weight's consumer reads it.
    static int stagingBuffer(ExecutionPlan plan, int slot) {
        return slotCount(plan) + plan.scratchRegions() + slot;
    }

    /// The expansion scratch's buffer index.
    static int scratchBuffer(ExecutionPlan plan) {
        return scratchBuffer(plan, 0);
    }

    /// Scratch buffer `region`'s buffer index.
    static int scratchBuffer(ExecutionPlan plan, int region) {
        return slotCount(plan) + region;
    }

    /// Slot `slot`'s allocation; `bytes` must fit the size it was given at load.
    long acquire(int slot, long bytes) {
        if (this.closed) throw new IllegalStateException("the workspace is closed");
        if (bytes > this.capacities[slot])
            throw new IllegalArgumentException("the quantum exceeds the workspace sized at load for " + this.maxRows
                    + " rows (slot " + slot + " needs " + bytes + " bytes, holds " + this.capacities[slot] + ")");
        return this.addresses[slot];
    }

    /// The expansion scratch's address, or 0 when no declared stage takes scratch on this GPU.
    long scratchAddress() {
        return scratchAddress(0);
    }

    long scratchBytes() {
        return scratchBytes(0);
    }

    long scratchAddress(int region) {
        return region < this.scratchAddresses.length ? this.scratchAddresses[region] : 0;
    }

    long scratchBytes(int region) {
        return region < this.scratchSizes.length ? this.scratchSizes[region] : 0;
    }

    /// The rows the workspace was sized for.
    public int maxRows() {
        return this.maxRows;
    }

    void fingerprint(CaptureFingerprint fingerprint) {
        for (int region = 0; region < this.scratchAddresses.length; region++)
            fingerprint.add(this.gpu, this.scratchAddresses[region]).add(this.scratchSizes[region]);
        for (int slot = 0; slot < this.addresses.length; slot++)
            fingerprint.add(this.gpu, this.addresses[slot]).add(this.capacities[slot]);
    }

    /// Device bytes the workspace holds.
    public long retainedBytes() {
        if (this.closed) return 0;
        long total = 0;
        for (int region = 0; region < this.scratchAddresses.length; region++)
            if (this.scratchAddresses[region] != 0) total += this.scratchSizes[region];
        for (int slot = 0; slot < this.addresses.length; slot++)
            if (this.addresses[slot] != 0) total += this.capacities[slot];
        return total;
    }

    /// Frees the workspace. Lifecycle: the runtime closes it after every quantum retired.
    @Override
    public void close() {
        RuntimeException failure = null;
        for (int region = 0; region < this.scratchAddresses.length; region++) {
            if (this.scratchAddresses[region] == 0) continue;
            try {
                this.gpu.free(this.scratchAddresses[region]);
                this.scratchAddresses[region] = 0;
            } catch (RuntimeException freeFailure) {
                if (failure == null) failure = freeFailure;
                else failure.addSuppressed(freeFailure);
            }
        }
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
