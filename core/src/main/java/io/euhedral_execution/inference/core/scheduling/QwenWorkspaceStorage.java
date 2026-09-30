package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.util.Arrays;
import java.util.Objects;

/// Device storage behind the workspaces of one reusable stage graph.
///
/// A graph runs one quantum at a time, and its pool recycles it only after that quantum's device work
/// retired, so the graph is the natural owner of storage whose shape repeats from quantum to quantum.
/// Each quantum's [QwenExecutionWorkspace] acquires its buffers here at admission and releases them at
/// retirement without freeing them; the next quantum on the same graph finds them already allocated.
///
/// Storage is a fixed table of slots, one per workspace buffer, not a general allocator. A slot keeps
/// the largest allocation any binding requested, so memory is bounded by the graph's largest quantum
/// rather than by token count. A request that exceeds a slot's capacity replaces its allocation; that
/// only happens at admission, while the graph is idle, so no queued work references the old one.
public final class QwenWorkspaceStorage implements AutoCloseable {

    private final GpuMemory gpu;
    private long[] addresses = new long[0];
    private long[] capacities = new long[0];
    private boolean closed;

    public QwenWorkspaceStorage(GpuMemory gpu) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
    }

    /// Returns slot `slot`'s allocation, replacing it first if it holds fewer than `bytes` bytes.
    long acquire(int slot, long bytes) {
        if (this.closed) throw new IllegalStateException("workspace storage is closed");
        if (slot < 0) throw new IllegalArgumentException("slot must not be negative");
        if (bytes <= 0) throw new IllegalArgumentException("bytes must be positive");
        ensureSlot(slot);
        if (this.capacities[slot] >= bytes) return this.addresses[slot];
        if (this.addresses[slot] != 0) {
            this.gpu.free(this.addresses[slot]);
            this.addresses[slot] = 0;
            this.capacities[slot] = 0;
        }
        long address = this.gpu.allocate(bytes);
        if (address == 0) throw new IllegalStateException("GPU returned a null workspace address");
        this.addresses[slot] = address;
        this.capacities[slot] = bytes;
        return address;
    }

    /// Transfers slot `slot`'s allocation to a caller that releases it independently; the next binding
    /// that needs the slot allocates a new one.
    long detach(int slot) {
        if (this.closed || slot < 0 || slot >= this.addresses.length || this.addresses[slot] == 0) {
            throw new IllegalStateException("workspace storage slot has no allocation to detach: " + slot);
        }
        long address = this.addresses[slot];
        this.addresses[slot] = 0;
        this.capacities[slot] = 0;
        return address;
    }

    /// Device bytes this storage currently holds.
    public long retainedBytes() {
        long total = 0;
        for (long capacity : this.capacities) total += capacity;
        return total;
    }

    public boolean isClosed() {
        return this.closed;
    }

    /// Frees every retained allocation. Only an owner whose device work provably stopped may call it; a
    /// failed free keeps that slot for a retry.
    @Override
    public void close() {
        if (this.closed) return;
        Throwable failure = null;
        for (int slot = 0; slot < this.addresses.length; slot++) {
            if (this.addresses[slot] == 0) continue;
            try {
                this.gpu.free(this.addresses[slot]);
                this.addresses[slot] = 0;
                this.capacities[slot] = 0;
            } catch (RuntimeException | Error cleanupFailure) {
                if (failure == null) failure = cleanupFailure;
                else failure.addSuppressed(cleanupFailure);
            }
        }
        if (failure instanceof Error error) throw error;
        if (failure != null) throw (RuntimeException) failure;
        this.closed = true;
    }

    private void ensureSlot(int slot) {
        if (slot < this.addresses.length) return;
        int length = Math.max(slot + 1, 2 * this.addresses.length);
        this.addresses = Arrays.copyOf(this.addresses, length);
        this.capacities = Arrays.copyOf(this.capacities, length);
    }
}
