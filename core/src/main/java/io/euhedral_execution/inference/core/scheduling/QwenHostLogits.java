package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/// A sequence's host copy of the final logits row of its latest sampling quantum.
///
/// The quantum that produces the row queues its device-to-host copy on its own stream, before its
/// retirement boundary, so a successful retirement proves that the row is complete; the CPU reads it
/// only after that, without a transfer or synchronization of its own. One sequence samples serially,
/// so one pinned row per session suffices and concurrent sessions never share one.
public final class QwenHostLogits implements AutoCloseable {

    private final ExecutionGpu gpu;
    private final int vocabularySize;
    private final long rowBytes;
    private ExecutionGpu.ReadbackBuffer row;
    private boolean queued;
    private boolean ready;
    private boolean closed;

    public QwenHostLogits(ExecutionGpu gpu, int vocabularySize) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (vocabularySize <= 0) throw new IllegalArgumentException("vocabularySize must be positive");
        this.vocabularySize = vocabularySize;
        this.rowBytes = Math.multiplyExact((long) vocabularySize, Short.BYTES);
    }

    public int vocabularySize() {
        return this.vocabularySize;
    }

    /// Queues the copy of the last of `rows` BF16 logits rows at `logitsAddress`. The quantum's logits
    /// stage calls it with the quantum's stream selected, after launching the rows' producer.
    void queueFinalRow(long logitsAddress, int rows) {
        if (this.closed) throw new IllegalStateException("host logits are closed");
        if (rows <= 0) throw new IllegalArgumentException("rows must be positive");
        this.ready = false;
        // Pinned once per session, while the host runs ahead of the device.
        if (this.row == null) this.row = this.gpu.allocateReadbackBuffer(this.rowBytes);
        long finalRow = Math.addExact(logitsAddress, Math.multiplyExact((long) (rows - 1), this.rowBytes));
        this.gpu.copyDeviceToReadback(this.row, finalRow, this.rowBytes);
        this.queued = true;
    }

    /// Publishes a queued row once its quantum retired; only a successful quantum's row becomes
    /// readable.
    void retired(boolean succeeded) {
        this.ready = succeeded && this.queued;
        this.queued = false;
    }

    /// The final logits row, BF16 in vocabulary order. Valid after a successful quantum retired and
    /// until the next sampling quantum is admitted.
    public MemorySegment row() {
        if (!this.ready || this.closed) throw new IllegalStateException("no retired logits row is available");
        return this.row.segment();
    }

    /// Releases the pinned row. A GPU that cannot prove its submitted work stopped may still copy into
    /// it, so the row is then retained.
    @Override
    public void close() {
        if (this.closed) return;
        this.ready = false;
        if (this.row != null) {
            if (!this.gpu.completionProven()) return;
            this.row.close();
            this.row = null;
        }
        this.closed = true;
    }
}
