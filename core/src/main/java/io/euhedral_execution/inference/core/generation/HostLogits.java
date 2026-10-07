package io.euhedral_execution.inference.core.generation;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.CaptureFingerprint;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/// A sequence's host copy of the final logits row of its latest sampling quantum, or of the token a
/// greedy selection on the device chose from it.
///
/// The quantum that produces the row queues its device-to-host copy on its own stream, before its
/// retirement boundary, so a successful retirement proves that the row is complete; the CPU reads it
/// only after that, without a transfer or synchronization of its own. One sequence samples serially,
/// so one pinned row per session suffices and concurrent sessions never share one.
///
/// With device selection on, the quantum instead queues an argmax over the row and copies back its
/// 8-byte result: the token the host argmax would choose, without the 0.5 MB row, its conversion or the
/// host scan at the token boundary.
public final class HostLogits implements AutoCloseable {

    private final ExecutionGpu gpu;
    private final int vocabularySize;
    private final long rowBytes;
    private ExecutionGpu.ReadbackBuffer row;
    private ExecutionGpu.ReadbackBuffer selection;
    private long deviceSelection;
    private ExecutionGpu.ReadbackBuffer rowSelections;
    private long deviceRowSelections;
    private int rowSelectionCapacity;
    private int queuedRows;
    private int readyRows;
    private boolean selectOnDevice;
    private boolean queued;
    private boolean queuedSelection;
    private boolean ready;
    private boolean selectionReady;
    private boolean closed;

    public HostLogits(ExecutionGpu gpu, int vocabularySize) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (vocabularySize <= 0) throw new IllegalArgumentException("vocabularySize must be positive");
        this.vocabularySize = vocabularySize;
        this.rowBytes = Math.multiplyExact((long) vocabularySize, Short.BYTES);
    }

    public int vocabularySize() {
        return this.vocabularySize;
    }

    public void fingerprint(CaptureFingerprint fingerprint) {
        fingerprint.add(this.selectOnDevice ? 1 : 0).add(this.gpu, this.deviceSelection);
        fingerprint.add(this.gpu, this.deviceRowSelections);
        fingerprint.add(this.rowSelectionCapacity);
        for (ExecutionGpu.ReadbackBuffer buffer :
                new ExecutionGpu.ReadbackBuffer[] {this.row, this.selection, this.rowSelections})
            fingerprint.add(this.gpu, buffer == null ? 0 : buffer.segment().address());
    }

    /// Whether the next sampling quanta select greedily on the device instead of copying the row. Only
    /// for a greedy sampler with no vocabulary constraint; set between quanta.
    public void selectOnDevice(boolean enabled) {
        this.selectOnDevice = enabled;
    }

    /// Queues the copy of the last of `rows` BF16 logits rows at `logitsAddress`, or of its device
    /// selection. The quantum's logits stage calls it with the quantum's stream selected, after
    /// launching the rows' producer.
    public void queueFinalRow(long logitsAddress, int rows) {
        if (this.closed) throw new IllegalStateException("host logits are closed");
        if (rows <= 0) throw new IllegalArgumentException("rows must be positive");
        this.ready = false;
        this.selectionReady = false;
        long finalRow = Math.addExact(logitsAddress, Math.multiplyExact((long) (rows - 1), this.rowBytes));
        if (this.selectOnDevice) {
            // Allocated and pinned once per session, while the host runs ahead of the device.
            if (this.deviceSelection == 0) this.deviceSelection = this.gpu.allocate(Long.BYTES);
            if (this.selection == null) this.selection = this.gpu.allocateReadbackBuffer(Long.BYTES);
            if (this.gpu.argmaxBf16(finalRow, this.vocabularySize, this.deviceSelection)) {
                this.gpu.copyDeviceToReadback(this.selection, this.deviceSelection, Long.BYTES);
                this.queuedSelection = true;
                this.queued = false;
                return;
            }
        }
        // Pinned once per session, while the host runs ahead of the device.
        if (this.row == null) this.row = this.gpu.allocateReadbackBuffer(this.rowBytes);
        this.gpu.copyDeviceToReadback(this.row, finalRow, this.rowBytes);
        this.queued = true;
        this.queuedSelection = false;
    }

    /// Queues a greedy device selection for each of `rows` BF16 logits rows at `logitsAddress` and the
    /// copy of their 8-byte results: a verifier reads back only token IDs, never vocabulary rows.
    public void queueRowSelections(long logitsAddress, int rows) {
        if (this.closed) throw new IllegalStateException("host logits are closed");
        if (rows <= 0) throw new IllegalArgumentException("rows must be positive");
        if (!this.selectOnDevice) throw new IllegalStateException("row selections need device selection");
        this.ready = false;
        this.selectionReady = false;
        this.readyRows = 0;
        if (rows > this.rowSelectionCapacity) {
            // Grown between sessions' first verifications; the previous quantum retired before this one.
            if (this.rowSelections != null) this.rowSelections.close();
            if (this.deviceRowSelections != 0) this.gpu.free(this.deviceRowSelections);
            this.rowSelections = null;
            this.deviceRowSelections = 0;
            this.deviceRowSelections = this.gpu.allocate((long) rows * Long.BYTES);
            this.rowSelections = this.gpu.allocateReadbackBuffer((long) rows * Long.BYTES);
            this.rowSelectionCapacity = rows;
        }
        for (int row = 0; row < rows; row++) {
            long rowAddress = Math.addExact(logitsAddress, Math.multiplyExact((long) row, this.rowBytes));
            if (!this.gpu.argmaxBf16(
                    rowAddress, this.vocabularySize, this.deviceRowSelections + (long) row * Long.BYTES))
                throw new IllegalStateException("device greedy selection is unavailable");
        }
        this.gpu.copyDeviceToReadback(this.rowSelections, this.deviceRowSelections, (long) rows * Long.BYTES);
        this.queued = false;
        this.queuedSelection = false;
        this.queuedRows = rows;
    }

    /// Publishes a queued row or selection once its quantum retired; only a successful quantum's
    /// result becomes readable.
    public void retired(boolean succeeded) {
        this.ready = succeeded && this.queued;
        this.selectionReady = succeeded && this.queuedSelection;
        this.readyRows = succeeded ? this.queuedRows : 0;
        this.queued = false;
        this.queuedSelection = false;
        this.queuedRows = 0;
    }

    /// The device's greedy token for each row of the latest retired verification (lowest token ID among
    /// equal maxima). Readable from the verification's own retirement onward.
    public int[] selectedTokens() {
        int rows = this.queuedRows > 0 ? this.queuedRows : this.readyRows;
        if (rows == 0 || this.closed) throw new IllegalStateException("no verified row selections are available");
        int[] tokens = new int[rows];
        for (int row = 0; row < rows; row++) {
            long key = this.rowSelections.segment().get(ValueLayout.JAVA_LONG, (long) row * Long.BYTES);
            if (key == 0) throw new IllegalArgumentException("logit row " + row + " has no selectable token");
            tokens[row] = (int) (0xFFFF_FFFFL - (key & 0xFFFF_FFFFL));
        }
        return tokens;
    }

    /// Whether the latest retired sampling quantum selected its token on the device.
    public boolean hasSelection() {
        return this.selectionReady && !this.closed;
    }

    /// The token the device selected: the host argmax of the final row (lowest token ID among equal
    /// maxima). Valid after a successful quantum retired and until the next sampling quantum is admitted.
    public int selectedToken() {
        if (!hasSelection()) throw new IllegalStateException("no retired device selection is available");
        long key = this.selection.segment().get(ValueLayout.JAVA_LONG, 0);
        if (key == 0) throw new IllegalArgumentException("logit row has no selectable token");
        return (int) (0xFFFF_FFFFL - (key & 0xFFFF_FFFFL));
    }

    /// The final logits row, BF16 in vocabulary order. Valid after a successful quantum retired and
    /// until the next sampling quantum is admitted.
    public MemorySegment row() {
        if (!this.ready || this.closed) throw new IllegalStateException("no retired logits row is available");
        return this.row.segment();
    }

    /// Releases the pinned row, the selection and its device word. A GPU that cannot prove its
    /// submitted work stopped may still write them, so they are then retained.
    @Override
    public void close() {
        if (this.closed) return;
        this.ready = false;
        this.selectionReady = false;
        this.readyRows = 0;
        if (this.row != null
                || this.selection != null
                || this.deviceSelection != 0
                || this.rowSelections != null
                || this.deviceRowSelections != 0) {
            if (!this.gpu.completionProven()) return;
            if (this.row != null) this.row.close();
            if (this.selection != null) this.selection.close();
            if (this.deviceSelection != 0) this.gpu.free(this.deviceSelection);
            if (this.rowSelections != null) this.rowSelections.close();
            if (this.deviceRowSelections != 0) this.gpu.free(this.deviceRowSelections);
            this.row = null;
            this.selection = null;
            this.deviceSelection = 0;
            this.rowSelections = null;
            this.deviceRowSelections = 0;
        }
        this.closed = true;
    }
}
