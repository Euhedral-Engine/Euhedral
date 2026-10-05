package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/// A sequence's host copy of the latest DFlash2 block's proposal: its tokens and, for each, the selector's scores
/// of the candidates. The block's selector stage queues the copies on the quantum's stream before its retirement
/// boundary, so a successful retirement proves them complete; the host reads them only after that.
public final class DFlash2Proposal implements AutoCloseable {
    private final ExecutionGpu gpu;
    private final int positions;
    private final int candidates;
    private ExecutionGpu.ReadbackBuffer tokens;
    private ExecutionGpu.ReadbackBuffer scores;
    private boolean queued;
    private boolean ready;
    private boolean closed;

    public DFlash2Proposal(ExecutionGpu gpu, int positions, int candidates) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (positions <= 0 || candidates <= 0)
            throw new IllegalArgumentException("positions and candidates must be positive");
        this.positions = positions;
        this.candidates = candidates;
    }

    void fingerprint(CaptureFingerprint fingerprint) {
        fingerprint.add(
                this.gpu, this.tokens == null ? 0 : this.tokens.segment().address());
        fingerprint.add(
                this.gpu, this.scores == null ? 0 : this.scores.segment().address());
    }

    /// Queues the copies of the block's proposal tokens (int32) and scores (FP32) at the given device addresses.
    /// Called by the selector stage with the quantum's stream selected.
    public void queue(long deviceTokens, long deviceScores) {
        if (this.closed) throw new IllegalStateException("the proposal is closed");
        this.ready = false;
        // Pinned once per session, while the host runs ahead of the device.
        if (this.tokens == null) this.tokens = this.gpu.allocateReadbackBuffer((long) this.positions * Integer.BYTES);
        if (this.scores == null)
            this.scores = this.gpu.allocateReadbackBuffer((long) this.positions * this.candidates * Float.BYTES);
        this.gpu.copyDeviceToReadback(this.tokens, deviceTokens, (long) this.positions * Integer.BYTES);
        this.gpu.copyDeviceToReadback(this.scores, deviceScores, (long) this.positions * this.candidates * Float.BYTES);
        this.queued = true;
    }

    void retired(boolean succeeded) {
        this.ready = succeeded && this.queued;
        this.queued = false;
    }

    /// The proposed tokens of the latest retired block, in position order.
    public int[] tokens() {
        if (!this.ready || this.closed) throw new IllegalStateException("no retired proposal is available");
        return this.tokens
                .segment()
                .asSlice(0, (long) this.positions * Integer.BYTES)
                .toArray(ValueLayout.JAVA_INT);
    }

    /// The selector's scores of each position's candidates (in candidate order: logit descending), FP32.
    public float[] scores() {
        if (!this.ready || this.closed) throw new IllegalStateException("no retired proposal is available");
        return this.scores
                .segment()
                .asSlice(0, (long) this.positions * this.candidates * Float.BYTES)
                .toArray(ValueLayout.JAVA_FLOAT);
    }

    /// Releases the pinned copies; a GPU that cannot prove its work stopped keeps them.
    @Override
    public void close() {
        if (this.closed) return;
        this.ready = false;
        if ((this.tokens != null || this.scores != null) && !this.gpu.completionProven()) return;
        if (this.tokens != null) this.tokens.close();
        if (this.scores != null) this.scores.close();
        this.tokens = null;
        this.scores = null;
        this.closed = true;
    }
}
