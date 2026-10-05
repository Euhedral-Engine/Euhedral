package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/// A sequence's host copy of the latest DFlash2 block's proposal: its tokens, each position's candidates and the
/// selector's scores of them. The block's selector stage queues the copies on the quantum's stream before its
/// retirement boundary, so a successful retirement proves them complete; the host reads them only after that.
public final class DFlash2Proposal implements AutoCloseable {
    private final ExecutionGpu gpu;
    private final int positions;
    private final int width;
    private ExecutionGpu.ReadbackBuffer tokens;
    private ExecutionGpu.ReadbackBuffer scores;
    private ExecutionGpu.ReadbackBuffer candidates;
    private boolean queued;
    private boolean ready;
    private boolean closed;

    public DFlash2Proposal(ExecutionGpu gpu, int positions, int candidates) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (positions <= 0 || candidates <= 0)
            throw new IllegalArgumentException("positions and candidates must be positive");
        this.positions = positions;
        this.width = candidates;
    }

    void fingerprint(CaptureFingerprint fingerprint) {
        for (ExecutionGpu.ReadbackBuffer buffer :
                new ExecutionGpu.ReadbackBuffer[] {this.tokens, this.scores, this.candidates})
            fingerprint.add(this.gpu, buffer == null ? 0 : buffer.segment().address());
    }

    /// Queues the copies of the block's proposal tokens (int32), scores (FP32) and candidate tokens (int32) at the
    /// given device addresses. Called by the selector stage with the quantum's stream selected.
    public void queue(long deviceTokens, long deviceScores, long deviceCandidates) {
        if (this.closed) throw new IllegalStateException("the proposal is closed");
        this.ready = false;
        long table = (long) this.positions * this.width * Float.BYTES;
        // Pinned once per session, while the host runs ahead of the device.
        if (this.tokens == null) this.tokens = this.gpu.allocateReadbackBuffer((long) this.positions * Integer.BYTES);
        if (this.scores == null) this.scores = this.gpu.allocateReadbackBuffer(table);
        if (this.candidates == null) this.candidates = this.gpu.allocateReadbackBuffer(table);
        this.gpu.copyDeviceToReadback(this.tokens, deviceTokens, (long) this.positions * Integer.BYTES);
        this.gpu.copyDeviceToReadback(this.scores, deviceScores, table);
        this.gpu.copyDeviceToReadback(this.candidates, deviceCandidates, table);
        this.queued = true;
    }

    void retired(boolean succeeded) {
        this.ready = succeeded && this.queued;
        this.queued = false;
    }

    private void requireReady() {
        if (!this.ready || this.closed) throw new IllegalStateException("no retired proposal is available");
    }

    /// The proposed tokens of the latest retired block, in position order.
    public int[] tokens() {
        requireReady();
        return this.tokens
                .segment()
                .asSlice(0, (long) this.positions * Integer.BYTES)
                .toArray(ValueLayout.JAVA_INT);
    }

    /// The selector's scores of each position's candidates (in candidate order: logit descending), FP32.
    public float[] scores() {
        requireReady();
        return this.scores
                .segment()
                .asSlice(0, (long) this.positions * this.width * Float.BYTES)
                .toArray(ValueLayout.JAVA_FLOAT);
    }

    /// Each position's candidate tokens, logit descending, row-major.
    public int[] candidates() {
        requireReady();
        return this.candidates
                .segment()
                .asSlice(0, (long) this.positions * this.width * Integer.BYTES)
                .toArray(ValueLayout.JAVA_INT);
    }

    /// Releases the pinned copies; a GPU that cannot prove its work stopped keeps them.
    @Override
    public void close() {
        if (this.closed) return;
        this.ready = false;
        if ((this.tokens != null || this.scores != null || this.candidates != null) && !this.gpu.completionProven())
            return;
        for (ExecutionGpu.ReadbackBuffer buffer :
                new ExecutionGpu.ReadbackBuffer[] {this.tokens, this.scores, this.candidates})
            if (buffer != null) buffer.close();
        this.tokens = null;
        this.scores = null;
        this.candidates = null;
        this.closed = true;
    }
}
