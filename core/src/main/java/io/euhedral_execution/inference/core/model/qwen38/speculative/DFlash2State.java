package io.euhedral_execution.inference.core.model.qwen38.speculative;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Config;
import io.euhedral_execution.inference.core.runtime.graph.CaptureFingerprint;
import java.util.Objects;

/// The DFlash2 drafter's per-sequence state (docs/DFLASH2.md).
///
/// - **Tap rows:** the target's hidden rows after each tapped layer, `[row][tap][hidden]` BF16, for the rows of the
///   latest target quantum that seeds drafting (a prefill chunk or a verification). The next DRAFT_CONTEXT quantum
///   consumes them before the next target quantum overwrites them; a sequence's quanta are serial, so one buffer
///   serves. Sized by the largest quantum (a prefill chunk), and never holds more than that quantum's rows.
/// - **Context ring:** each draft layer's keys and values of the committed positions, `[slot][kvHeads * headDim]`
///   BF16 with position `p` at slot `p % window`. The drafter attends over the last `window` positions only, so the
///   ring is all the context it keeps, whatever the sequence's length: [#contextLength] positions are committed.
public final class DFlash2State implements AutoCloseable {
    private final ExecutionGpu gpu;
    private final DFlash2Config config;
    private final long[] ringKeys;
    private final long[] ringValues;
    private long taps;
    private int tapCapacity;
    private int contextLength;
    private boolean closed;

    public DFlash2State(ExecutionGpu gpu, DFlash2Config config) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.config = Objects.requireNonNull(config, "config");
        this.ringKeys = new long[config.layers()];
        this.ringValues = new long[config.layers()];
        try {
            for (int layer = 0; layer < config.layers(); layer++) {
                this.ringKeys[layer] = gpu.allocate(ringBytes());
                this.ringValues[layer] = gpu.allocate(ringBytes());
            }
        } catch (RuntimeException | Error failure) {
            close();
            throw failure;
        }
    }

    public DFlash2Config config() {
        return this.config;
    }

    /// Bytes of one layer's key (or value) ring.
    public long ringBytes() {
        return (long) this.config.slidingWindow() * this.config.keyValueWidth() * Short.BYTES;
    }

    /// Tap rows for at least `rows` rows (a quantum's), grown between quanta.
    public long taps(int rows) {
        if (this.closed) throw new IllegalStateException("DFlash2 state is closed");
        if (rows > this.tapCapacity) {
            if (this.taps != 0) this.gpu.free(this.taps);
            this.taps = 0;
            this.tapCapacity = 0;
            int capacity = Math.max(rows, 8);
            this.taps = this.gpu.allocate((long) capacity * this.config.tapWidth() * Short.BYTES);
            this.tapCapacity = capacity;
        }
        return this.taps;
    }

    public long ringKeys(int layer) {
        return this.ringKeys[layer];
    }

    public long ringValues(int layer) {
        return this.ringValues[layer];
    }

    /// Positions `[0, contextLength)` have their keys and values in the ring (the last `window` of them).
    public int contextLength() {
        return this.contextLength;
    }

    /// Publishes context rows through `length`, at the retirement of the quantum that wrote them, or a restore.
    public void commitContext(int length) {
        if (length < this.contextLength) throw new IllegalArgumentException("the drafter's context never shrinks");
        this.contextLength = length;
    }

    public void fingerprint(CaptureFingerprint fingerprint) {
        fingerprint.add(this.gpu, this.taps).add(this.tapCapacity);
        for (int layer = 0; layer < this.ringKeys.length; layer++)
            fingerprint.add(this.gpu, this.ringKeys[layer]).add(this.gpu, this.ringValues[layer]);
    }

    @Override
    public void close() {
        Throwable failure = null;
        // Each buffer is forgotten only once freed, so a failed close can be retried.
        long[][] rings = {this.ringKeys, this.ringValues};
        for (long[] ring : rings) {
            for (int layer = 0; layer < ring.length; layer++) {
                try {
                    if (ring[layer] != 0) this.gpu.free(ring[layer]);
                    ring[layer] = 0;
                } catch (Throwable cleanupFailure) {
                    if (failure == null) failure = cleanupFailure;
                    else failure.addSuppressed(cleanupFailure);
                }
            }
        }
        try {
            if (this.taps != 0) this.gpu.free(this.taps);
            this.taps = 0;
        } catch (Throwable cleanupFailure) {
            if (failure == null) failure = cleanupFailure;
            else failure.addSuppressed(cleanupFailure);
        }
        this.closed = true;
        if (failure instanceof Error error) throw error;
        if (failure instanceof RuntimeException runtime) throw runtime;
        if (failure != null) throw new IllegalStateException("failed to release DFlash2 state", failure);
    }

    public boolean released() {
        if (this.taps != 0) return false;
        for (int layer = 0; layer < this.ringKeys.length; layer++)
            if (this.ringKeys[layer] != 0 || this.ringValues[layer] != 0) return false;
        return true;
    }
}
