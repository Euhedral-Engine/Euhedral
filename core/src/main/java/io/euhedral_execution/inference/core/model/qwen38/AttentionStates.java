package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.speculative.DFlash2State;
import io.euhedral_execution.inference.core.runtime.graph.CaptureFingerprint;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import java.util.Objects;

/// Persistent per-layer key/value caches for one Qwen sequence.
public final class AttentionStates implements AutoCloseable {

    private final AttentionKvState[] states;
    private final ExecutionGpu gpu;
    private boolean closed;
    /// Speculative drafting: base hidden rows that seed MTP rows, and the MTP's own hidden for its next
    /// recursive row (docs/MTP_CONTRACT.md §4). Allocated by the first quantum that needs them.
    private long draftSeedRows;
    private int draftSeedCapacity;
    private long draftRecursionHidden;
    /// The DFlash2 drafter's tap rows and context ring; allocated by the first quantum that needs them.
    private DFlash2State dflash2;
    /// Split-KV decode scratch shared by every attention layer of the sequence: a quantum's attention
    /// layers run one after another (each needs the previous layer's output) and the sequence's quanta are
    /// serialized by its lease, so one area serves them all.
    private long decodeScratch;
    private int decodeScratchHeads;
    private int decodeScratchRows;

    private AttentionStates(ExecutionGpu gpu, AttentionKvState[] states) {
        this.gpu = gpu;
        this.states = states;
    }

    public static AttentionStates allocate(ExecutionGpu gpu, LayerType[] layerTypes, int keyValueWidth) {
        return allocate(gpu, layerTypes, keyValueWidth, false);
    }

    /// With `mtpLayer`, the MTP layer's own cache follows the base layers, at index layerTypes.length.
    public static AttentionStates allocate(
            ExecutionGpu gpu, LayerType[] layerTypes, int keyValueWidth, boolean mtpLayer) {
        Objects.requireNonNull(gpu, "gpu");
        Objects.requireNonNull(layerTypes, "layerTypes");
        AttentionKvState[] states = new AttentionKvState[layerTypes.length + (mtpLayer ? 1 : 0)];
        for (int index = 0; index < states.length; index++) {
            if (index == layerTypes.length || layerTypes[index] == LayerType.FULL_ATTENTION) {
                states[index] = new AttentionKvState(gpu, keyValueWidth);
            }
        }
        return new AttentionStates(gpu, states);
    }

    /// Base post-final-norm hidden rows (BF16) for the next MTP catch-up, holding at least `rows` rows.
    public long draftSeedRows(int rows, int hidden) {
        if (this.closed) throw new IllegalStateException("attention sequence states are closed");
        if (rows > this.draftSeedCapacity) {
            if (this.draftSeedRows != 0) this.gpu.free(this.draftSeedRows);
            this.draftSeedRows = 0;
            this.draftSeedCapacity = 0;
            int capacity = Math.max(rows, 8);
            this.draftSeedRows = this.gpu.allocate((long) capacity * hidden * Short.BYTES);
            this.draftSeedCapacity = capacity;
        }
        return this.draftSeedRows;
    }

    /// The latest MTP row's post-`mtp.norm` hidden (BF16), seeding the next recursive draft row.
    public long draftRecursionHidden(int hidden) {
        if (this.closed) throw new IllegalStateException("attention sequence states are closed");
        if (this.draftRecursionHidden == 0) this.draftRecursionHidden = this.gpu.allocate((long) hidden * Short.BYTES);
        return this.draftRecursionHidden;
    }

    /// The sequence's DFlash2 state, allocated on first use.
    public DFlash2State dflash2(io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Config config) {
        if (this.closed) throw new IllegalStateException("attention sequence states are closed");
        if (this.dflash2 == null) this.dflash2 = new DFlash2State(this.gpu, config);
        return this.dflash2;
    }

    /// The sequence's DFlash2 state, or null before its first DFlash2 quantum.
    public DFlash2State dflash2() {
        return this.dflash2;
    }

    /// Split-KV scratch for `rows` decode rows: one one-row area (queryHeads x 64 splits x 258 floats) per
    /// row, as one-row decode (one area) and the row-exact attention twins of a verification (one per
    /// verified row) need. It grows when a verification first needs more rows.
    public long decodeScratch(int queryHeads, int rows) {
        if (this.closed) throw new IllegalStateException("attention sequence states are closed");
        if (queryHeads <= 0 || rows <= 0) throw new IllegalArgumentException("queryHeads and rows must be positive");
        if (this.decodeScratch != 0 && queryHeads != this.decodeScratchHeads)
            throw new IllegalArgumentException("decode head geometry changed");
        if (this.decodeScratch == 0 || rows > this.decodeScratchRows) {
            if (this.decodeScratch != 0) this.gpu.free(this.decodeScratch);
            this.decodeScratch = 0;
            this.decodeScratchRows = 0;
            this.decodeScratch =
                    this.gpu.allocate(Math.multiplyExact((long) queryHeads * rows, 64L * 258 * Float.BYTES));
            this.decodeScratchHeads = queryHeads;
            this.decodeScratchRows = rows;
        }
        return this.decodeScratch;
    }

    /// Whether every cache a quantum of `rows` rows at `start` appends to is already reserved: the base
    /// layers', or the MTP layer's (the last state) for a draft.
    boolean reserves(long start, int rows, boolean draft) {
        if (this.closed) return false;
        for (int index = 0; index < this.states.length; index++) {
            AttentionKvState state = this.states[index];
            if (state == null || (index == this.states.length - 1) != draft) continue;
            if (!state.reserves(start, rows)) return false;
        }
        return true;
    }

    void fingerprint(CaptureFingerprint fingerprint) {
        for (AttentionKvState state : this.states) if (state != null) state.fingerprint(fingerprint);
        fingerprint.add(this.gpu, this.draftSeedRows).add(this.draftSeedCapacity);
        fingerprint.add(this.gpu, this.draftRecursionHidden);
        fingerprint.add(this.gpu, this.decodeScratch).add(this.decodeScratchRows);
        if (this.dflash2 != null) this.dflash2.fingerprint(fingerprint);
    }

    /// Layer slots: the model's layers, plus the MTP layer's when the sequence has it.
    public int layerCount() {
        return this.states.length;
    }

    public AttentionKvState forLayer(int layerIndex) {
        if (this.closed) throw new IllegalStateException("attention sequence states are closed");
        if (layerIndex < 0 || layerIndex >= this.states.length || this.states[layerIndex] == null) {
            throw new IllegalArgumentException("layer does not have full-attention KV state: " + layerIndex);
        }
        return this.states[layerIndex];
    }

    @Override
    public void close() {
        if (this.closed) return;
        Throwable failure = null;
        // Each buffer is forgotten only once freed, so a failed close can be retried.
        try {
            if (this.draftSeedRows != 0) this.gpu.free(this.draftSeedRows);
            this.draftSeedRows = 0;
        } catch (Throwable cleanupFailure) {
            failure = cleanupFailure;
        }
        try {
            if (this.draftRecursionHidden != 0) this.gpu.free(this.draftRecursionHidden);
            this.draftRecursionHidden = 0;
        } catch (Throwable cleanupFailure) {
            if (failure == null) failure = cleanupFailure;
            else failure.addSuppressed(cleanupFailure);
        }
        try {
            if (this.decodeScratch != 0) this.gpu.free(this.decodeScratch);
            this.decodeScratch = 0;
        } catch (Throwable cleanupFailure) {
            if (failure == null) failure = cleanupFailure;
            else failure.addSuppressed(cleanupFailure);
        }
        try {
            if (this.dflash2 != null) this.dflash2.close();
        } catch (Throwable cleanupFailure) {
            if (failure == null) failure = cleanupFailure;
            else failure.addSuppressed(cleanupFailure);
        }
        for (int index = this.states.length - 1; index >= 0; index--) {
            AttentionKvState state = this.states[index];
            if (state == null) continue;
            try {
                state.close();
                this.states[index] = null;
            } catch (Throwable cleanupFailure) {
                if (failure == null) failure = cleanupFailure;
                else failure.addSuppressed(cleanupFailure);
            }
        }
        this.closed = this.draftSeedRows == 0
                && this.draftRecursionHidden == 0
                && this.decodeScratch == 0
                && (this.dflash2 == null || this.dflash2.released());
        for (AttentionKvState state : this.states) this.closed &= state == null;
        if (failure instanceof Error error) throw error;
        if (failure instanceof RuntimeException runtimeException) throw runtimeException;
        if (failure != null) throw new IllegalStateException("failed to release attention sequence states", failure);
    }
}
