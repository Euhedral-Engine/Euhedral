package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import java.util.Objects;

/// Persistent per-layer key/value caches for one Qwen sequence.
public final class AttentionSequenceStates implements AutoCloseable {

    private final AttentionKvState[] states;
    private final ExecutionGpu gpu;
    private boolean closed;
    /// Speculative drafting: base hidden rows that seed MTP rows, and the MTP's own hidden for its next
    /// recursive row (docs/MTP_CONTRACT.md §4). Allocated by the first quantum that needs them.
    private long draftSeedRows;
    private int draftSeedCapacity;
    private long draftRecursionHidden;

    private AttentionSequenceStates(ExecutionGpu gpu, AttentionKvState[] states) {
        this.gpu = gpu;
        this.states = states;
    }

    public static AttentionSequenceStates allocate(ExecutionGpu gpu, QwenLayerType[] layerTypes, int keyValueWidth) {
        return allocate(gpu, layerTypes, keyValueWidth, false);
    }

    /// With `mtpLayer`, the MTP layer's own cache follows the base layers, at index layerTypes.length.
    public static AttentionSequenceStates allocate(
            ExecutionGpu gpu, QwenLayerType[] layerTypes, int keyValueWidth, boolean mtpLayer) {
        Objects.requireNonNull(gpu, "gpu");
        Objects.requireNonNull(layerTypes, "layerTypes");
        AttentionKvState[] states = new AttentionKvState[layerTypes.length + (mtpLayer ? 1 : 0)];
        for (int index = 0; index < states.length; index++) {
            if (index == layerTypes.length || layerTypes[index] == QwenLayerType.FULL_ATTENTION) {
                states[index] = new AttentionKvState(gpu, keyValueWidth);
            }
        }
        return new AttentionSequenceStates(gpu, states);
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
        for (long address : new long[] {this.draftSeedRows, this.draftRecursionHidden}) {
            if (address == 0) continue;
            try {
                this.gpu.free(address);
            } catch (Throwable cleanupFailure) {
                failure = cleanupFailure;
            }
        }
        this.draftSeedRows = 0;
        this.draftRecursionHidden = 0;
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
        this.closed = true;
        for (AttentionKvState state : this.states) this.closed &= state == null;
        if (failure instanceof Error error) throw error;
        if (failure instanceof RuntimeException runtimeException) throw runtimeException;
        if (failure != null) throw new IllegalStateException("failed to release attention sequence states", failure);
    }
}
