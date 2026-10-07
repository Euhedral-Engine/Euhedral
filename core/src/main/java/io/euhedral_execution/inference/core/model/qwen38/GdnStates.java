package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.CaptureFingerprint;
import java.util.Objects;

/// Persistent convolution and recurrent buffers for every GDN layer in one sequence.
public final class GdnStates implements AutoCloseable {

    private final GdnState[] states;
    private boolean closed;

    private GdnStates(GdnState[] states) {
        this.states = states;
    }

    public static GdnStates allocate(
            ExecutionGpu gpu,
            LayerType[] layerTypes,
            int keyHeads,
            int valueHeads,
            int keyHeadDim,
            int valueHeadDim,
            int convolutionKernelDim) {
        Objects.requireNonNull(gpu, "gpu");
        Objects.requireNonNull(layerTypes, "layerTypes");
        GdnState[] states = new GdnState[layerTypes.length];
        try {
            for (int layerIndex = 0; layerIndex < layerTypes.length; layerIndex++) {
                if (layerTypes[layerIndex] == LayerType.GATED_DELTA_NET) {
                    states[layerIndex] = GdnState.allocate(
                            gpu, keyHeads, valueHeads, keyHeadDim, valueHeadDim, convolutionKernelDim);
                }
            }
            return new GdnStates(states);
        } catch (RuntimeException | Error failure) {
            closeAllocated(states, failure);
            throw failure;
        }
    }

    void fingerprint(CaptureFingerprint fingerprint) {
        for (GdnState state : this.states) if (state != null) state.fingerprint(fingerprint);
    }

    public GdnState forLayer(int layerIndex) {
        if (this.closed) throw new IllegalStateException("GDN sequence states are closed");
        if (layerIndex < 0 || layerIndex >= this.states.length || this.states[layerIndex] == null) {
            throw new IllegalArgumentException("layer does not have GDN sequence state: " + layerIndex);
        }
        return this.states[layerIndex];
    }

    @Override
    public void close() {
        if (this.closed) return;
        Throwable failure = closeAllocated(this.states, null);
        if (failure != null) throw propagate(failure);
        this.closed = true;
    }

    private static Throwable closeAllocated(GdnState[] states, Throwable failure) {
        for (int index = states.length - 1; index >= 0; index--) {
            GdnState state = states[index];
            if (state == null) continue;
            try {
                state.close();
                states[index] = null;
            } catch (Throwable cleanupFailure) {
                if (failure == null) failure = cleanupFailure;
                else failure.addSuppressed(cleanupFailure);
            }
        }
        return failure;
    }

    private static RuntimeException propagate(Throwable failure) {
        if (failure instanceof RuntimeException runtimeException) return runtimeException;
        if (failure instanceof Error error) throw error;
        return new IllegalStateException("failed to release GDN sequence states", failure);
    }
}
