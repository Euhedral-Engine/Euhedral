package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import java.util.Objects;

/// Persistent convolution and recurrent buffers for every GDN layer in one sequence.
public final class GdnSequenceStates implements AutoCloseable {

    private final QwenGdnSequenceState[] states;
    private boolean closed;

    private GdnSequenceStates(QwenGdnSequenceState[] states) {
        this.states = states;
    }

    public static GdnSequenceStates allocate(
            ExecutionGpu gpu,
            QwenLayerType[] layerTypes,
            int keyHeads,
            int valueHeads,
            int keyHeadDim,
            int valueHeadDim,
            int convolutionKernelDim) {
        Objects.requireNonNull(gpu, "gpu");
        Objects.requireNonNull(layerTypes, "layerTypes");
        QwenGdnSequenceState[] states = new QwenGdnSequenceState[layerTypes.length];
        try {
            for (int layerIndex = 0; layerIndex < layerTypes.length; layerIndex++) {
                if (layerTypes[layerIndex] == QwenLayerType.GATED_DELTA_NET) {
                    states[layerIndex] = QwenGdnSequenceState.allocate(
                            gpu, keyHeads, valueHeads, keyHeadDim, valueHeadDim, convolutionKernelDim);
                }
            }
            return new GdnSequenceStates(states);
        } catch (RuntimeException | Error failure) {
            closeAllocated(states, failure);
            throw failure;
        }
    }

    void fingerprint(CaptureFingerprint fingerprint) {
        for (QwenGdnSequenceState state : this.states) if (state != null) state.fingerprint(fingerprint);
    }

    public QwenGdnSequenceState forLayer(int layerIndex) {
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

    private static Throwable closeAllocated(QwenGdnSequenceState[] states, Throwable failure) {
        for (int index = states.length - 1; index >= 0; index--) {
            QwenGdnSequenceState state = states[index];
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
