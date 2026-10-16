package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import java.util.Map;

public record Weights(
        Qwen38Config config,
        TensorHandle tokenEmbedding,
        LayerWeights[] layers,
        TensorHandle finalNorm,
        TensorHandle lmHead,
        MtpWeights mtp,
        Map<String, TensorHandle> runtimeObjects,
        DFlash2Weights dflash2,
        Map<String, int[]> segmentEnds) {

    public Weights(
            Qwen38Config config,
            TensorHandle tokenEmbedding,
            LayerWeights[] layers,
            TensorHandle finalNorm,
            TensorHandle lmHead,
            MtpWeights mtp,
            Map<String, TensorHandle> runtimeObjects,
            DFlash2Weights dflash2) {
        this(config, tokenEmbedding, layers, finalNorm, lmHead, mtp, runtimeObjects, dflash2, Map.of());
    }

    public Weights(
            Qwen38Config config,
            TensorHandle tokenEmbedding,
            LayerWeights[] layers,
            TensorHandle finalNorm,
            TensorHandle lmHead,
            MtpWeights mtp,
            Map<String, TensorHandle> runtimeObjects) {
        this(config, tokenEmbedding, layers, finalNorm, lmHead, mtp, runtimeObjects, null);
    }

    public Weights(
            Qwen38Config config,
            TensorHandle tokenEmbedding,
            LayerWeights[] layers,
            TensorHandle finalNorm,
            TensorHandle lmHead,
            MtpWeights mtp) {
        this(config, tokenEmbedding, layers, finalNorm, lmHead, mtp, Map.of());
    }

    public Weights {
        runtimeObjects = runtimeObjects == null ? Map.of() : Map.copyOf(runtimeObjects);
        segmentEnds = segmentEnds == null ? Map.of() : Map.copyOf(segmentEnds);
    }

    /// These weights with the host-known segment ends of their trellis objects, by object name.
    public Weights withSegmentEnds(Map<String, int[]> ends) {
        return new Weights(config, tokenEmbedding, layers, finalNorm, lmHead, mtp, runtimeObjects, dflash2, ends);
    }

    /// The disk-backed objects: those execution reads from the artifact file.
    public java.util.List<TensorHandle> diskBacked() {
        return this.runtimeObjects.values().stream()
                .filter(TensorHandle::diskBacked)
                .toList();
    }
}
