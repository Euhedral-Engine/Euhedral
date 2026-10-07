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
        DFlash2Weights dflash2) {

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
    }
}
