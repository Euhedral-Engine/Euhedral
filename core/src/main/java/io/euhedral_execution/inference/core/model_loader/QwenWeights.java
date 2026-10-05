package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.layer_weights.DFlash2Weights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenMtpWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import java.util.Map;

public record QwenWeights(
        QwenConfig config,
        TensorHandle tokenEmbedding,
        QwenLayerWeights[] layers,
        TensorHandle finalNorm,
        TensorHandle lmHead,
        QwenMtpWeights mtp,
        Map<String, TensorHandle> runtimeObjects,
        DFlash2Weights dflash2) {

    public QwenWeights(
            QwenConfig config,
            TensorHandle tokenEmbedding,
            QwenLayerWeights[] layers,
            TensorHandle finalNorm,
            TensorHandle lmHead,
            QwenMtpWeights mtp,
            Map<String, TensorHandle> runtimeObjects) {
        this(config, tokenEmbedding, layers, finalNorm, lmHead, mtp, runtimeObjects, null);
    }

    public QwenWeights(
            QwenConfig config,
            TensorHandle tokenEmbedding,
            QwenLayerWeights[] layers,
            TensorHandle finalNorm,
            TensorHandle lmHead,
            QwenMtpWeights mtp) {
        this(config, tokenEmbedding, layers, finalNorm, lmHead, mtp, Map.of());
    }

    public QwenWeights {
        runtimeObjects = runtimeObjects == null ? Map.of() : Map.copyOf(runtimeObjects);
    }
}
