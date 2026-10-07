package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorHandle;

public record MtpWeights(
        TensorHandle embeddingNorm,
        TensorHandle hiddenNorm,
        TensorHandle projection,
        LayerWeights layer,
        TensorHandle finalNorm) {}
