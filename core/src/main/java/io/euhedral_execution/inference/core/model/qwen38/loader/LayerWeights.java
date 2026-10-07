package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorHandle;

public record LayerWeights(
        int index, TensorHandle inputNorm, TensorHandle postAttentionNorm, MixerWeights mixer, FfnWeights ffn) {}
