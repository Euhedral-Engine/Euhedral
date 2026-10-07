package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorHandle;

/// Fused full-attention runtime objects from the compact NInfer-compatible inventory.
public record AttentionWeights(
        TensorHandle queryKey,
        TensorHandle gateValue,
        TensorHandle queryNorm,
        TensorHandle keyNorm,
        TensorHandle output)
        implements MixerWeights {}
