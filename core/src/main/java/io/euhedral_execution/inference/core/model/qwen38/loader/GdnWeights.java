package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorHandle;

/// Fused GDN runtime objects from the compact NInfer-compatible inventory.
public record GdnWeights(
        TensorHandle aLog,
        TensorHandle dtBias,
        TensorHandle convolution,
        TensorHandle aProjection,
        TensorHandle bProjection,
        TensorHandle queryKey,
        TensorHandle valueZ,
        TensorHandle norm,
        TensorHandle output)
        implements MixerWeights {}
