package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorHandle;

/// Four-way fused attention projection used by the compact MTP object inventory.
public record MtpAttentionWeights(
        TensorHandle queryKeyGateValue, TensorHandle queryNorm, TensorHandle keyNorm, TensorHandle output)
        implements MixerWeights {}
