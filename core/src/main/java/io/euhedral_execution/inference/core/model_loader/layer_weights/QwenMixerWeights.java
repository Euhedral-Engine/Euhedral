package io.euhedral_execution.inference.core.model_loader.layer_weights;

public sealed interface QwenMixerWeights
        permits QwenCompactAttentionWeights, QwenCompactGatedDeltaNetWeights, QwenCompactMtpAttentionWeights {}
