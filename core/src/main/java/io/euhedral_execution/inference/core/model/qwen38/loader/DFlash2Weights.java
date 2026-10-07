package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorHandle;

/// The DFlash2 drafter's objects (`dflash2/` in the artifact). Its projections are BF16 or NVFP4; norms, dynamic
/// convolutions, the feature fusion and the selector are BF16. The selector's codebooks are read in place from
/// mapped host memory: a step gathers only the rows of its anchor and candidates.
public record DFlash2Weights(
        DFlash2Config config,
        TensorHandle fc,
        TensorHandle hiddenNorm,
        TensorHandle finalNorm,
        Layer[] layers,
        TensorHandle selectorProjection,
        TensorHandle predecessorCodebook,
        TensorHandle successorCodebook) {

    public DFlash2Weights {
        layers = layers.clone();
    }

    @Override
    public Layer[] layers() {
        return this.layers.clone();
    }

    /// One draft layer: Qwen3 sliding attention and SwiGLU, each wrapped in a grouped dynamic convolution.
    public record Layer(
            TensorHandle inputNorm,
            TensorHandle postAttentionNorm,
            TensorHandle query,
            TensorHandle keyValue,
            TensorHandle output,
            TensorHandle queryNorm,
            TensorHandle keyNorm,
            TensorHandle gateUp,
            TensorHandle down,
            TensorHandle attentionConvBase,
            TensorHandle attentionConvProjection,
            TensorHandle mlpConvBase,
            TensorHandle mlpConvProjection) {}
}
