package io.euhedral_execution.inference.core.model.qwen38.loader;

import java.util.Arrays;

/// The DFlash2 drafter's configuration, as the artifact's `dflash2/config` object records it (tools/euhedral_artifacts/
/// dflash2.py). The drafter reads the target's hidden rows after `targetLayers`, and drafts `blockSize - 1` tokens
/// after an anchor at once.
public record DFlash2Config(
        int layers,
        int hiddenSize,
        int intermediateSize,
        int attentionHeads,
        int keyValueHeads,
        int headDim,
        int blockSize,
        int maskToken,
        int convKernel,
        int convGroupSize,
        int selectorTopK,
        int selectorRank,
        int slidingWindow,
        int vocabSize,
        float rmsNormEpsilon,
        float ropeTheta,
        int[] targetLayers) {

    public static final int VERSION = 1;

    public DFlash2Config {
        targetLayers = targetLayers.clone();
        if (layers <= 0 || hiddenSize <= 0 || blockSize < 2 || targetLayers.length == 0 || headDim % 2 != 0)
            throw new IllegalArgumentException("invalid DFlash2 configuration");
        if (attentionHeads % keyValueHeads != 0 || hiddenSize % convGroupSize != 0)
            throw new IllegalArgumentException("invalid DFlash2 head or group geometry");
    }

    /// Parses the `dflash2/config` words.
    public static DFlash2Config parse(int[] words) {
        if (words.length < 18 || words[0] != VERSION || words.length != 18 + words[17])
            throw new IllegalArgumentException("unsupported dflash2/config: " + Arrays.toString(words));
        return new DFlash2Config(
                words[1],
                words[2],
                words[3],
                words[4],
                words[5],
                words[6],
                words[7],
                words[8],
                words[9],
                words[10],
                words[11],
                words[12],
                words[13],
                words[14],
                Float.intBitsToFloat(words[15]),
                Float.intBitsToFloat(words[16]),
                Arrays.copyOfRange(words, 18, words.length));
    }

    @Override
    public int[] targetLayers() {
        return this.targetLayers.clone();
    }

    public int taps() {
        return this.targetLayers.length;
    }

    /// Width of one row of tapped target hidden states: every tap's hidden row, in tap order.
    public int tapWidth() {
        return this.targetLayers.length * this.hiddenSize;
    }

    public int queryWidth() {
        return this.attentionHeads * this.headDim;
    }

    public int keyValueWidth() {
        return this.keyValueHeads * this.headDim;
    }

    /// Outputs of a dynamic convolution's kernel projection: the prepare and finish kernels, each `convKernel` taps
    /// per channel group.
    public int convProjectionWidth() {
        return 2 * this.convKernel * (this.hiddenSize / this.convGroupSize);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof DFlash2Config c
                && Arrays.equals(this.targetLayers, c.targetLayers)
                && this.layers == c.layers
                && this.hiddenSize == c.hiddenSize
                && this.intermediateSize == c.intermediateSize
                && this.attentionHeads == c.attentionHeads
                && this.keyValueHeads == c.keyValueHeads
                && this.headDim == c.headDim
                && this.blockSize == c.blockSize
                && this.maskToken == c.maskToken
                && this.convKernel == c.convKernel
                && this.convGroupSize == c.convGroupSize
                && this.selectorTopK == c.selectorTopK
                && this.selectorRank == c.selectorRank
                && this.slidingWindow == c.slidingWindow
                && this.vocabSize == c.vocabSize
                && Float.compare(this.rmsNormEpsilon, c.rmsNormEpsilon) == 0
                && Float.compare(this.ropeTheta, c.ropeTheta) == 0;
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(this.targetLayers) * 31 + Integer.hashCode(this.layers * 7919 + this.blockSize);
    }

    @Override
    public String toString() {
        return "DFlash2Config[layers=" + this.layers + ", block=" + this.blockSize + ", taps="
                + Arrays.toString(this.targetLayers) + ", window=" + this.slidingWindow + "]";
    }
}
