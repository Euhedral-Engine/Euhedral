package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.model.qwen38.speculative.MtpDecoder;
import java.util.Objects;
import java.util.OptionalInt;

public record Qwen38Config(
        int vocabSize,
        int hiddenSize,
        int numHiddenLayers,

        int numAttentionHeads,
        int numKeyValueHeads,
        int attentionHeadDim,

        int intermediateSize,

        int linearNumKeyHeads,
        int linearNumValueHeads,
        int linearKeyHeadDim,
        int linearValueHeadDim,
        int linearConvKernelDim,

        double rmsNormEpsilon,

        double ropeTheta,
        double partialRotaryFactor,
        int maxPositionEmbeddings,

        String hiddenActivation,

        LayerType[] layerTypes,

        int numExperts,
        int numExpertsPerToken,
        int moeIntermediateSize,
        int sharedExpertIntermediateSize,

        boolean tieWordEmbeddings,
        boolean attentionOutputGate,

        int mtpLayerCount,

        /// The MTP draft depth the artifact was built for, when it stores one (absent in artifacts that leave the
        /// choice to the engine's policy for their quantization).
        OptionalInt draftDepth) {

    public Qwen38Config {
        Objects.requireNonNull(draftDepth, "draftDepth");
        if (draftDepth.isPresent() && (draftDepth.getAsInt() < 1 || draftDepth.getAsInt() > MtpDecoder.MAX_DRAFTS))
            throw new IllegalArgumentException("draft depth must be 1 to " + MtpDecoder.MAX_DRAFTS);
    }

    /// A configuration that stores no draft depth.
    public Qwen38Config(
            int vocabSize,
            int hiddenSize,
            int numHiddenLayers,
            int numAttentionHeads,
            int numKeyValueHeads,
            int attentionHeadDim,
            int intermediateSize,
            int linearNumKeyHeads,
            int linearNumValueHeads,
            int linearKeyHeadDim,
            int linearValueHeadDim,
            int linearConvKernelDim,
            double rmsNormEpsilon,
            double ropeTheta,
            double partialRotaryFactor,
            int maxPositionEmbeddings,
            String hiddenActivation,
            LayerType[] layerTypes,
            int numExperts,
            int numExpertsPerToken,
            int moeIntermediateSize,
            int sharedExpertIntermediateSize,
            boolean tieWordEmbeddings,
            boolean attentionOutputGate,
            int mtpLayerCount) {
        this(
                vocabSize,
                hiddenSize,
                numHiddenLayers,
                numAttentionHeads,
                numKeyValueHeads,
                attentionHeadDim,
                intermediateSize,
                linearNumKeyHeads,
                linearNumValueHeads,
                linearKeyHeadDim,
                linearValueHeadDim,
                linearConvKernelDim,
                rmsNormEpsilon,
                ropeTheta,
                partialRotaryFactor,
                maxPositionEmbeddings,
                hiddenActivation,
                layerTypes,
                numExperts,
                numExpertsPerToken,
                moeIntermediateSize,
                sharedExpertIntermediateSize,
                tieWordEmbeddings,
                attentionOutputGate,
                mtpLayerCount,
                OptionalInt.empty());
    }

    /// This configuration storing `depth` as the MTP draft depth.
    public Qwen38Config withDraftDepth(int depth) {
        return new Qwen38Config(
                this.vocabSize,
                this.hiddenSize,
                this.numHiddenLayers,
                this.numAttentionHeads,
                this.numKeyValueHeads,
                this.attentionHeadDim,
                this.intermediateSize,
                this.linearNumKeyHeads,
                this.linearNumValueHeads,
                this.linearKeyHeadDim,
                this.linearValueHeadDim,
                this.linearConvKernelDim,
                this.rmsNormEpsilon,
                this.ropeTheta,
                this.partialRotaryFactor,
                this.maxPositionEmbeddings,
                this.hiddenActivation,
                this.layerTypes,
                this.numExperts,
                this.numExpertsPerToken,
                this.moeIntermediateSize,
                this.sharedExpertIntermediateSize,
                this.tieWordEmbeddings,
                this.attentionOutputGate,
                this.mtpLayerCount,
                OptionalInt.of(depth));
    }
}
