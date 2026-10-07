package io.euhedral_execution.inference.core.gpu;

import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.bf16ToFloat;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.floatToBf16;
import static io.euhedral_execution.inference.core.gpu.FirstLayerCpuReference.bf16Linear;
import static io.euhedral_execution.inference.core.gpu.FirstLayerCpuReference.convolution;
import static io.euhedral_execution.inference.core.gpu.FirstLayerCpuReference.gatedRmsNorm;
import static io.euhedral_execution.inference.core.gpu.FirstLayerCpuReference.quantizedLinear;
import static io.euhedral_execution.inference.core.gpu.FirstLayerCpuReference.read;
import static io.euhedral_execution.inference.core.gpu.FirstLayerCpuReference.readFp32;
import static io.euhedral_execution.inference.core.gpu.FirstLayerCpuReference.recurrence;
import static io.euhedral_execution.inference.core.gpu.FirstLayerCpuReference.residualAdd;
import static io.euhedral_execution.inference.core.gpu.FirstLayerCpuReference.rmsNormUnitOffset;
import static io.euhedral_execution.inference.core.gpu.FirstLayerCpuReference.sigmoid;
import static io.euhedral_execution.inference.core.gpu.FirstLayerCpuReference.softplus;
import static io.euhedral_execution.inference.core.gpu.FirstLayerCpuReference.swiGlu;

import io.euhedral_execution.inference.core.model.qwen38.loader.AttentionWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.DenseFfnWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.GdnWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.LayerWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.Weights;
import java.util.LinkedHashMap;
import java.util.Map;

/// Independent scalar CPU reference for a single-token compact Qwen text-model forward pass.
final class FullModelCpuReference {
    record Result(Map<Integer, short[]> layerOutputs, short[] finalNormalized, short[] logits) {}

    private FullModelCpuReference() {}

    static Result run(Weights weights, ExecutionGpu gpu, int tokenId) {
        int hidden = weights.config().hiddenSize();
        int intermediate = weights.config().intermediateSize();
        float epsilon = (float) weights.config().rmsNormEpsilon();
        short[] hiddenState = FirstLayerCpuReference.q3Embedding(
                read(gpu, weights.tokenEmbedding()),
                tokenId,
                hidden,
                weights.config().vocabSize());
        Map<Integer, short[]> layerOutputs = new LinkedHashMap<>();

        for (int layerIndex = 0; layerIndex < weights.layers().length; layerIndex++) {
            LayerWeights layer = weights.layers()[layerIndex];
            short[] inputNorm = rmsNormUnitOffset(hiddenState, read(gpu, layer.inputNorm()), 1, hidden, epsilon);
            short[] mixerDelta;
            if (layer.mixer() instanceof GdnWeights gdn) {
                int keyHeads = weights.config().linearNumKeyHeads();
                int valueHeads = weights.config().linearNumValueHeads();
                int keyHeadDim = weights.config().linearKeyHeadDim();
                int valueHeadDim = weights.config().linearValueHeadDim();
                int valueWidth = valueHeads * valueHeadDim;
                int queryKeyWidth = 2 * keyHeads * keyHeadDim;
                int convolutionWidth = queryKeyWidth + valueWidth;
                short[] queryKey = quantizedLinear(inputNorm, read(gpu, gdn.queryKey()), 1, hidden, queryKeyWidth, 4);
                short[] valueZ = quantizedLinear(inputNorm, read(gpu, gdn.valueZ()), 1, hidden, valueWidth * 2, 5);
                float[] a = bf16Linear(inputNorm, read(gpu, gdn.aProjection()), hidden, valueHeads);
                float[] b = bf16Linear(inputNorm, read(gpu, gdn.bProjection()), hidden, valueHeads);
                float[] aLog = readFp32(gpu, gdn.aLog());
                float[] dtBias = readFp32(gpu, gdn.dtBias());
                float[] g = new float[valueHeads];
                float[] beta = new float[valueHeads];
                for (int head = 0; head < valueHeads; head++) {
                    g[head] = (float) Math.exp((float) (-Math.exp(aLog[head]) * softplus(a[head] + dtBias[head])));
                    beta[head] = sigmoid(b[head]);
                }
                short[] convolved = convolution(
                        queryKey,
                        valueZ,
                        read(gpu, gdn.convolution()),
                        convolutionWidth,
                        queryKeyWidth,
                        valueWidth,
                        weights.config().linearConvKernelDim());
                FirstLayerCpuReference.Recurrence recurrent =
                        recurrence(convolved, g, beta, keyHeads, valueHeads, keyHeadDim, valueHeadDim, (float)
                                (1.0 / Math.sqrt(keyHeadDim)));
                short[] normalized = gatedRmsNorm(
                        recurrent.output(), valueZ, read(gpu, gdn.norm()), valueHeads, valueHeadDim, epsilon);
                mixerDelta = quantizedLinear(normalized, read(gpu, gdn.output()), 1, valueWidth, hidden, 3);
            } else if (layer.mixer() instanceof AttentionWeights attention) {
                int queryHeads = weights.config().numAttentionHeads();
                int keyValueHeads = weights.config().numKeyValueHeads();
                int headDim = weights.config().attentionHeadDim();
                int queryWidth = queryHeads * headDim;
                int valueWidth = keyValueHeads * headDim;
                short[] gateValue = quantizedLinear(
                        inputNorm, read(gpu, attention.gateValue()), 1, hidden, queryWidth + valueWidth, 5);
                // With one causal key, softmax is exactly one; Q/K and RoPE therefore cannot affect this value.
                short[] context = singleTokenAttentionContext(gateValue, queryHeads, keyValueHeads, headDim);
                mixerDelta = quantizedLinear(context, read(gpu, attention.output()), 1, queryWidth, hidden, 3);
            } else {
                throw new IllegalArgumentException("unsupported mixer in compact CPU reference: " + layer.mixer());
            }

            short[] mixerHidden = residualAdd(hiddenState, mixerDelta);
            short[] postNorm = rmsNormUnitOffset(mixerHidden, read(gpu, layer.postAttentionNorm()), 1, hidden, epsilon);
            DenseFfnWeights ffn = (DenseFfnWeights) layer.ffn();
            short[] gateUp = quantizedLinear(postNorm, read(gpu, ffn.gateUp()), 1, hidden, intermediate * 2, 3);
            short[] swiglu = swiGlu(gateUp, intermediate);
            short[] ffnDelta = quantizedLinear(swiglu, read(gpu, ffn.down()), 1, intermediate, hidden, 3);
            hiddenState = residualAdd(mixerHidden, ffnDelta);
            layerOutputs.put(layerIndex, hiddenState);
        }

        short[] finalNormalized = rmsNormUnitOffset(hiddenState, read(gpu, weights.finalNorm()), 1, hidden, epsilon);
        short[] logits = quantizedLinear(
                finalNormalized,
                read(gpu, weights.lmHead()),
                1,
                hidden,
                weights.config().vocabSize(),
                3);
        return new Result(Map.copyOf(layerOutputs), finalNormalized, logits);
    }

    static short[] singleTokenAttentionContext(short[] gateValue, int queryHeads, int keyValueHeads, int headDim) {
        if (gateValue == null
                || queryHeads <= 0
                || keyValueHeads <= 0
                || queryHeads % keyValueHeads != 0
                || headDim != 256
                || gateValue.length != Math.multiplyExact(queryHeads + keyValueHeads, headDim)) {
            throw new IllegalArgumentException("attention gate/value dimensions are invalid");
        }
        short[] output = new short[queryHeads * headDim];
        int valuesOffset = queryHeads * headDim;
        int groupsPerKeyValueHead = queryHeads / keyValueHeads;
        for (int queryHead = 0; queryHead < queryHeads; queryHead++) {
            int valueHead = queryHead / groupsPerKeyValueHead;
            double[] represented = Nvfp4KvReference.represented(gateValue, valuesOffset + valueHead * headDim);
            for (int lane = 0; lane < headDim; lane++) {
                int outputIndex = queryHead * headDim + lane;
                float gate = bf16ToFloat(gateValue[outputIndex]);
                float value = (float) represented[lane];
                float sigmoid = 1.0f / (1.0f + (float) Math.exp(-gate));
                output[outputIndex] = floatToBf16(value * sigmoid);
            }
        }
        return output;
    }
}
