package io.euhedral_execution.inference.core.gpu;

import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.bf16ToFloat;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.floatToBf16;

import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.loader.DenseFfnWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.GdnWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.LayerWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.Weights;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.EnumMap;

/// Independent CPU reference for the loaded compact GDN layer-zero dataflow.
final class FirstLayerCpuReference {

    record Result(EnumMap<ExecutionPlan.Buffer, short[]> buffers, float[] a, float[] b, float[] g, float[] beta) {}

    private FirstLayerCpuReference() {}

    static Result run(Weights weights, ExecutionGpu gpu, int tokenId) {
        LayerWeights layer = weights.layers()[0];
        GdnWeights gdn = (GdnWeights) layer.mixer();
        DenseFfnWeights ffn = (DenseFfnWeights) layer.ffn();
        int hidden = weights.config().hiddenSize();
        int intermediate = weights.config().intermediateSize();
        int keyHeads = weights.config().linearNumKeyHeads();
        int valueHeads = weights.config().linearNumValueHeads();
        int headDim = weights.config().linearKeyHeadDim();
        int valueWidth = valueHeads * weights.config().linearValueHeadDim();
        int queryKeyWidth = 2 * keyHeads * headDim;
        int convolutionWidth = queryKeyWidth + valueWidth;
        float epsilon = (float) weights.config().rmsNormEpsilon();

        EnumMap<ExecutionPlan.Buffer, short[]> buffers = new EnumMap<>(ExecutionPlan.Buffer.class);
        short[] hiddenState = q3Embedding(
                read(gpu, weights.tokenEmbedding()),
                tokenId,
                hidden,
                weights.config().vocabSize());
        buffers.put(ExecutionPlan.Buffer.HIDDEN_STATE, hiddenState);

        short[] inputNorm = rmsNormUnitOffset(hiddenState, read(gpu, layer.inputNorm()), 1, hidden, epsilon);
        buffers.put(ExecutionPlan.Buffer.INPUT_NORMALIZED, inputNorm);

        short[] queryKey = quantizedLinear(inputNorm, read(gpu, gdn.queryKey()), 1, hidden, queryKeyWidth, 4);
        buffers.put(ExecutionPlan.Buffer.QK_PROJECTED, queryKey);
        short[] valueZ = quantizedLinear(inputNorm, read(gpu, gdn.valueZ()), 1, hidden, valueWidth * 2, 5);
        buffers.put(ExecutionPlan.Buffer.VALUE_Z_PROJECTED, valueZ);

        float[] a = bf16Linear(inputNorm, read(gpu, gdn.aProjection()), hidden, valueHeads);
        float[] b = bf16Linear(inputNorm, read(gpu, gdn.bProjection()), hidden, valueHeads);
        float[] aLog = readFp32(gpu, gdn.aLog());
        float[] dtBias = readFp32(gpu, gdn.dtBias());
        float[] g = new float[valueHeads];
        float[] beta = new float[valueHeads];
        for (int head = 0; head < valueHeads; head++) {
            float shifted = a[head] + dtBias[head];
            g[head] = (float) Math.exp((float) (-Math.exp(aLog[head]) * softplus(shifted)));
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
        buffers.put(ExecutionPlan.Buffer.GDN_CONVOLVED, convolved);

        Recurrence recurrence = recurrence(
                convolved,
                g,
                beta,
                keyHeads,
                valueHeads,
                headDim,
                weights.config().linearValueHeadDim(),
                (float) (1.0 / Math.sqrt(headDim)));
        buffers.put(ExecutionPlan.Buffer.GDN_RECURRENT, recurrence.output());

        short[] gdnNormalized = gatedRmsNorm(
                recurrence.output(),
                valueZ,
                read(gpu, gdn.norm()),
                valueHeads,
                weights.config().linearValueHeadDim(),
                epsilon);
        buffers.put(ExecutionPlan.Buffer.GDN_NORMALIZED, gdnNormalized);

        short[] mixerDelta = quantizedLinear(gdnNormalized, read(gpu, gdn.output()), 1, valueWidth, hidden, 3);
        buffers.put(ExecutionPlan.Buffer.MIXER_DELTA, mixerDelta);
        short[] mixerHidden = residualAdd(hiddenState, mixerDelta);
        buffers.put(ExecutionPlan.Buffer.MIXER_HIDDEN, mixerHidden);

        short[] postMixerNormalized =
                rmsNormUnitOffset(mixerHidden, read(gpu, layer.postAttentionNorm()), 1, hidden, epsilon);
        buffers.put(ExecutionPlan.Buffer.POST_MIXER_NORMALIZED, postMixerNormalized);

        short[] gateUp = quantizedLinear(postMixerNormalized, read(gpu, ffn.gateUp()), 1, hidden, intermediate * 2, 3);
        buffers.put(ExecutionPlan.Buffer.GATE_UP, gateUp);
        short[] swiglu = swiGlu(gateUp, intermediate);
        buffers.put(ExecutionPlan.Buffer.SWIGLU, swiglu);
        short[] ffnDelta = quantizedLinear(swiglu, read(gpu, ffn.down()), 1, intermediate, hidden, 3);
        buffers.put(ExecutionPlan.Buffer.FFN_DELTA, ffnDelta);
        buffers.put(ExecutionPlan.Buffer.FINAL_HIDDEN_STATE, residualAdd(mixerHidden, ffnDelta));
        return new Result(buffers, a, b, g, beta);
    }

    static short[] q3Embedding(byte[] weights, int tokenId, int hidden, int vocabulary) {
        if (tokenId < 0 || tokenId >= vocabulary) throw new IllegalArgumentException("token ID");
        int groups = ((hidden + 127) / 128) * 2;
        int scaleOffset = align256(vocabulary * groups * 24);
        short[] output = new short[hidden];
        for (int k = 0; k < hidden; k++) {
            int group = k / 64;
            int bit = (k % 64) * 3;
            int offset = tokenId * groups * 24 + group * 24 + (bit >>> 3);
            int packed = (weights[offset] & 0xff) | ((weights[offset + 1] & 0xff) << 8);
            int code = (packed >>> (bit & 7)) & 7;
            if (code >= 4) code -= 8;
            float scale = fp16(weights, scaleOffset + (tokenId * groups + group) * 2);
            output[k] = floatToBf16(code * scale);
        }
        return output;
    }

    static short[] quantizedLinear(short[] input, byte[] weights, int rows, int width, int outputs, int bits) {
        int groups = bits == 3 ? ((width + 127) / 128) * 2 : width / 64;
        int codePlane = bits == 3 ? align256(outputs * groups * 24) : align256(outputs * groups * 32);
        int highBytes = bits == 5 ? outputs * groups * 8 : 0;
        int scaleOffset = codePlane + align256(highBytes);
        short[] output = new short[rows * outputs];
        for (int row = 0; row < rows; row++) {
            for (int out = 0; out < outputs; out++) {
                float sum = 0.0f;
                for (int k = 0; k < width; k++) {
                    int group = k / 64;
                    int lane = k % 64;
                    int groupOffset = out * groups + group;
                    int code;
                    if (bits == 3) {
                        int bit = lane * 3;
                        int offset = out * groups * 24 + group * 24 + (bit >>> 3);
                        int packed = (weights[offset] & 0xff) | ((weights[offset + 1] & 0xff) << 8);
                        code = (packed >>> (bit & 7)) & 7;
                        if (code >= 4) code -= 8;
                    } else {
                        int packed = weights[groupOffset * 32 + (lane >>> 1)] & 0xff;
                        code = (packed >>> ((lane & 1) * 4)) & 0xf;
                    }
                    if (bits == 5) {
                        int high = (weights[codePlane + groupOffset * 8 + (lane >>> 3)] >>> (lane & 7)) & 1;
                        code |= high << 4;
                        if (code >= 16) code -= 32;
                    } else if (bits == 4) {
                        if (code >= 8) code -= 16;
                    }
                    sum += bf16ToFloat(input[row * width + k])
                            * code
                            * fp16(weights, scaleOffset + (out * groups + group) * 2);
                }
                output[row * outputs + out] = floatToBf16(sum);
            }
        }
        return output;
    }

    static float[] bf16Linear(short[] input, byte[] weights, int width, int outputs) {
        float[] result = new float[outputs];
        for (int out = 0; out < outputs; out++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += bf16ToFloat(input[k]) * bf16(weights, (out * width + k) * 2);
            }
            result[out] = sum;
        }
        return result;
    }

    static short[] rmsNormUnitOffset(short[] input, byte[] weights, int rows, int width, float epsilon) {
        short[] output = new short[input.length];
        for (int row = 0; row < rows; row++) {
            float sum = 0.0f;
            for (int column = 0; column < width; column++) {
                float value = bf16ToFloat(input[row * width + column]);
                sum += value * value;
            }
            float inverse = (float) (1.0 / Math.sqrt(sum / width + epsilon));
            for (int column = 0; column < width; column++) {
                int index = row * width + column;
                output[index] = floatToBf16(bf16ToFloat(input[index]) * inverse * (1.0f + bf16(weights, column * 2)));
            }
        }
        return output;
    }

    static short[] convolution(
            short[] queryKey,
            short[] valueZ,
            byte[] weights,
            int channels,
            int queryKeyWidth,
            int valueWidth,
            int kernel) {
        short[] output = new short[channels];
        for (int channel = 0; channel < channels; channel++) {
            float sum = 0.0f;
            float current = channel < queryKeyWidth
                    ? bf16ToFloat(queryKey[channel])
                    : bf16ToFloat(valueZ[channel - queryKeyWidth]);
            for (int tap = 0; tap < kernel; tap++) {
                float value = tap == kernel - 1 ? current : 0.0f;
                sum += value * bf16(weights, (tap * channels + channel) * 2);
            }
            output[channel] = floatToBf16(silu(sum));
        }
        return output;
    }

    static Recurrence recurrence(
            short[] convolved,
            float[] g,
            float[] beta,
            int keyHeads,
            int valueHeads,
            int keyHeadDim,
            int valueHeadDim,
            float outputScale) {
        int queryKeyWidth = 2 * keyHeads * keyHeadDim;
        int valueWidth = valueHeads * valueHeadDim;
        int convolvedWidth = queryKeyWidth + valueWidth;
        float[] state = new float[valueHeads * valueHeadDim * keyHeadDim];
        short[] output = new short[valueWidth];
        int groups = valueHeads / keyHeads;
        for (int valueHead = 0; valueHead < valueHeads; valueHead++) {
            int keyHead = valueHead / groups;
            int queryBase = keyHead * keyHeadDim;
            int keyBase = keyHeads * keyHeadDim + queryBase;
            int valueBase = queryKeyWidth + valueHead * valueHeadDim;
            float[] query = normalized(convolved, queryBase, keyHeadDim);
            float[] key = normalized(convolved, keyBase, keyHeadDim);
            for (int valueColumn = 0; valueColumn < valueHeadDim; valueColumn++) {
                int stateOffset = (valueHead * valueHeadDim + valueColumn) * keyHeadDim;
                float dot = 0.0f;
                for (int k = 0; k < keyHeadDim; k++) dot += state[stateOffset + k] * key[k];
                float alpha = g[valueHead];
                float delta = beta[valueHead] * (bf16ToFloat(convolved[valueBase + valueColumn]) - alpha * dot);
                float sum = 0.0f;
                for (int k = 0; k < keyHeadDim; k++) {
                    state[stateOffset + k] = alpha * state[stateOffset + k] + delta * key[k];
                    sum += state[stateOffset + k] * query[k];
                }
                output[valueHead * valueHeadDim + valueColumn] = floatToBf16(sum * outputScale);
            }
        }
        return new Recurrence(output, state);
    }

    static short[] gatedRmsNorm(
            short[] recurrent, short[] valueZ, byte[] weights, int heads, int headDim, float epsilon) {
        short[] output = new short[recurrent.length];
        int width = heads * headDim;
        for (int head = 0; head < heads; head++) {
            int base = head * headDim;
            float sum = 0.0f;
            for (int i = 0; i < headDim; i++) {
                float value = bf16ToFloat(recurrent[base + i]);
                sum += value * value;
            }
            float inverse = (float) (1.0 / Math.sqrt(sum / headDim + epsilon));
            int zBase = width + base;
            for (int i = 0; i < headDim; i++) {
                float x = bf16ToFloat(recurrent[base + i]);
                float z = bf16ToFloat(valueZ[zBase + i]);
                output[base + i] = floatToBf16(x * inverse * bf16(weights, i * 2) * silu(z));
            }
        }
        return output;
    }

    static short[] residualAdd(short[] left, short[] right) {
        short[] output = new short[left.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = floatToBf16(bf16ToFloat(left[i]) + bf16ToFloat(right[i]));
        }
        return output;
    }

    static short[] swiGlu(short[] gateUp, int intermediate) {
        short[] output = new short[intermediate];
        for (int i = 0; i < intermediate; i++) {
            output[i] = floatToBf16(silu(bf16ToFloat(gateUp[i])) * bf16ToFloat(gateUp[intermediate + i]));
        }
        return output;
    }

    private static float[] normalized(short[] values, int offset, int width) {
        float sum = 0.0f;
        for (int i = 0; i < width; i++) {
            float value = bf16ToFloat(values[offset + i]);
            sum += value * value;
        }
        float inverse = (float) (1.0 / Math.sqrt(sum + 1.0e-6f));
        float[] output = new float[width];
        for (int i = 0; i < width; i++) output[i] = bf16ToFloat(values[offset + i]) * inverse;
        return output;
    }

    static byte[] read(ExecutionGpu gpu, TensorHandle handle) {
        if (handle.byteSize() > Integer.MAX_VALUE) throw new IllegalArgumentException("reference tensor too large");
        byte[] bytes = new byte[(int) handle.byteSize()];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(handle.byteSize(), 1);
            gpu.copyDeviceToHost(host, handle.deviceAddress(), handle.byteSize());
            host.asByteBuffer().get(bytes);
        }
        return bytes;
    }

    static float[] readFp32(ExecutionGpu gpu, TensorHandle handle) {
        byte[] bytes = read(gpu, handle);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[bytes.length / Float.BYTES];
        for (int i = 0; i < values.length; i++) values[i] = buffer.getFloat();
        return values;
    }

    private static float bf16(byte[] bytes, int offset) {
        int bits = (bytes[offset] & 0xff) | ((bytes[offset + 1] & 0xff) << 8);
        return bf16ToFloat((short) bits);
    }

    private static float fp16(byte[] bytes, int offset) {
        int bits = (bytes[offset] & 0xff) | ((bytes[offset + 1] & 0xff) << 8);
        return Float.float16ToFloat((short) bits);
    }

    private static float silu(float value) {
        return value / (1.0f + (float) Math.exp(-value));
    }

    static float sigmoid(float value) {
        return (float) (1.0 / (1.0 + Math.exp(-value)));
    }

    static float softplus(float value) {
        return (float) (Math.max(value, 0.0) + Math.log1p(Math.exp(-Math.abs(value))));
    }

    private static int align256(int value) {
        return (value + 255) & ~255;
    }

    record Recurrence(short[] output, float[] state) {}
}
