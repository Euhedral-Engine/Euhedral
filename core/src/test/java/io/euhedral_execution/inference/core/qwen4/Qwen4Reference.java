package io.euhedral_execution.inference.core.qwen4;

/// CPU references of the Flash-Next operators, written as upstream's PyTorch computes them: BF16 tensors, every
/// elementwise operation rounded to BF16, reductions accumulated wider (here in double, where the kernels use FP32).
final class Qwen4Reference {

    private Qwen4Reference() {}

    static float bf(short bits) {
        return Float.intBitsToFloat((bits & 0xffff) << 16);
    }

    static short bits(float value) {
        int raw = Float.floatToRawIntBits(value);
        if (Float.isNaN(value)) return (short) ((raw >>> 16) | 0x40);
        raw += 0x7fff + ((raw >>> 16) & 1);
        return (short) (raw >>> 16);
    }

    /// `value` rounded to BF16.
    static float r(double value) {
        return bf(bits((float) value));
    }

    static float sigmoid(float value) {
        return (float) (1.0 / (1.0 + Math.exp(-value)));
    }

    static float silu(float value) {
        return (float) (value / (1.0 + Math.exp(-value)));
    }

    /// Result of a linear: the BF16 output and, per element, the sum of absolute products (for tolerances).
    record Linear(short[] output, double[] magnitude) {}

    static Linear linear(short[] input, short[] weights, int rows, int k, int n) {
        short[] output = new short[rows * n];
        double[] magnitude = new double[rows * n];
        for (int row = 0; row < rows; row++)
            for (int j = 0; j < n; j++) {
                double sum = 0, abs = 0;
                for (int i = 0; i < k; i++) {
                    double product = (double) bf(input[row * k + i]) * bf(weights[j * k + i]);
                    sum += product;
                    abs += Math.abs(product);
                }
                output[row * n + j] = bits((float) sum);
                magnitude[row * n + j] = abs;
            }
        return new Linear(output, magnitude);
    }

    static short[] groupedRmsNorm(short[] input, short[] weight, int rows, int groups, int width, float epsilon) {
        short[] output = new short[rows * groups * width];
        for (int row = 0; row < rows; row++)
            for (int group = 0; group < groups; group++) {
                int base = (row * groups + group) * width;
                double sum = 0;
                for (int i = 0; i < width; i++) sum += (double) bf(input[base + i]) * bf(input[base + i]);
                float inverse = (float) (1.0 / Math.sqrt(sum / width + epsilon));
                for (int i = 0; i < width; i++)
                    output[base + i] = bits(bf(input[base + i]) * inverse * (1.0f + bf(weight[group * width + i])));
            }
        return output;
    }

    static short[] scaledSilu(short[] input, float divisor) {
        short[] output = new short[input.length];
        for (int i = 0; i < input.length; i++) output[i] = bits(silu(r(bf(input[i]) / divisor)));
        return output;
    }

    static short[] hcMix(short[] normed, short[] up, int rows, int streams, int width) {
        short[] mixed = new short[rows * width];
        for (int row = 0; row < rows; row++)
            for (int d = 0; d < width; d++) {
                double sum = 0;
                for (int s = 0; s < streams; s++) {
                    int at = (row * streams + s) * width + d;
                    sum += r(r(sigmoid(bf(up[at]))) * bf(normed[at]));
                }
                mixed[row * width + d] = bits((float) (sum / streams));
            }
        return mixed;
    }

    static short[] hcInject(short[] hyper, short[] block, short[] raw, int rows, int streams, int width) {
        short[] output = new short[rows * streams * width];
        for (int row = 0; row < rows; row++)
            for (int s = 0; s < streams; s++) {
                float weight = 2.0f * r(sigmoid(r(bf(raw[row * streams + s]) / streams)));
                for (int d = 0; d < width; d++) {
                    int at = (row * streams + s) * width + d;
                    output[at] = bits(bf(hyper[at]) + r(bf(block[row * width + d]) * weight));
                }
            }
        return output;
    }

    static float e4m3(int code) {
        int exponent = (code >> 3) & 15, mantissa = code & 7;
        if (exponent == 0) return mantissa / 8.0f * (float) Math.pow(2, -6);
        return (1.0f + mantissa / 8.0f) * (float) Math.pow(2, exponent - 7);
    }

    static final float[] E2M1 = {0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f};

    /// One staged record (codes, scales, padding, little-endian global scale) expanded to BF16 values.
    static short[] expandRecord(byte[] record, int offset, int recordBytes, int width) {
        float global = Float.intBitsToFloat((record[offset + recordBytes - 4] & 0xff)
                | (record[offset + recordBytes - 3] & 0xff) << 8
                | (record[offset + recordBytes - 2] & 0xff) << 16
                | (record[offset + recordBytes - 1] & 0xff) << 24);
        short[] values = new short[width];
        for (int i = 0; i < width; i++) {
            int byteValue = record[offset + (i >> 1)] & 0xff;
            int nibble = (i & 1) != 0 ? byteValue >> 4 : byteValue & 15;
            float magnitude = E2M1[nibble & 7];
            float value = (nibble & 8) != 0 ? -magnitude : magnitude;
            float block = e4m3(record[offset + (width >> 1) + (i >> 4)] & 0xff) * global;
            values[i] = bits(value * block);
        }
        return values;
    }

    static short[] pleGate(short[] key, short[] query, short[] value, int rows, int streams, int width) {
        short[] output = new short[rows * streams * width];
        for (int row = 0; row < rows; row++)
            for (int s = 0; s < streams; s++) {
                int base = (row * streams + s) * width;
                double sum = 0;
                for (int d = 0; d < width; d++) sum += r(bf(key[base + d]) * bf(query[base + d]));
                float gate = r(r(sum) * (1.0f / (float) Math.sqrt(width)));
                float magnitude = r(Math.sqrt(r(Math.max(Math.abs(gate), 1e-6f))));
                gate = magnitude * Math.signum(gate);
                float scale = r(sigmoid(gate));
                for (int d = 0; d < width; d++) output[base + d] = bits(scale * bf(value[row * width + d]));
            }
        return output;
    }

    /// The PLE convolution over `rows` rows given the `history` rows before them; returns [output].
    static short[] pleConv(
            short[] normed,
            short[] gated,
            short[] history,
            short[] weights,
            int rows,
            int channels,
            int taps,
            int dilation) {
        int historyRows = (taps - 1) * dilation;
        short[] output = new short[rows * channels];
        for (int t = 0; t < rows; t++)
            for (int c = 0; c < channels; c++) {
                double sum = 0;
                for (int i = 0; i < taps; i++) {
                    int source = t - historyRows + dilation * i;
                    float x = source >= 0
                            ? bf(normed[source * channels + c])
                            : bf(history[(historyRows + source) * channels + c]);
                    sum += (double) bf(weights[c * taps + i]) * x;
                }
                float convolved = r(silu(r(sum)));
                output[t * channels + c] = bits(bf(gated[t * channels + c]) + convolved);
            }
        return output;
    }

    /// The last `history` rows of (history ++ x).
    static short[] convHistory(short[] x, short[] history, int rows, int channels, int historyRows) {
        short[] next = new short[historyRows * channels];
        for (int i = 0; i < historyRows; i++)
            for (int c = 0; c < channels; c++) {
                int combined = rows + i;
                next[i * channels + c] = combined >= historyRows
                        ? x[(combined - historyRows) * channels + c]
                        : history[combined * channels + c];
            }
        return next;
    }

    /// The GDN convolution: silu(bf16(depthwise causal conv)) over (history ++ x).
    static short[] gdnConv(short[] x, short[] history, short[] weights, int rows, int channels, int taps) {
        int historyRows = taps - 1;
        short[] output = new short[rows * channels];
        for (int t = 0; t < rows; t++)
            for (int c = 0; c < channels; c++) {
                double sum = 0;
                for (int i = 0; i < taps; i++) {
                    int source = t - historyRows + i;
                    float value = source >= 0
                            ? bf(x[source * channels + c])
                            : bf(history[(historyRows + source) * channels + c]);
                    sum += (double) bf(weights[c * taps + i]) * value;
                }
                output[t * channels + c] = bits(silu(r(sum)));
            }
        return output;
    }

    /// alpha = exp(g) and beta = bf16(sigmoid(b)) per (row, head).
    record Control(float[] alpha, float[] beta) {}

    static Control gdnControl(short[] a, short[] b, short[] aLog, short[] dtBias, int rows, int heads) {
        float[] alpha = new float[rows * heads], beta = new float[rows * heads];
        for (int i = 0; i < rows * heads; i++) {
            int head = i % heads;
            double shifted = (double) bf(a[i]) + bf(dtBias[head]);
            double softplus = shifted > 20 ? shifted : Math.log1p(Math.exp(shifted));
            alpha[i] = (float) Math.exp(-Math.exp(bf(aLog[head])) * softplus);
            beta[i] = r(sigmoid(bf(b[i])));
        }
        return new Control(alpha, beta);
    }

    /// Qwen4ExpTextRMSNormGated with the three BF16 roundings.
    static short[] gdnGatedNorm(
            short[] core, short[] z, short[] weight, int rows, int heads, int headDim, float epsilon, boolean sigmoid) {
        short[] output = new short[rows * heads * headDim];
        for (int vector = 0; vector < rows * heads; vector++) {
            int base = vector * headDim;
            double sum = 0;
            for (int i = 0; i < headDim; i++) sum += (double) bf(core[base + i]) * bf(core[base + i]);
            float inverse = (float) (1.0 / Math.sqrt(sum / headDim + epsilon));
            for (int i = 0; i < headDim; i++) {
                float normalized = r(bf(core[base + i]) * inverse);
                float weighted = r(bf(weight[i]) * normalized);
                float gate = bf(z[base + i]);
                output[base + i] = bits(weighted * (sigmoid ? sigmoid(gate) : silu(gate)));
            }
        }
        return output;
    }
}
