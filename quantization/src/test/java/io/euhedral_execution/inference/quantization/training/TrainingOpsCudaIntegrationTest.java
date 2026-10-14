package io.euhedral_execution.inference.quantization.training;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.quantization.gpu.Device;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/// Every training kernel against a float64 reference at small dimensions: the forward pass directly, the backward
/// pass against central differences of the reference under a random projection of the output (L = sum r * y, so
/// dy = r).
class TrainingOpsCudaIntegrationTest {

    private static Device device;
    private static Ops ops;
    private static final List<Long> allocations = new ArrayList<>();
    private final Random random = new Random(11);

    @BeforeAll
    static void start() {
        device = new Device(Path.of(System.getProperty("euhedral.cuda.library")));
        ops = new Ops(device.cuda, device.stream, device.cublas);
    }

    @AfterAll
    static void stop() {
        for (long address : allocations) device.gpu.free(address);
        device.close();
    }

    // ---------------------------------------------------------------- helpers

    private double[] random(int count, double scale) {
        double[] values = new double[count];
        for (int i = 0; i < count; i++) values[i] = (float) (random.nextGaussian() * scale);
        return values;
    }

    private static long upload(double[] values) {
        long address = device.gpu.allocate(4L * Math.max(1, values.length));
        allocations.add(address);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(4L * Math.max(1, values.length), 4);
            for (int i = 0; i < values.length; i++) host.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) values[i]);
            device.gpu.copyHostToDevice(address, host, host.byteSize());
        }
        return address;
    }

    private static long zeros(int count) {
        return upload(new double[count]);
    }

    private static double[] download(long address, int count) {
        device.synchronize();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(4L * count, 4);
            device.gpu.copyDeviceToHost(host, address, host.byteSize());
            double[] values = new double[count];
            for (int i = 0; i < count; i++) values[i] = host.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            return values;
        }
    }

    private static void assertClose(double[] expected, double[] actual, double tolerance, String what) {
        double scale = 0;
        for (double value : expected) scale = Math.max(scale, Math.abs(value));
        double worst = 0;
        int at = -1;
        for (int i = 0; i < expected.length; i++) {
            double error = Math.abs(expected[i] - actual[i]) / Math.max(scale, 1e-6);
            if (error > worst) {
                worst = error;
                at = i;
            }
        }
        System.out.printf("%s: worst relative error %.2e (at %d)%n", what, worst, at);
        assertTrue(worst <= tolerance, what + ": relative error " + worst + " at " + at);
    }

    /// d(sum r * f(x)) / dx by central differences.
    private static double[] numericGradient(Function<double[], double[]> f, double[] x, double[] r) {
        double[] gradient = new double[x.length];
        double h = 1e-5;
        for (int i = 0; i < x.length; i++) {
            double saved = x[i];
            x[i] = saved + h;
            double up = dot(r, f.apply(x));
            x[i] = saved - h;
            double down = dot(r, f.apply(x));
            x[i] = saved;
            gradient[i] = (up - down) / (2 * h);
        }
        return gradient;
    }

    private static double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    private static double silu(double x) {
        return x / (1 + Math.exp(-x));
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // ---------------------------------------------------------------- tests

    @Test
    void rmsNorm() {
        int rows = 5, width = 96;
        for (float offset : new float[] {1, 0}) {
            double[] x = random(rows * width, 1), w = random(width, 0.3), r = random(rows * width, 1);
            Function<double[], double[]> reference = input -> {
                double[] y = new double[rows * width];
                for (int row = 0; row < rows; row++) {
                    double sum = 0;
                    for (int i = 0; i < width; i++) sum += input[row * width + i] * input[row * width + i];
                    double inv = 1 / Math.sqrt(sum / width + 1e-6);
                    for (int i = 0; i < width; i++) y[row * width + i] = input[row * width + i] * inv * (offset + w[i]);
                }
                return y;
            };
            long dx = upload(x), dw = upload(w), y = zeros(rows * width), inv = zeros(rows);
            ops.rmsNormForward(dx, dw, y, inv, rows, width, 1e-6f, offset);
            assertClose(reference.apply(x), download(y, rows * width), 1e-5, "rms norm forward " + offset);
            long gx = zeros(rows * width), gw = zeros(width);
            ops.rmsNormBackward(dx, dw, inv, upload(r), gx, gw, rows, width, offset);
            assertClose(numericGradient(reference, x, r), download(gx, rows * width), 1e-3, "rms norm dx " + offset);
            double[] wCopy = w.clone();
            Function<double[], double[]> byWeight = weights -> {
                System.arraycopy(weights, 0, w, 0, width);
                return reference.apply(x);
            };
            double[] expected = numericGradient(byWeight, wCopy, r);
            System.arraycopy(wCopy, 0, w, 0, width);
            assertClose(expected, download(gw, width), 1e-3, "rms norm dw " + offset);
        }
    }

    @Test
    void swiglu() {
        int rows = 4, width = 40;
        double[] gu = random(rows * 2 * width, 1), r = random(rows * width, 1);
        Function<double[], double[]> reference = input -> {
            double[] h = new double[rows * width];
            for (int row = 0; row < rows; row++)
                for (int j = 0; j < width; j++)
                    h[row * width + j] = silu(input[row * 2 * width + j]) * input[row * 2 * width + width + j];
            return h;
        };
        long d = upload(gu), h = zeros(rows * width), g = zeros(rows * 2 * width);
        ops.swigluForward(d, h, rows, width);
        assertClose(reference.apply(gu), download(h, rows * width), 1e-5, "swiglu forward");
        ops.swigluBackward(d, upload(r), g, rows, width);
        assertClose(numericGradient(reference, gu, r), download(g, rows * 2 * width), 1e-3, "swiglu backward");
    }

    @Test
    void groupedWeights() {
        int n = 6, k = 128, group = 64;
        double[] codes = new double[n * k], scales = random(n * k / group, 0.1), r = random(n * k, 1);
        byte[] bytes = new byte[n * k];
        for (int i = 0; i < n * k; i++) {
            bytes[i] = (byte) (random.nextInt(8) - 4);
            codes[i] = bytes[i];
        }
        long codesDevice = device.gpu.allocate(n * k);
        allocations.add(codesDevice);
        io.euhedral_execution.inference.quantization.gpu.Upload.bytes(device.gpu, codesDevice, bytes);
        Function<double[], double[]> reference = s -> {
            double[] w = new double[n * k];
            for (int i = 0; i < n * k; i++) w[i] = codes[i] * s[(i / k) * (k / group) + (i % k) / group];
            return w;
        };
        long s = upload(scales), w = zeros(n * k), gs = zeros(scales.length);
        ops.dequantize(codesDevice, s, w, n, k, group);
        assertClose(reference.apply(scales), download(w, n * k), 1e-6, "dequantize");
        ops.scaleGradient(codesDevice, upload(r), gs, n, k, group);
        assertClose(numericGradient(reference, scales, r), download(gs, scales.length), 1e-4, "scale gradient");
    }

    @Test
    void gdnControl() {
        int rows = 7, heads = 6;
        double[] a = random(rows * heads, 1), b = random(rows * heads, 1), aLog = random(heads, 0.5);
        double[] dt = random(heads, 0.5), rAlpha = random(rows * heads, 1), rBeta = random(rows * heads, 1);
        Function<double[], double[]> reference = ab -> {
            double[] out = new double[2 * rows * heads];
            for (int i = 0; i < rows * heads; i++) {
                double shifted = ab[i] + dt[i % heads];
                double softplus = Math.log1p(Math.exp(shifted));
                out[i] = Math.exp(-Math.exp(aLog[i % heads]) * softplus);
                out[rows * heads + i] = sigmoid(ab[rows * heads + i]);
            }
            return out;
        };
        double[] ab = new double[2 * rows * heads], r = new double[2 * rows * heads];
        System.arraycopy(a, 0, ab, 0, a.length);
        System.arraycopy(b, 0, ab, a.length, b.length);
        System.arraycopy(rAlpha, 0, r, 0, a.length);
        System.arraycopy(rBeta, 0, r, a.length, b.length);
        long da = upload(a), db = upload(b), alog = upload(aLog), ddt = upload(dt);
        long alpha = zeros(rows * heads), beta = zeros(rows * heads);
        ops.gdnControlForward(da, db, alog, ddt, alpha, beta, rows, heads);
        double[] expected = reference.apply(ab), actual = new double[2 * rows * heads];
        System.arraycopy(download(alpha, rows * heads), 0, actual, 0, rows * heads);
        System.arraycopy(download(beta, rows * heads), 0, actual, rows * heads, rows * heads);
        assertClose(expected, actual, 1e-5, "gdn control forward");
        long ga = zeros(rows * heads), gb = zeros(rows * heads);
        ops.gdnControlBackward(da, alog, ddt, alpha, beta, upload(rAlpha), upload(rBeta), ga, gb, rows, heads);
        double[] numeric = numericGradient(reference, ab, r), gradient = new double[2 * rows * heads];
        System.arraycopy(download(ga, rows * heads), 0, gradient, 0, rows * heads);
        System.arraycopy(download(gb, rows * heads), 0, gradient, rows * heads, rows * heads);
        assertClose(numeric, gradient, 1e-3, "gdn control backward");
    }

    @Test
    void gdnConvolution() {
        int rows = 9, qkWidth = 8, valueWidth = 6, kernel = 4, width = qkWidth + valueWidth;
        double[] qk = random(rows * qkWidth, 1), vz = random(rows * 2 * valueWidth, 1), w = random(kernel * width, 0.5);
        double[] r = random(rows * width, 1);
        double[] joint = new double[qk.length + vz.length];
        System.arraycopy(qk, 0, joint, 0, qk.length);
        System.arraycopy(vz, 0, joint, qk.length, vz.length);
        Function<double[], double[]> reference = input -> {
            double[] out = new double[rows * width];
            for (int t = 0; t < rows; t++)
                for (int c = 0; c < width; c++) {
                    double sum = 0;
                    for (int tap = 0; tap < kernel; tap++) {
                        int s = t - kernel + 1 + tap;
                        if (s < 0) continue;
                        double value = c < qkWidth
                                ? input[s * qkWidth + c]
                                : input[qk.length + s * 2 * valueWidth + c - qkWidth];
                        sum += w[tap * width + c] * value;
                    }
                    out[t * width + c] = silu(sum);
                }
            return out;
        };
        long dqk = upload(qk), dvz = upload(vz), dw = upload(w), pre = zeros(rows * width), out = zeros(rows * width);
        ops.gdnConvolutionForward(dqk, dvz, dw, pre, out, rows, qkWidth, valueWidth, kernel);
        assertClose(reference.apply(joint), download(out, rows * width), 1e-5, "gdn convolution forward");
        long gqk = zeros(qk.length), gvz = zeros(vz.length);
        ops.gdnConvolutionBackward(pre, upload(r), dw, gqk, gvz, rows, qkWidth, valueWidth, kernel);
        double[] gradient = new double[joint.length];
        System.arraycopy(download(gqk, qk.length), 0, gradient, 0, qk.length);
        System.arraycopy(download(gvz, vz.length), 0, gradient, qk.length, vz.length);
        double[] numeric = numericGradient(reference, joint, r);
        // The z half of valueZ is not an input of the convolution.
        for (int t = 0; t < rows; t++)
            for (int j = 0; j < valueWidth; j++) numeric[qk.length + t * 2 * valueWidth + valueWidth + j] = 0;
        assertClose(numeric, gradient, 1e-3, "gdn convolution backward");
    }

    @Test
    void gdnRecurrence() {
        int rows = 37, keyHeads = 2, valueHeads = 4, dk = 32, dv = 24, segment = 8;
        int qkWidth = 2 * keyHeads * dk, width = qkWidth + valueHeads * dv;
        double[] convolved = random(rows * width, 1), r = random(rows * valueHeads * dv, 1);
        double[] alphaBeta = new double[2 * rows * valueHeads];
        for (int i = 0; i < rows * valueHeads; i++) {
            alphaBeta[i] = 0.5 + 0.49 * random.nextDouble();
            alphaBeta[rows * valueHeads + i] = random.nextDouble();
        }
        double[] joint = new double[convolved.length + alphaBeta.length];
        System.arraycopy(convolved, 0, joint, 0, convolved.length);
        System.arraycopy(alphaBeta, 0, joint, convolved.length, alphaBeta.length);
        int n = rows * valueHeads;
        Function<double[], double[]> reference = input -> {
            double[] out = new double[rows * valueHeads * dv];
            for (int h = 0; h < valueHeads; h++) {
                int kh = h / (valueHeads / keyHeads);
                double[][] state = new double[dv][dk];
                for (int t = 0; t < rows; t++) {
                    int row = t * width;
                    double[] q = new double[dk], k = new double[dk];
                    double nq = 0, nk = 0;
                    for (int j = 0; j < dk; j++) {
                        q[j] = input[row + kh * dk + j];
                        k[j] = input[row + keyHeads * dk + kh * dk + j];
                        nq += q[j] * q[j];
                        nk += k[j] * k[j];
                    }
                    nq = Math.sqrt(nq + 1e-6);
                    nk = Math.sqrt(nk + 1e-6);
                    for (int j = 0; j < dk; j++) {
                        q[j] /= nq;
                        k[j] /= nk;
                    }
                    double a = input[convolved.length + t * valueHeads + h];
                    double b = input[convolved.length + n + t * valueHeads + h];
                    for (int c = 0; c < dv; c++) {
                        double u = 0;
                        for (int j = 0; j < dk; j++) u += state[c][j] * k[j];
                        double delta = b * (input[row + qkWidth + h * dv + c] - a * u);
                        for (int j = 0; j < dk; j++) state[c][j] = a * state[c][j] + delta * k[j];
                        double o = 0;
                        for (int j = 0; j < dk; j++) o += state[c][j] * q[j];
                        out[t * valueHeads * dv + h * dv + c] = o / Math.sqrt(dk);
                    }
                }
            }
            return out;
        };
        long dConv = upload(convolved), alpha = upload(java.util.Arrays.copyOfRange(alphaBeta, 0, n));
        long beta = upload(java.util.Arrays.copyOfRange(alphaBeta, n, 2 * n));
        long out = zeros(rows * valueHeads * dv);
        long checkpoints = zeros((int) Ops.checkpointFloats(rows, valueHeads, dk, dv, segment));
        ops.gdnRecurrenceForward(dConv, alpha, beta, out, checkpoints, rows, keyHeads, valueHeads, dk, dv, segment);
        assertClose(reference.apply(joint), download(out, rows * valueHeads * dv), 1e-4, "gdn recurrence forward");
        long scratch = zeros((int) Ops.recurrenceScratchFloats(rows, valueHeads, dk, dv, segment));
        long gConv = zeros(convolved.length), gAlpha = zeros(n), gBeta = zeros(n);
        ops.gdnRecurrenceBackward(
                dConv,
                alpha,
                beta,
                upload(r),
                checkpoints,
                scratch,
                gConv,
                gAlpha,
                gBeta,
                rows,
                keyHeads,
                valueHeads,
                dk,
                dv,
                segment);
        double[] gradient = new double[joint.length];
        System.arraycopy(download(gConv, convolved.length), 0, gradient, 0, convolved.length);
        System.arraycopy(download(gAlpha, n), 0, gradient, convolved.length, n);
        System.arraycopy(download(gBeta, n), 0, gradient, convolved.length + n, n);
        assertClose(numericGradient(reference, joint, r), gradient, 2e-3, "gdn recurrence backward");
    }

    @Test
    void gdnGatedNorm() {
        int rows = 3, heads = 4, dv = 32, width = heads * dv;
        double[] o = random(rows * width, 1), vz = random(rows * 2 * width, 1), w = random(dv, 0.5);
        double[] r = random(rows * width, 1);
        double[] joint = new double[o.length + vz.length];
        System.arraycopy(o, 0, joint, 0, o.length);
        System.arraycopy(vz, 0, joint, o.length, vz.length);
        Function<double[], double[]> reference = input -> {
            double[] y = new double[rows * width];
            for (int row = 0; row < rows; row++)
                for (int h = 0; h < heads; h++) {
                    double sum = 0;
                    for (int i = 0; i < dv; i++) sum += Math.pow(input[row * width + h * dv + i], 2);
                    double inv = 1 / Math.sqrt(sum / dv + 1e-6);
                    for (int i = 0; i < dv; i++)
                        y[row * width + h * dv + i] = input[row * width + h * dv + i]
                                * inv
                                * w[i]
                                * silu(input[o.length + row * 2 * width + width + h * dv + i]);
                }
            return y;
        };
        long dO = upload(o), dVz = upload(vz), dw = upload(w), y = zeros(rows * width), inv = zeros(rows * heads);
        ops.gdnGatedNormForward(dO, dVz, dw, y, inv, rows, heads, dv, 1e-6f);
        assertClose(reference.apply(joint), download(y, rows * width), 1e-5, "gated norm forward");
        long gO = zeros(o.length), gVz = zeros(vz.length), gw = zeros(dv);
        ops.gdnGatedNormBackward(dO, dVz, dw, inv, upload(r), gO, gVz, gw, rows, heads, dv);
        double[] gradient = new double[joint.length];
        System.arraycopy(download(gO, o.length), 0, gradient, 0, o.length);
        System.arraycopy(download(gVz, vz.length), 0, gradient, o.length, vz.length);
        double[] numeric = numericGradient(reference, joint, r);
        assertClose(numeric, gradient, 1e-3, "gated norm backward");
    }

    @Test
    void qkNormRope() {
        int rows = 3, qHeads = 4, kvHeads = 2, d = 64, rotary = 16, heads = qHeads + kvHeads;
        long position0 = 1000;
        double theta = 1e7;
        double[] qk = random(rows * heads * d, 1), qn = random(d, 0.3), kn = random(d, 0.3);
        double[] r = random(rows * heads * d, 1);
        Function<double[], double[]> reference = input -> {
            double[] out = new double[rows * heads * d];
            for (int row = 0; row < rows; row++)
                for (int head = 0; head < heads; head++) {
                    int base = (row * heads + head) * d;
                    double sum = 0;
                    for (int i = 0; i < d; i++) sum += input[base + i] * input[base + i];
                    double inv = 1 / Math.sqrt(sum / d + 1e-6);
                    double[] n = new double[d];
                    for (int i = 0; i < d; i++) n[i] = input[base + i] * inv * (1 + (head < qHeads ? qn : kn)[i]);
                    for (int i = 0; i < d; i++) {
                        double value = n[i];
                        if (i < rotary) {
                            int half = rotary / 2, pair = i < half ? i + half : i - half, f = i < half ? i : pair;
                            double angle = (position0 + row) * Math.pow(theta, -(2.0 * f) / rotary);
                            value = i < half
                                    ? n[i] * Math.cos(angle) - n[pair] * Math.sin(angle)
                                    : n[i] * Math.cos(angle) + n[pair] * Math.sin(angle);
                        }
                        out[base + i] = value;
                    }
                }
            return out;
        };
        long dqk = upload(qk), dqn = upload(qn), dkn = upload(kn);
        long out = zeros(qk.length), inv = zeros(rows * heads);
        ops.qkNormRopeForward(dqk, dqn, dkn, out, inv, rows, qHeads, kvHeads, d, rotary, position0, theta, 1e-6f);
        assertClose(reference.apply(qk), download(out, qk.length), 1e-5, "qk norm rope forward");
        long g = zeros(qk.length), gqn = zeros(d), gkn = zeros(d);
        ops.qkNormRopeBackward(
                dqk, dqn, dkn, inv, upload(r), g, gqn, gkn, rows, qHeads, kvHeads, d, rotary, position0, theta);
        assertClose(numericGradient(reference, qk, r), download(g, qk.length), 1e-3, "qk norm rope backward");
    }

    @Test
    void causalSoftmax() {
        int heads = 3, rows = 11;
        float scale = 0.3f;
        double[] scores = random(heads * rows * rows, 2), r = random(heads * rows * rows, 1);
        Function<double[], double[]> reference = input -> {
            double[] p = new double[input.length];
            for (int b = 0; b < heads * rows; b++) {
                int row = b % rows;
                double max = Double.NEGATIVE_INFINITY, sum = 0;
                for (int j = 0; j <= row; j++) max = Math.max(max, input[b * rows + j] * scale);
                for (int j = 0; j <= row; j++) sum += Math.exp(input[b * rows + j] * scale - max);
                for (int j = 0; j <= row; j++) p[b * rows + j] = Math.exp(input[b * rows + j] * scale - max) / sum;
            }
            return p;
        };
        long s = upload(scores);
        ops.causalSoftmax(s, heads, rows, scale);
        double[] p = download(s, scores.length);
        assertClose(reference.apply(scores), p, 1e-5, "causal softmax");
        long g = upload(r);
        ops.causalSoftmaxBackward(s, g, heads, rows, scale);
        double[] numeric = numericGradient(reference, scores, r);
        assertClose(numeric, download(g, scores.length), 1e-3, "causal softmax backward");
    }

    @Test
    void gate() {
        int rows = 4, width = 12, rowWidth = 20;
        double[] context = random(rows * width, 1), gv = random(rows * rowWidth, 1), r = random(rows * width, 1);
        double[] joint = new double[context.length + gv.length];
        System.arraycopy(context, 0, joint, 0, context.length);
        System.arraycopy(gv, 0, joint, context.length, gv.length);
        Function<double[], double[]> reference = input -> {
            double[] out = new double[rows * width];
            for (int i = 0; i < rows * width; i++)
                out[i] = input[i] * sigmoid(input[context.length + (i / width) * rowWidth + i % width]);
            return out;
        };
        long c = upload(context), g = upload(gv), out = zeros(rows * width);
        ops.gateForward(c, g, out, rows, width, rowWidth);
        assertClose(reference.apply(joint), download(out, rows * width), 1e-5, "gate forward");
        long gc = zeros(context.length), gg = zeros(gv.length);
        ops.gateBackward(c, g, upload(r), gc, gg, rows, width, rowWidth);
        double[] gradient = new double[joint.length];
        System.arraycopy(download(gc, context.length), 0, gradient, 0, context.length);
        System.arraycopy(download(gg, gv.length), 0, gradient, context.length, gv.length);
        assertClose(numericGradient(reference, joint, r), gradient, 1e-3, "gate backward");
    }

    @Test
    void linear() {
        int rows = 5, in = 24, out = 16;
        double[] x = random(rows * in, 1), w = random(out * in, 0.3), r = random(rows * out, 1);
        Function<double[], double[]> byInput = input -> {
            double[] y = new double[rows * out];
            for (int row = 0; row < rows; row++)
                for (int o = 0; o < out; o++) {
                    double sum = 0;
                    for (int i = 0; i < in; i++) sum += input[row * in + i] * w[o * in + i];
                    y[row * out + o] = sum;
                }
            return y;
        };
        long dx = upload(x), dw = upload(w), y = zeros(rows * out);
        ops.linear(dx, dw, y, rows, in, out, false);
        assertClose(byInput.apply(x), download(y, rows * out), 1e-5, "linear forward");
        long gx = zeros(rows * in), gw = zeros(out * in), dy = upload(r);
        ops.linearInputGradient(dy, dw, gx, rows, in, out);
        assertClose(numericGradient(byInput, x, r), download(gx, rows * in), 1e-4, "linear dx");
        ops.linearWeightGradient(dy, dx, gw, rows, in, out);
        double[] expected = new double[out * in];
        for (int o = 0; o < out; o++)
            for (int i = 0; i < in; i++)
                for (int row = 0; row < rows; row++) expected[o * in + i] += r[row * out + o] * x[row * in + i];
        assertClose(expected, download(gw, out * in), 1e-5, "linear dw");
    }

    @Test
    void distillation() {
        int rows = 3, vocabulary = 3000, k = 64;
        double[] logits = random(rows * vocabulary, 3), student = random(rows * vocabulary, 3);
        long teacher = upload(logits), ids = zeros(rows * k), logProbs = zeros(rows * k);
        ops.topKLogSoftmax(teacher, rows, vocabulary, k, ids, logProbs);
        double[] idBits = download(ids, rows * k);
        int[] chosen = new int[rows * k];
        double[] lp = download(logProbs, rows * k);
        for (int i = 0; i < rows * k; i++) chosen[i] = Float.floatToRawIntBits((float) idBits[i]);
        double[] expected = new double[rows * k], expectedIds = new double[rows * k], actualIds = new double[rows * k];
        for (int r = 0; r < rows; r++) {
            Integer[] order = new Integer[vocabulary];
            for (int j = 0; j < vocabulary; j++) order[j] = j;
            int row = r;
            java.util.Arrays.sort(
                    order, (a, b) -> Double.compare(logits[row * vocabulary + b], logits[row * vocabulary + a]));
            double max = logits[r * vocabulary + order[0]], sum = 0;
            for (int j = 0; j < vocabulary; j++) sum += Math.exp(logits[r * vocabulary + j] - max);
            for (int i = 0; i < k; i++) {
                expectedIds[r * k + i] = order[i];
                actualIds[r * k + i] = chosen[r * k + i];
                expected[r * k + i] = logits[r * vocabulary + order[i]] - max - Math.log(sum);
            }
        }
        assertClose(expectedIds, actualIds, 0, "top-k ids");
        assertClose(expected, lp, 1e-5, "top-k log-probabilities");
        Function<double[], double[]> loss = z -> {
            double total = 0;
            for (int r = 0; r < rows; r++) {
                double max = Double.NEGATIVE_INFINITY, sum = 0, norm = 0;
                for (int j = 0; j < vocabulary; j++) max = Math.max(max, z[r * vocabulary + j]);
                for (int j = 0; j < vocabulary; j++) sum += Math.exp(z[r * vocabulary + j] - max);
                for (int i = 0; i < k; i++) norm += Math.exp(lp[r * k + i]);
                for (int i = 0; i < k; i++) {
                    double p = Math.exp(lp[r * k + i]) / norm;
                    total += p * (Math.log(p) - (z[r * vocabulary + chosen[r * k + i]] - max - Math.log(sum)));
                }
            }
            return new double[] {total};
        };
        long z = upload(student), dz = zeros(rows * vocabulary), scalar = zeros(2);
        ops.topKDistillation(z, rows, vocabulary, k, ids, logProbs, 1, dz, scalar);
        double[] numeric = numericGradient(loss, student, new double[] {1});
        assertClose(numeric, download(dz, rows * vocabulary), 1e-3, "distillation gradient");
    }
}
