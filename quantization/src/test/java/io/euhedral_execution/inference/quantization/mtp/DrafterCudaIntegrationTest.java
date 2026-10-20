package io.euhedral_execution.inference.quantization.mtp;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.quantization.gpu.Device;
import io.euhedral_execution.inference.quantization.gpu.Upload;
import io.euhedral_execution.inference.quantization.training.Ops;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/// The drafter's training kernels and its backward pass, at small dimensions: NVFP4 rounding byte for byte as the
/// converter rounds (tools/euhedral_artifacts/nvfp4.py, transcribed below in float32), the decoder of artifact rows,
/// the draft head's cross-entropy against double arithmetic, and the unrolled drafter's gradients against central
/// differences of its forward pass (projections unrounded), across a window that attends an earlier window's keys.
class DrafterCudaIntegrationTest {
    private static Device device;
    private static Ops ops;
    private static DrafterKernels kernels;
    private static final List<Long> allocations = new ArrayList<>();
    private final Random random = new Random(19);

    @BeforeAll
    static void start() {
        device = new Device(Path.of(System.getProperty("euhedral.cuda.library")));
        ops = new Ops(device.cuda, device.stream, device.cublas);
        kernels = new DrafterKernels(device.cuda, device.stream);
    }

    @AfterAll
    static void stop() {
        for (long address : allocations) device.gpu.free(address);
        device.close();
    }

    private static long allocate(long bytes) {
        long address = device.gpu.allocate(Math.max(4, bytes));
        allocations.add(address);
        return address;
    }

    private static long upload(float[] values) {
        long address = allocate(4L * values.length);
        Upload.floats(device.gpu, address, values);
        return address;
    }

    private float[] random(int count, double scale) {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float) (random.nextGaussian() * scale);
        return values;
    }

    private static MemorySegment download(long address, long bytes) {
        device.synchronize();
        MemorySegment host = Arena.ofAuto().allocate(bytes, 8);
        device.gpu.copyDeviceToHost(host, address, bytes);
        return host;
    }

    private static float[] floats(long address, long count) {
        return download(address, 4 * count).toArray(ValueLayout.JAVA_FLOAT);
    }

    // ------------------------------------------------------------ NVFP4

    private static final float[] E2M1 = {0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f};

    private static float e4m3(int code) {
        return code < 8 ? (code & 7) / 8f * 0.015625f : (1 + (code & 7) / 8f) * (float) Math.pow(2, (code >> 3) - 7);
    }

    /// nvfp4.py round_to_table over an ascending table, in float32.
    private static int roundToTable(float value, float[] table) {
        int upper = 0;
        while (upper < table.length && table[upper] < value) upper++;
        upper = Math.min(Math.max(upper, 1), table.length - 1);
        int lower = upper - 1;
        float below = value - table[lower], above = table[upper] - value;
        int index = above < below || (above == below && upper % 2 == 0) ? upper : lower;
        if (value >= table[table.length - 1]) index = table.length - 1;
        if (value <= table[0]) index = 0;
        return index;
    }

    @Test
    void nvfp4RoundingIsTheConvertersByteForByte() {
        int rows = 12, k = 64;
        float[] w = random(rows * k, 0.05);
        for (int i = 0; i < 16; i++) w[3 * k + i] = 0f; // an all-zero block
        w[5 * k + 7] = 1.7f; // an outlier that sets the global scale
        w[7 * k + 20] = -0.25f;
        float[] e4m3 = new float[127];
        for (int c = 0; c < 127; c++) e4m3[c] = e4m3(c);
        float amax = 0;
        for (float v : w) amax = Math.max(amax, Math.abs(v));
        float global = amax / 2688f;
        byte[] expectedPacked = new byte[rows * k / 2], expectedScales = new byte[rows * k / 16];
        float[] expectedValues = new float[rows * k];
        for (int b = 0; b < rows * k / 16; b++) {
            float largest = 0;
            for (int i = 0; i < 16; i++) largest = Math.max(largest, Math.abs(w[b * 16 + i]));
            int scale = roundToTable(largest / 6f / global, e4m3);
            float decode = e4m3[scale] * global;
            expectedScales[b] = (byte) scale;
            int[] codes = new int[16];
            for (int i = 0; i < 16; i++) {
                float v = w[b * 16 + i];
                int magnitude = roundToTable(decode > 0 ? Math.abs(v) / decode : 0f, E2M1);
                codes[i] = magnitude | (v < 0 && magnitude > 0 ? 8 : 0);
                expectedValues[b * 16 + i] = E2M1[magnitude] * ((codes[i] & 8) != 0 ? -1f : 1f) * decode;
            }
            for (int i = 0; i < 8; i++) expectedPacked[b * 8 + i] = (byte) (codes[2 * i] | (codes[2 * i + 1] << 4));
        }
        long values = upload(w), out = allocate(4L * rows * k), maximum = allocate(4);
        long packed = allocate(rows * k / 2), scales = allocate(rows * k / 16), global32 = allocate(4);
        ops.zero(maximum, 1);
        kernels.absoluteMaximum(values, (long) rows * k, maximum);
        kernels.nvfp4Round(values, out, packed, scales, rows, k, maximum);
        assertEquals(amax, Float.intBitsToFloat(download(maximum, 4).get(ValueLayout.JAVA_INT, 0)));
        assertArrayEquals(expectedValues, floats(out, (long) rows * k));
        assertArrayEquals(expectedPacked, download(packed, rows * k / 2).toArray(ValueLayout.JAVA_BYTE));
        assertArrayEquals(expectedScales, download(scales, rows * k / 16).toArray(ValueLayout.JAVA_BYTE));
        // The artifact decoder reads the same values back (as BF16).
        Upload.floats(device.gpu, global32, new float[] {global});
        long bf16 = allocate(2L * rows * k);
        kernels.nvfp4DecodeBf16(packed, scales, global32, rows, k, k, bf16);
        short[] decoded = download(bf16, 2L * rows * k).toArray(ValueLayout.JAVA_SHORT);
        for (int i = 0; i < rows * k; i++) {
            int bits = Float.floatToRawIntBits(expectedValues[i]);
            short expected = (short) ((bits + 0x7fff + ((bits >> 16) & 1)) >>> 16);
            assertEquals(expected, decoded[i], "decoded value " + i);
        }
    }

    // ------------------------------------------------------------ cross-entropy

    @Test
    void crossEntropyMatchesDoubleArithmetic() {
        int rows = 3, vocabulary = 1000;
        float[] logits = random(rows * vocabulary, 2);
        logits[2 * vocabulary + 10] = 9f;
        logits[2 * vocabulary + 400] = 9f; // a tie: the lower index wins
        int[] labels = {5, -1, 999};
        float scale = 0.25f;
        long x = upload(logits), label = allocate(4L * rows), predicted = allocate(4L * rows);
        long logProbability = allocate(4L * rows), loss = allocate(8);
        Upload.ints(device.gpu, label, labels);
        ops.zero(loss, 2);
        kernels.crossEntropy(x, rows, vocabulary, label, scale, loss, predicted, logProbability);
        float[] gradient = floats(x, (long) rows * vocabulary);
        int[] argmax = download(predicted, 4L * rows).toArray(ValueLayout.JAVA_INT);
        double expectedLoss = 0;
        for (int r = 0; r < rows; r++) {
            double maximum = Double.NEGATIVE_INFINITY, sum = 0;
            int best = 0;
            for (int i = 0; i < vocabulary; i++) {
                if (logits[r * vocabulary + i] > maximum) {
                    maximum = logits[r * vocabulary + i];
                    best = i;
                }
            }
            for (int i = 0; i < vocabulary; i++) sum += Math.exp(logits[r * vocabulary + i] - maximum);
            assertEquals(best, argmax[r], "argmax of row " + r);
            for (int i = 0; i < vocabulary; i++) {
                double p = Math.exp(logits[r * vocabulary + i] - maximum) / sum;
                double expected = labels[r] < 0 ? 0 : (p - (i == labels[r] ? 1 : 0)) * scale;
                assertEquals(expected, gradient[r * vocabulary + i], 1e-6, "gradient " + r + "," + i);
            }
            if (labels[r] >= 0) expectedLoss -= logits[r * vocabulary + labels[r]] - maximum - Math.log(sum);
        }
        assertEquals(10, argmax[2]);
        assertEquals(expectedLoss, download(loss, 8).get(ValueLayout.JAVA_DOUBLE, 0), 1e-4);
    }

    // ------------------------------------------------------------ gradients

    private static final int ROWS = 6, DEPTHS = 3, CONTEXT = 4, COUNT = 5;

    private Drafter drafter() {
        var shape = new Drafter.Shape(64, 96, 4, 2, 32, 16, 1e7, 1e-6f);
        var drafter = new Drafter(device.gpu, device.cuda, device.cublas, ops, kernels, shape, DEPTHS, ROWS, 16, 32);
        for (Drafter.Parameter p : drafter.parameters.values()) {
            double scale = p.rows == 1 ? 0.1 : 1 / Math.sqrt(p.columns);
            Upload.floats(device.gpu, p.values, random((int) p.count, scale));
        }
        drafter.rounded = false;
        drafter.measure();
        return drafter;
    }

    /// sum over depths of (m_k - target_k)^2 for the window at position CONTEXT, in double.
    private double loss(Drafter drafter, long hidden, long[] embedded, long[] targets, long scalar, boolean gradients) {
        int h = drafter.shape.hidden();
        long[] outputs = new long[DEPTHS];
        long input = hidden;
        for (int k = 1; k <= DEPTHS; k++) {
            outputs[k - 1] = drafter.forward(k, CONTEXT, 0, COUNT, input, embedded[k - 1]);
            input = outputs[k - 1];
        }
        ops.zero(scalar, 2);
        if (gradients) drafter.startBackward(0, CONTEXT, COUNT);
        for (int k = 1; k <= DEPTHS; k++)
            ops.squaredError(
                    outputs[k - 1],
                    targets[k - 1],
                    (long) COUNT * h,
                    1,
                    gradients ? drafter.outputGradient(k) : 0,
                    scalar);
        if (gradients) for (int k = DEPTHS; k >= 1; k--) drafter.backward(k, CONTEXT, 0, COUNT);
        return download(scalar, 8).get(ValueLayout.JAVA_DOUBLE, 0);
    }

    @Test
    void unrolledDraftsBackwardMatchesCentralDifferences() {
        try (Drafter drafter = drafter()) {
            int h = drafter.shape.hidden();
            // An earlier window's depth-1 forward leaves its keys for the trained window to attend.
            long contextHidden = upload(random(CONTEXT * h, 1)), contextEmbedded = upload(random(CONTEXT * h, 1));
            drafter.forward(1, 0, 0, CONTEXT, contextHidden, contextEmbedded);
            long hidden = upload(random(COUNT * h, 1)), scalar = allocate(8);
            long[] embedded = new long[DEPTHS], targets = new long[DEPTHS];
            for (int k = 0; k < DEPTHS; k++) {
                embedded[k] = upload(random(COUNT * h, 1));
                targets[k] = upload(random(COUNT * h, 1));
            }
            loss(drafter, hidden, embedded, targets, scalar, true);
            List<String> failures = new ArrayList<>();
            for (Drafter.Parameter p : drafter.parameters.values()) {
                float[] original = floats(p.values, p.count), gradient = floats(p.gradient, p.count);
                float[] direction = random((int) p.count, 1);
                double analytic = 0, magnitude = 0;
                for (int i = 0; i < p.count; i++) {
                    analytic += gradient[i] * direction[i];
                    magnitude = Math.max(magnitude, Math.abs(original[i]));
                }
                double epsilon = (p.rows == 1 ? 1e-2 : 1e-3) * Math.max(magnitude, 1e-2);
                float[] shifted = new float[(int) p.count];
                for (int i = 0; i < p.count; i++) shifted[i] = (float) (original[i] + epsilon * direction[i]);
                Upload.floats(device.gpu, p.values, shifted);
                double up = loss(drafter, hidden, embedded, targets, scalar, false);
                for (int i = 0; i < p.count; i++) shifted[i] = (float) (original[i] - epsilon * direction[i]);
                Upload.floats(device.gpu, p.values, shifted);
                double down = loss(drafter, hidden, embedded, targets, scalar, false);
                Upload.floats(device.gpu, p.values, original);
                double numeric = (up - down) / (2 * epsilon);
                double error = Math.abs(numeric - analytic) / Math.max(Math.abs(numeric), 1e-6);
                System.out.printf(
                        "%s: analytic %.6g, numeric %.6g, relative error %.2e%n", p.name, analytic, numeric, error);
                if (error > 2e-2) failures.add(p.name + " " + analytic + " vs " + numeric);
            }
            assertTrue(failures.isEmpty(), String.join("; ", failures));
        }
    }

    @Test
    void roundedProjectionsRunFinite() {
        try (Drafter drafter = drafter()) {
            drafter.rounded = true;
            int h = drafter.shape.hidden();
            long hidden = upload(random(COUNT * h, 1)), embedded = upload(random(COUNT * h, 1));
            float[] out = floats(drafter.forward(1, 0, 0, COUNT, hidden, embedded), (long) COUNT * h);
            for (float v : out) assertTrue(Float.isFinite(v));
        }
    }
}
