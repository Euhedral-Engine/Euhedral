package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bf;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bits;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.KernelArguments;
import io.euhedral_execution.inference.core.gpu.Qwen4Kernel;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The Gated DeltaNet operators of Flash-Next against CPU references: the fused-QKV convolution with carried
/// history, the decay/beta control and the gated output norm with the sigmoid gate. The recurrence is the dense
/// engine's and the whole layer is checked against the upstream fixtures (Qwen4FixtureCudaIntegrationTest).
class Qwen4GdnCudaIntegrationTest {

    private static CudaGpuMemory open() {
        return new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
    }

    private static short[] random(SplittableRandom rng, int count, double scale) {
        short[] values = new short[count];
        for (int i = 0; i < count; i++) {
            double u = rng.nextDouble(), v = rng.nextDouble();
            values[i] =
                    bits((float) (Math.sqrt(-2 * Math.log(Math.max(u, 1e-12))) * Math.cos(2 * Math.PI * v) * scale));
        }
        return values;
    }

    private static long upload(CudaGpuMemory gpu, Arena arena, short[] values) {
        long address = gpu.allocate(Math.max(16, (long) values.length * 2));
        MemorySegment host = arena.allocate((long) values.length * 2, 16);
        MemorySegment.copy(values, 0, host, ValueLayout.JAVA_SHORT, 0, values.length);
        gpu.copyHostToDevice(address, host, (long) values.length * 2);
        return address;
    }

    private static short[] downloadBf16(CudaGpuMemory gpu, Arena arena, long address, int count) {
        MemorySegment host = arena.allocate((long) count * 2, 16);
        gpu.copyDeviceToHost(host, address, (long) count * 2);
        short[] values = new short[count];
        MemorySegment.copy(host, ValueLayout.JAVA_SHORT, 0, values, 0, count);
        return values;
    }

    private static float[] downloadFloat(CudaGpuMemory gpu, Arena arena, long address, int count) {
        MemorySegment host = arena.allocate((long) count * 4, 16);
        gpu.copyDeviceToHost(host, address, (long) count * 4);
        float[] values = new float[count];
        MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0, values, 0, count);
        return values;
    }

    private static void assertNear(short[] expected, short[] actual, String what) {
        assertEquals(expected.length, actual.length, what);
        int differing = 0;
        for (int i = 0; i < expected.length; i++) {
            float e = bf(expected[i]), a = bf(actual[i]);
            if (Math.abs(e - a) > Math.abs(e) * 0.0079 + 1e-30)
                throw new AssertionError(what + ": element " + i + " expected " + e + " but was " + a);
            if (expected[i] != actual[i]) differing++;
        }
        assertTrue(
                differing <= expected.length * 0.02 + 2,
                what + ": " + differing + " of " + expected.length + " differ");
    }

    private static final KernelArguments ARGUMENTS = new KernelArguments();

    @Test
    void theConvolutionCarriesItsHistoryAcrossChunks() {
        SplittableRandom rng = new SplittableRandom(31);
        int channels = 300, taps = 4, total = 21, historyRows = taps - 1;
        short[] x = random(rng, total * channels, 1.5), weights = random(rng, channels * taps, 0.5);
        short[] whole = Qwen4Reference.gdnConv(x, new short[historyRows * channels], weights, total, channels, taps);
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            long xa = upload(gpu, arena, x), wa = upload(gpu, arena, weights);
            long history = upload(gpu, arena, new short[historyRows * channels]);
            long out = gpu.allocate((long) total * channels * 2);
            try {
                int row = 0;
                for (int rows : new int[] {1, 2, 5, 1, 12}) {
                    long offset = (long) row * channels * 2;
                    gpu.launchTableKernel(
                            Qwen4Kernel.GDN_CONV_BF16,
                            (channels + 255) / 256,
                            rows,
                            1,
                            256,
                            1,
                            1,
                            0,
                            ARGUMENTS
                                    .clear()
                                    .pointer(xa + offset)
                                    .pointer(history)
                                    .pointer(wa)
                                    .pointer(out + offset)
                                    .int32(rows)
                                    .int32(channels)
                                    .int32(taps));
                    Qwen4Ops.convHistory(gpu, xa + offset, history, rows, channels, historyRows);
                    row += rows;
                }
                assertNear(whole, downloadBf16(gpu, arena, out, total * channels), "convolution");
            } finally {
                gpu.free(out);
                gpu.free(history);
                gpu.free(wa);
                gpu.free(xa);
            }
        }
    }

    @Test
    void theControlMatchesTheReference() {
        SplittableRandom rng = new SplittableRandom(32);
        int rows = 7, heads = 48;
        short[] a = random(rng, rows * heads, 2.0), b = random(rng, rows * heads, 2.0);
        short[] aLog = random(rng, heads, 1.0), dtBias = random(rng, heads, 1.0);
        a[3] = bits(30f); // beyond the softplus threshold
        Qwen4Reference.Control expected = Qwen4Reference.gdnControl(a, b, aLog, dtBias, rows, heads);
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            long aa = upload(gpu, arena, a),
                    ba = upload(gpu, arena, b),
                    la = upload(gpu, arena, aLog),
                    da = upload(gpu, arena, dtBias);
            long alpha = gpu.allocate((long) rows * heads * 4), beta = gpu.allocate((long) rows * heads * 4);
            try {
                gpu.launchTableKernel(
                        Qwen4Kernel.GDN_CONTROL_BF16,
                        (rows * heads + 255) / 256,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        ARGUMENTS
                                .clear()
                                .pointer(aa)
                                .pointer(ba)
                                .pointer(la)
                                .pointer(da)
                                .pointer(alpha)
                                .pointer(beta)
                                .int32(rows)
                                .int32(heads));
                float[] alphaActual = downloadFloat(gpu, arena, alpha, rows * heads);
                float[] betaActual = downloadFloat(gpu, arena, beta, rows * heads);
                for (int i = 0; i < rows * heads; i++) {
                    assertEquals(
                            expected.alpha()[i],
                            alphaActual[i],
                            Math.abs(expected.alpha()[i]) * 1e-5 + 1e-30,
                            "alpha " + i);
                    assertEquals(expected.beta()[i], betaActual[i], 0.0, "beta " + i);
                }
            } finally {
                for (long address : new long[] {beta, alpha, da, la, ba, aa}) gpu.free(address);
            }
        }
    }

    @Test
    void theGatedNormMatchesTheReferenceForBothGates() {
        SplittableRandom rng = new SplittableRandom(33);
        int rows = 5, heads = 48, headDim = 128;
        short[] core = random(rng, rows * heads * headDim, 1.0), z = random(rng, rows * heads * headDim, 2.0);
        short[] weight = random(rng, headDim, 1.0);
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            long ca = upload(gpu, arena, core), za = upload(gpu, arena, z), wa = upload(gpu, arena, weight);
            long out = gpu.allocate((long) rows * heads * headDim * 2);
            try {
                for (int activation = 0; activation < 2; activation++) {
                    short[] expected =
                            Qwen4Reference.gdnGatedNorm(core, z, weight, rows, heads, headDim, 1e-6f, activation == 1);
                    gpu.launchTableKernel(
                            Qwen4Kernel.GDN_GATED_NORM_BF16,
                            rows * heads,
                            1,
                            1,
                            128,
                            1,
                            1,
                            0,
                            ARGUMENTS
                                    .clear()
                                    .pointer(ca)
                                    .pointer(za)
                                    .pointer(wa)
                                    .pointer(out)
                                    .int32(rows)
                                    .int32(heads)
                                    .int32(headDim)
                                    .float32(1e-6f)
                                    .int32(activation));
                    assertNear(expected, downloadBf16(gpu, arena, out, expected.length), "activation " + activation);
                }
            } finally {
                gpu.free(out);
                gpu.free(wa);
                gpu.free(za);
                gpu.free(ca);
            }
        }
    }
}
