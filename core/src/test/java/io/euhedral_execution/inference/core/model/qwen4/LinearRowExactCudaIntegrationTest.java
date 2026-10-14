package io.euhedral_execution.inference.core.model.qwen4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The row-exact BF16 linear (the unquantized Qwen3.8 model's projections) on that model's shapes: each row of a
/// 1-, 8-, 9-, 16-, 17- or 32-row launch equals the same row launched alone, bit for bit, and the rows are within
/// FP32 summation-order noise of the exact route.
class LinearRowExactCudaIntegrationTest {

    private static float bf(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }

    private static short[] random(SplittableRandom random, int count, double scale) {
        short[] values = new short[count];
        for (int i = 0; i < count; i++)
            values[i] = (short) (Float.floatToIntBits((float) (random.nextGaussian() * scale)) >>> 16);
        return values;
    }

    @Test
    void everyRowEqualsItsOneRowLaunch() throws Exception {
        // (k, n): GDN a/b, attention key/value, a split-K narrow output, the MLP down projection, the MTP input
        // projection, a wide output that takes the unsplit kernels.
        int[][] shapes = {{5120, 48}, {5120, 1024}, {5120, 1536}, {17408, 5120}, {10240, 5120}, {5120, 8192}};
        int[] counts = {1, 8, 9, 16, 17, 32};
        try (CudaGpuMemory gpu = TestSupport.openGpu();
                Arena arena = Arena.ofConfined()) {
            SplittableRandom random = new SplittableRandom(29);
            for (int[] shape : shapes) {
                int k = shape[0], n = shape[1], rows = 32;
                long weights = TestSupport.upload(gpu, arena, random(random, k * n, 0.02));
                short[] inputValues = random(random, rows * k, 1.0);
                long input = TestSupport.upload(gpu, arena, inputValues);
                long many = gpu.allocate((long) rows * n * 2), one = gpu.allocate((long) n * 2);
                long exact = gpu.allocate((long) rows * n * 2);
                boolean before = gpu.selectExactNumerics(true);
                try {
                    Ops.linearBf16(gpu, input, weights, exact, rows, k, n);
                } finally {
                    gpu.selectExactNumerics(before);
                }
                for (int count : counts) {
                    Ops.linearBf16RowExact(gpu, input, weights, many, count, k, n);
                    gpu.synchronize();
                    short[] together = TestSupport.downloadBf16(gpu, arena, many, count * n);
                    for (int row = 0; row < count; row++) {
                        Ops.linearBf16RowExact(gpu, input + 2L * row * k, weights, one, 1, k, n);
                        gpu.synchronize();
                        short[] alone = TestSupport.downloadBf16(gpu, arena, one, n);
                        assertArrayEquals(
                                alone,
                                Arrays.copyOfRange(together, row * n, (row + 1) * n),
                                k + "x" + n + ", row " + row + " of " + count);
                    }
                }
                gpu.synchronize();
                short[] reference = TestSupport.downloadBf16(gpu, arena, exact, rows * n);
                short[] candidate = TestSupport.downloadBf16(gpu, arena, many, rows * n);
                double error = 0, norm = 0;
                for (int i = 0; i < reference.length; i++) {
                    double d = bf(candidate[i]) - bf(reference[i]);
                    error += d * d;
                    norm += (double) bf(reference[i]) * bf(reference[i]);
                }
                assertTrue(Math.sqrt(error / norm) < 4e-3, k + "x" + n + ": relative RMS " + Math.sqrt(error / norm));
                for (long address : new long[] {weights, input, many, one, exact}) gpu.free(address);
            }
        }
    }
}
