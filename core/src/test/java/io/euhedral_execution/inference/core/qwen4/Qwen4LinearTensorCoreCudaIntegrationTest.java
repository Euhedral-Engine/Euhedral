package io.euhedral_execution.inference.core.qwen4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The BF16 linears on tensor cores (prefill chunks) against the FP32 kernel (decode and the exact route), on the
/// model's shapes: within FP32 summation-order noise, and a row's bits the same however many rows run with it.
class Qwen4LinearTensorCoreCudaIntegrationTest {

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
    void tensorCoreLinearsMatchTheFp32KernelAndARowDoesNotDependOnItsCompany() throws Exception {
        // (k, n): the hyper-connection's down and up projections, the router, a GDN gate projection.
        int[][] shapes = {{10240, 320}, {320, 10240}, {2560, 512}, {2560, 32}};
        try (CudaGpuMemory gpu = Qwen4TestSupport.openGpu();
                Arena arena = Arena.ofConfined()) {
            SplittableRandom random = new SplittableRandom(23);
            for (int[] shape : shapes) {
                int k = shape[0], n = shape[1], rows = 64;
                long weights = Qwen4TestSupport.upload(gpu, arena, random(random, k * n, 0.05));
                short[] inputValues = random(random, rows * k, 1.0);
                long input = Qwen4TestSupport.upload(gpu, arena, inputValues);
                int few = Qwen4Ops.TC_MIN_ROWS;
                long one = Qwen4TestSupport.upload(gpu, arena, Arrays.copyOfRange(inputValues, 5 * k, (5 + few) * k));
                long exact = gpu.allocate((long) rows * n * 2);
                long fast = gpu.allocate((long) rows * n * 2);
                long fastOne = gpu.allocate((long) few * n * 2);
                boolean before = gpu.selectExactNumerics(true);
                try {
                    Qwen4Ops.linearBf16(gpu, input, weights, exact, rows, k, n);
                } finally {
                    gpu.selectExactNumerics(before);
                }
                Qwen4Ops.linearBf16(gpu, input, weights, fast, rows, k, n);
                Qwen4Ops.linearBf16(gpu, one, weights, fastOne, few, k, n);
                gpu.synchronize();
                short[] reference = Qwen4TestSupport.downloadBf16(gpu, arena, exact, rows * n);
                short[] candidate = Qwen4TestSupport.downloadBf16(gpu, arena, fast, rows * n);
                double error = 0, norm = 0;
                for (int i = 0; i < reference.length; i++) {
                    double d = bf(candidate[i]) - bf(reference[i]);
                    error += d * d;
                    norm += (double) bf(reference[i]) * bf(reference[i]);
                }
                double relative = Math.sqrt(error / norm);
                System.out.printf("k %d, n %d: tensor cores against FP32, relative RMS %.2e%n", k, n, relative);
                assertTrue(relative < 5e-3, "summation-order noise only: " + relative);
                assertArrayEquals(
                        Arrays.copyOfRange(candidate, 5 * n, 6 * n),
                        Qwen4TestSupport.downloadBf16(gpu, arena, fastOne, n),
                        "row 5 among 9 rows has the bits it has among 64, k " + k + " n " + n);
                for (long address : new long[] {weights, input, one, exact, fast, fastOne}) gpu.free(address);
            }
        }
    }

    /// Decode's BF16 linears with K split across warps, on the shapes that use them: within summation-order noise of
    /// the FP32 kernel, and a row's bits the same alone or among eight.
    @Test
    void splitDecodeLinearsMatchTheFp32KernelAndARowDoesNotDependOnItsCompany() throws Exception {
        int[][] shapes = {{2560, 1}, {10240, 4}, {10240, 320}, {2560, 512}, {2560, 32}};
        try (CudaGpuMemory gpu = Qwen4TestSupport.openGpu();
                Arena arena = Arena.ofConfined()) {
            SplittableRandom random = new SplittableRandom(29);
            for (int[] shape : shapes) {
                int k = shape[0], n = shape[1], rows = 8;
                assertTrue(Qwen4Ops.slices(k, n) > 1, "the shape " + k + "x" + n + " splits K");
                long weights = Qwen4TestSupport.upload(gpu, arena, random(random, k * n, 0.05));
                short[] inputValues = random(random, rows * k, 1.0);
                long input = Qwen4TestSupport.upload(gpu, arena, inputValues);
                long one = Qwen4TestSupport.upload(gpu, arena, Arrays.copyOfRange(inputValues, 3 * k, 4 * k));
                long exact = gpu.allocate((long) rows * n * 2);
                long fast = gpu.allocate((long) rows * n * 2);
                long fastOne = gpu.allocate((long) n * 2 + 16);
                boolean before = gpu.selectExactNumerics(true);
                try {
                    Qwen4Ops.linearBf16(gpu, input, weights, exact, rows, k, n);
                } finally {
                    gpu.selectExactNumerics(before);
                }
                Qwen4Ops.linearBf16(gpu, input, weights, fast, rows, k, n);
                Qwen4Ops.linearBf16(gpu, one, weights, fastOne, 1, k, n);
                gpu.synchronize();
                short[] reference = Qwen4TestSupport.downloadBf16(gpu, arena, exact, rows * n);
                short[] candidate = Qwen4TestSupport.downloadBf16(gpu, arena, fast, rows * n);
                double error = 0, norm = 0;
                for (int i = 0; i < reference.length; i++) {
                    double d = bf(candidate[i]) - bf(reference[i]);
                    error += d * d;
                    norm += (double) bf(reference[i]) * bf(reference[i]);
                }
                double relative = Math.sqrt(error / norm);
                System.out.printf(
                        "k %d, n %d: split K against one warp per column, relative RMS %.2e%n", k, n, relative);
                assertTrue(relative < 5e-3, "summation-order noise only: " + relative);
                assertArrayEquals(
                        Arrays.copyOfRange(candidate, 3 * n, 4 * n),
                        Qwen4TestSupport.downloadBf16(gpu, arena, fastOne, n),
                        "row 3 alone has the bits it has among 8, k " + k + " n " + n);
                for (long address : new long[] {weights, input, one, exact, fast, fastOne}) gpu.free(address);
            }
        }
    }
}
