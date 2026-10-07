package io.euhedral_execution.inference.core.qwen4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The shared expert's down projection with its input width padded from 640 to 768 (zero columns in the weights and
/// in the activation): bit for bit the unpadded projection on the exact route, and within the native route's error
/// of it on the native prefill kernels, which take only the padded width.
class Qwen4SharedDownPaddingCudaIntegrationTest {

    private static final int HIDDEN = 2560, WIDTH = 640, PADDED = Qwen4MoeLayer.paddedWidth(WIDTH);

    private static float bf(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }

    private static short[] random(SplittableRandom random, int count, double scale) {
        short[] values = new short[count];
        for (int i = 0; i < count; i++)
            values[i] = (short) (Float.floatToIntBits((float) (random.nextGaussian() * scale)) >>> 16);
        return values;
    }

    /// A plain NVFP4 tensor of `rows` rows of `k` values: random codes, E4M3 scales between 0.5 and 2, global 1/64.
    private static long weights(CudaGpuMemory gpu, Arena arena, SplittableRandom random, int rows, int k) {
        long bytes = Qwen4MoeOps.nvfp4Bytes(rows, k);
        MemorySegment host = arena.allocate(bytes, 16);
        long scales = (rows * (long) k / 2 + 255) & ~255L;
        for (long i = 0; i < (long) rows * k / 2; i++) host.set(ValueLayout.JAVA_BYTE, i, (byte) random.nextInt(256));
        for (long i = 0; i < (long) rows * k / 16; i++)
            host.set(ValueLayout.JAVA_BYTE, scales + i, (byte) (0x30 + random.nextInt(0x10)));
        long global = (scales + (long) rows * k / 16 + 255) & ~255L;
        host.set(ValueLayout.JAVA_FLOAT_UNALIGNED, global, 1f / 64);
        long address = gpu.allocate(bytes);
        gpu.copyHostToDevice(address, host, bytes);
        return address;
    }

    @Test
    void thePaddedDownProjectionIsTheUnpaddedOneOnTheExactRouteAndCloseOnTheNativeOne() throws Exception {
        try (CudaGpuMemory gpu = Qwen4TestSupport.openGpu();
                Arena arena = Arena.ofConfined()) {
            SplittableRandom random = new SplittableRandom(17);
            long down = weights(gpu, arena, random, HIDDEN, WIDTH);
            long padded = gpu.allocate(Qwen4MoeOps.nvfp4Bytes(HIDDEN, PADDED));
            Qwen4MoeOps.nvfp4PadK(gpu, down, padded, HIDDEN, WIDTH, PADDED);
            for (int rows : new int[] {16, 128}) {
                long gate = Qwen4TestSupport.upload(gpu, arena, random(random, rows * WIDTH, 1.0));
                long up = Qwen4TestSupport.upload(gpu, arena, random(random, rows * WIDTH, 1.0));
                long act = gpu.allocate((long) rows * WIDTH * 2);
                long actPadded = gpu.allocate((long) rows * PADDED * 2);
                Qwen4MoeOps.swiGlu(gpu, gate, up, act, rows * WIDTH);
                Qwen4MoeOps.swiGluPadded(gpu, gate, up, actPadded, rows, WIDTH, PADDED);
                gpu.synchronize();
                short[] plain = Qwen4TestSupport.downloadBf16(gpu, arena, act, rows * WIDTH);
                short[] wide = Qwen4TestSupport.downloadBf16(gpu, arena, actPadded, rows * PADDED);
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < PADDED; c++)
                        assertEquals(
                                c < WIDTH ? plain[r * WIDTH + c] : 0,
                                wide[r * PADDED + c],
                                "activation row " + r + " column " + c);

                long exact = gpu.allocate((long) rows * HIDDEN * 2);
                long exactPadded = gpu.allocate((long) rows * HIDDEN * 2);
                long nativePadded = gpu.allocate((long) rows * HIDDEN * 2);
                boolean before = gpu.selectExactNumerics(true);
                try {
                    gpu.linearNvfp4Bf16(act, down, exact, rows, WIDTH, HIDDEN, Qwen4MoeOps.nvfp4Bytes(HIDDEN, WIDTH));
                    gpu.linearNvfp4Bf16(
                            actPadded,
                            padded,
                            exactPadded,
                            rows,
                            PADDED,
                            HIDDEN,
                            Qwen4MoeOps.nvfp4Bytes(HIDDEN, PADDED));
                } finally {
                    gpu.selectExactNumerics(before);
                }
                gpu.linearNvfp4Bf16(
                        actPadded, padded, nativePadded, rows, PADDED, HIDDEN, Qwen4MoeOps.nvfp4Bytes(HIDDEN, PADDED));
                gpu.synchronize();
                short[] reference = Qwen4TestSupport.downloadBf16(gpu, arena, exact, rows * HIDDEN);
                assertArrayEquals(
                        reference,
                        Qwen4TestSupport.downloadBf16(gpu, arena, exactPadded, rows * HIDDEN),
                        "the zero columns change nothing on the exact route, " + rows + " rows");
                short[] fast = Qwen4TestSupport.downloadBf16(gpu, arena, nativePadded, rows * HIDDEN);
                double error = 0, norm = 0;
                for (int i = 0; i < fast.length; i++) {
                    double d = bf(fast[i]) - bf(reference[i]);
                    error += d * d;
                    norm += (double) bf(reference[i]) * bf(reference[i]);
                }
                double relative = Math.sqrt(error / norm);
                System.out.printf("%d rows: native padded against exact, relative RMS error %.4f%n", rows, relative);
                assertTrue(relative < 0.02, "the native route is within its error: " + relative);
                for (long address : new long[] {gate, up, act, actPadded, exact, exactPadded, nativePadded})
                    gpu.free(address);
            }
            gpu.free(down);
            gpu.free(padded);
        }
    }
}
