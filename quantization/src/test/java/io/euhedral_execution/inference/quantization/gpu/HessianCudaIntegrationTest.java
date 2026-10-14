package io.euhedral_execution.inference.quantization.gpu;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Test;

/// The symmetric Hessian (upper blocks, then mirrored) against a float64 X^T X on the host, for a width that is not
/// a multiple of the block, accumulated over two calls.
class HessianCudaIntegrationTest {

    @Test
    void symmetricAccumulationIsXTransposeX() {
        int rows = 300, k = 1100;
        Random random = new Random(3);
        short[] x = new short[rows * k];
        for (int i = 0; i < x.length; i++) x[i] = (short) (Float.floatToIntBits((float) random.nextGaussian()) >>> 16);
        try (Device device = new Device(Path.of(System.getProperty("euhedral.cuda.library")));
                Arena arena = Arena.ofConfined()) {
            long xDevice = device.gpu.allocate(2L * x.length), h = device.gpu.allocate(4L * k * k);
            device.gpu.zeroDeviceMemory(h, 4L * k * k);
            Upload.segment(device.gpu, xDevice, MemorySegment.ofArray(x), 2L * x.length);
            device.accumulateHessian(h, xDevice, rows / 2, k);
            device.accumulateHessian(h, xDevice + 2L * (rows / 2) * k, rows - rows / 2, k);
            device.symmetrize(h, k);
            device.synchronize();
            MemorySegment host = arena.allocate(4L * k * k);
            device.gpu.copyDeviceToHost(host, h, host.byteSize());
            double worst = 0;
            for (int a = 0; a < k; a += 7) {
                for (int b = 0; b < k; b += 5) {
                    double want = 0;
                    for (int r = 0; r < rows; r++) {
                        want += (double) Float.intBitsToFloat(x[r * k + a] << 16)
                                * Float.intBitsToFloat(x[r * k + b] << 16);
                    }
                    float got = host.getAtIndex(ValueLayout.JAVA_FLOAT, (long) b * k + a);
                    worst = Math.max(worst, Math.abs(got - want) / (Math.abs(want) + 1));
                }
            }
            System.out.printf("worst relative difference %.2e%n", worst);
            assertTrue(worst < 1e-4, "worst relative difference " + worst);
            device.gpu.free(xDevice);
            device.gpu.free(h);
        }
    }
}
