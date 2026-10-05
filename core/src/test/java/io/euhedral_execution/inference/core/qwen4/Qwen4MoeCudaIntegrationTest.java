package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bf;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bits;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The routing and shared-expert kernels of the MoE block against CPU references.
class Qwen4MoeCudaIntegrationTest {

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
        MemorySegment host = arena.allocate(Math.max(16, (long) values.length * 2), 16);
        MemorySegment.copy(values, 0, host, ValueLayout.JAVA_SHORT, 0, values.length);
        gpu.copyHostToDevice(address, host, (long) values.length * 2);
        return address;
    }

    private static short[] download(CudaGpuMemory gpu, Arena arena, long address, int count) {
        MemorySegment host = arena.allocate(Math.max(16, (long) count * 2), 16);
        gpu.copyDeviceToHost(host, address, (long) count * 2);
        short[] values = new short[count];
        MemorySegment.copy(host, ValueLayout.JAVA_SHORT, 0, values, 0, count);
        return values;
    }

    private static int[] downloadInts(CudaGpuMemory gpu, Arena arena, long address, int count) {
        MemorySegment host = arena.allocate(Math.max(16, (long) count * 4), 16);
        gpu.copyDeviceToHost(host, address, (long) count * 4);
        int[] values = new int[count];
        MemorySegment.copy(host, ValueLayout.JAVA_INT, 0, values, 0, count);
        return values;
    }

    @Test
    void theRouterPicksTheTopExpertsDeterministicallyAndRenormalizes() {
        SplittableRandom rng = new SplittableRandom(41);
        int rows = 37, experts = 512, k = 10;
        short[] logits = random(rng, rows * experts, 2.0);
        // Ties: rows whose top logits repeat exactly (BF16 logits collide often), including at the boundary.
        for (int row = 0; row < rows; row += 3) {
            short tied = bits(6.0f + row);
            for (int j : new int[] {5, 90, 91, 300, 301, 302, 400, 401, 402, 403, 404, 405})
                logits[row * experts + j] = tied;
        }
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            long la = upload(gpu, arena, logits);
            long ids = gpu.allocate((long) rows * k * 4), weights = gpu.allocate((long) rows * k * 2);
            try {
                Qwen4MoeOps.router(gpu, la, ids, weights, rows, experts, k);
                int[] actualIds = downloadInts(gpu, arena, ids, rows * k);
                short[] actualWeights = download(gpu, arena, weights, rows * k);
                for (int row = 0; row < rows; row++) {
                    Qwen4Reference.Routing expected = Qwen4Reference.router(logits, row, experts, k);
                    assertArrayEquals(
                            expected.ids(), Arrays.copyOfRange(actualIds, row * k, (row + 1) * k), "ids row " + row);
                    short[] w = Arrays.copyOfRange(actualWeights, row * k, (row + 1) * k);
                    for (int t = 0; t < k; t++)
                        assertEquals(
                                bf(expected.weights()[t]),
                                bf(w[t]),
                                Math.abs(bf(expected.weights()[t])) * 0.008 + 1e-9,
                                "weight row " + row + " slot " + t);
                }
            } finally {
                gpu.free(weights);
                gpu.free(ids);
                gpu.free(la);
            }
        }
    }

    @Test
    void swiGluAndTheMoeFinishMatchTheReference() {
        SplittableRandom rng = new SplittableRandom(42);
        int rows = 5, width = 700;
        short[] gate = random(rng, rows * width, 2.0), up = random(rng, rows * width, 1.0);
        short[] shared = random(rng, rows * width, 1.0),
                routed = random(rng, rows * width, 1.0),
                gateRaw = random(rng, rows, 2.0);
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            long ga = upload(gpu, arena, gate), ua = upload(gpu, arena, up), oa = gpu.allocate((long) rows * width * 2);
            long sa = upload(gpu, arena, shared), ra = upload(gpu, arena, routed), na = upload(gpu, arena, gateRaw);
            try {
                Qwen4MoeOps.swiGlu(gpu, ga, ua, oa, rows * width);
                short[] act = download(gpu, arena, oa, rows * width);
                short[] expectedAct = Qwen4Reference.swiGlu(gate, up);
                int differing = 0;
                for (int i = 0; i < act.length; i++) {
                    if (Math.abs(bf(act[i]) - bf(expectedAct[i])) > Math.abs(bf(expectedAct[i])) * 0.008 + 1e-30)
                        throw new AssertionError("swiglu element " + i);
                    if (act[i] != expectedAct[i]) differing++;
                }
                assert differing < act.length / 20 : differing;
                Qwen4MoeOps.finish(gpu, ra, sa, na, oa, rows, width);
                short[] out = download(gpu, arena, oa, rows * width);
                short[] expected = Qwen4Reference.moeFinish(routed, shared, gateRaw, rows, width);
                for (int i = 0; i < out.length; i++)
                    if (Math.abs(bf(out[i]) - bf(expected[i])) > Math.abs(bf(expected[i])) * 0.008 + 1e-30)
                        throw new AssertionError("finish element " + i);
            } finally {
                for (long address : new long[] {na, ra, sa, oa, ua, ga}) gpu.free(address);
            }
        }
    }
}
