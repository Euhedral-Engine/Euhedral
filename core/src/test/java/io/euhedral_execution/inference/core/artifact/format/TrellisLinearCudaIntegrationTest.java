package io.euhedral_execution.inference.core.artifact.format;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ScratchUse;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

/// One launch per route of [TrellisFormat#linear] on the device, over synthetic trellis objects (random tile words,
/// signs and output scales), against the scalar reference route of the same launch.
class TrellisLinearCudaIntegrationTest {

    private static final int W = 56;
    private static final ValueLayout.OfShort SHORT =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static CudaGpuMemory open() {
        return new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
    }

    private static short fp16(float value) {
        return Float.floatToFloat16(value);
    }

    private static short bf16(float value) {
        int bits = Float.floatToIntBits(value);
        return (short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16);
    }

    private static float bf16Value(short bits) {
        return Float.intBitsToFloat((bits & 0xffff) << 16);
    }

    /// The device copies of one synthetic object.
    private static final class Synthetic implements AutoCloseable {
        final CudaGpuMemory gpu;
        final List<TensorHandle> weights = new ArrayList<>();
        final List<Long> allocations = new ArrayList<>();
        final int n, k;
        final int[] ends;

        Synthetic(CudaGpuMemory gpu, Arena arena, Random random, int n, int k, int... ends) {
            this.gpu = gpu;
            this.n = n;
            this.k = k;
            this.ends = ends;
            long[] shape = {k / 16, n / 16, W};
            long words = shape[0] * shape[1] * W;
            MemorySegment payload = arena.allocate(words * 2, 2);
            for (long i = 0; i < words; i++) payload.set(SHORT, i * 2, (short) random.nextInt());
            weights.add(upload(
                    "synthetic",
                    shape,
                    TensorDataType.BF16,
                    WeightFormat.TRELLIS,
                    WeightLayout.TRELLIS_TILE16_V1,
                    payload,
                    words * 2));
            int s = ends.length;
            MemorySegment suh = arena.allocate((long) s * k * 2, 2);
            for (long i = 0; i < (long) s * k; i++) suh.set(SHORT, i * 2, fp16(random.nextBoolean() ? 1f : -1f));
            weights.add(upload(
                    "synthetic.suh",
                    new long[] {s, k},
                    TensorDataType.FP16,
                    WeightFormat.FP16,
                    WeightLayout.CONTIGUOUS_LE_V1,
                    suh,
                    (long) s * k * 2));
            MemorySegment svh = arena.allocate(n * 2L, 2);
            for (int i = 0; i < n; i++) svh.set(SHORT, i * 2L, fp16(0.5f + random.nextFloat()));
            weights.add(upload(
                    "synthetic.svh",
                    new long[] {n},
                    TensorDataType.FP16,
                    WeightFormat.FP16,
                    WeightLayout.CONTIGUOUS_LE_V1,
                    svh,
                    n * 2L));
            MemorySegment segments = arena.allocate(s * 4L, 4);
            for (int i = 0; i < s; i++) segments.set(INT, i * 4L, ends[i]);
            weights.add(upload(
                    "synthetic.segments",
                    new long[] {s},
                    TensorDataType.INT32,
                    WeightFormat.I32,
                    WeightLayout.CONTIGUOUS_LE_V1,
                    segments,
                    s * 4L));
        }

        private TensorHandle upload(
                String name,
                long[] shape,
                TensorDataType type,
                WeightFormat format,
                WeightLayout layout,
                MemorySegment host,
                long bytes) {
            long address = this.gpu.allocate(bytes);
            this.allocations.add(address);
            this.gpu.copyHostToDevice(address, host, bytes);
            return new TensorHandle(name, shape, type, format, layout, address, bytes);
        }

        @Override
        public void close() {
            for (long address : this.allocations) this.gpu.free(address);
        }
    }

    /// The BF16 output of `rows` rows through `format` with `exact` numerics or not.
    private static float[] run(
            CudaGpuMemory gpu,
            Arena arena,
            TrellisFormat format,
            Synthetic object,
            MemorySegment x,
            int rows,
            boolean verify,
            boolean exact) {
        long input = gpu.allocate((long) rows * object.k * 2), output = gpu.allocate((long) rows * object.n * 2);
        long scratchBytes =
                format.scratchBytes(gpu, ScratchUse.TRELLIS_LINEAR, rows, object.k, object.n, object.weights);
        long scratch = gpu.allocate(scratchBytes);
        try {
            gpu.copyHostToDevice(input, x, (long) rows * object.k * 2);
            gpu.selectExactNumerics(exact);
            try {
                format.linear(
                        gpu,
                        new LinearLaunch(
                                ExecutionPlan.Kind.TRELLIS_LINEAR,
                                object.weights,
                                input,
                                output,
                                rows,
                                object.k,
                                object.n,
                                scratch,
                                scratchBytes,
                                verify,
                                false,
                                object.ends));
            } finally {
                gpu.selectExactNumerics(false);
            }
            gpu.synchronize();
            MemorySegment host = arena.allocate((long) rows * object.n * 2, 2);
            gpu.copyDeviceToHost(host, output, (long) rows * object.n * 2);
            float[] values = new float[rows * object.n];
            for (int i = 0; i < values.length; i++) values[i] = bf16Value(host.get(SHORT, i * 2L));
            return values;
        } finally {
            gpu.free(input);
            gpu.free(output);
            gpu.free(scratch);
        }
    }

    private static MemorySegment activations(Arena arena, Random random, int rows, int k) {
        MemorySegment x = arena.allocate((long) rows * k * 2, 2);
        for (long i = 0; i < (long) rows * k; i++) x.set(SHORT, i * 2, bf16((float) random.nextGaussian()));
        return x;
    }

    /// `actual` is within `relative` of `wanted` per element, plus `relative` of the row's rms.
    private static void assertClose(float[] wanted, float[] actual, int n, double relative, String label) {
        int rows = wanted.length / n;
        double worst = 0;
        for (int row = 0; row < rows; row++) {
            double sum = 0;
            for (int c = 0; c < n; c++) sum += (double) wanted[row * n + c] * wanted[row * n + c];
            double rms = Math.sqrt(sum / n);
            assertTrue(rms > 0 && Double.isFinite(rms), label + " row " + row + " has no energy");
            for (int c = 0; c < n; c++) {
                double allowed = relative * (Math.abs(wanted[row * n + c]) + rms);
                double error = Math.abs(wanted[row * n + c] - actual[row * n + c]);
                worst = Math.max(worst, error / allowed);
                assertTrue(
                        error <= allowed,
                        label + " row " + row + " column " + c + ": " + actual[row * n + c] + " against "
                                + wanted[row * n + c]);
            }
        }
        assertTrue(Double.isFinite(worst));
    }

    @Test
    void everyRouteAgreesWithTheScalarReference() {
        TrellisFormat format = new TrellisFormat(new TrellisSettings(32, 140));
        Random random = new Random(7);
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            // n/128 = 4 blocks: ceil(140 / 4) = 35 splits, capped at k/256 = 4.
            try (Synthetic split = new Synthetic(gpu, arena, random, 512, 1024, 128, 512)) {
                assertEquals(4, TrellisFormat.splits(1024, 512, 140));
                MemorySegment x = activations(arena, random, 40, 1024);
                float[] reference = run(gpu, arena, format, split, x, 40, false, true);
                float[] tensorCore = run(gpu, arena, format, split, x, 8, false, false);
                assertClose(Arrays.copyOf(reference, 8 * 512), tensorCore, 512, 0.02, "tensor-core, K split");
                // A verification of 40 rows, over the limit of 32, stays on the same route and equals one-row launches.
                float[] verify = run(gpu, arena, format, split, x, 40, true, false);
                assertClose(reference, verify, 512, 0.02, "verification");
                for (int row : new int[] {0, 17, 39}) {
                    MemorySegment single = arena.allocate(1024 * 2L, 2);
                    MemorySegment.copy(x, row * 1024L * 2, single, 0, 1024 * 2L);
                    float[] one = run(gpu, arena, format, split, single, 1, false, false);
                    assertArrayEquals(one, Arrays.copyOfRange(verify, row * 512, (row + 1) * 512), "row " + row);
                }
                // 40 rows, not a verification, above the limit: decode to BF16 and the row-exact GEMM.
                float[] prefill = run(gpu, arena, format, split, x, 40, false, false);
                assertClose(reference, prefill, 512, 0.04, "prefill");
            }
            // 140 blocks of 128: one split, the epilogue fused into the GEMM.
            try (Synthetic single = new Synthetic(gpu, arena, random, 17920, 1024, 128, 17920)) {
                assertEquals(1, TrellisFormat.splits(1024, 17920, 140));
                MemorySegment x = activations(arena, random, 3, 1024);
                float[] reference = run(gpu, arena, format, single, x, 3, false, true);
                assertClose(reference, run(gpu, arena, format, single, x, 3, false, false), 17920, 0.02, "tensor-core");
            }
        }
    }
}
