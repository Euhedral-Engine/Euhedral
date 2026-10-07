package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bf;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bits;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.GpuMemoryException;
import io.euhedral_execution.inference.core.gpu.KernelArguments;
import io.euhedral_execution.inference.core.gpu.Qwen4Kernel;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The Flash-Next operators on the GPU against their CPU references (Qwen4Reference), on synthetic data: every
/// kernel's arithmetic, the shapes and the chunk-independence of the convolution state. The real weights and the
/// upstream fixtures are in Qwen4FixtureCudaIntegrationTest.
class Qwen4OperatorCudaIntegrationTest {

    private static CudaGpuMemory open() {
        return new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
    }

    private static short[] random(SplittableRandom rng, int count, double scale) {
        short[] values = new short[count];
        for (int i = 0; i < count; i++) {
            double u = rng.nextDouble(), v = rng.nextDouble();
            double normal = Math.sqrt(-2 * Math.log(Math.max(u, 1e-12))) * Math.cos(2 * Math.PI * v);
            values[i] = bits((float) (normal * scale));
        }
        return values;
    }

    private static long upload(CudaGpuMemory gpu, Arena arena, short[] values) {
        long address = gpu.allocate(Math.max(16, (long) values.length * Short.BYTES));
        MemorySegment host = arena.allocate((long) values.length * Short.BYTES, 16);
        MemorySegment.copy(values, 0, host, ValueLayout.JAVA_SHORT, 0, values.length);
        gpu.copyHostToDevice(address, host, (long) values.length * Short.BYTES);
        return address;
    }

    private static long upload(CudaGpuMemory gpu, Arena arena, byte[] values) {
        long address = gpu.allocate(Math.max(16, values.length));
        MemorySegment host = arena.allocate(values.length, 16);
        MemorySegment.copy(values, 0, host, ValueLayout.JAVA_BYTE, 0, values.length);
        gpu.copyHostToDevice(address, host, values.length);
        return address;
    }

    private static short[] download(CudaGpuMemory gpu, Arena arena, long address, int count) {
        MemorySegment host = arena.allocate((long) count * Short.BYTES, 16);
        gpu.copyDeviceToHost(host, address, (long) count * Short.BYTES);
        short[] values = new short[count];
        MemorySegment.copy(host, ValueLayout.JAVA_SHORT, 0, values, 0, count);
        return values;
    }

    /// Elements agree within one BF16 step (the reference accumulates in double, the kernel in FP32 in another
    /// order, so a value on a rounding boundary may land on the other side); `slack` widens it by an absolute
    /// amount per element, for sums that cancel.
    private static void assertNear(short[] expected, short[] actual, double[] slack, String what) {
        assertEquals(expected.length, actual.length, what);
        int off = 0;
        for (int i = 0; i < expected.length; i++) {
            float e = bf(expected[i]), a = bf(actual[i]);
            double bound = Math.abs(e) * 0.0079 + (slack == null ? 0 : slack[i] * 1e-6) + 1e-30;
            if (Math.abs(e - a) > bound)
                throw new AssertionError(what + ": element " + i + " expected " + e + " but was " + a);
            if (expected[i] != actual[i]) off++;
        }
        assertTrue(
                off <= expected.length * 0.02 + 2, what + ": " + off + " of " + expected.length + " elements differ");
    }

    @Test
    void theKernelTableMatchesTheEnum() {
        try (CudaGpuMemory gpu = open()) {
            List<String> names = gpu.tableKernelNames();
            Qwen4Kernel[] kernels = Qwen4Kernel.values();
            assertEquals(kernels.length, names.size());
            for (int i = 0; i < kernels.length; i++) assertEquals(kernels[i].symbol(), names.get(i));
        }
    }

    @Test
    void theLauncherRejectsArgumentsThatDisagreeWithTheKernel() {
        try (CudaGpuMemory gpu = open()) {
            // euhedral_q4_scaled_silu_bf16 takes (ptr, ptr, u32, f32): a missing argument and a wrong width both fail.
            KernelArguments missing =
                    new KernelArguments().pointer(1).pointer(1).int32(1);
            assertThrows(
                    GpuMemoryException.class,
                    () -> gpu.launchTableKernel(Qwen4Kernel.SCALED_SILU_BF16, 1, 1, 1, 32, 1, 1, 0, missing));
            KernelArguments wide =
                    new KernelArguments().pointer(1).pointer(1).pointer(1).float32(1);
            assertThrows(
                    GpuMemoryException.class,
                    () -> gpu.launchTableKernel(Qwen4Kernel.SCALED_SILU_BF16, 1, 1, 1, 32, 1, 1, 0, wide));
        }
    }

    @Test
    void bf16LinearMatchesTheReferenceForDecodeAndPrefillShapes() {
        SplittableRandom rng = new SplittableRandom(11);
        int[][] shapes = {
            {1, 2560, 640}, {1, 320, 10240}, {3, 10240, 320}, {8, 2560, 512}, {9, 2560, 7}, {20, 320, 33}, {1, 2560, 4}
        };
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            for (int[] shape : shapes) {
                int rows = shape[0], k = shape[1], n = shape[2];
                short[] x = random(rng, rows * k, 1.0), w = random(rng, n * k, 1.0 / Math.sqrt(k));
                Qwen4Reference.Linear expected = Qwen4Reference.linear(x, w, rows, k, n);
                long xa = upload(gpu, arena, x), wa = upload(gpu, arena, w), ya = gpu.allocate((long) rows * n * 2);
                try {
                    Qwen4Ops.linearBf16(gpu, xa, wa, ya, rows, k, n);
                    assertNear(
                            expected.output(),
                            download(gpu, arena, ya, rows * n),
                            expected.magnitude(),
                            Arrays.toString(shape));
                } finally {
                    gpu.free(ya);
                    gpu.free(wa);
                    gpu.free(xa);
                }
            }
        }
    }

    @Test
    void groupedRmsNormMatchesTheReference() {
        SplittableRandom rng = new SplittableRandom(12);
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            for (int[] shape : new int[][] {{1, 4, 2560}, {5, 4, 2560}, {3, 1, 128}, {2, 4, 100}}) {
                int rows = shape[0], groups = shape[1], width = shape[2];
                short[] x = random(rng, rows * groups * width, 2.0), w = random(rng, groups * width, 0.3);
                short[] expected = Qwen4Reference.groupedRmsNorm(x, w, rows, groups, width, 1e-6f);
                long xa = upload(gpu, arena, x), wa = upload(gpu, arena, w), ya = gpu.allocate((long) x.length * 2);
                try {
                    Qwen4Ops.groupedRmsNorm(gpu, xa, wa, ya, rows, groups, width, 1e-6f);
                    assertNear(expected, download(gpu, arena, ya, x.length), null, Arrays.toString(shape));
                } finally {
                    gpu.free(ya);
                    gpu.free(wa);
                    gpu.free(xa);
                }
            }
        }
    }

    @Test
    void aGatedResidualMixesAndInjectsAsUpstream() {
        SplittableRandom rng = new SplittableRandom(13);
        int streams = 4, hidden = 256, lowrank = 40;
        int width = streams * hidden;
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            for (int rows : new int[] {1, 2, 17}) {
                short[] state = random(rng, rows * width, 1.5);
                short[] norm = random(rng, width, 0.2);
                short[] down = random(rng, lowrank * width, 1.0 / Math.sqrt(width));
                short[] up = random(rng, width * lowrank, 1.0 / Math.sqrt(lowrank));
                short[] inject = random(rng, streams * width, 1.0 / Math.sqrt(width));
                short[] block = random(rng, rows * hidden, 1.0);

                short[] normed = Qwen4Reference.groupedRmsNorm(state, norm, rows, streams, hidden, 1e-6f);
                short[] lowDown = Qwen4Reference.linear(normed, down, rows, width, lowrank)
                        .output();
                short[] act = Qwen4Reference.scaledSilu(lowDown, streams);
                short[] upOut =
                        Qwen4Reference.linear(act, up, rows, lowrank, width).output();
                short[] mixedExpected = Qwen4Reference.hcMix(normed, upOut, rows, streams, hidden);
                short[] raw = Qwen4Reference.linear(normed, inject, rows, width, streams)
                        .output();
                short[] stateExpected = Qwen4Reference.hcInject(state, block, raw, rows, streams, hidden);

                var connection = new Qwen4HyperConnection(streams, hidden, lowrank, 1e-6f);
                long stateAddress = upload(gpu, arena, state), blockAddress = upload(gpu, arena, block);
                long weightsNorm = upload(gpu, arena, norm), weightsDown = upload(gpu, arena, down);
                long weightsUp = upload(gpu, arena, up), weightsInject = upload(gpu, arena, inject);
                long scratchAddress = gpu.allocate(connection.scratchBytes(rows));
                long mixedAddress = gpu.allocate((long) rows * hidden * 2);
                long outputAddress = gpu.allocate((long) rows * width * 2);
                try {
                    var scratch = connection.scratch(scratchAddress, rows);
                    connection.mix(
                            gpu,
                            new Qwen4HyperConnection.Weights(
                                    Qwen4Weight.of(weightsNorm, 0),
                                    Qwen4Weight.of(weightsDown, 0),
                                    Qwen4Weight.of(weightsUp, 0),
                                    Qwen4Weight.of(weightsInject, 0)),
                            stateAddress,
                            scratch,
                            mixedAddress,
                            rows);
                    assertNear(
                            mixedExpected,
                            download(gpu, arena, mixedAddress, rows * hidden),
                            null,
                            "mixed rows=" + rows);
                    // The block runs on the mixed input; here it is the given block. Inject from the original state.
                    connection.inject(gpu, stateAddress, blockAddress, scratch, outputAddress, rows);
                    assertNear(
                            stateExpected,
                            download(gpu, arena, outputAddress, rows * width),
                            null,
                            "injected rows=" + rows);
                    // In place.
                    connection.inject(gpu, stateAddress, blockAddress, scratch, stateAddress, rows);
                    assertArrayEquals(
                            download(gpu, arena, outputAddress, rows * width),
                            download(gpu, arena, stateAddress, rows * width));
                } finally {
                    for (long address : new long[] {
                        outputAddress,
                        mixedAddress,
                        scratchAddress,
                        weightsInject,
                        weightsUp,
                        weightsDown,
                        weightsNorm,
                        blockAddress,
                        stateAddress
                    }) gpu.free(address);
                }
            }
        }
    }

    @Test
    void theEmbeddingRepeatsOverTheStreams() {
        SplittableRandom rng = new SplittableRandom(14);
        int rows = 3, streams = 4, width = 100;
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            short[] x = random(rng, rows * width, 1.0);
            long xa = upload(gpu, arena, x), ya = gpu.allocate((long) rows * streams * width * 2);
            try {
                Qwen4Ops.repeatStreams(gpu, xa, ya, rows, streams, width);
                short[] actual = download(gpu, arena, ya, rows * streams * width);
                for (int r = 0; r < rows; r++)
                    for (int s = 0; s < streams; s++)
                        assertArrayEquals(
                                Arrays.copyOfRange(x, r * width, (r + 1) * width),
                                Arrays.copyOfRange(actual, (r * streams + s) * width, (r * streams + s + 1) * width));
            } finally {
                gpu.free(ya);
                gpu.free(xa);
            }
        }
    }

    @Test
    void theEmbeddingGatherCopiesTheChosenRows() {
        SplittableRandom rng = new SplittableRandom(18);
        int vocabulary = 300, width = 2560, rows = 9;
        short[] table = random(rng, vocabulary * width, 1.0);
        int[] ids = new int[rows];
        for (int i = 0; i < rows; i++) ids[i] = rng.nextInt(vocabulary);
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            long ta = upload(gpu, arena, table);
            long ia = gpu.allocate(rows * 4L);
            MemorySegment host = arena.allocate(rows * 4L, 16);
            MemorySegment.copy(ids, 0, host, ValueLayout.JAVA_INT, 0, rows);
            gpu.copyHostToDevice(ia, host, rows * 4L);
            long oa = gpu.allocate((long) rows * width * 2);
            try {
                Qwen4Ops.embedding(gpu, ta, ia, oa, rows, width, vocabulary);
                short[] actual = download(gpu, arena, oa, rows * width);
                for (int r = 0; r < rows; r++)
                    assertArrayEquals(
                            Arrays.copyOfRange(table, ids[r] * width, (ids[r] + 1) * width),
                            Arrays.copyOfRange(actual, r * width, (r + 1) * width));
            } finally {
                gpu.free(oa);
                gpu.free(ia);
                gpu.free(ta);
            }
        }
    }

    @Test
    void ngramRecordsExpandToTheNvfp4Values() {
        SplittableRandom rng = new SplittableRandom(15);
        int width = 160, recordBytes = 96, count = 37;
        byte[] records = new byte[count * recordBytes];
        for (int i = 0; i < count; i++) {
            int at = i * recordBytes;
            for (int b = 0; b < width / 2; b++) records[at + b] = (byte) rng.nextInt(256);
            for (int b = 0; b < width / 16; b++) {
                int code;
                do code = rng.nextInt(127); // not the NaN code
                while (false);
                records[at + width / 2 + b] = (byte) code;
            }
            float global = (float) (rng.nextDouble() * 0.01);
            int raw = Float.floatToRawIntBits(global);
            for (int b = 0; b < 4; b++) records[at + recordBytes - 4 + b] = (byte) (raw >> (8 * b));
        }
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            long ra = upload(gpu, arena, records), ya = gpu.allocate((long) count * width * 2);
            try {
                Qwen4Ops.ngramExpand(gpu, ra, ya, count, width, recordBytes);
                short[] actual = download(gpu, arena, ya, count * width);
                for (int i = 0; i < count; i++)
                    assertArrayEquals(
                            Qwen4Reference.expandRecord(records, i * recordBytes, recordBytes, width),
                            Arrays.copyOfRange(actual, i * width, (i + 1) * width),
                            "record " + i);
            } finally {
                gpu.free(ya);
                gpu.free(ra);
            }
        }
    }

    @Test
    void thePleGateMatchesTheReference() {
        SplittableRandom rng = new SplittableRandom(16);
        int streams = 4, width = 320;
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            for (int rows : new int[] {1, 6}) {
                short[] key = random(rng, rows * streams * width, 1.0),
                        query = random(rng, rows * streams * width, 1.0);
                short[] value = random(rng, rows * width, 1.0);
                short[] expected = Qwen4Reference.pleGate(key, query, value, rows, streams, width);
                long ka = upload(gpu, arena, key), qa = upload(gpu, arena, query), va = upload(gpu, arena, value);
                long ya = gpu.allocate((long) rows * streams * width * 2);
                try {
                    Qwen4Ops.pleGate(gpu, ka, qa, va, ya, rows, streams, width);
                    assertNear(expected, download(gpu, arena, ya, expected.length), null, "rows=" + rows);
                } finally {
                    gpu.free(ya);
                    gpu.free(va);
                    gpu.free(qa);
                    gpu.free(ka);
                }
            }
        }
    }

    @Test
    void thePleConvolutionCarriesItsHistoryAcrossChunks() {
        SplittableRandom rng = new SplittableRandom(17);
        int channels = 300, taps = 4, dilation = 3, total = 23, historyRows = (taps - 1) * dilation;
        short[] normed = random(rng, total * channels, 1.0), gated = random(rng, total * channels, 1.0);
        short[] weights = random(rng, channels * taps, 0.5);
        // The reference runs the whole sequence at once from a zero history.
        short[] whole = Qwen4Reference.pleConv(
                normed, gated, new short[historyRows * channels], weights, total, channels, taps, dilation);
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            long na = upload(gpu, arena, normed), ga = upload(gpu, arena, gated), wa = upload(gpu, arena, weights);
            long history = upload(gpu, arena, new short[historyRows * channels]);
            long out = gpu.allocate((long) total * channels * 2);
            try {
                // The kernel, in the chunks the engine uses for prefill and decode, including chunks shorter than the
                // history.
                int[] chunks = {9, 1, 1, 5, 7};
                int row = 0;
                short[] collected = new short[total * channels];
                short[] stateExpected = new short[historyRows * channels];
                for (int rows : chunks) {
                    long offset = (long) row * channels * 2;
                    Qwen4Ops.pleConv(
                            gpu, na + offset, ga + offset, history, wa, out + offset, rows, channels, taps, dilation);
                    Qwen4Ops.convHistory(gpu, na + offset, history, rows, channels, historyRows);
                    short[] chunkNormed = Arrays.copyOfRange(normed, row * channels, (row + rows) * channels);
                    stateExpected = Qwen4Reference.convHistory(chunkNormed, stateExpected, rows, channels, historyRows);
                    assertArrayEquals(
                            stateExpected,
                            download(gpu, arena, history, historyRows * channels),
                            "history after " + (row + rows));
                    row += rows;
                }
                System.arraycopy(download(gpu, arena, out, total * channels), 0, collected, 0, collected.length);
                assertNear(whole, collected, null, "chunked convolution");
            } finally {
                gpu.free(out);
                gpu.free(history);
                gpu.free(wa);
                gpu.free(ga);
                gpu.free(na);
            }
        }
    }
}
