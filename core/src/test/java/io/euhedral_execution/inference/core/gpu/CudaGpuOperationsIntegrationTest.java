package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.sampling.TokenSampler;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.SplittableRandom;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class CudaGpuOperationsIntegrationTest {

    /// Shapes a decode kernel takes (K in whole 1024-value slices, 16-row CTAs) run the contiguous kernels,
    /// within one BF16 step of the scalar reference; every row of a multi-row launch is bit for bit the
    /// one-row result.
    @Test
    void decodeKernelsMatchReferenceAndMultiRowLaunchesMatchOneRowLaunches() throws Throwable {
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Arena arena = Arena.ofConfined()) {
            for (int width : new int[] {1024, 2048}) {
                int outputs = 48;
                byte[] packed = q3Weights(outputs, width);
                int maxRows = 8;
                short[] input = new short[maxRows * width];
                for (int i = 0; i < input.length; i++) input[i] = floatToBf16((i % 23 - 11) * 0.125f);
                long x = upload(gpu, arena, input), w = upload(gpu, arena, packed);
                long y = gpu.allocate((long) maxRows * outputs * Short.BYTES);
                try {
                    short[][] oneRow = new short[maxRows][];
                    for (int row = 0; row < maxRows; row++) {
                        gpu.linearQ3Bf16(x + (long) row * width * Short.BYTES, w, y, 1, width, outputs, packed.length);
                        oneRow[row] = download(gpu, arena, y, outputs);
                        gpu.referenceLinearQ3Bf16(
                                x + (long) row * width * Short.BYTES, w, y, 1, width, outputs, packed.length);
                        assertBf16Equals(download(gpu, arena, y, outputs), oneRow[row], 0.05f);
                    }
                    for (int rows = 2; rows <= maxRows; rows++) {
                        gpu.linearQ3Bf16(x, w, y, rows, width, outputs, packed.length);
                        short[] actual = download(gpu, arena, y, rows * outputs);
                        for (int row = 0; row < rows; row++)
                            assertArrayEquals(
                                    oneRow[row],
                                    java.util.Arrays.copyOfRange(actual, row * outputs, (row + 1) * outputs),
                                    "rows=" + rows + " row=" + row);
                    }
                } finally {
                    gpu.free(y);
                    gpu.free(w);
                    gpu.free(x);
                }
            }
        }
    }

    /// Shapes and row counts no production kernel takes (odd K, partial CTAs, no FP8 route) run the scalar
    /// reference, so the dispatch answers every shape with the reference's bits.
    @Test
    void unqualifiedShapesRunTheScalarReference() throws Throwable {
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Arena arena = Arena.ofConfined()) {
            for (int width : new int[] {65, 192}) {
                for (int rows : new int[] {1, 2, 9, 33, 65}) {
                    for (int outputs : new int[] {35, 65}) {
                        byte[] packed = q3Weights(outputs, width);
                        short[] input = new short[rows * width];
                        for (int i = 0; i < input.length; i++) input[i] = floatToBf16((i % 23 - 11) * 0.125f);
                        long x = upload(gpu, arena, input), w = upload(gpu, arena, packed);
                        long y = gpu.allocate((long) rows * outputs * Short.BYTES);
                        try {
                            gpu.referenceLinearQ3Bf16(x, w, y, rows, width, outputs, packed.length);
                            short[] expected = download(gpu, arena, y, rows * outputs);
                            gpu.linearQ3Bf16(x, w, y, rows, width, outputs, packed.length);
                            assertArrayEquals(
                                    expected,
                                    download(gpu, arena, y, rows * outputs),
                                    "rows=" + rows + " outputs=" + outputs);
                        } finally {
                            gpu.free(y);
                            gpu.free(w);
                            gpu.free(x);
                        }
                    }
                }
            }
        }
    }

    @Test
    void q3RoutesPreserveScaleEdges() {
        try (var gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                var arena = Arena.ofConfined()) {
            for (int width : new int[] {1024, 2048}) {
                int rows = 33, outputs = 128, groups = ((width + 127) / 128) * 2;
                byte[] packed = q3Weights(outputs, width);
                int scaleOffset = (outputs * groups * 24 + 255) & ~255;
                var scales = ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN);
                int[] edgeScales = {0, 1, 0x8001, 0x03ff, 0x0400, 0x3555, 0xb555, 0x7bff};
                for (int out = 0; out < outputs; out++)
                    for (int group = 0; group < groups; group++)
                        scales.putShort(
                                scaleOffset + (out * groups + group) * 2, (short) edgeScales[out % edgeScales.length]);
                short[] input = new short[rows * width];
                for (int i = 0; i < input.length; i++) input[i] = floatToBf16((i % 23 - 11) * 0.125f);
                long x = upload(gpu, arena, input), w = upload(gpu, arena, packed);
                long y = gpu.allocate((long) rows * outputs * Short.BYTES);
                boolean previous = gpu.selectExactNumerics(true);
                try {
                    gpu.referenceLinearQ3Bf16(x, w, y, rows, width, outputs, packed.length);
                    short[] expected = download(gpu, arena, y, rows * outputs);
                    // Exact numerics: the production route matches the scalar reference to one BF16 step.
                    gpu.linearQ3Bf16(x, w, y, rows, width, outputs, packed.length);
                    short[] actual = download(gpu, arena, y, rows * outputs);
                    for (int i = 0; i < actual.length; i++) {
                        float error = Math.abs(bf16ToFloat(expected[i]) - bf16ToFloat(actual[i]));
                        boolean adjacent = (expected[i] < 0) == (actual[i] < 0)
                                && Math.abs((expected[i] & 0xffff) - (actual[i] & 0xffff)) <= 1;
                        assertTrue(Float.isFinite(error) && (error <= 0.001f || adjacent), "scale edge index " + i);
                    }
                    // Relaxed numerics stage the BF16 rounding of each code * scale: the result stays within
                    // 1% of the output RMS of the reference, at every scale edge.
                    gpu.selectExactNumerics(false);
                    gpu.linearQ3Bf16(x, w, y, rows, width, outputs, packed.length);
                    short[] relaxed = download(gpu, arena, y, rows * outputs);
                    double sum = 0;
                    for (short value : expected) sum += (double) bf16ToFloat(value) * bf16ToFloat(value);
                    double rms = Math.sqrt(sum / expected.length);
                    for (int i = 0; i < relaxed.length; i++) {
                        float error = Math.abs(bf16ToFloat(expected[i]) - bf16ToFloat(relaxed[i]));
                        assertTrue(
                                Float.isFinite(error) && error <= 0.01 * rms + 0.001, "relaxed scale edge index " + i);
                    }
                } finally {
                    gpu.selectExactNumerics(previous);
                    gpu.free(y);
                    gpu.free(w);
                    gpu.free(x);
                }
            }
        }
    }

    @Test
    void q3LinearReferenceDecodesSubnormalScale() {
        int width = 128;
        byte[] weights = q3Weights(1, width);
        int scaleOffset = (2 * 24 + 255) & ~255;
        ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN).putShort(scaleOffset, (short) 1);
        short[] input = new short[width];
        input[0] = floatToBf16(1.0f);

        assertEquals(
                floatToBf16(-4.0f * Float.float16ToFloat((short) 1)), linearReference(input, weights, 1, width, 1)[0]);
    }

    @Test
    void bf16RmsNormMatchesCpuReference() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        assertTrue(Files.isRegularFile(libraryPath));
        int rows = 2;
        int width = 128;
        float epsilon = 1.0e-5f;
        short[] input = new short[rows * width];
        short[] weight = new short[width];
        for (int i = 0; i < input.length; i++) input[i] = floatToBf16((i % 19 - 9) * 0.125f);
        for (int i = 0; i < weight.length; i++) weight[i] = floatToBf16(0.75f + (i % 7) * 0.03125f);
        short[] expected = rmsNorm(input, weight, rows, width, epsilon);

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena arena = Arena.ofConfined()) {
            long inputDevice = upload(gpu, arena, input);
            long weightDevice = upload(gpu, arena, weight);
            long outputDevice = gpu.allocate((long) expected.length * 2);
            try {
                gpu.rmsNormBf16(inputDevice, weightDevice, outputDevice, rows, width, epsilon);
                short[] actual = download(gpu, arena, outputDevice, expected.length);
                assertBf16Equals(expected, actual, 0.002f);
            } finally {
                gpu.free(outputDevice);
                gpu.free(weightDevice);
                gpu.free(inputDevice);
            }
        }
    }

    @Test
    void deviceArgmaxSelectsTheHostArgmaxIncludingTiesAndSpecialValues() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        int vocabulary = 248_320;
        SplittableRandom random = new SplittableRandom(91);
        short[] row = new short[vocabulary];
        for (int i = 0; i < row.length; i++) row[i] = floatToBf16((float) random.nextGaussian() * 3.0f);
        row[17] = (short) 0x7FC1; // NaN never wins
        row[1000] = (short) 0xFF80; // negative infinity never wins
        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena arena = Arena.ofConfined()) {
            long result = gpu.allocate(Long.BYTES);
            try {
                for (int peak : new int[] {vocabulary - 1, 151_000, 40, 0}) {
                    row[peak] = floatToBf16(40.0f); // equal maxima: the lowest token ID wins
                    float[] logits = new float[vocabulary];
                    for (int i = 0; i < row.length; i++) logits[i] = Float.intBitsToFloat(row[i] << 16);
                    int expected = new TokenSampler(GenerationConfig.greedy(1L), vocabulary).selectToken(logits);
                    long device = upload(gpu, arena, row);
                    try {
                        assertTrue(gpu.argmaxBf16(device, vocabulary, result));
                        MemorySegment key = arena.allocate(Long.BYTES);
                        gpu.copyDeviceToHost(key, result, Long.BYTES);
                        long bits = key.get(ValueLayout.JAVA_LONG, 0);
                        assertEquals(expected, (int) (0xFFFF_FFFFL - (bits & 0xFFFF_FFFFL)), "peak " + peak);
                    } finally {
                        gpu.free(device);
                    }
                }
            } finally {
                gpu.free(result);
            }
        }
    }

    @Test
    void firstRmsNormOnAnotherWorkerInitializesCudaContext() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        int width = 128;
        short[] input = new short[width];
        short[] weight = new short[width];
        for (int col = 0; col < width; col++) {
            input[col] = floatToBf16((col % 13 - 6) * 0.125f);
            weight[col] = floatToBf16(0.75f);
        }
        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena arena = Arena.ofConfined();
                var worker = Executors.newSingleThreadExecutor()) {
            long inputDevice = upload(gpu, arena, input);
            long weightDevice = upload(gpu, arena, weight);
            long outputDevice = gpu.allocate((long) width * Short.BYTES);
            try {
                worker.submit(() -> gpu.rmsNormBf16(inputDevice, weightDevice, outputDevice, 1, width, 1.0e-5f))
                        .get(10, TimeUnit.SECONDS);
                assertBf16Equals(
                        rmsNorm(input, weight, 1, width, 1.0e-5f), download(gpu, arena, outputDevice, width), 0.002f);
            } finally {
                gpu.free(outputDevice);
                gpu.free(weightDevice);
                gpu.free(inputDevice);
            }
        }
    }

    @Test
    void q3RowSplitLinearMatchesCpuReference() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        int rows = 2;
        int inFeatures = 128;
        int outFeatures = 3;
        short[] input = new short[rows * inFeatures];
        for (int i = 0; i < input.length; i++) input[i] = floatToBf16((i % 13 - 6) * 0.2f);
        byte[] weights = q3Weights(outFeatures, inFeatures);
        short[] expected = linearReference(input, weights, rows, inFeatures, outFeatures);

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena arena = Arena.ofConfined()) {
            long inputDevice = upload(gpu, arena, input);
            long weightsDevice = upload(gpu, arena, weights);
            long outputDevice = gpu.allocate((long) expected.length * 2);
            try {
                gpu.linearQ3Bf16(
                        inputDevice, weightsDevice, outputDevice, rows, inFeatures, outFeatures, weights.length);
                short[] actual = download(gpu, arena, outputDevice, expected.length);
                assertBf16Equals(expected, actual, 0.02f);
            } finally {
                gpu.free(outputDevice);
                gpu.free(weightsDevice);
                gpu.free(inputDevice);
            }
        }
    }

    @Test
    void q3LinearHandlesZeroAndSubnormalScalesOnGpu() {
        int width = 128;
        byte[] weights = q3Weights(2, width);
        int scaleOffset = (2 * 2 * 24 + 255) & ~255;
        ByteBuffer scales = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
        scales.putShort(scaleOffset, (short) 0);
        scales.putShort(scaleOffset + 2 * Short.BYTES, (short) 1);
        short[] input = new short[width];
        input[0] = floatToBf16(1.0f);
        short[] expected = linearReference(input, weights, 1, width, 2);

        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Arena arena = Arena.ofConfined()) {
            long inputDevice = upload(gpu, arena, input);
            long weightsDevice = upload(gpu, arena, weights);
            long outputDevice = gpu.allocate((long) expected.length * Short.BYTES);
            try {
                gpu.linearQ3Bf16(inputDevice, weightsDevice, outputDevice, 1, width, 2, weights.length);
                assertBf16Equals(expected, download(gpu, arena, outputDevice, expected.length), 1.0e-8f);
            } finally {
                gpu.free(outputDevice);
                gpu.free(weightsDevice);
                gpu.free(inputDevice);
            }
        }
    }

    @Test
    void paddedQ3LinearOnAnotherWorkerMatchesCpuReference() throws Exception {
        int width = 192;
        int outputs = 5;
        short[] input = new short[width];
        for (int col = 0; col < width; col++) input[col] = floatToBf16((col % 17 - 8) * 0.125f);
        byte[] weights = q3Weights(outputs, width);
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Arena arena = Arena.ofConfined();
                var worker = Executors.newSingleThreadExecutor()) {
            long inputDevice = upload(gpu, arena, input);
            long weightDevice = upload(gpu, arena, weights);
            long outputDevice = gpu.allocate((long) outputs * Short.BYTES);
            try {
                worker.submit(() -> gpu.linearQ3Bf16(
                                inputDevice, weightDevice, outputDevice, 1, width, outputs, weights.length))
                        .get(10, TimeUnit.SECONDS);
                assertBf16Equals(
                        linearReference(input, weights, 1, width, outputs),
                        download(gpu, arena, outputDevice, outputs),
                        0.03f);
            } finally {
                gpu.free(outputDevice);
                gpu.free(weightDevice);
                gpu.free(inputDevice);
            }
        }
    }

    static long upload(CudaGpuMemory gpu, Arena arena, short[] values) {
        MemorySegment host = arena.allocate((long) values.length * 2, 2);
        ByteBuffer buffer = host.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        for (short value : values) buffer.putShort(value);
        long device = gpu.allocate(host.byteSize());
        gpu.copyHostToDevice(device, host, host.byteSize());
        return device;
    }

    static long upload(CudaGpuMemory gpu, Arena arena, byte[] values) {
        MemorySegment host = arena.allocate(values.length, 1);
        host.asByteBuffer().put(values);
        long device = gpu.allocate(values.length);
        gpu.copyHostToDevice(device, host, values.length);
        return device;
    }

    static short[] download(CudaGpuMemory gpu, Arena arena, long device, int count) {
        MemorySegment host = arena.allocate((long) count * 2, 2);
        gpu.copyDeviceToHost(host, device, host.byteSize());
        ByteBuffer buffer = host.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        short[] result = new short[count];
        for (int i = 0; i < count; i++) result[i] = buffer.getShort();
        return result;
    }

    static void assertBf16Equals(short[] expected, short[] actual, float tolerance) {
        for (int i = 0; i < expected.length; i++) {
            float difference = Math.abs(bf16ToFloat(expected[i]) - bf16ToFloat(actual[i]));
            assertTrue(difference <= tolerance, "index " + i + ": " + difference);
        }
    }

    static short[] rmsNorm(short[] input, short[] weight, int rows, int width, float epsilon) {
        short[] result = new short[input.length];
        for (int row = 0; row < rows; row++) {
            float sum = 0;
            for (int col = 0; col < width; col++) {
                float value = bf16ToFloat(input[row * width + col]);
                sum += value * value;
            }
            float inverse = (float) (1.0 / Math.sqrt(sum / width + epsilon));
            for (int col = 0; col < width; col++) {
                result[row * width + col] =
                        floatToBf16(bf16ToFloat(input[row * width + col]) * inverse * bf16ToFloat(weight[col]));
            }
        }
        return result;
    }

    static byte[] q3Weights(int rows, int width) {
        int groups = ((width + 127) / 128) * 2;
        int base = rows * groups * 24;
        int scaleOffset = (base + 255) & ~255;
        byte[] payload = new byte[scaleOffset + rows * groups * 2];
        ByteBuffer scales = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < rows; row++) {
            for (int group = 0; group < groups; group++) {
                int offset = scaleOffset + (row * groups + group) * 2;
                scales.putShort(offset, floatToFp16(0.125f * (group + 1)));
                for (int k = 0; k < 64; k++) {
                    int code = ((row * 11 + group * 5 + k) % 8) - 4;
                    int bit = k * 3;
                    int byteOffset = row * groups * 24 + group * 24 + (bit >>> 3);
                    int shift = bit & 7;
                    int packed = (code & 7) << shift;
                    payload[byteOffset] |= (byte) packed;
                    if (shift > 5) payload[byteOffset + 1] |= (byte) (packed >>> 8);
                }
            }
        }
        return payload;
    }

    static short[] linearReference(short[] input, byte[] weights, int rows, int width, int outputs) {
        int groups = ((width + 127) / 128) * 2;
        int scaleOffset = (outputs * groups * 24 + 255) & ~255;
        short[] result = new short[rows * outputs];
        ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < rows; row++)
            for (int out = 0; out < outputs; out++) {
                float sum = 0;
                for (int k = 0; k < width; k++) {
                    int group = k / 64;
                    int bit = (k % 64) * 3;
                    int offset = out * groups * 24 + group * 24 + (bit >>> 3);
                    int packed = (weights[offset] & 255) | ((weights[offset + 1] & 255) << 8);
                    int code = (packed >>> (bit & 7)) & 7;
                    if (code >= 4) code -= 8;
                    float scale = Float.float16ToFloat(buffer.getShort(scaleOffset + (out * groups + group) * 2));
                    sum += bf16ToFloat(input[row * width + k]) * code * scale;
                }
                result[row * outputs + out] = floatToBf16(sum);
            }
        return result;
    }

    static float bf16ToFloat(short value) {
        return Float.intBitsToFloat((value & 0xffff) << 16);
    }

    static short floatToBf16(float value) {
        int bits = Float.floatToRawIntBits(value);
        return (short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16);
    }

    private static short floatToFp16(float value) {
        int bits = Float.floatToRawIntBits(value), sign = (bits >>> 16) & 0x8000;
        int exponent = ((bits >>> 23) & 255) - 112, mantissa = (bits >>> 13) & 1023;
        return (short) (sign | Math.max(0, Math.min(31, exponent)) << 10 | mantissa);
    }
}
