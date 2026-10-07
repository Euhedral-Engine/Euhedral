package io.euhedral_execution.inference.core.gpu;

import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.assertBf16Equals;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.bf16ToFloat;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.download;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.floatToBf16;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.upload;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class LayerGpuOperationsIntegrationTest {

    @Test
    @Timeout(5)
    void standaloneOperationsForTheLoadedGdnLayerAreInTheGpuContract() {
        assertDoesNotThrow(() -> ExecutionGpu.class.getMethod(
                "linearQ4Bf16", long.class, long.class, long.class, int.class, int.class, int.class, long.class));
        assertDoesNotThrow(() -> ExecutionGpu.class.getMethod(
                "linearQ5Bf16", long.class, long.class, long.class, int.class, int.class, int.class, long.class));
        assertDoesNotThrow(() -> ExecutionGpu.class.getMethod(
                "linearBf16ToFloat", long.class, long.class, long.class, int.class, int.class, int.class));
        assertDoesNotThrow(() -> ExecutionGpu.class.getMethod(
                "gdnControlFp32",
                long.class,
                long.class,
                long.class,
                long.class,
                long.class,
                long.class,
                int.class,
                int.class));
        assertDoesNotThrow(() -> ExecutionGpu.class.getMethod(
                "gdnConvolutionBf16",
                long.class,
                long.class,
                long.class,
                long.class,
                long.class,
                int.class,
                int.class,
                int.class,
                int.class,
                int.class));
        assertDoesNotThrow(() -> ExecutionGpu.class.getMethod(
                "gdnRecurrenceBf16",
                long.class,
                long.class,
                long.class,
                long.class,
                long.class,
                int.class,
                int.class,
                int.class,
                int.class,
                int.class,
                float.class));
        assertDoesNotThrow(() -> ExecutionGpu.class.getMethod(
                "gdnGatedRmsNormBf16",
                long.class,
                long.class,
                long.class,
                long.class,
                int.class,
                int.class,
                int.class,
                float.class));
        assertDoesNotThrow(() -> ExecutionGpu.class.getMethod(
                "residualAddBf16", long.class, long.class, long.class, int.class, int.class));
        assertDoesNotThrow(
                () -> ExecutionGpu.class.getMethod("swiGluBf16", long.class, long.class, int.class, int.class));
        assertDoesNotThrow(() -> ExecutionGpu.class.getMethod("zeroDeviceMemory", long.class, long.class));
    }

    @Test
    void unitOffsetRmsNormMatchesIndependentCpuReference() throws Exception {
        Method operation = assertDoesNotThrow(() -> ExecutionGpu.class.getMethod(
                "rmsNormUnitOffsetBf16", long.class, long.class, long.class, int.class, int.class, float.class));
        int rows = 2;
        int width = 128;
        short[] input = new short[rows * width];
        short[] weight = new short[width];
        for (int index = 0; index < input.length; index++) {
            input[index] = floatToBf16((index % 31 - 15) * 0.125f);
        }
        for (int index = 0; index < weight.length; index++) {
            weight[index] = floatToBf16((index % 17 - 8) * 0.03125f);
        }
        short[] expected = rmsNormUnitOffsetReference(input, weight, rows, width, 1.0e-6f);

        try (CudaGpuMemory gpu = new CudaGpuMemory(cudaLibrary());
                Arena arena = Arena.ofConfined()) {
            long inputAddress = upload(gpu, arena, input);
            long weightAddress = upload(gpu, arena, weight);
            long outputAddress = gpu.allocate((long) input.length * Short.BYTES);
            try {
                operation.invoke(gpu, inputAddress, weightAddress, outputAddress, rows, width, 1.0e-6f);
                assertBf16Equals(expected, download(gpu, arena, outputAddress, input.length), 0.002f);
            } finally {
                gpu.free(outputAddress);
                gpu.free(weightAddress);
                gpu.free(inputAddress);
            }
        }
    }

    @Test
    void q4AndQ5LinearMatchIndependentCpuDequantization() throws Exception {
        Path library = cudaLibrary();
        int rows = 2;
        int width = 128;
        int outputs = 5;
        short[] input = new short[rows * width];
        for (int i = 0; i < input.length; i++) input[i] = floatToBf16((i % 23 - 11) * 0.0625f);

        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Arena arena = Arena.ofConfined()) {
            long inputAddress = upload(gpu, arena, input);
            long outputAddress = gpu.allocate((long) rows * outputs * Short.BYTES);
            try {
                for (int bits : new int[] {4, 5}) {
                    byte[] weights = quantizedWeights(outputs, width, bits);
                    long weightAddress = upload(gpu, arena, weights);
                    try {
                        if (bits == 4) {
                            gpu.linearQ4Bf16(
                                    inputAddress, weightAddress, outputAddress, rows, width, outputs, weights.length);
                        } else {
                            gpu.linearQ5Bf16(
                                    inputAddress, weightAddress, outputAddress, rows, width, outputs, weights.length);
                        }
                        assertBf16Equals(
                                quantizedLinearReference(input, weights, rows, width, outputs, bits),
                                download(gpu, arena, outputAddress, rows * outputs),
                                0.025f);
                    } finally {
                        gpu.free(weightAddress);
                    }
                }
            } finally {
                gpu.free(outputAddress);
                gpu.free(inputAddress);
            }
        }
    }

    @Test
    void q4AndQ5TiledLinearMatchIndependentCpuOnPartialTiles() throws Exception {
        try (CudaGpuMemory gpu = new CudaGpuMemory(cudaLibrary());
                Arena arena = Arena.ofConfined()) {
            for (int bits : new int[] {4, 5}) {
                for (int rows : new int[] {1, 4, 8, 9, 16, 32, 33, 256}) {
                    int width = 128;
                    int outputs = 37;
                    short[] input = new short[rows * width];
                    for (int i = 0; i < input.length; i++) input[i] = floatToBf16((i % 23 - 11) * 0.0625f);
                    byte[] weights = quantizedWeights(outputs, width, bits);
                    long inputAddress = upload(gpu, arena, input);
                    long weightAddress = upload(gpu, arena, weights);
                    long outputAddress = gpu.allocate((long) rows * outputs * Short.BYTES);
                    try {
                        if (bits == 4) {
                            gpu.linearQ4Bf16(
                                    inputAddress, weightAddress, outputAddress, rows, width, outputs, weights.length);
                        } else {
                            gpu.linearQ5Bf16(
                                    inputAddress, weightAddress, outputAddress, rows, width, outputs, weights.length);
                        }
                        assertBf16Equals(
                                quantizedLinearReference(input, weights, rows, width, outputs, bits),
                                download(gpu, arena, outputAddress, rows * outputs),
                                0.025f);
                    } finally {
                        gpu.free(outputAddress);
                        gpu.free(weightAddress);
                        gpu.free(inputAddress);
                    }
                }
            }
        }
    }

    @Test
    void q4AndQ5LinearAcceptTwoByteAlignedWeightPayload() throws Exception {
        // A valid packed payload need not start on a 4-byte boundary; every row
        // count that selects a different optimized route must still be correct.
        try (CudaGpuMemory gpu = new CudaGpuMemory(cudaLibrary());
                Arena arena = Arena.ofConfined()) {
            for (int bits : new int[] {4, 5}) {
                for (int rows : new int[] {1, 2, 4, 9, 33, 64}) {
                    int width = 256;
                    int outputs = 37;
                    short[] input = new short[rows * width];
                    for (int i = 0; i < input.length; i++) input[i] = floatToBf16((i % 23 - 11) * 0.0625f);
                    byte[] weights = quantizedWeights(outputs, width, bits);
                    MemorySegment host = arena.allocate(weights.length);
                    MemorySegment.copy(weights, 0, host, ValueLayout.JAVA_BYTE, 0, weights.length);
                    long inputAddress = upload(gpu, arena, input);
                    long allocation = gpu.allocate(weights.length + 4L);
                    long weightAddress = allocation + 2;
                    long outputAddress = gpu.allocate((long) rows * outputs * Short.BYTES);
                    try {
                        gpu.copyHostToDevice(weightAddress, host, weights.length);
                        if (bits == 4) {
                            gpu.linearQ4Bf16(
                                    inputAddress, weightAddress, outputAddress, rows, width, outputs, weights.length);
                        } else {
                            gpu.linearQ5Bf16(
                                    inputAddress, weightAddress, outputAddress, rows, width, outputs, weights.length);
                        }
                        assertBf16Equals(
                                quantizedLinearReference(input, weights, rows, width, outputs, bits),
                                download(gpu, arena, outputAddress, rows * outputs),
                                0.025f);
                    } finally {
                        gpu.free(outputAddress);
                        gpu.free(allocation);
                        gpu.free(inputAddress);
                    }
                }
            }
        }
    }

    @Test
    void bf16LinearPreservesFp32ControlProjectionResults() throws Exception {
        int rows = 2;
        int width = 128;
        int outputs = 3;
        short[] input = new short[rows * width];
        short[] weights = new short[outputs * width];
        for (int i = 0; i < input.length; i++) input[i] = floatToBf16((i % 17 - 8) * 0.03125f);
        for (int i = 0; i < weights.length; i++) weights[i] = floatToBf16((i % 11 - 5) * 0.015625f);
        float[] expected = new float[rows * outputs];
        for (int row = 0; row < rows; row++) {
            for (int output = 0; output < outputs; output++) {
                float sum = 0.0f;
                for (int k = 0; k < width; k++) {
                    sum += bf16ToFloat(input[row * width + k]) * bf16ToFloat(weights[output * width + k]);
                }
                expected[row * outputs + output] = sum;
            }
        }

        try (CudaGpuMemory gpu = new CudaGpuMemory(cudaLibrary());
                Arena arena = Arena.ofConfined()) {
            long inputAddress = upload(gpu, arena, input);
            long weightAddress = upload(gpu, arena, weights);
            long outputAddress = gpu.allocate((long) expected.length * Float.BYTES);
            try {
                gpu.linearBf16ToFloat(inputAddress, weightAddress, outputAddress, rows, width, outputs);
                assertFloatEquals(expected, downloadFloats(gpu, arena, outputAddress, expected.length), 0.0002f);
            } finally {
                gpu.free(outputAddress);
                gpu.free(weightAddress);
                gpu.free(inputAddress);
            }
        }
    }

    @Test
    void gdnControlMatchesSoftplusAndSigmoidReference() throws Exception {
        int rows = 2;
        int heads = 3;
        float[] a = {-1.5f, -0.25f, 0.5f, 1.0f, -2.0f, 3.0f};
        float[] b = {-3.0f, 0.0f, 2.0f, 1.0f, -1.0f, 0.5f};
        float[] aLog = {-1.0f, -0.25f, 0.2f};
        float[] dtBias = {0.1f, -0.2f, 0.3f};
        float[] expectedAlpha = new float[a.length];
        float[] expectedBeta = new float[b.length];
        for (int i = 0; i < a.length; i++) {
            float shifted = a[i] + dtBias[i % heads];
            expectedAlpha[i] = (float) Math.exp((float) (-Math.exp(aLog[i % heads]) * softplus(shifted)));
            expectedBeta[i] = (float) (1.0 / (1.0 + Math.exp(-b[i])));
        }

        try (CudaGpuMemory gpu = new CudaGpuMemory(cudaLibrary());
                Arena arena = Arena.ofConfined()) {
            long aAddress = uploadFloats(gpu, arena, a);
            long bAddress = uploadFloats(gpu, arena, b);
            long aLogAddress = uploadFloats(gpu, arena, aLog);
            long dtBiasAddress = uploadFloats(gpu, arena, dtBias);
            long alphaAddress = gpu.allocate((long) a.length * Float.BYTES);
            long betaAddress = gpu.allocate((long) b.length * Float.BYTES);
            try {
                gpu.gdnControlFp32(
                        aAddress, bAddress, aLogAddress, dtBiasAddress, alphaAddress, betaAddress, rows, heads);
                assertFloatEquals(expectedAlpha, downloadFloats(gpu, arena, alphaAddress, a.length), 0.00002f);
                assertFloatEquals(expectedBeta, downloadFloats(gpu, arena, betaAddress, b.length), 0.00002f);
            } finally {
                gpu.free(betaAddress);
                gpu.free(alphaAddress);
                gpu.free(dtBiasAddress);
                gpu.free(aLogAddress);
                gpu.free(bAddress);
                gpu.free(aAddress);
            }
        }
    }

    @Test
    void gdnCausalConvolutionMatchesReferenceAndCarriesTheTail() throws Exception {
        int rows = 3;
        int queryKeyWidth = 256;
        int valueWidth = 128;
        int channels = queryKeyWidth + valueWidth;
        int kernel = 4;
        short[] queryKey = new short[rows * queryKeyWidth];
        short[] valueZ = new short[rows * valueWidth * 2];
        short[] weights = new short[kernel * channels];
        short[] initialState = new short[channels * (kernel - 1)];
        for (int i = 0; i < queryKey.length; i++) queryKey[i] = floatToBf16((i % 29 - 14) * 0.015625f);
        for (int i = 0; i < valueZ.length; i++) valueZ[i] = floatToBf16((i % 31 - 15) * 0.015625f);
        for (int i = 0; i < weights.length; i++) weights[i] = floatToBf16((i % 13 - 6) * 0.03125f);
        for (int i = 0; i < initialState.length; i++) initialState[i] = floatToBf16((i % 7 - 3) * 0.0625f);
        short[][] expected =
                convolutionReference(queryKey, valueZ, weights, initialState, rows, queryKeyWidth, valueWidth, kernel);

        try (CudaGpuMemory gpu = new CudaGpuMemory(cudaLibrary());
                Arena arena = Arena.ofConfined()) {
            long queryAddress = upload(gpu, arena, queryKey);
            long valueZAddress = upload(gpu, arena, valueZ);
            long weightsAddress = upload(gpu, arena, weights);
            long stateAddress = upload(gpu, arena, initialState);
            long outputAddress = gpu.allocate((long) rows * channels * Short.BYTES);
            try {
                gpu.gdnConvolutionBf16(
                        queryAddress,
                        valueZAddress,
                        weightsAddress,
                        stateAddress,
                        outputAddress,
                        rows,
                        queryKeyWidth,
                        valueWidth,
                        channels,
                        kernel);
                assertBf16Equals(expected[0], download(gpu, arena, outputAddress, rows * channels), 0.004f);
                assertBf16Equals(expected[1], download(gpu, arena, stateAddress, channels * (kernel - 1)), 0.0f);
            } finally {
                gpu.free(outputAddress);
                gpu.free(stateAddress);
                gpu.free(weightsAddress);
                gpu.free(valueZAddress);
                gpu.free(queryAddress);
            }
        }
    }

    @Test
    void gdnRecurrenceMatchesNormalizedGroupedHeadReference() throws Exception {
        int rows = 2;
        int keyHeads = 1;
        int valueHeads = 3;
        int headDim = 128;
        int queryKeyWidth = 2 * keyHeads * headDim;
        int valueWidth = valueHeads * headDim;
        int convolvedWidth = queryKeyWidth + valueWidth;
        float scale = (float) (1.0 / Math.sqrt(headDim));
        short[] convolved = new short[rows * convolvedWidth];
        float[] g = new float[rows * valueHeads];
        float[] beta = new float[rows * valueHeads];
        float[] initialState = new float[valueHeads * headDim * headDim];
        for (int i = 0; i < convolved.length; i++) convolved[i] = floatToBf16((i % 37 - 18) * 0.0078125f);
        for (int i = 0; i < g.length; i++) {
            g[i] = (float) Math.exp(-0.1f - (i % 5) * 0.03f);
            beta[i] = 0.2f + (i % 7) * 0.08f;
        }
        for (int i = 0; i < initialState.length; i++) initialState[i] = (i % 19 - 9) * 0.0005f;
        RecurrenceResult expected =
                recurrenceReference(convolved, g, beta, initialState, rows, keyHeads, valueHeads, headDim, scale);

        try (CudaGpuMemory gpu = new CudaGpuMemory(cudaLibrary());
                Arena arena = Arena.ofConfined()) {
            long convolvedAddress = upload(gpu, arena, convolved);
            long alphaAddress = uploadFloats(gpu, arena, g);
            long betaAddress = uploadFloats(gpu, arena, beta);
            long stateAddress = uploadFloats(gpu, arena, initialState);
            long outputAddress = gpu.allocate((long) rows * valueWidth * Short.BYTES);
            try {
                gpu.gdnRecurrenceBf16(
                        convolvedAddress,
                        alphaAddress,
                        betaAddress,
                        stateAddress,
                        outputAddress,
                        rows,
                        keyHeads,
                        valueHeads,
                        headDim,
                        headDim,
                        scale);
                assertBf16Equals(expected.output(), download(gpu, arena, outputAddress, rows * valueWidth), 0.008f);
                assertFloatEquals(
                        expected.state(), downloadFloats(gpu, arena, stateAddress, initialState.length), 0.0001f);
            } finally {
                gpu.free(outputAddress);
                gpu.free(stateAddress);
                gpu.free(betaAddress);
                gpu.free(alphaAddress);
                gpu.free(convolvedAddress);
            }
        }
    }

    @Test
    void gdnGatedRmsNormMatchesPerHeadReference() throws Exception {
        int rows = 2;
        int heads = 3;
        int headDim = 128;
        int width = heads * headDim;
        float epsilon = 1.0e-5f;
        short[] recurrent = new short[rows * width];
        short[] valueZ = new short[rows * width * 2];
        short[] weight = new short[headDim];
        for (int i = 0; i < recurrent.length; i++) recurrent[i] = floatToBf16((i % 23 - 11) * 0.015625f);
        for (int i = 0; i < valueZ.length; i++) valueZ[i] = floatToBf16((i % 17 - 8) * 0.0625f);
        for (int i = 0; i < weight.length; i++) weight[i] = floatToBf16(0.8f + (i % 9) * 0.02f);
        short[] expected = gatedNormReference(recurrent, valueZ, weight, rows, heads, headDim, epsilon);

        try (CudaGpuMemory gpu = new CudaGpuMemory(cudaLibrary());
                Arena arena = Arena.ofConfined()) {
            long recurrentAddress = upload(gpu, arena, recurrent);
            long valueZAddress = upload(gpu, arena, valueZ);
            long weightAddress = upload(gpu, arena, weight);
            long outputAddress = gpu.allocate((long) recurrent.length * Short.BYTES);
            try {
                gpu.gdnGatedRmsNormBf16(
                        recurrentAddress, valueZAddress, weightAddress, outputAddress, rows, heads, headDim, epsilon);
                assertBf16Equals(expected, download(gpu, arena, outputAddress, recurrent.length), 0.003f);
            } finally {
                gpu.free(outputAddress);
                gpu.free(weightAddress);
                gpu.free(valueZAddress);
                gpu.free(recurrentAddress);
            }
        }
    }

    @Test
    void residualAddAndSwiGluMatchCpuReferences() throws Exception {
        int rows = 2;
        int width = 128;
        short[] residual = new short[rows * width];
        short[] delta = new short[rows * width];
        short[] expectedResidual = new short[rows * width];
        for (int i = 0; i < residual.length; i++) {
            residual[i] = floatToBf16((i % 19 - 9) * 0.125f);
            delta[i] = floatToBf16((i % 13 - 6) * 0.0625f);
            expectedResidual[i] = floatToBf16(bf16ToFloat(residual[i]) + bf16ToFloat(delta[i]));
        }
        short[] gateUp = new short[rows * width * 2];
        short[] expectedSwiGlu = new short[rows * width];
        for (int i = 0; i < rows * width; i++) {
            gateUp[i] = floatToBf16((i % 17 - 8) * 0.125f);
            gateUp[rows * width + i] = floatToBf16((i % 11 - 5) * 0.25f);
        }
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < width; col++) {
                float gate = bf16ToFloat(gateUp[row * 2 * width + col]);
                float up = bf16ToFloat(gateUp[row * 2 * width + width + col]);
                expectedSwiGlu[row * width + col] = floatToBf16(gate * sigmoid(gate) * up);
            }
        }

        try (CudaGpuMemory gpu = new CudaGpuMemory(cudaLibrary());
                Arena arena = Arena.ofConfined()) {
            long residualAddress = upload(gpu, arena, residual);
            long deltaAddress = upload(gpu, arena, delta);
            long sumAddress = gpu.allocate((long) residual.length * Short.BYTES);
            long gateUpAddress = upload(gpu, arena, gateUp);
            long activationAddress = gpu.allocate((long) rows * width * Short.BYTES);
            try {
                gpu.residualAddBf16(residualAddress, deltaAddress, sumAddress, rows, width);
                gpu.swiGluBf16(gateUpAddress, activationAddress, rows, width);
                assertBf16Equals(expectedResidual, download(gpu, arena, sumAddress, residual.length), 0.0f);
                assertBf16Equals(expectedSwiGlu, download(gpu, arena, activationAddress, rows * width), 0.002f);
            } finally {
                gpu.free(activationAddress);
                gpu.free(gateUpAddress);
                gpu.free(sumAddress);
                gpu.free(deltaAddress);
                gpu.free(residualAddress);
            }
        }
    }

    @Test
    void zeroDeviceMemoryInitializesSequenceState() throws Exception {
        byte[] nonzero = new byte[1024];
        java.util.Arrays.fill(nonzero, (byte) 0x7f);
        try (CudaGpuMemory gpu = new CudaGpuMemory(cudaLibrary());
                Arena arena = Arena.ofConfined()) {
            long device = upload(gpu, arena, nonzero);
            try {
                gpu.zeroDeviceMemory(device, nonzero.length);
                MemorySegment output = arena.allocate(nonzero.length, 1);
                gpu.copyDeviceToHost(output, device, nonzero.length);
                for (long i = 0; i < output.byteSize(); i++)
                    assertTrue(output.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i) == 0);
            } finally {
                gpu.free(device);
            }
        }
    }

    private static Path cudaLibrary() {
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        assertTrue(Files.isRegularFile(library));
        return library;
    }

    private static short[] rmsNormUnitOffsetReference(
            short[] input, short[] weight, int rows, int width, float epsilon) {
        short[] output = new short[input.length];
        for (int row = 0; row < rows; row++) {
            float squares = 0.0f;
            for (int column = 0; column < width; column++) {
                float value = bf16ToFloat(input[row * width + column]);
                squares += value * value;
            }
            float inverse = (float) (1.0 / Math.sqrt(squares / width + epsilon));
            for (int column = 0; column < width; column++) {
                int index = row * width + column;
                output[index] = floatToBf16(bf16ToFloat(input[index]) * inverse * (1.0f + bf16ToFloat(weight[column])));
            }
        }
        return output;
    }

    private static byte[] quantizedWeights(int rows, int width, int bits) {
        int groups = width / 64;
        int codeBytes = rows * groups * 32;
        int highBytes = bits == 5 ? rows * groups * 8 : 0;
        int codePlaneBytes = align256(codeBytes);
        int scaleOffset = codePlaneBytes + align256(highBytes);
        byte[] weights = new byte[scaleOffset + rows * groups * Short.BYTES];
        ByteBuffer scales = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < rows; row++) {
            for (int group = 0; group < groups; group++) {
                int groupOffset = row * groups + group;
                scales.putShort(scaleOffset + groupOffset * Short.BYTES, floatToFp16(0.0625f * (group + 1)));
                for (int lane = 0; lane < 64; lane++) {
                    int code =
                            bits == 4 ? (row * 3 + group * 5 + lane) % 15 - 7 : (row * 7 + group * 3 + lane) % 31 - 15;
                    int codeOffset = groupOffset * 32 + (lane >>> 1);
                    int shift = (lane & 1) * 4;
                    weights[codeOffset] |= (byte) ((code & 0x0f) << shift);
                    if (bits == 5 && code < 0) {
                        int highBitOffset = codePlaneBytes + groupOffset * 8 + (lane >>> 3);
                        weights[highBitOffset] |= (byte) (1 << (lane & 7));
                    }
                }
            }
        }
        return weights;
    }

    private static short[] quantizedLinearReference(
            short[] input, byte[] weights, int rows, int width, int outputs, int bits) {
        int groups = width / 64;
        int codePlaneBytes = align256(outputs * groups * 32);
        int highBytes = bits == 5 ? outputs * groups * 8 : 0;
        int scaleOffset = codePlaneBytes + align256(highBytes);
        ByteBuffer data = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
        short[] result = new short[rows * outputs];
        for (int row = 0; row < rows; row++) {
            for (int output = 0; output < outputs; output++) {
                float sum = 0.0f;
                for (int k = 0; k < width; k++) {
                    int group = k / 64;
                    int lane = k % 64;
                    int groupOffset = output * groups + group;
                    int packed = weights[groupOffset * 32 + (lane >>> 1)] & 0xff;
                    int code = (packed >>> ((lane & 1) * 4)) & 0x0f;
                    if (bits == 4) {
                        if (code >= 8) code -= 16;
                    } else {
                        int high = (weights[codePlaneBytes + groupOffset * 8 + (lane >>> 3)] >>> (lane & 7)) & 1;
                        code |= high << 4;
                        if (code >= 16) code -= 32;
                    }
                    float scale = Float.float16ToFloat(data.getShort(scaleOffset + groupOffset * Short.BYTES));
                    sum += bf16ToFloat(input[row * width + k]) * code * scale;
                }
                result[row * outputs + output] = floatToBf16(sum);
            }
        }
        return result;
    }

    private static short[][] convolutionReference(
            short[] queryKey,
            short[] valueZ,
            short[] weights,
            short[] initialState,
            int rows,
            int queryKeyWidth,
            int valueWidth,
            int kernel) {
        int channels = queryKeyWidth + valueWidth;
        short[] output = new short[rows * channels];
        short[] state = initialState.clone();
        for (int channel = 0; channel < channels; channel++) {
            for (int row = 0; row < rows; row++) {
                float sum = 0.0f;
                for (int tap = 0; tap < kernel; tap++) {
                    int sourceRow = row + tap - (kernel - 1);
                    float value;
                    if (sourceRow < 0) {
                        value = bf16ToFloat(state[channel * (kernel - 1) + sourceRow + kernel - 1]);
                    } else {
                        value = projectedValue(queryKey, valueZ, sourceRow, channel, queryKeyWidth, valueWidth);
                    }
                    sum += value * bf16ToFloat(weights[tap * channels + channel]);
                }
                output[row * channels + channel] = floatToBf16(sum * sigmoid(sum));
            }
            for (int history = 0; history < kernel - 1; history++) {
                int sourceRow = rows - (kernel - 1) + history;
                short next = sourceRow < 0
                        ? initialState[channel * (kernel - 1) + sourceRow + kernel - 1]
                        : floatToBf16(projectedValue(queryKey, valueZ, sourceRow, channel, queryKeyWidth, valueWidth));
                state[channel * (kernel - 1) + history] = next;
            }
        }
        return new short[][] {output, state};
    }

    private static float projectedValue(
            short[] queryKey, short[] valueZ, int row, int channel, int queryKeyWidth, int valueWidth) {
        return channel < queryKeyWidth
                ? bf16ToFloat(queryKey[row * queryKeyWidth + channel])
                : bf16ToFloat(valueZ[row * valueWidth * 2 + channel - queryKeyWidth]);
    }

    private static RecurrenceResult recurrenceReference(
            short[] convolved,
            float[] g,
            float[] beta,
            float[] initialState,
            int rows,
            int keyHeads,
            int valueHeads,
            int headDim,
            float scale) {
        int queryKeyWidth = 2 * keyHeads * headDim;
        int valueWidth = valueHeads * headDim;
        int convolvedWidth = queryKeyWidth + valueWidth;
        float[] state = initialState.clone();
        short[] output = new short[rows * valueWidth];
        for (int row = 0; row < rows; row++) {
            for (int valueHead = 0; valueHead < valueHeads; valueHead++) {
                int keyHead = valueHead / (valueHeads / keyHeads);
                int queryBase = row * convolvedWidth + keyHead * headDim;
                int keyBase = row * convolvedWidth + keyHeads * headDim + keyHead * headDim;
                int valueBase = row * convolvedWidth + queryKeyWidth + valueHead * headDim;
                float[] query = normalized(convolved, queryBase, headDim);
                float[] key = normalized(convolved, keyBase, headDim);
                float alpha = g[row * valueHeads + valueHead];
                float gate = beta[row * valueHeads + valueHead];
                for (int dv = 0; dv < headDim; dv++) {
                    int stateOffset = (valueHead * headDim + dv) * headDim;
                    float dot = 0.0f;
                    for (int k = 0; k < headDim; k++) dot += state[stateOffset + k] * key[k];
                    float delta = gate * (bf16ToFloat(convolved[valueBase + dv]) - alpha * dot);
                    float outputSum = 0.0f;
                    for (int k = 0; k < headDim; k++) {
                        state[stateOffset + k] = alpha * state[stateOffset + k] + delta * key[k];
                        outputSum += state[stateOffset + k] * query[k];
                    }
                    output[row * valueWidth + valueHead * headDim + dv] = floatToBf16(outputSum * scale);
                }
            }
        }
        return new RecurrenceResult(output, state);
    }

    private static float[] normalized(short[] values, int offset, int width) {
        float sum = 0.0f;
        for (int i = 0; i < width; i++) {
            float value = bf16ToFloat(values[offset + i]);
            sum += value * value;
        }
        float inverse = (float) (1.0 / Math.sqrt(sum + 1.0e-6f));
        float[] result = new float[width];
        for (int i = 0; i < width; i++) result[i] = bf16ToFloat(values[offset + i]) * inverse;
        return result;
    }

    private static short[] gatedNormReference(
            short[] recurrent, short[] valueZ, short[] weight, int rows, int heads, int headDim, float epsilon) {
        short[] output = new short[recurrent.length];
        for (int row = 0; row < rows; row++) {
            for (int head = 0; head < heads; head++) {
                int base = row * heads * headDim + head * headDim;
                float squareSum = 0.0f;
                for (int i = 0; i < headDim; i++) {
                    float x = bf16ToFloat(recurrent[base + i]);
                    squareSum += x * x;
                }
                float inverse = (float) (1.0 / Math.sqrt(squareSum / headDim + epsilon));
                int zBase = row * heads * headDim * 2 + heads * headDim + head * headDim;
                for (int i = 0; i < headDim; i++) {
                    float x = bf16ToFloat(recurrent[base + i]);
                    float z = bf16ToFloat(valueZ[zBase + i]);
                    float silu = z * sigmoid(z);
                    output[base + i] = floatToBf16(x * inverse * bf16ToFloat(weight[i]) * silu);
                }
            }
        }
        return output;
    }

    private static long uploadFloats(CudaGpuMemory gpu, Arena arena, float[] values) {
        MemorySegment host = arena.allocate((long) values.length * Float.BYTES, Float.BYTES);
        ByteBuffer buffer = host.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) buffer.putFloat(value);
        long device = gpu.allocate(host.byteSize());
        gpu.copyHostToDevice(device, host, host.byteSize());
        return device;
    }

    private static float[] downloadFloats(CudaGpuMemory gpu, Arena arena, long device, int count) {
        MemorySegment host = arena.allocate((long) count * Float.BYTES, Float.BYTES);
        gpu.copyDeviceToHost(host, device, host.byteSize());
        ByteBuffer buffer = host.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = buffer.getFloat();
        return values;
    }

    private static void assertFloatEquals(float[] expected, float[] actual, float tolerance) {
        assertTrue(expected.length == actual.length, "array lengths differ");
        for (int i = 0; i < expected.length; i++) {
            float difference = Math.abs(expected[i] - actual[i]);
            assertTrue(difference <= tolerance, "index " + i + ": expected " + expected[i] + ", actual " + actual[i]);
        }
    }

    private static int align256(int size) {
        return (size + 255) & ~255;
    }

    private static short floatToFp16(float value) {
        int bits = Float.floatToRawIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int exponent = ((bits >>> 23) & 0xff) - 112;
        int mantissa = (bits >>> 13) & 0x3ff;
        return (short) (sign | Math.max(0, Math.min(31, exponent)) << 10 | mantissa);
    }

    private static float sigmoid(float value) {
        return (float) (1.0 / (1.0 + Math.exp(-value)));
    }

    private static float softplus(float value) {
        return (float) (Math.max(value, 0.0) + Math.log1p(Math.exp(-Math.abs(value))));
    }

    private record RecurrenceResult(short[] output, float[] state) {}
}
