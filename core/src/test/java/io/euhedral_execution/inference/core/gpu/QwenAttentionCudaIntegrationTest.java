package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.scheduling.AttentionKvState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class QwenAttentionCudaIntegrationTest {

    private static final int QUERY_HEADS = 2;
    private static final int KEY_VALUE_HEADS = 1;
    private static final int HEAD_DIM = 256;
    private static final int ROTARY_DIM = 64;
    private static final int QUERY_WIDTH = QUERY_HEADS * HEAD_DIM;
    private static final int KEY_VALUE_WIDTH = KEY_VALUE_HEADS * HEAD_DIM;
    private static final int QUERY_KEY_WIDTH = QUERY_WIDTH + KEY_VALUE_WIDTH;
    private static final int GATE_VALUE_WIDTH = QUERY_WIDTH + KEY_VALUE_WIDTH;
    private static final float EPSILON = 1.0e-6f;
    private static final double ROPE_THETA = 1_000_000.0;

    @Test
    @Timeout(90)
    void standaloneAttentionOperationsMatchCpuReferenceAcrossPrefillAndDecodeAppend() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        int prefillRows = 3;
        short[] queryKey = projections(prefillRows, QUERY_KEY_WIDTH, 0.13f);
        short[] gateValue = projections(prefillRows, GATE_VALUE_WIDTH, 0.21f);
        short[] queryNorm = normWeights(HEAD_DIM, 0.02f);
        short[] keyNorm = normWeights(HEAD_DIM, -0.015f);
        short[] normalized = new short[queryKey.length];
        short[] expectedNormalized = qkNormRope(queryKey, queryNorm, keyNorm, prefillRows, 0);
        short[] expectedOutput;

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena arena = Arena.ofConfined();
                AttentionKvState cache = new AttentionKvState(gpu, KEY_VALUE_WIDTH)) {
            long queryKeyDevice = CudaGpuOperationsIntegrationTest.upload(gpu, arena, queryKey);
            long gateValueDevice = CudaGpuOperationsIntegrationTest.upload(gpu, arena, gateValue);
            long queryNormDevice = CudaGpuOperationsIntegrationTest.upload(gpu, arena, queryNorm);
            long keyNormDevice = CudaGpuOperationsIntegrationTest.upload(gpu, arena, keyNorm);
            long normalizedDevice = gpu.allocate((long) normalized.length * Short.BYTES);
            long prefillPosition = position(gpu, arena, 0);
            try {
                gpu.attentionQkNormRopeBf16(
                        queryKeyDevice,
                        queryNormDevice,
                        keyNormDevice,
                        normalizedDevice,
                        prefillRows,
                        QUERY_HEADS,
                        KEY_VALUE_HEADS,
                        HEAD_DIM,
                        ROTARY_DIM,
                        0,
                        prefillPosition,
                        EPSILON,
                        ROPE_THETA);
                normalized = CudaGpuOperationsIntegrationTest.download(gpu, arena, normalizedDevice, normalized.length);
                assertBf16Close(expectedNormalized, normalized, 0.012f);

                cache.prepareAppend(0, prefillRows);
                long gateValueAppendDevice = gateValueDevice;
                gpu.attentionKvAppendNvfp4(
                        normalizedDevice,
                        gateValueAppendDevice,
                        cache.keyCacheAddress(),
                        cache.valueCacheAddress(),
                        prefillRows,
                        QUERY_WIDTH,
                        KEY_VALUE_WIDTH,
                        0,
                        prefillPosition);
                cache.appendSubmitted(prefillRows);
                cache.commitSubmitted();
                expectedOutput = causalAttention(
                        expectedNormalized, expectedNormalized, gateValue, gateValue, prefillRows, QUERY_HEADS, 0);
                long outputDevice = gpu.allocate((long) prefillRows * QUERY_WIDTH * Short.BYTES);
                try {
                    gpu.attentionCausalNvfp4(
                            normalizedDevice,
                            gateValueDevice,
                            cache.keyCacheAddress(),
                            cache.valueCacheAddress(),
                            outputDevice,
                            prefillRows,
                            QUERY_HEADS,
                            KEY_VALUE_HEADS,
                            HEAD_DIM,
                            cache.length(),
                            0,
                            prefillPosition,
                            cache.decodeScratchAddress(QUERY_HEADS));
                    short[] actual =
                            CudaGpuOperationsIntegrationTest.download(gpu, arena, outputDevice, expectedOutput.length);
                    assertBf16Close(expectedOutput, actual, 0.03f);
                } finally {
                    gpu.free(outputDevice);
                }

                int decodePosition = prefillRows;
                short[] decodeQueryKey = projections(1, QUERY_KEY_WIDTH, 0.37f);
                short[] decodeGateValue = projections(1, GATE_VALUE_WIDTH, 0.43f);
                long decodeQueryKeyDevice = CudaGpuOperationsIntegrationTest.upload(gpu, arena, decodeQueryKey);
                long decodeGateValueDevice = CudaGpuOperationsIntegrationTest.upload(gpu, arena, decodeGateValue);
                long decodeNormalizedDevice = gpu.allocate((long) QUERY_KEY_WIDTH * Short.BYTES);
                long decodeOutputDevice = gpu.allocate((long) QUERY_WIDTH * Short.BYTES);
                long decodePositionDevice = position(gpu, arena, decodePosition);
                try {
                    gpu.attentionQkNormRopeBf16(
                            decodeQueryKeyDevice,
                            queryNormDevice,
                            keyNormDevice,
                            decodeNormalizedDevice,
                            1,
                            QUERY_HEADS,
                            KEY_VALUE_HEADS,
                            HEAD_DIM,
                            ROTARY_DIM,
                            decodePosition,
                            decodePositionDevice,
                            EPSILON,
                            ROPE_THETA);
                    short[] expectedDecodeNormalized =
                            qkNormRope(decodeQueryKey, queryNorm, keyNorm, 1, decodePosition);
                    assertBf16Close(
                            expectedDecodeNormalized,
                            CudaGpuOperationsIntegrationTest.download(
                                    gpu, arena, decodeNormalizedDevice, QUERY_KEY_WIDTH),
                            0.012f);

                    cache.prepareAppend(decodePosition, 1);
                    gpu.attentionKvAppendNvfp4(
                            decodeNormalizedDevice,
                            decodeGateValueDevice,
                            cache.keyCacheAddress(),
                            cache.valueCacheAddress(),
                            1,
                            QUERY_WIDTH,
                            KEY_VALUE_WIDTH,
                            decodePosition,
                            decodePositionDevice);
                    cache.appendSubmitted(1);
                    cache.commitSubmitted();
                    assertTrue(cache.capacity() >= 4);
                    assertTrue(cache.length() == 4);
                    short[] joinedNormalized = concat(expectedNormalized, expectedDecodeNormalized);
                    short[] joinedGateValue = concat(gateValue, decodeGateValue);
                    short[] expectedDecodeOutput = causalAttention(
                            expectedDecodeNormalized,
                            joinedNormalized,
                            decodeGateValue,
                            joinedGateValue,
                            1,
                            QUERY_HEADS,
                            decodePosition);
                    gpu.attentionCausalNvfp4(
                            decodeNormalizedDevice,
                            decodeGateValueDevice,
                            cache.keyCacheAddress(),
                            cache.valueCacheAddress(),
                            decodeOutputDevice,
                            1,
                            QUERY_HEADS,
                            KEY_VALUE_HEADS,
                            HEAD_DIM,
                            cache.length(),
                            decodePosition,
                            decodePositionDevice,
                            cache.decodeScratchAddress(QUERY_HEADS));
                    assertBf16Close(
                            expectedDecodeOutput,
                            CudaGpuOperationsIntegrationTest.download(
                                    gpu, arena, decodeOutputDevice, expectedDecodeOutput.length),
                            0.03f);
                } finally {
                    gpu.free(decodePositionDevice);
                    gpu.free(decodeOutputDevice);
                    gpu.free(decodeNormalizedDevice);
                    gpu.free(decodeGateValueDevice);
                    gpu.free(decodeQueryKeyDevice);
                }
            } finally {
                gpu.free(prefillPosition);
                gpu.free(normalizedDevice);
                gpu.free(keyNormDevice);
                gpu.free(queryNormDevice);
                gpu.free(gateValueDevice);
                gpu.free(queryKeyDevice);
            }
        }
    }

    /// A device copy of a quantum's start position, which position-dependent kernels read.
    private static long position(CudaGpuMemory gpu, Arena arena, long value) {
        MemorySegment host = arena.allocate(Long.BYTES, Long.BYTES);
        host.set(ValueLayout.JAVA_LONG, 0, value);
        long device = gpu.allocate(Long.BYTES);
        gpu.copyHostToDevice(device, host, Long.BYTES);
        return device;
    }

    private static short[] qkNormRope(short[] input, short[] queryNorm, short[] keyNorm, int rows, long position) {
        short[] result = new short[input.length];
        int headCount = QUERY_HEADS + KEY_VALUE_HEADS;
        for (int row = 0; row < rows; row++) {
            for (int head = 0; head < headCount; head++) {
                int vectorOffset = row * QUERY_KEY_WIDTH + head * HEAD_DIM;
                short[] weights = head < QUERY_HEADS ? queryNorm : keyNorm;
                float sumSquares = 0.0f;
                for (int dimension = 0; dimension < HEAD_DIM; dimension++) {
                    float value = CudaGpuOperationsIntegrationTest.bf16ToFloat(input[vectorOffset + dimension]);
                    sumSquares += value * value;
                }
                float inverse = (float) (1.0 / Math.sqrt(sumSquares / HEAD_DIM + EPSILON));
                float[] normalized = new float[HEAD_DIM];
                for (int dimension = 0; dimension < HEAD_DIM; dimension++) {
                    normalized[dimension] =
                            CudaGpuOperationsIntegrationTest.bf16ToFloat(input[vectorOffset + dimension])
                                    * inverse
                                    * (1.0f + CudaGpuOperationsIntegrationTest.bf16ToFloat(weights[dimension]));
                }
                int rotaryHalf = ROTARY_DIM / 2;
                for (int dimension = 0; dimension < HEAD_DIM; dimension++) {
                    float value = normalized[dimension];
                    if (dimension < ROTARY_DIM) {
                        int frequencyIndex = dimension < rotaryHalf ? dimension : dimension - rotaryHalf;
                        int pairedDimension = dimension < rotaryHalf ? dimension + rotaryHalf : dimension - rotaryHalf;
                        double angle = (position + row) * Math.pow(ROPE_THETA, -2.0 * frequencyIndex / ROTARY_DIM);
                        float cosine = (float) Math.cos(angle);
                        float sine = (float) Math.sin(angle);
                        value = dimension < rotaryHalf
                                ? normalized[dimension] * cosine - normalized[pairedDimension] * sine
                                : normalized[dimension] * cosine + normalized[pairedDimension] * sine;
                    }
                    result[vectorOffset + dimension] = CudaGpuOperationsIntegrationTest.floatToBf16(value);
                }
            }
        }
        return result;
    }

    private static short[] causalAttention(
            short[] queryRows,
            short[] cachedQueryKey,
            short[] queryGateValue,
            short[] cachedGateValue,
            int rows,
            int queryHeads,
            long startPosition) {
        double[][] keys = new double[cachedQueryKey.length / QUERY_KEY_WIDTH * KEY_VALUE_HEADS][];
        double[][] values = new double[keys.length][];
        for (int i = 0; i < keys.length; i++) {
            int offset = (i / KEY_VALUE_HEADS) * QUERY_KEY_WIDTH + QUERY_WIDTH + (i % KEY_VALUE_HEADS) * HEAD_DIM;
            keys[i] = Nvfp4KvReference.represented(cachedQueryKey, offset);
            values[i] = Nvfp4KvReference.represented(cachedGateValue, offset);
        }
        List<Short> output = new ArrayList<>();
        int groupSize = queryHeads / KEY_VALUE_HEADS;
        for (int row = 0; row < rows; row++) {
            int absolutePosition = Math.toIntExact(startPosition + row);
            for (int head = 0; head < queryHeads; head++) {
                int queryOffset = row * QUERY_KEY_WIDTH + head * HEAD_DIM;
                int keyHead = head / groupSize;
                List<Float> scores = new ArrayList<>(absolutePosition + 1);
                float maximum = Float.NEGATIVE_INFINITY;
                for (int keyPosition = 0; keyPosition <= absolutePosition; keyPosition++) {
                    float dot = 0.0f;
                    for (int dimension = 0; dimension < HEAD_DIM; dimension++) {
                        dot += CudaGpuOperationsIntegrationTest.bf16ToFloat(queryRows[queryOffset + dimension])
                                * keys[keyPosition * KEY_VALUE_HEADS + keyHead][dimension];
                    }
                    float score = dot * (float) (1.0 / Math.sqrt(HEAD_DIM));
                    scores.add(score);
                    maximum = Math.max(maximum, score);
                }
                float denominator = 0.0f;
                for (int keyPosition = 0; keyPosition < scores.size(); keyPosition++) {
                    float probability = (float) Math.exp(scores.get(keyPosition) - maximum);
                    scores.set(keyPosition, probability);
                    denominator += probability;
                }
                for (int dimension = 0; dimension < HEAD_DIM; dimension++) {
                    float value = 0.0f;
                    for (int keyPosition = 0; keyPosition < scores.size(); keyPosition++) {
                        value += scores.get(keyPosition)
                                / denominator
                                * values[keyPosition * KEY_VALUE_HEADS + keyHead][dimension];
                    }
                    int gateOffset = row * GATE_VALUE_WIDTH + head * HEAD_DIM + dimension;
                    float gate = CudaGpuOperationsIntegrationTest.bf16ToFloat(queryGateValue[gateOffset]);
                    float gated = value * sigmoid(gate);
                    output.add(CudaGpuOperationsIntegrationTest.floatToBf16(gated));
                }
            }
        }
        short[] result = new short[output.size()];
        for (int index = 0; index < result.length; index++) result[index] = output.get(index);
        return result;
    }

    private static short[] projections(int rows, int width, float seed) {
        short[] values = new short[rows * width];
        for (int index = 0; index < values.length; index++) {
            values[index] = CudaGpuOperationsIntegrationTest.floatToBf16((float) Math.sin(seed * (index + 1)) * 0.6f);
        }
        return values;
    }

    private static short[] normWeights(int width, float seed) {
        short[] values = new short[width];
        for (int index = 0; index < width; index++) {
            values[index] = CudaGpuOperationsIntegrationTest.floatToBf16(seed * (index % 9 - 4));
        }
        return values;
    }

    private static short[] concat(short[] left, short[] right) {
        short[] result = new short[left.length + right.length];
        System.arraycopy(left, 0, result, 0, left.length);
        System.arraycopy(right, 0, result, left.length, right.length);
        return result;
    }

    private static float sigmoid(float value) {
        return (float) (1.0 / (1.0 + Math.exp(-value)));
    }

    private static void assertBf16Close(short[] expected, short[] actual, float tolerance) {
        for (int index = 0; index < expected.length; index++) {
            float difference = Math.abs(CudaGpuOperationsIntegrationTest.bf16ToFloat(expected[index])
                    - CudaGpuOperationsIntegrationTest.bf16ToFloat(actual[index]));
            assertTrue(difference <= tolerance, "index " + index + ": " + difference);
        }
    }
}
