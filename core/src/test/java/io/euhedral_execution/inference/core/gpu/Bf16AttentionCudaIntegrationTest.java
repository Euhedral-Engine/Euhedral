package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.state.AttentionKvState;
import io.euhedral_execution.inference.core.state.KvFormat;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The engine's BF16 attention cache through [CudaGpuMemory]: the append stores rows as computed, prefill and decode
/// over it match a float64 reference of gated causal GQA attention, and verification rows equal one-row decode bit for
/// bit, at the Qwen3.8 attention geometry (24 query heads, 4 KV heads of 256).
class Bf16AttentionCudaIntegrationTest {

    private static final int QUERY_HEADS = 24;
    private static final int KEY_HEADS = 4;
    private static final int QUERY_WIDTH = QUERY_HEADS * 256;
    private static final int KEY_WIDTH = KEY_HEADS * 256;
    private static final int WIDTH = QUERY_WIDTH + KEY_WIDTH;
    /// Relative RMS error a route may add to the float64 reference beyond the BF16 rounding of its output.
    private static final double EXCESS = 1e-4;

    @Test
    @Timeout(120)
    void prefillDecodeAndVerificationOverABf16Cache() {
        int prompt = 2100;
        int verified = 4;
        int tokens = prompt + 1 + verified;
        var random = new SplittableRandom(0xbf16);
        short[] queryKey = bf16Rows(random, tokens, 0.5);
        short[] gateValue = bf16Rows(random, tokens, 0.7);
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Arena arena = Arena.ofConfined();
                AttentionKvState cache = new AttentionKvState(gpu, KEY_WIDTH, KvFormat.BF16)) {
            long queryKeyDevice = CudaGpuOperationsIntegrationTest.upload(gpu, arena, queryKey);
            long gateValueDevice = CudaGpuOperationsIntegrationTest.upload(gpu, arena, gateValue);
            long output = gpu.allocate((long) tokens * QUERY_WIDTH * Short.BYTES);
            long scratch = gpu.allocate((long) verified * QUERY_HEADS * 64 * 258 * Float.BYTES);
            long rowOutput = gpu.allocate((long) QUERY_WIDTH * Short.BYTES);
            long[] positions = new long[tokens + 1];
            try {
                for (int start = 0; start < tokens; start += 512) {
                    int rows = Math.min(512, tokens - start);
                    positions[start] = position(gpu, arena, start);
                    cache.prepareAppend(start, rows);
                    gpu.attentionKvAppendBf16(
                            queryKeyDevice + (long) start * WIDTH * Short.BYTES,
                            gateValueDevice + (long) start * WIDTH * Short.BYTES,
                            cache.keyCacheAddress(),
                            cache.valueCacheAddress(),
                            rows,
                            QUERY_WIDTH,
                            KEY_WIDTH,
                            start,
                            positions[start]);
                    cache.appendSubmitted(rows);
                }
                cache.commitSubmitted();
                assertPagesHoldTheRows(gpu, arena, cache, queryKey, gateValue, tokens);

                // Prompt chunks of 512 (FlashAttention-2 route) and the 52-row tail (32-row tile route).
                for (int start = 0; start < prompt; start += 512) {
                    int rows = Math.min(512, prompt - start);
                    attend(gpu, cache, queryKeyDevice, gateValueDevice, output, start, rows, positions[start], 0);
                }
                // One decode row past 2048 keys (the GQA route).
                positions[prompt] = position(gpu, arena, prompt);
                attend(gpu, cache, queryKeyDevice, gateValueDevice, output, prompt, 1, positions[prompt], scratch);
                short[] actual =
                        CudaGpuOperationsIntegrationTest.download(gpu, arena, output, (prompt + 1) * QUERY_WIDTH);
                assertMatchesReference(queryKey, gateValue, actual, 0, prompt - 64, 64);
                assertMatchesReference(queryKey, gateValue, actual, 0, 0, 40);
                assertMatchesReference(queryKey, gateValue, actual, 0, prompt, 1);

                // Verification: rows prompt + 1 .. prompt + 4 in one launch, each bit for bit its one-row decode.
                int first = prompt + 1;
                positions[first] = position(gpu, arena, first);
                gpu.selectRowExact(true);
                try {
                    attend(
                            gpu,
                            cache,
                            queryKeyDevice,
                            gateValueDevice,
                            output,
                            first,
                            verified,
                            positions[first],
                            scratch);
                } finally {
                    gpu.selectRowExact(false);
                }
                short[] together = CudaGpuOperationsIntegrationTest.download(
                        gpu, arena, output + (long) first * QUERY_WIDTH * Short.BYTES, verified * QUERY_WIDTH);
                for (int row = 0; row < verified; row++) {
                    long at = position(gpu, arena, first + row);
                    try {
                        gpu.attentionCausalBf16(
                                queryKeyDevice + (long) (first + row) * WIDTH * Short.BYTES,
                                gateValueDevice + (long) (first + row) * WIDTH * Short.BYTES,
                                cache.keyCacheAddress(),
                                cache.valueCacheAddress(),
                                rowOutput,
                                1,
                                QUERY_HEADS,
                                KEY_HEADS,
                                256,
                                first + row + 1,
                                first + row,
                                at,
                                scratch);
                        short[] alone = CudaGpuOperationsIntegrationTest.download(gpu, arena, rowOutput, QUERY_WIDTH);
                        assertArrayEquals(
                                alone,
                                Arrays.copyOfRange(together, row * QUERY_WIDTH, (row + 1) * QUERY_WIDTH),
                                "verified row " + row);
                    } finally {
                        gpu.free(at);
                    }
                }
                assertMatchesReference(queryKey, gateValue, together, first, first, verified);
            } finally {
                for (long position : positions) if (position != 0) gpu.free(position);
                gpu.free(rowOutput);
                gpu.free(scratch);
                gpu.free(output);
                gpu.free(gateValueDevice);
                gpu.free(queryKeyDevice);
            }
        }
    }

    private static void attend(
            CudaGpuMemory gpu,
            AttentionKvState cache,
            long queryKey,
            long gateValue,
            long output,
            int start,
            int rows,
            long position,
            long scratch) {
        gpu.attentionCausalBf16(
                queryKey + (long) start * WIDTH * Short.BYTES,
                gateValue + (long) start * WIDTH * Short.BYTES,
                cache.keyCacheAddress(),
                cache.valueCacheAddress(),
                output + (long) start * QUERY_WIDTH * Short.BYTES,
                rows,
                QUERY_HEADS,
                KEY_HEADS,
                256,
                start + rows,
                start,
                position,
                scratch);
    }

    /// Every page holds its tokens' K (from `queryKey`) and V (from `gateValue`) rows bit for bit.
    private static void assertPagesHoldTheRows(
            CudaGpuMemory gpu, Arena arena, AttentionKvState cache, short[] queryKey, short[] gateValue, int tokens) {
        long plane = cache.planePageBytes();
        assertEquals(256L * 512 * KEY_HEADS, plane);
        MemorySegment page = arena.allocate(2 * plane);
        var addresses = cache.pageAddresses();
        for (int index = 0; index < addresses.size(); index++) {
            gpu.copyDeviceToHost(page, addresses.get(index), 2 * plane);
            for (int token = index * 256; token < Math.min(tokens, (index + 1) * 256); token++) {
                for (int element = 0; element < KEY_WIDTH; element++) {
                    long offset = ((long) (token % 256) * KEY_WIDTH + element) * Short.BYTES;
                    int source = token * WIDTH + QUERY_WIDTH + element;
                    assertEquals(queryKey[source], page.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset));
                    assertEquals(gateValue[source], page.get(ValueLayout.JAVA_SHORT_UNALIGNED, plane + offset));
                }
            }
        }
    }

    /// Rows `[from, from + count)` of the attention output (`actual` starts at row `actualStart`) against float64
    /// causal
    /// GQA attention over every earlier token, gated by the sigmoid of the gate.
    private static void assertMatchesReference(
            short[] queryKey, short[] gateValue, short[] actual, int actualStart, int from, int count) {
        double error = 0;
        double floor = 0;
        double norm = 0;
        for (int row = from; row < from + count; row++) {
            for (int head = 0; head < QUERY_HEADS; head++) {
                int keyHead = head / (QUERY_HEADS / KEY_HEADS);
                double[] scores = new double[row + 1];
                double maximum = Double.NEGATIVE_INFINITY;
                for (int key = 0; key <= row; key++) {
                    double dot = 0;
                    for (int d = 0; d < 256; d++)
                        dot += value(queryKey, row * WIDTH + head * 256 + d)
                                * value(queryKey, key * WIDTH + QUERY_WIDTH + keyHead * 256 + d);
                    scores[key] = dot / 16;
                    maximum = Math.max(maximum, scores[key]);
                }
                double sum = 0;
                for (int key = 0; key <= row; key++) sum += scores[key] = Math.exp(scores[key] - maximum);
                for (int d = 0; d < 256; d++) {
                    double mixed = 0;
                    for (int key = 0; key <= row; key++)
                        mixed += scores[key] * value(gateValue, key * WIDTH + QUERY_WIDTH + keyHead * 256 + d);
                    double gate = value(gateValue, row * WIDTH + head * 256 + d);
                    double expected = mixed / sum / (1 + Math.exp(-gate));
                    double got = value(actual, (row - actualStart) * QUERY_WIDTH + head * 256 + d);
                    double rounded = CudaGpuOperationsIntegrationTest.bf16ToFloat(
                            CudaGpuOperationsIntegrationTest.floatToBf16((float) expected));
                    error += (got - expected) * (got - expected);
                    floor += (rounded - expected) * (rounded - expected);
                    norm += expected * expected;
                }
            }
        }
        double relative = Math.sqrt(error / norm);
        double bound = Math.sqrt(floor / norm) + EXCESS;
        assertTrue(relative <= bound, "rows " + from + "+" + count + ": " + relative + " > " + bound);
    }

    private static double value(short[] values, int index) {
        return CudaGpuOperationsIntegrationTest.bf16ToFloat(values[index]);
    }

    private static short[] bf16Rows(SplittableRandom random, int rows, double scale) {
        short[] values = new short[rows * WIDTH];
        for (int index = 0; index < values.length; index++)
            values[index] =
                    CudaGpuOperationsIntegrationTest.floatToBf16((float) (scale * (random.nextDouble() * 2 - 1)));
        return values;
    }

    private static long position(CudaGpuMemory gpu, Arena arena, long value) {
        MemorySegment host = arena.allocate(Long.BYTES, Long.BYTES);
        host.set(ValueLayout.JAVA_LONG, 0, value);
        long device = gpu.allocate(Long.BYTES);
        gpu.copyHostToDevice(device, host, Long.BYTES);
        return device;
    }
}
