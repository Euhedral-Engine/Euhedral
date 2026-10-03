package io.euhedral_execution.inference.core.model_loader.artifact;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Test;

class P2e2LayoutTest {

    private static final ValueLayout.OfInt WORD = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    @Test
    void planeOffsetsFollowTheConverter() {
        // tools/euhedral_artifacts/q3_p2e2.py p2e2_offsets(3, 1024): (768, 1024, 1280)
        assertEquals(768, P2e2Layout.rowBaseOffset(3, 1024));
        assertEquals(1024, P2e2Layout.scaleOffset(3, 1024));
        assertEquals(1280, P2e2Layout.payloadOffset(3, 1024));
        assertEquals(
                CompactTensorLayout.expectedByteSize(
                        new long[] {3, 1024},
                        TensorDataType.BF16,
                        WeightFormat.Q3_G64_FP16,
                        WeightLayout.ROW_SPLIT_K128_V1),
                P2e2Layout.expandedByteSize(3, 1024));
    }

    @Test
    void sizesBetweenAnEmptyAndAFullPayloadAreAccepted() {
        long payload = P2e2Layout.payloadOffset(4, 2048);
        long empty = payload + 4L * P2e2Layout.PAD_WORDS;
        long full = empty + 4L * (4 * 2048 / 16);
        assertTrue(P2e2Layout.acceptsByteSize(4, 2048, empty));
        assertTrue(P2e2Layout.acceptsByteSize(4, 2048, full));
        assertFalse(P2e2Layout.acceptsByteSize(4, 2048, empty - 4));
        assertFalse(P2e2Layout.acceptsByteSize(4, 2048, full + 4));
        assertFalse(P2e2Layout.acceptsByteSize(4, 2048, empty + 2));
    }

    @Test
    void compactLayoutValidationAcceptsOnlyQ3WholeSliceTensors() {
        long[] shape = {4, 2048};
        long size = P2e2Layout.payloadOffset(4, 2048) + 4L * P2e2Layout.PAD_WORDS;
        assertTrue(CompactTensorLayout.acceptsByteSize(
                shape, TensorDataType.BF16, WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_P2E2_V1, size));
        assertThrows(
                IllegalArgumentException.class,
                () -> CompactTensorLayout.acceptsByteSize(
                        shape, TensorDataType.BF16, WeightFormat.Q4_G64_FP16, WeightLayout.ROW_SPLIT_P2E2_V1, size));
        assertThrows(
                IllegalArgumentException.class,
                () -> CompactTensorLayout.acceptsByteSize(
                        new long[] {4, 1536},
                        TensorDataType.BF16,
                        WeightFormat.Q3_G64_FP16,
                        WeightLayout.ROW_SPLIT_P2E2_V1,
                        size));
        assertThrows(
                IllegalArgumentException.class,
                () -> CompactTensorLayout.expectedByteSize(
                        shape, TensorDataType.BF16, WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_P2E2_V1));
    }

    @Test
    void consistentTensorsValidate() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment tensor = encode(arena, randomCodes(5, 2048, new Random(3)));
            assertDoesNotThrow(() -> P2e2Layout.validate(tensor, 5, 2048));
        }
    }

    @Test
    void rowBaseThatDisagreesWithThePrimaryPlaneIsRejected() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment tensor = encode(arena, randomCodes(5, 2048, new Random(4)));
            long base = P2e2Layout.rowBaseOffset(5, 2048);
            tensor.set(WORD, base + 8, tensor.get(WORD, base + 8) + 1);
            assertThrows(IllegalArgumentException.class, () -> P2e2Layout.validate(tensor, 5, 2048));
        }
    }

    @Test
    void payloadShorterThanThePrimaryPlaneIsRejected() {
        try (Arena arena = Arena.ofConfined()) {
            int[][] codes = new int[2][1024];
            for (int[] row : codes) java.util.Arrays.fill(row, 3);
            MemorySegment tensor = encode(arena, codes);
            MemorySegment truncated = tensor.asSlice(0, P2e2Layout.payloadOffset(2, 1024) + 4L * P2e2Layout.PAD_WORDS);
            assertThrows(IllegalArgumentException.class, () -> P2e2Layout.validate(truncated, 2, 1024));
        }
    }

    private static int[][] randomCodes(int rows, int k, Random random) {
        int[] values = {-3, -2, -1, 0, 1, 2, 3};
        int[][] codes = new int[rows][k];
        for (int[] row : codes) for (int i = 0; i < k; i++) row[i] = values[random.nextInt(values.length)];
        return codes;
    }

    /// The layout of docs/COMPRESSED_Q3.md with zero scales.
    private static MemorySegment encode(Arena arena, int[][] codes) {
        int rows = codes.length, k = codes[0].length;
        int units = 0;
        for (int[] row : codes) for (int code : row) if (Math.abs(code) >= 2) units++;
        long payload = P2e2Layout.payloadOffset(rows, k);
        MemorySegment tensor = arena.allocate(payload + 4L * ((units + 15) / 16 + P2e2Layout.PAD_WORDS), 4);
        long base = P2e2Layout.rowBaseOffset(rows, k);
        int unit = 0;
        for (int r = 0; r < rows; r++) {
            tensor.set(WORD, base + 4L * r, unit);
            for (int j = 0; j < k; j++) {
                int code = codes[r][j];
                int symbol = Math.abs(code) >= 2 ? 3 : code + 1;
                long word = (long) r * k / 16 + j / 16;
                tensor.set(WORD, 4 * word, tensor.get(WORD, 4 * word) | symbol << (2 * (j % 16)));
                if (symbol == 3) {
                    int value = code == -3 ? 0 : code == -2 ? 1 : code == 2 ? 2 : 3;
                    long at = payload + 4L * (unit / 16);
                    tensor.set(WORD, at, tensor.get(WORD, at) | value << (2 * (unit % 16)));
                    unit++;
                }
            }
        }
        return tensor;
    }
}
