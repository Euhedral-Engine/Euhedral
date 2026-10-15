package io.euhedral_execution.inference.core.generation;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionFixtures;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class LogitsSamplerTest {

    private static final long LOGITS_ADDRESS = 4096L;

    @Test
    void pairedConversionMatchesTheExactPerValueConversionForEveryBf16PatternAndAnOddTail() {
        for (int length : new int[] {65_536, 65_537, 1, 3}) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment row = arena.allocate((long) length * Short.BYTES, Short.BYTES);
                for (int index = 0; index < length; index++) {
                    row.set(ValueLayout.JAVA_SHORT, (long) index * Short.BYTES, (short) (index * 40503 + 7));
                }
                if (length >= 65_536) {
                    for (int bits = 0; bits < 65_536; bits++)
                        row.set(ValueLayout.JAVA_SHORT, (long) bits * Short.BYTES, (short) bits);
                }
                float[] converted = new float[length];
                LogitsSampler.convertBf16(row, converted);
                for (int index = 0; index < length; index++) {
                    short bits = row.get(ValueLayout.JAVA_SHORT, (long) index * Short.BYTES);
                    float expected = Float.intBitsToFloat(Short.toUnsignedInt(bits) << 16);
                    if (Float.isNaN(expected)) assertTrue(Float.isNaN(converted[index]), "pattern " + index);
                    else
                        assertEquals(
                                Float.floatToRawIntBits(expected),
                                Float.floatToRawIntBits(converted[index]),
                                "length " + length + " index " + index);
                }
            }
        }
    }

    @Test
    void greedySelectionCopiesOnlyTheFinalBf16VocabularyRowAndPreservesLogitsOwnership() {
        short[] values = {
            bf16(9.0f), bf16(0.0f), bf16(1.0f),
            bf16(-2.0f), bf16(4.0f), bf16(3.0f)
        };
        FakeGpu gpu = new FakeGpu(LOGITS_ADDRESS, values);
        LogitsSampler sampler = new LogitsSampler(GenerationConfig.greedy(7L), 3);

        try (DeviceLogits logits = new DeviceLogits(gpu, LOGITS_ADDRESS, 2, 3)) {
            assertEquals(1, sampler.selectToken(logits, gpu));
            assertEquals(LOGITS_ADDRESS + 3L * Short.BYTES, gpu.copiedSource);
            assertEquals(3L * Short.BYTES, gpu.copiedBytes);
            assertEquals(1, gpu.copyCount);
            assertEquals(List.of(), gpu.freedAddresses);
        }

        assertEquals(List.of(LOGITS_ADDRESS), gpu.freedAddresses);
    }

    @Test
    void constrainedSelectionMasksTokensBeforeApplyingTopK() {
        FakeGpu gpu = new FakeGpu(LOGITS_ADDRESS, new short[] {bf16(10.0f), bf16(9.0f), bf16(1.0f)});
        LogitsSampler sampler = new LogitsSampler(new GenerationConfig(1.0f, 1, 1.0f, 7L, false), 3);

        try (DeviceLogits logits = new DeviceLogits(gpu, LOGITS_ADDRESS, 1, 3)) {
            assertEquals(1, sampler.selectToken(logits, gpu, tokenId -> tokenId == 1));
            assertThrows(IllegalArgumentException.class, () -> sampler.selectToken(logits, gpu, tokenId -> false));
        }
        assertEquals(List.of(LOGITS_ADDRESS), gpu.freedAddresses);
    }

    @Test
    void selectionRejectsDifferentGpuAndVocabularyMismatch() {
        FakeGpu owner = new FakeGpu(LOGITS_ADDRESS, new short[] {bf16(1.0f), bf16(2.0f)});
        FakeGpu otherGpu = new FakeGpu(LOGITS_ADDRESS, new short[] {bf16(1.0f), bf16(2.0f)});
        LogitsSampler matchingSampler = new LogitsSampler(GenerationConfig.greedy(1L), 2);

        try (DeviceLogits logits = new DeviceLogits(owner, LOGITS_ADDRESS, 1, 2)) {
            assertThrows(IllegalArgumentException.class, () -> matchingSampler.selectToken(logits, otherGpu));
            assertEquals(0, owner.copyCount);

            LogitsSampler mismatchedSampler = new LogitsSampler(GenerationConfig.greedy(1L), 3);
            assertThrows(IllegalArgumentException.class, () -> mismatchedSampler.selectToken(logits, owner));
            assertEquals(0, owner.copyCount);
        }
    }

    @Test
    void directCopyRejectsDifferentGpuWithoutCopying() {
        FakeGpu owner = new FakeGpu(LOGITS_ADDRESS, new short[] {bf16(1.0f), bf16(2.0f)});
        FakeGpu otherGpu = new FakeGpu(LOGITS_ADDRESS, new short[] {bf16(1.0f), bf16(2.0f)});

        try (Arena arena = Arena.ofConfined();
                DeviceLogits logits = new DeviceLogits(owner, LOGITS_ADDRESS, 1, 2)) {
            MemorySegment destination = arena.allocate(2L * Short.BYTES, Short.BYTES);
            assertThrows(IllegalArgumentException.class, () -> logits.copyFinalTokenRowToHost(otherGpu, destination));
            assertEquals(0, owner.copyCount);
        }
    }

    @Test
    void directCopyRejectsClosedLogitsWithoutCopying() {
        FakeGpu gpu = new FakeGpu(LOGITS_ADDRESS, new short[] {bf16(1.0f), bf16(2.0f)});
        DeviceLogits logits = new DeviceLogits(gpu, LOGITS_ADDRESS, 1, 2);
        logits.close();

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment destination = arena.allocate(2L * Short.BYTES, Short.BYTES);
            assertThrows(IllegalStateException.class, () -> logits.copyFinalTokenRowToHost(gpu, destination));
            assertEquals(0, gpu.copyCount);
        }
    }

    @Test
    void deviceBridgePreservesBf16NonFiniteLogitSemantics() {
        FakeGpu gpu = new FakeGpu(
                LOGITS_ADDRESS,
                new short[] {bf16(Float.NaN), bf16(Float.NEGATIVE_INFINITY), bf16(Float.POSITIVE_INFINITY)});
        LogitsSampler sampler = new LogitsSampler(new GenerationConfig(1.0f, 0, 1.0f, 17L, false), 3);

        try (DeviceLogits logits = new DeviceLogits(gpu, LOGITS_ADDRESS, 1, 3)) {
            for (int i = 0; i < 16; i++) assertEquals(2, sampler.selectToken(logits, gpu));
        }
    }

    @Test
    void allUnselectableBf16DeviceLogitsAreRejected() {
        FakeGpu gpu = new FakeGpu(LOGITS_ADDRESS, new short[] {bf16(Float.NaN), bf16(Float.NEGATIVE_INFINITY)});
        LogitsSampler sampler = new LogitsSampler(new GenerationConfig(1.0f, 0, 1.0f, 17L, false), 2);

        try (DeviceLogits logits = new DeviceLogits(gpu, LOGITS_ADDRESS, 1, 2)) {
            assertThrows(IllegalArgumentException.class, () -> sampler.selectToken(logits, gpu));
            assertEquals(1, gpu.copyCount);
        }
    }

    @Test
    void tooSmallHostDestinationIsRejectedWithoutCopying() {
        FakeGpu gpu = new FakeGpu(LOGITS_ADDRESS, new short[] {bf16(1.0f), bf16(2.0f)});

        try (Arena arena = Arena.ofConfined();
                DeviceLogits logits = new DeviceLogits(gpu, LOGITS_ADDRESS, 1, 2)) {
            MemorySegment tooSmall = arena.allocate(Short.BYTES, Short.BYTES);
            assertThrows(IllegalArgumentException.class, () -> logits.copyFinalTokenRowToHost(gpu, tooSmall));
            assertEquals(0, gpu.copyCount);
        }
    }

    @Test
    void selectionRejectsClosedDeviceLogitsWithoutCopying() {
        FakeGpu gpu = new FakeGpu(LOGITS_ADDRESS, new short[] {bf16(1.0f), bf16(2.0f)});
        DeviceLogits logits = new DeviceLogits(gpu, LOGITS_ADDRESS, 1, 2);
        logits.close();
        LogitsSampler sampler = new LogitsSampler(GenerationConfig.greedy(1L), 2);

        assertThrows(IllegalStateException.class, () -> sampler.selectToken(logits, gpu));
        assertEquals(0, gpu.copyCount);
        assertEquals(List.of(LOGITS_ADDRESS), gpu.freedAddresses);
    }

    @Test
    void stochasticSelectionSamplesTheCopiedBf16Logits() {
        FakeGpu gpu = new FakeGpu(LOGITS_ADDRESS, new short[] {bf16(0.0f), bf16(10.0f), bf16(9.0f)});
        LogitsSampler sampler = new LogitsSampler(new GenerationConfig(1.0f, 0, 0.8f, 17L, false), 3);
        boolean selectedSecondCandidate = false;

        try (DeviceLogits logits = new DeviceLogits(gpu, LOGITS_ADDRESS, 1, 3)) {
            for (int i = 0; i < 32; i++) {
                int tokenId = sampler.selectToken(logits, gpu);
                assertTrue(tokenId == 1 || tokenId == 2);
                selectedSecondCandidate |= tokenId == 2;
            }
            assertTrue(selectedSecondCandidate);
            assertEquals(32, gpu.copyCount);
        }
    }

    @Test
    void multiRowStochasticSelectionSamplesOnlyTheFinalDeviceRow() {
        short[] values = {
            bf16(9.0f), bf16(0.0f), bf16(1.0f),
            bf16(-2.0f), bf16(4.0f), bf16(3.0f)
        };
        FakeGpu gpu = new FakeGpu(LOGITS_ADDRESS, values);
        GenerationConfig config = new GenerationConfig(1.0f, 2, 0.8f, 17L, false);
        LogitsSampler firstSampler = new LogitsSampler(config, 3);
        LogitsSampler secondSampler = new LogitsSampler(config, 3);
        int[] firstSequence = new int[32];
        int[] secondSequence = new int[32];

        try (DeviceLogits logits = new DeviceLogits(gpu, LOGITS_ADDRESS, 2, 3)) {
            for (int i = 0; i < firstSequence.length; i++) {
                int tokenId = firstSampler.selectToken(logits, gpu);
                assertTrue(tokenId == 1 || tokenId == 2);
                firstSequence[i] = tokenId;
                secondSequence[i] = secondSampler.selectToken(logits, gpu);
            }
            assertArrayEquals(firstSequence, secondSequence);
            assertEquals(LOGITS_ADDRESS + 3L * Short.BYTES, gpu.copiedSource);
            assertEquals(3L * Short.BYTES, gpu.copiedBytes);
            assertEquals(64, gpu.copyCount);
            assertEquals(List.of(), gpu.freedAddresses);
        }

        assertEquals(List.of(LOGITS_ADDRESS), gpu.freedAddresses);
    }

    @Test
    void selectRowsSamplesEachRowAtItsPosition() {
        int rows = 24;
        var gpu = new RowsGpu(rows, 4);
        var logits = new HostLogits(gpu, 4);
        logits.queueRows(1 << 20, rows);
        logits.retired(true);
        var config = new GenerationConfig(1.0f, 0, 1.0f, 99L, false);

        int[] selected = new LogitsSampler(config, 4).selectRows(logits, 100);
        assertEquals(rows, selected.length);
        var sampler = new LogitsSampler(config, 4);
        for (int row = 0; row < rows; row++) assertEquals(sampler.selectRow(logits, row, 100 + row), selected[row]);

        var reverse = new LogitsSampler(config, 4);
        for (int row = rows - 1; row >= 0; row--)
            assertEquals(selected[row], reverse.selectRow(logits, row, 100 + row), "row " + row);
        assertArrayEquals(selected, new LogitsSampler(config, 4).selectRows(logits, 100), "a fresh sampler agrees");
        assertTrue(java.util.Arrays.stream(selected).distinct().count() > 1, "the rows are not all one token");

        logits.queueFinalRow(1 << 20, 1);
        logits.retired(true);
        var keyed = new LogitsSampler(config, 4);
        for (long position = 100; position < 100 + rows; position++)
            assertEquals(
                    keyed.selectToken(logits, null, position),
                    keyed.selectToken(logits, token -> true, position),
                    "a keyed selection of a position repeats");
    }

    @Test
    void selectRowsOfADeviceSelectingCopyAreTheDeviceChoices() {
        var gpu = new RowsGpu(3, 4);
        gpu.selectable = true;
        var logits = new HostLogits(gpu, 4);
        logits.selectOnDevice(true);
        logits.queueRowSelections(1 << 20, 3);
        logits.retired(true);
        assertArrayEquals(
                logits.selectedTokens(), new LogitsSampler(GenerationConfig.greedy(1L), 4).selectRows(logits, 0));
    }

    /// Fills each copied row with the same flat logits; device selection, when enabled, picks token 2.
    private static final class RowsGpu extends ExecutionFixtures.RecordingGpu {
        boolean selectable;

        RowsGpu(int rows, int vocabulary) {
            this.nextAddress.set(1L << 40);
        }

        @Override
        public boolean argmaxBf16(long logitsAddress, int count, long resultAddress) {
            return this.selectable;
        }

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            if (byteSize % 8 != 0) return;
            if (this.selectable && byteSize <= 3 * Long.BYTES) {
                for (long offset = 0; offset < byteSize; offset += Long.BYTES)
                    destination.set(ValueLayout.JAVA_LONG, offset, 0xFFFF_FFFFL - 2);
                return;
            }
            short[] flat = {bf16(0.1f), bf16(0.4f), bf16(0.2f), bf16(0.3f)};
            for (long i = 0; i * Short.BYTES < byteSize; i++)
                destination.set(ValueLayout.JAVA_SHORT, i * Short.BYTES, flat[(int) (i % 4)]);
        }
    }

    private static short bf16(float value) {
        return (short) (Float.floatToRawIntBits(value) >>> 16);
    }

    private static final class FakeGpu extends ExecutionGpu {

        private final long baseAddress;
        private final short[] deviceValues;
        private final List<Long> freedAddresses = new ArrayList<>();
        private long copiedSource;
        private long copiedBytes;
        private int copyCount;

        private FakeGpu(long baseAddress, short[] deviceValues) {
            this.baseAddress = baseAddress;
            this.deviceValues = deviceValues;
        }

        @Override
        public long allocate(long byteSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            this.copiedSource = source;
            this.copiedBytes = byteSize;
            this.copyCount++;
            long byteOffset = source - this.baseAddress;
            if (byteOffset < 0 || byteOffset + byteSize > (long) this.deviceValues.length * Short.BYTES) {
                throw new IllegalArgumentException("copy outside fake device allocation");
            }
            int sourceOffset = Math.toIntExact(byteOffset / Short.BYTES);
            for (int i = 0; i < byteSize / Short.BYTES; i++) {
                destination.set(ValueLayout.JAVA_SHORT, (long) i * Short.BYTES, this.deviceValues[sourceOffset + i]);
            }
        }

        @Override
        public void free(long address) {
            this.freedAddresses.add(address);
        }

        @Override
        public void embedQ3(
                long tokenIdsAddress,
                long embeddingAddress,
                long embeddingByteSize,
                long hiddenStateAddress,
                int tokenCount,
                int vocabularySize,
                int hiddenSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void synchronize() {
            throw new UnsupportedOperationException();
        }
    }
}
