package io.euhedral_execution.inference.core.artifact.format;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile.Quantization;
import java.util.List;
import org.junit.jupiter.api.Test;

class TrellisFormatTest {

    private static final LinearFormat FORMAT = LinearFormats.find(WeightFormat.TRELLIS, WeightLayout.TRELLIS_TILE16_V1)
            .orElseThrow();

    /// mlp/gate_up: k=5120, n=34816.
    private static final long[] GATE_UP = {320, 2176, 56};

    private static TensorDescriptor descriptor(long[] shape) {
        return new TensorDescriptor(
                "text/layers/0/mlp/gate_up",
                shape,
                TensorDataType.BF16,
                WeightFormat.TRELLIS,
                WeightLayout.TRELLIS_TILE16_V1,
                0,
                1);
    }

    private static ExpectedTensor expected(long n, long k) {
        return new ExpectedTensor(
                new long[] {n, k}, TensorDataType.BF16, WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_K128_V1);
    }

    @Test
    void theByteSizeIsTwoBytesPerWordOfEveryTile() {
        for (int w : new int[] {40, 48, 56, 64}) {
            long[] shape = {320, 2176, w};
            assertEquals(320L * 2176 * w * 2, FORMAT.byteSize(shape, TensorDataType.BF16));
            assertTrue(FORMAT.acceptsByteSize(shape, TensorDataType.BF16, 320L * 2176 * w * 2));
            assertFalse(FORMAT.acceptsByteSize(shape, TensorDataType.BF16, 320L * 2176 * w * 2 + 1));
        }
        assertThrows(IllegalArgumentException.class, () -> FORMAT.byteSize(GATE_UP, TensorDataType.FP16));
    }

    @Test
    void onlyOneToEightBitsInHalfStepsAreStoredWidths() {
        for (int k = 1; k <= 8; k++) assertTrue(TrellisLayout.wordsValid(16 * k), "w=" + 16 * k);
        for (int k = 1; k <= 7; k++) assertTrue(TrellisLayout.wordsValid(16 * k + 8), "w=" + (16 * k + 8));
        for (int w : new int[] {0, 8, 12, 17, 129, 136, -16, 144}) {
            assertFalse(TrellisLayout.wordsValid(w), "w=" + w);
            if (w > 0) {
                long[] shape = {320, 2176, w};
                assertThrows(IllegalArgumentException.class, () -> FORMAT.byteSize(shape, TensorDataType.BF16));
                assertFalse(FORMAT.accepts(descriptor(shape), expected(34816, 5120)), "w=" + w);
            }
        }
        assertEquals(3.5, TrellisLayout.bits(56));
        assertEquals(3.0, TrellisLayout.bits(48));
        assertEquals(1.5, TrellisLayout.bits(24));
    }

    @Test
    void kAndNMustBeMultiplesOfOneHundredTwentyEight() {
        // k = 80, then n = 34832.
        for (long[] shape : new long[][] {{5, 2176, 56}, {320, 2177, 56}}) {
            assertThrows(IllegalArgumentException.class, () -> FORMAT.byteSize(shape, TensorDataType.BF16));
            assertFalse(FORMAT.accepts(descriptor(shape), expected(shape[1] * 16, shape[0] * 16)));
        }
        assertThrows(
                IllegalArgumentException.class, () -> FORMAT.byteSize(new long[] {320, 2176}, TensorDataType.BF16));
        assertThrows(
                IllegalArgumentException.class,
                () -> FORMAT.byteSize(new long[] {320, 2176, 56, 1}, TensorDataType.BF16));
        assertFalse(FORMAT.accepts(descriptor(new long[] {320, 2176}), expected(34816, 5120)));
    }

    @Test
    void thePayloadStandsForTheLogicalNByKOfAnyQuantizedRole() {
        assertTrue(FORMAT.accepts(descriptor(GATE_UP), expected(34816, 5120)));
        for (WeightFormat role : List.of(
                WeightFormat.Q3_G64_FP16, WeightFormat.Q4_G64_FP16, WeightFormat.Q5_G64_FP16, WeightFormat.NVFP4))
            assertTrue(FORMAT.accepts(
                    descriptor(GATE_UP),
                    new ExpectedTensor(
                            new long[] {34816, 5120}, TensorDataType.BF16, role, WeightLayout.ROW_SPLIT_K128_V1)));
        assertFalse(FORMAT.accepts(descriptor(GATE_UP), expected(5120, 34816)));
        assertFalse(FORMAT.accepts(
                descriptor(GATE_UP),
                new ExpectedTensor(
                        new long[] {34816, 5120},
                        TensorDataType.BF16,
                        WeightFormat.BF16,
                        WeightLayout.CONTIGUOUS_LE_V1)));
        TensorDescriptor fp16 = new TensorDescriptor(
                "n", GATE_UP, TensorDataType.FP16, WeightFormat.TRELLIS, WeightLayout.TRELLIS_TILE16_V1, 0, 1);
        assertFalse(FORMAT.accepts(fp16, expected(34816, 5120)));
    }

    @Test
    void theFormatAnswersItsStorageQuestions() {
        assertEquals(List.of("text/x.suh", "text/x.svh", "text/x.segments"), FORMAT.companions("text/x"));
        assertTrue(FORMAT.stageable());
        assertTrue(FORMAT.splitsRows());
        assertFalse(FORMAT.compressed());
        assertEquals(Quantization.TRELLIS, FORMAT.quantization());
        assertEquals(FORMAT, LinearFormats.of(descriptor(GATE_UP)));
    }

    @Test
    void segmentsAreIncreasingAlignedAndEndAtN() {
        TrellisLayout.validateSegments(new int[] {17408, 34816}, 34816, 2);
        TrellisLayout.validateSegments(new int[] {5120}, 5120, 1);
        int[] eight = new int[8];
        for (int index = 0; index < 8; index++) eight[index] = 128 * (index + 1);
        TrellisLayout.validateSegments(eight, 1024, 8);

        // not increasing, repeated, unaligned, last is not n, count differs from suh rows, more than 8, none
        for (Object[] bad : new Object[][] {
            {new int[] {34816, 17408}, 34816L, 2L},
            {new int[] {17408, 17408}, 17408L, 2L},
            {new int[] {17400, 34816}, 34816L, 2L},
            {new int[] {17408, 34688}, 34816L, 2L},
            {new int[] {17408, 34816}, 34816L, 3L},
            {new int[] {128, 256, 384, 512, 640, 768, 896, 1024, 1152}, 1152L, 9L},
            {new int[] {}, 128L, 0L},
            {new int[] {0, 128}, 128L, 2L}
        })
            assertThrows(
                    IllegalArgumentException.class,
                    () -> TrellisLayout.validateSegments((int[]) bad[0], (long) bad[1], (long) bad[2]));
        int[] ends = {6144, 7168};
        assertEquals(0, TrellisLayout.segmentOf(ends, 0));
        assertEquals(0, TrellisLayout.segmentOf(ends, 6143));
        assertEquals(1, TrellisLayout.segmentOf(ends, 6144));
        assertEquals(1, TrellisLayout.segmentOf(ends, 7167));
        assertThrows(IllegalArgumentException.class, () -> TrellisLayout.segmentOf(ends, 7168));
    }

    @Test
    void thePrefillScratchHoldsTheDecodedObjectTheRotatedActivationsAndTheOutputs() {
        long n = 34816, k = 5120, rows = 512;
        long expected = 8 * rows * k * 2 + n * k * 2 + rows * n * 2;
        // A row limit below the rows reserves the prefill route.
        TrellisFormat prefill = new TrellisFormat(new TrellisSettings(64, 560, 128));
        long aligned = prefill.planScratchBytes(descriptor(GATE_UP), 512);
        // Every region is 256-aligned, and these are.
        assertEquals(expected, aligned);
        assertTrue(prefill.planScratchBytes(descriptor(GATE_UP), 1024) > aligned);
        // At the default limit every 512-row chunk stays on the tensor-core route, which needs less.
        assertTrue(FORMAT.planScratchBytes(descriptor(GATE_UP), 512) < aligned);
        long small = prefill.planScratchBytes(descriptor(new long[] {40, 8, 56}), 100);
        assertTrue(small >= 128L * 640 * 2 + 100L * 128 * 2 + 8 * 100 * 640 * 2);
        assertEquals(0, small % 256);
        assertEquals(
                0,
                LinearFormats.find(WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1)
                        .orElseThrow()
                        .planScratchBytes(descriptor(GATE_UP), 512));
    }

    @Test
    void launchesFromTheWholeKRowsKeepNoSplitPartials() {
        // gate_up: k = 5120, n = 34816, three K splits at 560 target blocks.
        long k = 5120, n = 34816, splits = 3;
        TrellisFormat format = new TrellisFormat(new TrellisSettings(512, 560, 128));
        assertEquals(8 * 127 * k * 2 + splits * 127 * n * 4, format.planScratchBytes(descriptor(GATE_UP), 127));
        assertEquals(8 * 512 * k * 2 + splits * 127 * n * 4, format.planScratchBytes(descriptor(GATE_UP), 512));
        assertEquals(8 * 16 * k * 2 + splits * 16 * n * 4, format.planScratchBytes(descriptor(GATE_UP), 16));
    }
}
