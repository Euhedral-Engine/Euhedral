package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.state.KvFormat;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class AttentionLaunchGeometryTest {

    /// The model's query heads per KV head.
    private static final int GROUP = 6;

    /// `fused_from` in native/src/host/qwen_layer_ops.c for rows 2 to 8.
    private static final long[] FUSED_FROM_BY_ROWS = {49152, 16384, 4096, 2048, 16384, 16384, 12288};

    private static long key(KvFormat format, int rows, long last) {
        return AttentionLaunchGeometry.of(GROUP, format, last - rows, rows);
    }

    @Test
    void thresholdsAreThoseOfTheNativeDispatch() {
        for (int rows = 2; rows <= 8; rows++)
            assertEquals(FUSED_FROM_BY_ROWS[rows - 2], AttentionLaunchGeometry.fusedFrom(rows), "rows " + rows);
        assertEquals(Long.MAX_VALUE, AttentionLaunchGeometry.fusedFrom(0));
        assertEquals(Long.MAX_VALUE, AttentionLaunchGeometry.fusedFrom(1));
        assertEquals(Long.MAX_VALUE, AttentionLaunchGeometry.fusedFrom(9));
    }

    @Test
    void theFusedChoiceFlipsExactlyWhereTheLastRowCrossesTheThreshold() {
        for (int rows = 2; rows <= 8; rows++) {
            long from = FUSED_FROM_BY_ROWS[rows - 2];
            for (long last = Math.max(rows, 1); last <= 70_000; last++) {
                assertEquals(
                        last >= from,
                        AttentionLaunchGeometry.fused(GROUP, KvFormat.NVFP4, rows, last),
                        "rows " + rows + " last " + last);
                assertEquals(
                        last >= from,
                        (key(KvFormat.NVFP4, rows, last) >> 16 & 1) == 1,
                        "key bit, rows " + rows + " last " + last);
            }
        }
    }

    @Test
    void thePositionChangesTheKeyOnlyWhereTheFusedChoiceFlipsOnceSplitsReachTheirCap() {
        // From 2048 keys both split counts are capped at 64, so the fused choice is all that varies with position.
        // The scan starts at 4097 keys, where every row is past the switch; a threshold at or under that (rows 4 and
        // 5) leaves no change in range.
        for (int rows = 2; rows <= 8; rows++) {
            long from = FUSED_FROM_BY_ROWS[rows - 2];
            List<Long> changes = new ArrayList<>();
            long start = 4097;
            for (long last = start; last <= 70_000; last++)
                if (key(KvFormat.NVFP4, rows, last) != key(KvFormat.NVFP4, rows, last - 1)) changes.add(last);
            assertEquals(from > start ? List.of(from) : List.of(), changes, "rows " + rows);
        }
    }

    @Test
    void theKeyDiffersAcrossTheThresholdAndNotWithinASide() {
        for (int rows = 2; rows <= 8; rows++) {
            long from = FUSED_FROM_BY_ROWS[rows - 2];
            assertNotEquals(key(KvFormat.NVFP4, rows, from - 1), key(KvFormat.NVFP4, rows, from), "rows " + rows);
            // Eight keys on, the first row of the step is past the 2048-key switch too.
            assertEquals(key(KvFormat.NVFP4, rows, from + 8), key(KvFormat.NVFP4, rows, from + 1000), "rows " + rows);
            assertEquals(
                    key(KvFormat.NVFP4, rows, from + 3000), key(KvFormat.NVFP4, rows, from + 1000), "rows " + rows);
            if (from > 4096) {
                assertEquals(key(KvFormat.NVFP4, rows, 3000), key(KvFormat.NVFP4, rows, from - 1), "rows " + rows);
            }
        }
    }

    @Test
    void theKeyKeepsTheSplitCountsBeneathTheFusedChoice() {
        // 4100 keys: 129 splits of 32 keys capped at 64; 2000 keys: 42 splits of 48 keys.
        assertEquals(64L, key(KvFormat.NVFP4, 3, 4100) & 0xFF);
        assertEquals(42L, key(KvFormat.NVFP4, 3, 2000) >> 8 & 0xFF);
        assertEquals(0L, key(KvFormat.NVFP4, 3, 2000) & 0xFF);
    }

    @Test
    void rowsAndCachesWithoutTheFusedKernelNeverFuse() {
        for (long last = 100; last <= 70_000; last += 7) {
            assertFalse(AttentionLaunchGeometry.fused(GROUP, KvFormat.BF16, 4, last), "bf16 " + last);
            assertFalse(AttentionLaunchGeometry.fused(GROUP, KvFormat.NVFP4, 1, last), "one row " + last);
            assertFalse(AttentionLaunchGeometry.fused(GROUP, KvFormat.NVFP4, 9, last), "nine rows " + last);
            // 8 rows of 8 query heads need 8 tiles; a group above 8 never takes the GQA kernels.
            assertFalse(AttentionLaunchGeometry.fused(8, KvFormat.NVFP4, 8, last), "eight tiles " + last);
            assertFalse(AttentionLaunchGeometry.fused(9, KvFormat.NVFP4, 4, last), "group nine " + last);
        }
        assertTrue(AttentionLaunchGeometry.fused(GROUP, KvFormat.NVFP4, 8, 70_000));
    }
}
