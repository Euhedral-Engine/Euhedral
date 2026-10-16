package io.euhedral_execution.inference.quantization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/// The number of chunks a scorer forces together, from the memory figures of a model.
class ScorerBatchTest {

    private static final long GIB = 1L << 30;

    @Test
    void memoryLimitsTheBatch() {
        // 10 GiB free, one sequence needs 6 GiB, each more 1 GiB: 6, 7, 8, 9, 10 GiB fit, so 5 sequences.
        assertEquals(5, Scorer.maxBatch(10 * GIB, 6 * GIB, GIB, 8, 512, 64));
    }

    @Test
    void theRowLimitLimitsTheBatch() {
        // Plenty of memory; 32 rows of 8 per chunk is 4 chunks; 512 rows is 64.
        assertEquals(4, Scorer.maxBatch(100 * GIB, 6 * GIB, GIB, 8, 32, 64));
        assertEquals(64, Scorer.maxBatch(100 * GIB, 6 * GIB, GIB, 8, 512, 100));
        assertEquals(5, Scorer.maxBatch(100 * GIB, 6 * GIB, GIB, 8, 45, 100));
    }

    @Test
    void theChunkCountLimitsTheBatch() {
        assertEquals(8, Scorer.maxBatch(100 * GIB, 6 * GIB, GIB, 8, 512, 8));
    }

    @Test
    void oneChunkAtATimeWhenOneSequenceBarelyFitsOrNot() {
        assertEquals(1, Scorer.maxBatch(6 * GIB, 6 * GIB, GIB, 8, 512, 8));
        assertEquals(1, Scorer.maxBatch(5 * GIB, 6 * GIB, GIB, 8, 512, 8));
    }

    @Test
    void aRowCountOverTheLimitIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> Scorer.maxBatch(GIB, GIB, GIB, 8, 4, 8));
        assertThrows(IllegalArgumentException.class, () -> Scorer.maxBatch(GIB, GIB, 0, 8, 32, 8));
    }

    @Test
    void anExplicitBatchIsCheckedAgainstTheRowLimit() {
        assertEquals(4, Scorer.checkedBatch(4, 8, 32));
        assertThrows(IllegalArgumentException.class, () -> Scorer.checkedBatch(5, 8, 32));
    }
}
