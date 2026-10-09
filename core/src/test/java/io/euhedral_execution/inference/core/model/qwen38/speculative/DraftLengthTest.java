package io.euhedral_execution.inference.core.model.qwen38.speculative;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DraftLengthTest {

    /// Drafts a step makes when its drafts carry these log-probabilities.
    private static int drafted(DraftLength length, double... logProbabilities) {
        int drafted = 1;
        double sum = logProbabilities[0];
        while (length.continues(drafted, sum)) sum += logProbabilities[drafted++];
        return drafted;
    }

    @Test
    void aFixedLengthIgnoresConfidence() {
        DraftLength four = DraftLength.fixed(4);
        assertFalse(four.gated());
        assertEquals(4, drafted(four, -9, -9, -9, -9, -9));
        assertEquals(4, drafted(four, 0, 0, 0, 0, 0));
    }

    @Test
    void aGatedLengthDraftsTheLeastThenWhileTheSumHolds() {
        DraftLength gated = new DraftLength(3, 7, 1, -0.5f);
        assertTrue(gated.gated());
        // Unconfident drafts still reach the least.
        assertEquals(3, drafted(gated, -2, -2, -2, -2, -2, -2, -2));
        // Confident drafts run to the most.
        assertEquals(7, drafted(gated, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01));
        // The sum crosses the threshold with the fifth draft, which is still made; the sixth is not.
        assertEquals(5, drafted(gated, -0.1, -0.1, -0.1, -0.1, -0.2, -0.1, -0.1));
        // A sum exactly at the threshold continues.
        assertEquals(4, drafted(gated, -0.25, -0.25, 0, -0.1, 0, 0, 0));
    }

    @Test
    void aBlockStepChecksOnlyAtBlockBoundaries() {
        DraftLength blocks = new DraftLength(3, 6, 3, -0.5f);
        // Confident first block: the whole second block is drafted, however unconfident it is.
        assertEquals(6, drafted(blocks, -0.1, -0.1, -0.1, -5, -5, -5, -5));
        // Unconfident first block: the step stops at the least.
        assertEquals(3, drafted(blocks, -0.1, -0.1, -0.4, 0, 0, 0, 0));
        // The cap cuts a block short.
        assertEquals(7, drafted(new DraftLength(3, 7, 3, -0.5f), 0, 0, 0, 0, 0, 0, 0, 0));
    }

    @Test
    void anUnscoredDraftStopsAtTheLeast() {
        assertEquals(3, drafted(new DraftLength(3, 7, 1, -0.5f), Double.NaN, Double.NaN, Double.NaN, Double.NaN));
    }

    @Test
    void rejectsInvalidLengths() {
        assertThrows(IllegalArgumentException.class, () -> new DraftLength(0, 3, 1, -1));
        assertThrows(IllegalArgumentException.class, () -> new DraftLength(4, 3, 1, -1));
        assertThrows(IllegalArgumentException.class, () -> new DraftLength(1, MtpDecoder.MAX_DRAFTS + 1, 1, -1));
        assertThrows(IllegalArgumentException.class, () -> new DraftLength(1, 3, 1, Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new DraftLength(1, 3, 0, -1));
    }
}
