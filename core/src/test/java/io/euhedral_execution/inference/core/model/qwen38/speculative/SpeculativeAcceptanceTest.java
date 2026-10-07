package io.euhedral_execution.inference.core.model.qwen38.speculative;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class SpeculativeAcceptanceTest {
    private static final int EOS = 99;

    private static SpeculativeAcceptance resolve(int[] rows, int[] selected, int maxOutputs) {
        var acceptance = new SpeculativeAcceptance(rows, token -> token == EOS, maxOutputs);
        acceptance.resolve(selected);
        return acceptance;
    }

    @Test
    void acceptsTheMatchingDraftPrefixThenTheFirstUnmatchedBaseToken() {
        // Rows [t0, d1, d2, d3]; selections g0..g3.
        var none = resolve(new int[] {5, 6, 7, 8}, new int[] {1, 2, 3, 4}, 16);
        assertEquals(0, none.acceptedDrafts());
        assertArrayEquals(new int[] {1}, none.outputs());
        assertEquals(1, none.committedRows());

        var two = resolve(new int[] {5, 6, 7, 8}, new int[] {6, 7, 3, 4}, 16);
        assertEquals(2, two.acceptedDrafts());
        assertArrayEquals(new int[] {6, 7, 3}, two.outputs());
        assertEquals(3, two.committedRows());

        var all = resolve(new int[] {5, 6, 7, 8}, new int[] {6, 7, 8, 9}, 16);
        assertEquals(3, all.acceptedDrafts());
        assertArrayEquals(new int[] {6, 7, 8, 9}, all.outputs(), "the full step adds the verifier's token");
        assertEquals(4, all.committedRows());

        // The rejected draft's base token: g at the first unmatched draft; none when all were accepted.
        assertEquals(1, none.rejectedBaseToken());
        assertEquals(3, two.rejectedBaseToken());
        assertEquals(-1, all.rejectedBaseToken());
    }

    @Test
    void stopsAfterTheFirstEndTokenAndAtTheBudget() {
        var acceptedEnd = resolve(new int[] {5, EOS, 7, 8}, new int[] {EOS, 7, 8, 9}, 16);
        assertEquals(3, acceptedEnd.acceptedDrafts());
        assertArrayEquals(new int[] {EOS}, acceptedEnd.outputs(), "the end token is never fed back");
        assertEquals(1, acceptedEnd.committedRows());

        var baseEnd = resolve(new int[] {5, 6, 7, 8}, new int[] {6, EOS, 1, 2}, 16);
        assertArrayEquals(new int[] {6, EOS}, baseEnd.outputs());
        assertEquals(2, baseEnd.committedRows());

        var budget = resolve(new int[] {5, 6, 7, 8}, new int[] {6, 7, 8, 9}, 2);
        assertArrayEquals(new int[] {6, 7}, budget.outputs());
        assertEquals(2, budget.committedRows());
    }
}
