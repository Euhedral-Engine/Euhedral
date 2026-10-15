package io.euhedral_execution.inference.quantization.trellis;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class TrellisKernelsTest {

    @Test
    void scratchBelowOneTileIsRejected() {
        for (double bits : new double[] {2, 2.5, 3, 3.5, 4}) {
            long tile = TrellisKernels.scratchBytes(1, bits);
            assertThrows(IllegalArgumentException.class, () -> TrellisKernels.scratchTiles(tile - 1, bits));
            assertThrows(IllegalArgumentException.class, () -> TrellisKernels.scratchTiles(0, bits));
            assertEquals(1, TrellisKernels.scratchTiles(tile, bits));
            assertEquals(3, TrellisKernels.scratchTiles(3 * tile + 5, bits));
        }
    }

    @Test
    void unsupportedRatesAreRejected() {
        for (double bits : new double[] {1, 1.5, 4.5, 7, 8, 3.25})
            assertThrows(IllegalArgumentException.class, () -> TrellisKernels.wordsPerTile(bits));
    }
}
