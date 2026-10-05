package io.euhedral_execution.inference.core.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/// The layer's host-side decisions: the scratch layout, the key splits and the scores tiling, without a GPU.
class Qwen4QsaLayerTest {

    private static Qwen4QsaLayer.Config config(int maxTokens) {
        return new Qwen4QsaLayer.Config(2560, 24, 2, 256, 64, 1.0e7f, 1.0e-6f, 4, 128, 4, 2048, maxTokens);
    }

    @Test
    void scratchRegionsAreAlignedAndDisjoint() {
        Qwen4QsaLayer layer = new Qwen4QsaLayer(config(262144));
        for (int rows : new int[] {1, 2, 8, 64, 255, 256, 512, 3000}) {
            Qwen4QsaLayer.Scratch s = layer.scratch(4096, rows);
            long[] starts = {
                s.qProj(),
                s.kProj(),
                s.kNormed(),
                s.vProj(),
                s.indexProj(),
                s.scores(),
                s.ids(),
                s.counts(),
                s.partial(),
                s.gated(),
                s.end()
            };
            for (int i = 0; i < starts.length; i++) {
                assertEquals(0, starts[i] % 256, "region " + i + " at " + rows + " rows");
                if (i > 0) assertTrue(starts[i] >= starts[i - 1], "regions ascend");
            }
            assertEquals(layer.scratchBytes(rows), s.end() - 4096);
            assertTrue(s.ids() - s.scores() >= 262144 / 4 * 4L, "the scores hold one row of the longest sequence");
        }
    }

    @Test
    void partialsCoverEverySplitCountASmallerCallUses() {
        Qwen4QsaLayer layer = new Qwen4QsaLayer(config(262144));
        for (int sized : new int[] {1, 8, 64, 512}) {
            Qwen4QsaLayer.Scratch s = layer.scratch(0, sized);
            long available = s.gated() - s.partial();
            for (int rows = 1; rows <= sized; rows++) {
                int splits = layer.splitsFor(rows, 2051);
                if (splits == 1) continue;
                long needed = (long) rows * 24 * splits * Qwen4QsaOps.PARTIAL_FLOATS * Float.BYTES;
                assertTrue(needed <= available, rows + " rows with " + splits + " splits in scratch for " + sized);
            }
        }
    }

    @Test
    void splitsKeepEnoughKeysPerSplitAndDropOutForLargeChunks() {
        Qwen4QsaLayer layer = new Qwen4QsaLayer(config(262144));
        assertEquals(64, layer.splitsFor(1, 2051));
        assertEquals(1, layer.splitsFor(1, 20));
        assertEquals(1, layer.splitsFor(512, 2051));
        assertEquals(1, layer.splitsFor(300, 2051));
        for (int rows = 1; rows < 600; rows++) {
            int splits = layer.splitsFor(rows, 2051);
            assertTrue(splits >= 1 && splits <= 64);
            assertTrue((long) rows * 2 * splits <= Math.max(512, 2L * rows), "units stay bounded at " + rows);
        }
    }

    @Test
    void scoreTilesFitTheScratchAndCoverAtLeastOneRow() {
        Qwen4QsaLayer layer = new Qwen4QsaLayer(config(262144));
        assertEquals(64, layer.scoreTileRows(512, 262144));
        assertEquals(512, layer.scoreTileRows(512, 4096));
        Qwen4QsaLayer tiny = new Qwen4QsaLayer(config(262144), 1);
        assertEquals(1, tiny.scoreTileRows(512, 262144));
        assertEquals(1, tiny.scoreTileRows(1, 100));
    }

    @Test
    void unsupportedGeometryIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new Qwen4QsaLayer(
                        new Qwen4QsaLayer.Config(2560, 24, 2, 128, 64, 1.0e7f, 1.0e-6f, 4, 128, 4, 2048, 4096)));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Qwen4QsaLayer(
                        new Qwen4QsaLayer.Config(2560, 24, 2, 256, 64, 1.0e7f, 1.0e-6f, 4, 128, 8, 2048, 4096)));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Qwen4QsaLayer(
                        new Qwen4QsaLayer.Config(2560, 24, 2, 256, 64, 1.0e7f, 1.0e-6f, 4, 128, 4, 2048, 0)));
    }
}
