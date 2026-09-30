package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class QwenHostLogitsTest {

    @Test
    void theRowIsReadableOnlyAfterItsQuantumRetiredSuccessfully() {
        var gpu = new ReadbackGpu();
        var logits = new QwenHostLogits(gpu, 4);
        assertThrows(IllegalStateException.class, logits::row, "nothing has retired yet");

        logits.queueFinalRow(1000, 1);
        assertThrows(IllegalStateException.class, logits::row, "the copy has not retired");
        logits.retired(true);
        assertEquals(1000, gpu.sources.getLast());
        assertEquals((short) 1000, logits.row().get(ValueLayout.JAVA_SHORT, 0));

        logits.queueFinalRow(2000, 1);
        assertThrows(IllegalStateException.class, logits::row, "a new quantum's copy invalidates the old row");
        logits.retired(false);
        assertThrows(IllegalStateException.class, logits::row, "a failed quantum never exposes its row");
        logits.retired(true);
        assertThrows(IllegalStateException.class, logits::row, "a quantum that queued nothing exposes nothing");
        assertEquals(1, gpu.allocations, "one pinned row per session");
    }

    @Test
    void copiesTheFinalRowOfSeveralLogitsRows() {
        var gpu = new ReadbackGpu();
        var logits = new QwenHostLogits(gpu, 4);
        logits.queueFinalRow(1000, 3);
        logits.retired(true);
        assertEquals(1000 + 2L * 4 * Short.BYTES, gpu.sources.getLast());
    }

    @Test
    void anUnprovenGpuRetainsThePinnedRowAndAProvenOneReleasesItOnce() {
        var gpu = new ReadbackGpu();
        var logits = new QwenHostLogits(gpu, 4);
        logits.queueFinalRow(1000, 1);
        gpu.proven = false;
        logits.close();
        assertEquals(0, gpu.releases, "queued DMA may still write the row");

        gpu.proven = true;
        logits.close();
        logits.close();
        assertEquals(1, gpu.releases);
        assertThrows(IllegalStateException.class, () -> logits.queueFinalRow(1000, 1));
        assertThrows(IllegalStateException.class, logits::row);
    }

    @Test
    void aSessionThatNeverSampledHasNothingToRelease() {
        var gpu = new ReadbackGpu();
        new QwenHostLogits(gpu, 4).close();
        assertEquals(0, gpu.allocations);
        assertFalse(gpu.sources.iterator().hasNext());
        assertTrue(gpu.proven);
    }

    private static final class ReadbackGpu extends QwenExecutionFixtures.RecordingGpu {
        final List<Long> sources = new ArrayList<>();
        int allocations;
        int releases;
        boolean proven = true;

        @Override
        public ReadbackBuffer allocateReadbackBuffer(long bytes) {
            this.allocations++;
            ReadbackBuffer allocated = super.allocateReadbackBuffer(bytes);
            return new ReadbackBuffer(allocated.segment(), () -> {
                this.releases++;
                allocated.close();
            });
        }

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            this.sources.add(source);
            destination.set(ValueLayout.JAVA_SHORT, 0, (short) source);
        }

        @Override
        public boolean completionProven() {
            return this.proven;
        }
    }
}
