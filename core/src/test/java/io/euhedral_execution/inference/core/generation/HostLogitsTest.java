package io.euhedral_execution.inference.core.generation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model.qwen38.ExecutionFixtures;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class HostLogitsTest {

    @Test
    void theRowIsReadableOnlyAfterItsQuantumRetiredSuccessfully() {
        var gpu = new ReadbackGpu();
        var logits = new HostLogits(gpu, 4);
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
        var logits = new HostLogits(gpu, 4);
        logits.queueFinalRow(1000, 3);
        logits.retired(true);
        assertEquals(1000 + 2L * 4 * Short.BYTES, gpu.sources.getLast());
    }

    @Test
    void anUnprovenGpuRetainsThePinnedRowAndAProvenOneReleasesItOnce() {
        var gpu = new ReadbackGpu();
        var logits = new HostLogits(gpu, 4);
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
        new HostLogits(gpu, 4).close();
        assertEquals(0, gpu.allocations);
        assertFalse(gpu.sources.iterator().hasNext());
        assertTrue(gpu.proven);
    }

    @Test
    void deviceSelectionReadsBackOnlyTheSelectedToken() {
        var gpu = new ReadbackGpu();
        gpu.selectable = true;
        var logits = new HostLogits(gpu, 4);
        logits.selectOnDevice(true);
        logits.queueFinalRow(1000, 3);
        assertEquals(List.of(1000 + 2L * 4 * Short.BYTES), gpu.selected, "the argmax reads the final row");
        assertFalse(logits.hasSelection(), "the selection has not retired");
        logits.retired(true);
        assertTrue(logits.hasSelection());
        assertEquals(2, logits.selectedToken());
        assertThrows(IllegalStateException.class, logits::row, "no row was copied");
        assertEquals(List.of((long) Long.BYTES), gpu.copyBytes, "only the 8-byte key crosses to the host");

        logits.queueFinalRow(1000, 1);
        logits.retired(false);
        assertFalse(logits.hasSelection(), "a failed quantum never exposes its selection");
        assertThrows(IllegalStateException.class, logits::selectedToken);

        logits.selectOnDevice(false);
        logits.queueFinalRow(1000, 1);
        logits.retired(true);
        assertFalse(logits.hasSelection());
        assertEquals((short) 1000, logits.row().get(ValueLayout.JAVA_SHORT, 0), "host sampling copies the row");

        gpu.proven = true;
        logits.close();
        assertEquals(List.of(gpu.deviceWord), gpu.frees, "the device word is freed with the session");
        assertEquals(2, gpu.releases, "the pinned key and row are released");
    }

    @Test
    void aGpuWithoutDeviceSelectionCopiesTheRowInstead() {
        var gpu = new ReadbackGpu();
        var logits = new HostLogits(gpu, 4);
        logits.selectOnDevice(true);
        logits.queueFinalRow(1000, 1);
        logits.retired(true);
        assertFalse(logits.hasSelection());
        assertEquals((short) 1000, logits.row().get(ValueLayout.JAVA_SHORT, 0));
    }

    @Test
    void aRowWithNoSelectableLogitFailsLikeTheHostArgmax() {
        var gpu = new ReadbackGpu();
        gpu.selectable = true;
        gpu.key = 0;
        var logits = new HostLogits(gpu, 4);
        logits.selectOnDevice(true);
        logits.queueFinalRow(1000, 1);
        logits.retired(true);
        assertThrows(IllegalArgumentException.class, logits::selectedToken);
    }

    @Test
    void copiesEveryRowAndExposesThemFromRetirement() {
        var gpu = new ReadbackGpu();
        var logits = new HostLogits(gpu, 4);
        assertThrows(IllegalStateException.class, () -> logits.row(0), "nothing was queued");
        assertThrows(IllegalStateException.class, logits::rowCount);

        logits.queueRows(1000, 3);
        assertEquals(3, logits.rowCount(), "acceptance reads the rows before the quantum retires");
        assertEquals(List.of(1000L), gpu.sources, "one copy of all three rows");
        assertEquals(List.of(3L * 4 * Short.BYTES), gpu.copyBytes);
        logits.retired(true);
        assertEquals(3, logits.rowCount());
        assertEquals((short) 1000, logits.row(0).get(ValueLayout.JAVA_SHORT, 0));
        assertEquals(4L * Short.BYTES, logits.row(1).byteSize());
        for (int row = 0; row < 3; row++)
            for (int token = 0; token < 4; token++)
                assertEquals(
                        (short) (1000 + row * 4 + token),
                        logits.row(row).get(ValueLayout.JAVA_SHORT, (long) token * Short.BYTES),
                        "row " + row + " token " + token);
        assertThrows(IndexOutOfBoundsException.class, () -> logits.row(3));

        logits.retired(false);
        assertThrows(IllegalStateException.class, () -> logits.row(1), "nothing is queued, nothing stays readable");
        assertThrows(IllegalStateException.class, logits::rowCount);

        logits.queueRows(2000, 2);
        logits.retired(false);
        assertThrows(IllegalStateException.class, () -> logits.row(0), "a failed quantum never exposes its rows");
        logits.queueRows(2000, 5);
        logits.retired(true);
        assertEquals(5, logits.rowCount(), "the buffer grew between quanta");
        assertEquals(2, gpu.allocations, "the first buffer was replaced by a larger one");
        assertEquals(1, gpu.releases);
        logits.queueFinalRow(3000, 1);
        assertThrows(IllegalStateException.class, () -> logits.row(0), "a new quantum invalidates the old rows");
    }

    private static final class ReadbackGpu extends ExecutionFixtures.RecordingGpu {
        final List<Long> sources = new ArrayList<>();
        final List<Long> selected = new ArrayList<>();
        final List<Long> copyBytes = new ArrayList<>();
        int allocations;
        int releases;
        boolean proven = true;
        boolean selectable;
        long deviceWord;
        long key = 0xFFFF_FFFFL - 2;

        ReadbackGpu() {
            // Device words far from the logits addresses the tests pass.
            this.nextAddress.set(1L << 40);
        }

        @Override
        public long allocate(long byteSize) {
            this.deviceWord = super.allocate(byteSize);
            return this.deviceWord;
        }

        @Override
        public boolean argmaxBf16(long logitsAddress, int count, long resultAddress) {
            if (!this.selectable) return false;
            assertEquals(4, count);
            assertEquals(this.deviceWord, resultAddress);
            this.selected.add(logitsAddress);
            return true;
        }

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
            this.copyBytes.add(byteSize);
            if (source == this.deviceWord && byteSize == Long.BYTES) {
                destination.set(ValueLayout.JAVA_LONG, 0, this.key);
                return;
            }
            this.sources.add(source);
            // Element i of the copy is source + i, so every row's content is distinct.
            for (long i = 0; i * Short.BYTES < byteSize; i++)
                destination.set(ValueLayout.JAVA_SHORT, i * Short.BYTES, (short) (source + i));
        }

        @Override
        public boolean completionProven() {
            return this.proven;
        }
    }
}
