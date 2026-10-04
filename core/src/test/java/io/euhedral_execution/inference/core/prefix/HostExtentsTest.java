package io.euhedral_execution.inference.core.prefix;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class HostExtentsTest {
    private static final long KIB = 1024;

    @Test
    void allocatesFirstFitInAlignedUnits() {
        var extents = new HostExtents(64 * KIB);
        assertEquals(0, extents.allocate(1));
        assertEquals(4 * KIB, extents.allocate(5000));
        assertEquals(12 * KIB, extents.allocate(4 * KIB));
        assertEquals(64 * KIB - 16 * KIB, extents.freeBytes());
        assertEquals(16 * KIB, extents.usedBytes());
    }

    @Test
    void refusesWhatDoesNotFitAndKeepsTheRange() {
        var extents = new HostExtents(16 * KIB);
        assertEquals(0, extents.allocate(16 * KIB));
        assertEquals(-1, extents.allocate(1));
        extents.free(0, 16 * KIB);
        assertEquals(0, extents.allocate(16 * KIB));
    }

    @Test
    void coalescesNeighboursOnFree() {
        var extents = new HostExtents(24 * KIB);
        long a = extents.allocate(8 * KIB);
        long b = extents.allocate(8 * KIB);
        long c = extents.allocate(8 * KIB);
        extents.free(a, 8 * KIB);
        extents.free(c, 8 * KIB);
        assertEquals(-1, extents.allocate(16 * KIB), "the holes are not adjacent yet");
        extents.free(b, 8 * KIB);
        assertEquals(0, extents.allocate(24 * KIB));
    }

    @Test
    void rejectsDoubleFreeAndForeignExtents() {
        var extents = new HostExtents(16 * KIB);
        long a = extents.allocate(4 * KIB);
        extents.free(a, 4 * KIB);
        assertThrows(IllegalArgumentException.class, () -> extents.free(a, 4 * KIB));
        assertThrows(IllegalArgumentException.class, () -> extents.free(12 * KIB, 8 * KIB));
        assertThrows(IllegalArgumentException.class, () -> extents.free(1, 4 * KIB));
        assertThrows(IllegalArgumentException.class, () -> extents.allocate(0));
    }

    @Test
    void anEmptyRangeNeverAllocates() {
        var extents = new HostExtents(0);
        assertEquals(-1, extents.allocate(1));
        assertEquals(0, extents.totalBytes());
    }
}
