package io.euhedral_execution.inference.core.testing;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.BitSet;
import org.junit.jupiter.api.Test;

class WorkerCpusTest {

    private static BitSet cpus(int... values) {
        BitSet set = new BitSet();
        for (int value : values) set.set(value);
        return set;
    }

    @Test
    void parsesCpusAndInclusiveRanges() {
        assertEquals(cpus(2, 3, 4, 5), WorkerCpus.parse("2-5"));
        assertEquals(cpus(2, 3, 8), WorkerCpus.parse("2,3,8"));
        assertEquals(cpus(0, 1, 4), WorkerCpus.parse(" 0-1 , 4 "));
        assertThrows(IllegalArgumentException.class, () -> WorkerCpus.parse("5-2"));
        assertThrows(IllegalArgumentException.class, () -> WorkerCpus.parse(","));
    }
}
