package io.euhedral_execution.inference.core.generation;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class TokenRecordTest {
    @Test
    void growsAndSnapshotsInOrder() {
        var record = new TokenRecord();
        List<Integer> expected = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            record.add(i * 3);
            expected.add(i * 3);
        }
        assertEquals(expected, record.snapshot());
        assertEquals(List.of(), new TokenRecord().snapshot());
    }
}
