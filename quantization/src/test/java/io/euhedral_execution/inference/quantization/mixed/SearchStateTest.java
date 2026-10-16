package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.quantization.mixed.SearchState.Measured;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SearchStateTest {

    @TempDir
    Path directory;

    private static Allocation allocation(Option first, Option second) {
        return new Allocation(Map.of("text/layers/0/mlp/down", first, "text/layers/1/mlp/down", second));
    }

    private static Measured measured(Allocation allocation, double kl, int round) {
        return new Measured(allocation, kl, 0.001, 1234, 1.5, 2.5, true, round, "test");
    }

    @Test
    void whatIsRecordedSurvivesAStopAndComesBackWithoutDuplicates() throws Exception {
        Path file = directory.resolve("search.resume");
        try (SearchState state = SearchState.open(file, "fingerprint", 17, 1L << 30)) {
            assertTrue(state.add(measured(allocation(Option.NVFP4, Option.NVFP4), 0.1, 0)));
            assertTrue(state.add(measured(allocation(Option.NVFP4, Option.TRELLIS_3_0), 0.2, 1)));
            state.round(1);
            state.explore(0.4);
            state.stale(2);
            state.heldOut(0.07);
            // The stop: a candidate of the same batch is offered again after the restart.
        }
        try (SearchState state = SearchState.open(file, "fingerprint", 99, 5)) {
            assertEquals(2, state.measured().size());
            assertEquals(17, state.seed(), "the seed of the first run is kept");
            assertEquals(1L << 30, state.freeBytes());
            assertEquals(1, state.round());
            assertEquals(0.4, state.explore(), 0);
            assertEquals(2, state.stale());
            assertEquals(0.07, state.heldOut(), 0);
            assertTrue(state.contains(allocation(Option.NVFP4, Option.TRELLIS_3_0)));
            assertFalse(state.add(measured(allocation(Option.NVFP4, Option.TRELLIS_3_0), 0.9, 1)), "no duplicate");
            assertTrue(state.add(measured(allocation(Option.TRELLIS_3_0, Option.NVFP4), 0.3, 1)));
            assertEquals(3, state.measured().size());
            Measured first = state.measured().get(0);
            assertEquals(0.1, first.kl(), 0);
            assertEquals(1234, first.bytes());
            assertEquals("test", first.origin());
        }
        try (SearchState state = SearchState.open(file, "fingerprint", 0, 0)) {
            assertEquals(3, state.measured().size());
        }
    }

    @Test
    void aRecordThatWasBeingWrittenWhenTheRunStoppedIsNotKept() throws Exception {
        Path file = directory.resolve("search.resume");
        try (SearchState state = SearchState.open(file, "fingerprint", 1, 1)) {
            state.add(measured(allocation(Option.NVFP4, Option.NVFP4), 0.1, 0));
            // Bytes of a record that never reached the header: past the recorded end.
            state.writeUnrecordedBytesForTest(new byte[] {0, 0, 0, 9, '{', '"', 'a'});
        }
        try (SearchState state = SearchState.open(file, "fingerprint", 1, 1)) {
            assertEquals(1, state.measured().size());
            assertTrue(state.add(measured(allocation(Option.NVFP4, Option.TRELLIS_3_0), 0.2, 0)));
        }
        try (SearchState state = SearchState.open(file, "fingerprint", 1, 1)) {
            assertEquals(2, state.measured().size());
        }
    }

    @Test
    void aStateOfAnotherSearchIsRefused() throws Exception {
        Path file = directory.resolve("search.resume");
        try (SearchState state = SearchState.open(file, "one", 1, 1)) {
            state.add(measured(allocation(Option.NVFP4, Option.NVFP4), 0.1, 0));
        }
        IllegalStateException failure =
                assertThrows(IllegalStateException.class, () -> SearchState.open(file, "two", 1, 1));
        assertTrue(failure.getMessage().contains("different search"), failure.getMessage());
    }
}
