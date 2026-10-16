package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class AllocationTest {

    @TempDir
    Path directory;

    @Test
    void roundTripsWithSortedKeys() throws Exception {
        var allocation = new Allocation(Map.of(
                "text/layers/1/mlp/down", Option.TRELLIS_3_5,
                "text/layers/0/mlp/down", Option.NVFP4,
                "text/layers/0/mlp/gate_up", Option.TRELLIS_2_5));
        Path file = directory.resolve("allocation.json");
        allocation.write(file);
        assertEquals(allocation, Allocation.read(file));
        String text = Files.readString(file);
        assertTrue(text.indexOf("layers/0/mlp/down") < text.indexOf("layers/0/mlp/gate_up"));
        assertTrue(text.indexOf("layers/0/mlp/gate_up") < text.indexOf("layers/1/mlp/down"));
        assertTrue(text.contains("\"trellis-3.5\""));
    }

    @Test
    void readsTheDocumentedShape() throws Exception {
        Path file = directory.resolve("allocation.json");
        Files.writeString(file, "{\"objects\": {\"text/layers/0/mlp/down\": \"trellis-3.0\"}}");
        assertEquals(
                Map.of("text/layers/0/mlp/down", Option.TRELLIS_3_0),
                Allocation.read(file).choices());
    }

    @Test
    void refusesAnUnknownOptionKeyNamingItAndItsObject() throws Exception {
        Path file = directory.resolve("allocation.json");
        Files.writeString(file, "{\"objects\": {\"text/layers/0/mlp/down\": \"trellis-9.9\"}}");
        var failure = assertThrows(IllegalArgumentException.class, () -> Allocation.read(file));
        assertTrue(failure.getMessage().contains("trellis-9.9"), failure.getMessage());
        assertTrue(failure.getMessage().contains("text/layers/0/mlp/down"), failure.getMessage());
    }

    @Test
    void optionKeysRoundTrip() {
        for (Option option : Option.values()) assertEquals(option, Option.of(option.key()));
        assertEquals("trellis-2.5", Option.TRELLIS_2_5.key());
    }
}
