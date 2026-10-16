package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class PriorsSeedsTest {

    @TempDir
    Path directory;

    @Test
    void priorsReadTheErrorOfEachOptionOfEachObject() throws Exception {
        Path file = directory.resolve("priors.json");
        Files.writeString(file, """
                {"objects": {"text/layers/0/mlp/down": {"nvfp4": 0.0012, "trellis-2.5": 0.0031},
                             "text/layers/1/mlp/down": {"nvfp4": 0.002, "trellis-2.5": 0.004}}}
                """);
        Priors priors = Priors.read(file);
        assertEquals(0.0031, priors.error("text/layers/0/mlp/down", Option.TRELLIS_2_5), 0);
        assertEquals(List.of(Option.NVFP4, Option.TRELLIS_2_5), priors.options());
        IllegalArgumentException missing = assertThrows(
                IllegalArgumentException.class, () -> priors.error("text/layers/1/mlp/down", Option.TRELLIS_3_0));
        assertTrue(missing.getMessage().contains("trellis-3.0")
                && missing.getMessage().contains("text/layers/1/mlp/down"));
    }

    @Test
    void priorsRefuseAnUnknownOptionNamingItsObject() throws Exception {
        Path file = directory.resolve("priors.json");
        Files.writeString(file, "{\"objects\": {\"text/layers/0/mlp/down\": {\"fp8\": 1}}}");
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class, () -> Priors.read(file));
        assertTrue(refused.getMessage().contains("text/layers/0/mlp/down"), refused.getMessage());
    }

    @Test
    void seedsReadAListOrOneAllocation() throws Exception {
        Path list = directory.resolve("list.json");
        Files.writeString(
                list, "{\"allocations\": [{\"objects\": {\"text/layers/0/mlp/down\": \"nvfp4\"}}, {\"objects\": {}}]}");
        assertEquals(
                List.of(new Allocation(Map.of("text/layers/0/mlp/down", Option.NVFP4)), new Allocation(Map.of())),
                Seeds.read(list));
        Path one = directory.resolve("one.json");
        Files.writeString(one, "{\"objects\": {\"text/layers/1/mlp/down\": \"trellis-3.0\"}}");
        assertEquals(List.of(new Allocation(Map.of("text/layers/1/mlp/down", Option.TRELLIS_3_0))), Seeds.read(one));
    }
}
