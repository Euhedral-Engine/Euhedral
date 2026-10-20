package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.quantization.TestLibraries;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The launches of two options timed on the device, over the real objects of a trellis 3.0 artifact
/// (`euhedral.qwen.trellis30-artifact`) and an NVFP4 one (`euhedral.qwen.nvfp4-artifact`); skips without them and the
/// CUDA library.
@Tag("own-jvm")
class CostCudaIntegrationTest {

    @TempDir
    Path directory;

    private static Path property(String name) {
        String value = System.getProperty(name);
        assumeTrue(value != null && Files.exists(Path.of(value)), "needs " + name);
        return Path.of(value);
    }

    @Test
    void everyShapeAndOptionHasPositiveTimesAndPrefillCostsMoreThanADecodeLaunch() throws Exception {
        Path library = property("euhedral.cuda.library");
        Sources sources = Sources.open(
                Map.of(
                        Option.TRELLIS_3_0, property("euhedral.qwen.trellis30-artifact"),
                        Option.NVFP4, property("euhedral.qwen.nvfp4-artifact")),
                Option.TRELLIS_3_0);
        CostTable table = Cost.run(
                TestLibraries.of(library), sources, List.of(1, 4, 512), directory.resolve("costs.json"), 10, 64L << 20);
        for (String name : sources.projections()) {
            for (Option option : sources.options()) {
                Cost.Shape shape = Cost.shape(sources.descriptor(option, name));
                double one = table.micros(new Cost.Key(shape, option, 1));
                double four = table.micros(new Cost.Key(shape, option, 4));
                double prefill = table.micros(new Cost.Key(shape, option, 512));
                assertTrue(one > 0 && four > 0 && prefill > 0, name + " " + option.key());
                assertTrue(prefill >= one, name + " " + option.key() + ": 512 rows " + prefill + " us, 1 row " + one);
            }
        }
        CostTable read = CostTable.read(directory.resolve("costs.json"));
        Allocation base = new Allocation(Map.of());
        assertTrue(read.prefillMicrosPerToken(sources, base) > 0);
        assertTrue(read.verifyMicros(sources, base, 4) > 0);
    }
}
