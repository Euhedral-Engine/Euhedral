package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.quantization.Residency;
import io.euhedral_execution.inference.quantization.Scorer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.TreeMap;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/// Assembled artifacts scored on the device, against a llama.cpp logits file.
///
/// Needs a trellis artifact (`euhedral.qwen.trellis30-artifact`), an NVFP4 one (`euhedral.qwen.nvfp4-artifact`), a
/// reference (`euhedral.quantization.score-reference`) and the CUDA library; skips without them. The written
/// artifact goes beside the trellis artifact and is deleted afterwards.
@Tag("own-jvm")
class AssembleCudaIntegrationTest {

    private static final int ROWS = 4, BATCH = 2;

    private static Path property(String name) {
        String value = System.getProperty(name);
        assumeTrue(value != null && Files.exists(Path.of(value)), "needs " + name);
        return Path.of(value);
    }

    @Test
    void allFromOneSourceScoresAsThatArtifactAndTheWrittenFileAsItsView() throws Exception {
        Path library = property("euhedral.cuda.library");
        Path trellis = property("euhedral.qwen.trellis30-artifact");
        Path nvfp4 = property("euhedral.qwen.nvfp4-artifact");
        List<Path> references = List.of(property("euhedral.quantization.score-reference"));
        Sources sources = Sources.open(Map.of(Option.TRELLIS_3_0, trellis, Option.NVFP4, nvfp4), Option.TRELLIS_3_0);

        Map<String, Option> every = new TreeMap<>();
        for (String name : sources.projections()) every.put(name, Option.TRELLIS_3_0);
        Allocation pure = new Allocation(every);

        Map<String, Option> mixed = new TreeMap<>();
        for (String name : sources.projections()) if (name.endsWith("/mlp/down")) mixed.put(name, Option.NVFP4);
        Allocation allocation = new Allocation(mixed);
        Artifact view = Assemble.view(sources, allocation, OptionalInt.of(3));

        Path directory = Files.createTempDirectory(trellis.toAbsolutePath().getParent(), "mixed-");
        try (Scorer scorer = Scorer.open(library, 6)) {
            Scorer.Result source =
                    scorer.score(trellis, ArtifactReader.read(trellis), references, ROWS, BATCH, Residency.PLANNED);
            Scorer.Result identity = scorer.score(
                    trellis,
                    Assemble.view(sources, pure, OptionalInt.empty()),
                    references,
                    ROWS,
                    BATCH,
                    Residency.PLANNED);
            assertArrayEquals(source.chunkKl(), identity.chunkKl());
            assertEquals(source.meanKl(), identity.meanKl());

            Scorer.Result composite = scorer.score(trellis, view, references, ROWS, BATCH, Residency.PLANNED);
            Path out = directory.resolve("mixed.edrl");
            Assemble.write(sources, allocation, OptionalInt.of(3), out);
            Scorer.Result written =
                    scorer.score(out, ArtifactReader.read(out), references, ROWS, BATCH, Residency.PLANNED);
            assertArrayEquals(composite.chunkKl(), written.chunkKl());
        } finally {
            try (Stream<Path> files = Files.walk(directory)) {
                files.sorted(Comparator.reverseOrder())
                        .forEach(file -> file.toFile().delete());
            }
        }
    }
}
