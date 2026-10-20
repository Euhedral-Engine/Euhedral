package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.quantization.Residency;
import io.euhedral_execution.inference.quantization.Scorer;
import io.euhedral_execution.inference.quantization.TestLibraries;
import io.euhedral_execution.inference.quantization.mixed.SearchState.Measured;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The search on the device: two rounds with small batches, RCO settings and ensemble, scoring against a
/// reference, then a restart that scores nothing again.
///
/// Needs a trellis artifact (`euhedral.qwen.trellis30-artifact`), an NVFP4 one
/// (`euhedral.qwen.nvfp4-artifact`), a reference (`euhedral.quantization.score-reference`; the smaller the
/// faster) and the CUDA library; skips without them. The priors are the same for every object (a quarter of
/// the NVFP4 error for trellis 3.0), which is enough for the loop to run; the search is not expected to find
/// anything.
@Tag("own-jvm")
class SearchCudaIntegrationTest {

    private static final int ROWS = 4, BATCH = 2, VERIFY_ROWS = 4;

    @TempDir
    Path directory;

    private static Path property(String name) {
        String value = System.getProperty(name);
        assumeTrue(value != null && Files.exists(Path.of(value)), "needs " + name);
        return Path.of(value);
    }

    @Test
    void twoRoundsMeasureOnTheDeviceAndAResumeScoresNothingAgain() throws Exception {
        Path library = property("euhedral.cuda.library");
        List<Path> references = List.of(property("euhedral.quantization.score-reference"));
        Sources sources = Sources.open(
                Map.of(
                        Option.TRELLIS_3_0, property("euhedral.qwen.trellis30-artifact"),
                        Option.NVFP4, property("euhedral.qwen.nvfp4-artifact")),
                Option.TRELLIS_3_0);
        CostTable costs = Cost.run(
                TestLibraries.of(library),
                sources,
                List.of(VERIFY_ROWS, CostTable.PREFILL_ROWS),
                directory.resolve("costs.json"),
                5,
                64L << 20);
        Map<String, Map<Option, Double>> errors = new TreeMap<>();
        for (String name : sources.projections())
            errors.put(name, Map.of(Option.NVFP4, 0.004, Option.TRELLIS_3_0, 0.001));
        Priors priors = new Priors(errors);

        try (Scorer scorer = Scorer.open(TestLibraries.of(library), 6)) {
            long free = scorer.freeDeviceBytes();
            int[] scored = {0};
            Search.Scoring scoring = (path, artifact) -> {
                scored[0]++;
                return scorer.score(path, artifact, references, ROWS, BATCH, Residency.PLANNED);
            };
            Path state = directory.resolve("search.resume");
            Search.Settings settings = new Search.Settings(
                    sources,
                    priors,
                    costs,
                    4096,
                    state,
                    directory.resolve("out"),
                    List.of(),
                    2,
                    0.5,
                    3,
                    1,
                    FidelityModel.DEFAULT_BANDS,
                    VERIFY_ROWS,
                    free,
                    freeBytes -> Feasibility.planner(sources, freeBytes, 4096, 4),
                    2,
                    3,
                    1e-3,
                    List.of(0.0, 0.01),
                    List.of(1.0),
                    new Rco.Settings(20, 0.1, 2, 1),
                    1024,
                    2,
                    1,
                    "integration");
            Search.Outcome outcome = Search.run(settings, scoring);
            assertEquals(2, outcome.rounds());
            assertEquals(scored[0], outcome.measured().size());
            assertFalse(outcome.front().isEmpty());
            for (Measured m : outcome.measured()) assertTrue(m.kl() >= 0 && Double.isFinite(m.kl()), m.origin());
            assertTrue(Files.exists(directory.resolve("out/front.json")));

            int before = scored[0];
            Search.Outcome resumed = Search.run(settings, scoring);
            assertEquals(before, scored[0], "a finished search scores nothing on resume");
            assertEquals(outcome.measured().size(), resumed.measured().size());
        }
    }
}
