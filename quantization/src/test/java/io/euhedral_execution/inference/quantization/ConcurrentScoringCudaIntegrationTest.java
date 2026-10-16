package io.euhedral_execution.inference.quantization;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.quantization.mixed.Allocation;
import io.euhedral_execution.inference.quantization.mixed.Assemble;
import io.euhedral_execution.inference.quantization.mixed.Option;
import io.euhedral_execution.inference.quantization.mixed.Sources;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/// Two candidates that share most of their objects, scored at once over one cache, give each chunk the KL they give
/// scored alone (bitwise), and the device's peak allocation is below the two alone together.
///
/// The candidates are the base (trellis 3.0) artifact with two different projections each taken from the NVFP4
/// artifact. Needs both (`euhedral.qwen.trellis30-artifact`, `euhedral.qwen.nvfp4-artifact`), a reference
/// (`euhedral.quantization.score-reference`) and the CUDA library; skips without them, and when the device is too
/// small to hold both candidates at once.
@Tag("own-jvm")
class ConcurrentScoringCudaIntegrationTest {

    private static final int ROWS = 4, BATCH = 1;

    private static Path property(String name) {
        String value = System.getProperty(name);
        assumeTrue(value != null && Files.exists(Path.of(value)), "needs " + name);
        return Path.of(value);
    }

    @Test
    void twoCandidatesAtOnceScoreAsTheyDoAloneAndPeakBelowTheirSum() throws Exception {
        Path library = property("euhedral.cuda.library");
        List<Path> references = List.of(property("euhedral.quantization.score-reference"));
        Sources sources = Sources.open(
                Map.of(
                        Option.TRELLIS_3_0, property("euhedral.qwen.trellis30-artifact"),
                        Option.NVFP4, property("euhedral.qwen.nvfp4-artifact")),
                Option.TRELLIS_3_0);
        List<String> projections = List.copyOf(sources.projections());
        Allocation first = new Allocation(Map.of(projections.get(0), Option.NVFP4, projections.get(1), Option.NVFP4));
        Allocation second = new Allocation(Map.of(projections.get(2), Option.NVFP4, projections.get(3), Option.NVFP4));
        Artifact a = Assemble.view(sources, first, OptionalInt.of(3));
        Artifact b = Assemble.view(sources, second, OptionalInt.of(3));
        Path base = sources.path(sources.base());

        try (Scorer scorer = Scorer.open(library, 6)) {
            var gpu = scorer.device().gpu;
            gpu.resetPeakAllocatedBytes();
            Scorer.Result aloneA = scorer.score(base, a, references, ROWS, BATCH, Residency.PLANNED);
            long peakA = gpu.peakAllocatedBytes();
            gpu.resetPeakAllocatedBytes();
            Scorer.Result aloneB = scorer.score(base, b, references, ROWS, BATCH, Residency.PLANNED);
            long peakB = gpu.peakAllocatedBytes();

            gpu.resetPeakAllocatedBytes();
            CompletableFuture<Scorer.Result> together;
            CompletableFuture<Scorer.Result> alsoTogether;
            int peakRunning;
            try (ConcurrentScoring scoring = new ConcurrentScoring(scorer, references, ROWS, BATCH, 0, 256L << 20, 4)) {
                together = scoring.submit(base, a, () -> {});
                alsoTogether = scoring.submit(base, b, () -> {});
                long deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(30);
                while (!(together.isDone() && alsoTogether.isDone())) {
                    scoring.service();
                    try {
                        CompletableFuture.anyOf(together, alsoTogether, scoring.attention())
                                .get(1, TimeUnit.SECONDS);
                    } catch (TimeoutException ignored) {
                        assertTrue(System.nanoTime() < deadline, "scoring did not finish");
                    }
                }
                peakRunning = scoring.peakRunning();
            }
            long peakTogether = gpu.peakAllocatedBytes();
            assumeTrue(peakRunning == 2, "the device did not hold both candidates at once");

            Scorer.Result resultA = together.join(), resultB = alsoTogether.join();
            assertArrayEquals(aloneA.chunkKl(), resultA.chunkKl());
            assertArrayEquals(aloneB.chunkKl(), resultB.chunkKl());
            assertEquals(aloneA.meanKl(), resultA.meanKl());
            assertEquals(aloneB.meanKl(), resultB.meanKl());
            assertTrue(
                    peakTogether < peakA + peakB,
                    "peak " + (peakTogether >> 20) + " MiB against " + ((peakA + peakB) >> 20) + " MiB alone");
        }
    }
}
