package io.euhedral_execution.inference.quantization;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/// Scoring is row-exact, so the chunks forced together do not change a chunk's result: 4 at a time and as many as
/// the device holds give bitwise equal per-chunk KL.
///
/// Needs a trellis artifact (`euhedral.qwen.trellis30-artifact`), a reference (`euhedral.quantization.score-reference`)
/// and the CUDA library; skips without them.
@Tag("own-jvm")
class ScorerBatchCudaIntegrationTest {

    private static final int ROWS = 8;

    private static Path property(String name) {
        String value = System.getProperty(name);
        assumeTrue(value != null && Files.exists(Path.of(value)), "needs " + name);
        return Path.of(value);
    }

    @Test
    void theAutoSizedBatchScoresEveryChunkAsFourAtATimeDoes() throws Exception {
        Path library = property("euhedral.cuda.library");
        Path trellis = property("euhedral.qwen.trellis30-artifact");
        List<Path> references = List.of(property("euhedral.quantization.score-reference"));
        try (Scorer scorer = Scorer.open(TestLibraries.of(library), 6)) {
            var artifact = ArtifactReader.read(trellis);
            Scorer.Result four = scorer.score(trellis, artifact, references, ROWS, 4, Residency.PLANNED);
            Scorer.Result auto = scorer.score(trellis, artifact, references, ROWS, 0, Residency.PLANNED);
            assertEquals(four.chunkKl().length, auto.chunkKl().length);
            assertArrayEquals(four.chunkKl(), auto.chunkKl());
            assertEquals(four.meanKl(), auto.meanKl());
            assertEquals(four.top1(), auto.top1());
        }
    }
}
