package io.euhedral_execution.inference.quantization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/// A two-chunk reference of the trellis 3.0 artifact (`euhedral.qwen.trellis30-artifact`) over a text of the
/// repository's, written by the `reference` command, then the same artifact scored against it: the only difference
/// between the two is the 16-bit record quantization, so the mean KL is near zero. Needs the tokenizer directory
/// (`euhedral.qwen.tokenizer-dir`) and the CUDA library too; skips without them.
@Tag("own-jvm")
class ReferenceCudaIntegrationTest {

    private static final int CONTEXT = 512;

    private static Path property(String name) {
        String value = System.getProperty(name);
        assumeTrue(value != null && Files.exists(Path.of(value)), "needs " + name);
        return Path.of(value);
    }

    @Test
    void anArtifactScoredAgainstItsOwnReferenceDiffersOnlyByTheRecordQuantization() throws Exception {
        Path library = property("euhedral.cuda.library");
        Path artifact = property("euhedral.qwen.trellis30-artifact");
        Path tokenizer = property("euhedral.qwen.tokenizer-dir");
        Path text = Path.of(getClass().getResource("/reference-text.md").toURI());
        Path directory = Files.createTempDirectory("reference-");
        try {
            Path out = directory.resolve("self.kld");
            ReferenceLogits.run(new ReferenceLogits.Settings(
                    artifact, tokenizer, text, out, CONTEXT, 2, false, 4, 2, Residency.PLANNED, 6, library));
            try (Scorer.Reference reference = Scorer.Reference.open(out)) {
                assertEquals(CONTEXT, reference.context);
                assertEquals(2, reference.chunks);
            }
            try (Scorer scorer = Scorer.open(library, 6)) {
                Scorer.Result result =
                        scorer.score(artifact, ArtifactReader.read(artifact), List.of(out), 4, 2, Residency.PLANNED);
                assertTrue(result.meanKl() < 1e-4, "mean KL " + result.meanKl());
                assertEquals(1.0, result.top1(), 0.02);
            }
        } finally {
            try (var files = Files.list(directory)) {
                for (Path file : files.toList()) Files.delete(file);
            } catch (IOException ignored) {
                // A temporary directory: left behind at worst.
            }
            Files.deleteIfExists(directory);
        }
    }
}
