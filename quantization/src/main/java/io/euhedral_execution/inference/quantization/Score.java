package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Locale;

/// The `score` command: [Scorer] on one artifact file, each reference's summary appended to the results file (and
/// printed) as the reference completes.
public final class Score {

    /// Command-line settings.
    public record Settings(
            Path artifact,
            List<Path> references,
            List<String> names,
            Path results,
            int rows,
            long hostBytes,
            int batch,
            int workers,
            Path library) {}

    private Score() {}

    public static void run(Settings settings) throws Exception {
        if (settings.batch() < 1
                || settings.rows() < 1
                || settings.rows() > 8
                || settings.batch() * settings.rows() > 32)
            throw new IllegalArgumentException("--rows (at most 8) times --batch must be at most 32");
        var artifact = ArtifactReader.read(settings.artifact());
        try (Scorer scorer = Scorer.open(settings.library(), settings.workers())) {
            scorer.run(
                    settings.artifact(),
                    artifact,
                    settings.references(),
                    settings.rows(),
                    settings.batch(),
                    settings.hostBytes(),
                    (index, scored) -> {
                        String name = settings.names().get(index);
                        String summary = Scorer.Summary.lines(
                                name, scored.path(), settings.artifact(), scored.reference(), scored.values());
                        System.out.print(summary);
                        try {
                            Files.writeString(
                                    settings.results(),
                                    summary,
                                    StandardOpenOption.CREATE,
                                    StandardOpenOption.APPEND,
                                    StandardOpenOption.WRITE);
                        } catch (IOException failure) {
                            throw new UncheckedIOException(failure);
                        }
                        System.out.printf(
                                Locale.ROOT,
                                "scored %s: %d chunks in %.1f s%n",
                                name,
                                scored.reference().chunks,
                                scored.seconds());
                    });
        }
    }
}
