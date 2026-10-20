package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.quantization.gpu.CudaLibraries;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Locale;

/// The `score` command: [Scorer] on one artifact file, each reference's summary appended to the results file (and
/// printed) as the reference completes. With a values directory, each reference's per-token values are also written
/// there as `NAME.values`: three little-endian int32 (chunks, scored tokens per chunk, values per token) and then the
/// values as little-endian float64, chunk after chunk, in [Scorer]'s order (NLL, reference NLL, KL, top-1 agreement,
/// probability difference), for tools that aggregate them themselves.
public final class Score {

    /// Command-line settings.
    public record Settings(
            Path artifact,
            List<Path> references,
            List<String> names,
            Path results,
            Path values,
            int rows,
            long hostBytes,
            int batch,
            int workers,
            CudaLibraries library) {}

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
                        if (settings.values() != null)
                            writeValues(settings.values().resolve(name + ".values"), scored);
                        System.out.printf(
                                Locale.ROOT,
                                "scored %s: %d chunks in %.1f s%n",
                                name,
                                scored.reference().chunks,
                                scored.seconds());
                    });
        }
    }

    private static void writeValues(Path file, Scorer.Scored scored) {
        double[][] values = scored.values();
        int steps = scored.reference().records;
        int fields = values.length == 0 ? 0 : values[0].length / Math.max(1, steps);
        ByteBuffer buffer = ByteBuffer.allocate(12 + values.length * steps * fields * Double.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(values.length).putInt(steps).putInt(fields);
        for (double[] chunk : values) for (int i = 0; i < steps * fields; i++) buffer.putDouble(chunk[i]);
        try {
            Files.write(file, buffer.array());
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }
}
