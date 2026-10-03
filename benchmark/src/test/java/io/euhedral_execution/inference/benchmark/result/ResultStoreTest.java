package io.euhedral_execution.inference.benchmark.result;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.exc.UnrecognizedPropertyException;
import io.euhedral_execution.inference.benchmark.BenchmarkFixtures;
import io.euhedral_execution.inference.benchmark.config.Scenario;
import io.euhedral_execution.inference.benchmark.prompt.PromptMaterial;
import io.euhedral_execution.inference.benchmark.run.BenchmarkRunner;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ResultStoreTest {
    @TempDir
    Path directory;

    static List<BenchmarkResult> measuredRows() throws Exception {
        List<Scenario> scenarios =
                List.of(new Scenario(Scenario.Kind.PREFILL, 6, 0), new Scenario(Scenario.Kind.SUSTAINED_DECODE, 2, 3));
        List<BenchmarkResult> rows = new ArrayList<>();
        BenchmarkRunner.run(
                BenchmarkFixtures.options(scenarios, 1, 2),
                BenchmarkFixtures.prompts(scenarios),
                BenchmarkFixtures.FakeTarget::new,
                BenchmarkFixtures.context(),
                rows::add);
        return rows;
    }

    private Path sample() throws Exception {
        try (var stream = Objects.requireNonNull(getClass().getResourceAsStream("/ninfer-sample.jsonl"))) {
            Path file = this.directory.resolve("ninfer.jsonl");
            Files.write(file, stream.readAllBytes());
            return file;
        }
    }

    @Test
    void jsonlAndJsonDocumentsRoundTrip() throws Exception {
        var rows = measuredRows();
        Path lines = this.directory.resolve("r.jsonl");
        try (var writer = new ResultStore.Writer(lines, false, false, false)) {
            rows.forEach(writer::write);
        }
        assertEquals(rows.size(), Files.readAllLines(lines).size());
        assertEquals(rows, ResultStore.read(lines));

        Path document = this.directory.resolve("r.json");
        try (var writer = new ResultStore.Writer(document, true, false, false)) {
            rows.forEach(writer::write);
        }
        assertTrue(Files.readString(document).contains("\"schema\" : \"" + ResultStore.DOCUMENT_SCHEMA + "\""));
        assertEquals(rows, ResultStore.read(document));
    }

    @Test
    void rowFieldOrderIsStable() throws Exception {
        String json = measuredRows().getFirst().toJson();
        List<String> keys = List.of(
                "schema",
                "schemaVersion",
                "implementation",
                "provenance",
                "runId",
                "fork",
                "recordedAt",
                "scenario",
                "warmup",
                "iteration",
                "status",
                "statusReason",
                "work",
                "timings",
                "throughput",
                "engine",
                "gpuMemory");
        int previous = -1;
        for (String key : keys) {
            int position = json.indexOf("\"" + key + "\":");
            assertTrue(position > previous, key + " out of order in " + json);
            previous = position;
        }
        assertTrue(json.startsWith("{\"schema\":\"euhedral-inference.benchmark-result\",\"schemaVersion\":1,"));
    }

    @Test
    void existingOutputRequiresOverwriteOrAppend() throws Exception {
        var rows = measuredRows();
        Path lines = this.directory.resolve("r.jsonl");
        Files.writeString(lines, "");
        assertThrows(IllegalArgumentException.class, () -> new ResultStore.Writer(lines, false, false, false));
        try (var writer = new ResultStore.Writer(lines, false, false, true)) {
            writer.write(rows.getFirst());
        }
        try (var writer = new ResultStore.Writer(lines, false, false, true)) {
            writer.write(rows.getLast());
        }
        assertEquals(List.of(rows.getFirst(), rows.getLast()), ResultStore.read(lines));
    }

    @Test
    void importsExternalRowsInTheSameSchemaWithProvenance() throws Exception {
        Path source = sample();
        var imported = ResultStore.importRows(source, "ninfer", "5070 Ti baseline, NInfer build abc123");
        assertEquals(2, imported.size());
        var row = imported.getFirst();
        assertEquals("ninfer", row.implementation());
        assertEquals("imported", row.provenance().kind());
        assertEquals("ninfer-bench", row.provenance().tool());
        assertEquals(
                source.toAbsolutePath().normalize().toString(), row.provenance().source());
        assertEquals(
                PromptMaterial.sha256(Files.readString(source)),
                row.provenance().sourceSha256());
        assertEquals("5070 Ti baseline, NInfer build abc123", row.provenance().note());
        assertEquals("ext-fork-2", row.fork().id());
        assertEquals(512.0, row.throughput().prefillTokensPerSecond());
        assertNull(row.gpuMemory());

        Path store = this.directory.resolve("store.jsonl");
        try (var writer = new ResultStore.Writer(store, false, false, false)) {
            measuredRows().forEach(writer::write);
            imported.forEach(writer::write);
        }
        var combined = ResultStore.read(store);
        assertEquals(
                List.of("euhedral-inference", "ninfer"),
                combined.stream()
                        .map(BenchmarkResult::implementation)
                        .distinct()
                        .toList());
        assertTrue(Summary.format(combined).contains("ninfer | - | prefill-1024"), Summary.format(combined));
        assertThrows(
                IllegalArgumentException.class,
                () -> ResultStore.importRows(source, "euhedral-inference", null),
                "an import cannot relabel another implementation's rows");
    }

    @Test
    void rejectsRowsOutsideTheSchemaContract() throws Exception {
        String valid = Files.readAllLines(sample()).getFirst();
        Path file = this.directory.resolve("bad.jsonl");
        Files.writeString(file, valid.replace("\"schemaVersion\":1", "\"schemaVersion\":2"));
        assertThrows(IllegalArgumentException.class, () -> ResultStore.read(file));
        Files.writeString(file, valid.replace("\"status\":\"success\"", "\"status\":\"failed\""));
        assertThrows(IllegalArgumentException.class, () -> ResultStore.read(file), "failed rows carry no rates");
        Files.writeString(file, valid.replace("\"prefillTokensPerSecond\":512.0", "\"prefillTokensPerSecond\":0.0"));
        assertThrows(IllegalArgumentException.class, () -> ResultStore.read(file));
        Files.writeString(file, valid.replace("\"warmup\":false", "\"warmup\":false,\"extra\":1"));
        assertThrows(UnrecognizedPropertyException.class, () -> ResultStore.read(file));
        Files.writeString(file, "\n");
        assertThrows(IllegalArgumentException.class, () -> ResultStore.read(file));
    }
}
