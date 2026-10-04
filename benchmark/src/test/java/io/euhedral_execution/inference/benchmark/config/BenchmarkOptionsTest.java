package io.euhedral_execution.inference.benchmark.config;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.exc.MismatchedInputException;
import com.fasterxml.jackson.databind.exc.UnrecognizedPropertyException;
import io.euhedral_execution.inference.benchmark.prompt.PromptMaterial;
import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class BenchmarkOptionsTest {
    private static final Instant NOW = Instant.parse("2026-09-25T12:00:00Z");
    private static final String REQUIRED =
            "\"artifact\":\"/m.edrl\",\"tokenizer\":\"/tok\",\"cudaLibrary\":\"/lib.so\"";

    @TempDir
    Path directory;

    private BenchmarkOptions load(String extraFields) throws Exception {
        Path file = this.directory.resolve("bench.json");
        Files.writeString(file, "{" + REQUIRED + (extraFields.isEmpty() ? "" : "," + extraFields) + "}");
        return BenchmarkOptions.load(file, NOW);
    }

    /// Returns the validation message, whether Jackson wrapped it or not.
    private String rejection(String extraFields) {
        Exception failure = assertThrows(Exception.class, () -> load(extraFields));
        if (failure instanceof JsonMappingException mapping && mapping.getCause() instanceof IllegalArgumentException)
            return mapping.getCause().getMessage();
        return failure.getMessage();
    }

    @Test
    void defaultsMatchTheProductionEngineDefaults() throws Exception {
        var options = load("");
        assertEquals(Path.of("/m.edrl"), options.artifact());
        assertEquals(InferenceConfig.DEFAULT_MAX_CONTEXT_TOKENS, options.maxContextTokens());
        assertEquals(0L, options.prefixCacheBytes(), "benchmark iterations repeat prompts, so the cache is off");
        assertEquals("all", options.cpus());
        assertEquals(Scenario.DEFAULT_SUITE, options.scenarios());
        assertEquals(1, options.warmup());
        assertEquals(3, options.iterations());
        assertEquals(GenerationConfig.greedy(1L), options.generation().toConfig());
        assertEquals(PromptMaterial.DEFAULT_SEED, options.promptSeed());
        assertEquals(Path.of("benchmark-results/euhedral-20260925T120000Z.jsonl"), options.output());
        assertFalse(options.json());
        assertFalse(options.gpuMemory());
        assertEquals(10L, options.shutdownTimeout().toSeconds());
    }

    @Test
    void unknownEngineOptionsAreRejected() {
        assertInstanceOf(
                UnrecognizedPropertyException.class,
                assertThrows(Exception.class, () -> load("\"gpuExecutionMode\":\"SYNC\"")));
    }

    @Test
    void bindsEveryOption() throws Exception {
        var options = load("""
                "maxContextTokens":8192,"prefixCacheBytes":1073741824,"cpus":"2-5,8","excludeCpus":[3],"excludeCores":[1],
                "scenarios":["prefill:64","decode:32:128"],"warmup":0,"iterations":5,
                "generation":{"mode":"sample","seed":9,"temperature":0.6,"topK":10,"topP":0.9},
                "promptSeed":4,"output":"out/run.json","gpuMemory":true,
                "shutdownTimeoutSeconds":30""");
        assertEquals(8192, options.maxContextTokens());
        assertEquals(1073741824L, options.prefixCacheBytes());
        assertEquals("2-5,8", options.cpus());
        assertEquals(List.of(3), options.excludeCpus());
        assertEquals(List.of(1), options.excludeCores());
        assertEquals(
                List.of(
                        new Scenario(Scenario.Kind.PREFILL, 64, 0),
                        new Scenario(Scenario.Kind.SUSTAINED_DECODE, 32, 128)),
                options.scenarios());
        assertEquals(
                new GenerationConfig(0.6f, 10, 0.9f, 9L, false),
                options.generation().toConfig());
        assertTrue(options.json(), "a .json output writes one document");
        assertTrue(options.gpuMemory());
    }

    @Test
    void scenariosRoundTripAsTheirSpecificationStrings() throws Exception {
        var scenarios = List.of(
                Scenario.parse("prefill:512"),
                Scenario.parse("first-token:1024"),
                Scenario.parse("prompt-to-n:1024:64"),
                Scenario.parse("DECODE:32:256"));
        String json = BenchmarkOptions.JSON.writeValueAsString(scenarios);
        assertEquals("[\"prefill:512\",\"first-token:1024\",\"prompt-to-n:1024:64\",\"decode:32:256\"]", json);
        assertEquals(scenarios, List.of(BenchmarkOptions.JSON.readValue(json, Scenario[].class)));
        assertEquals("prompt-to-n-1024-64", scenarios.get(2).name());
    }

    @Test
    void rejectsInvalidConfigurationBeforeAnyModelWork() throws Exception {
        Path file = this.directory.resolve("missing.json");
        Files.writeString(file, "{\"artifact\":\"/m\",\"tokenizer\":\"/t\"}");
        assertThrows(MismatchedInputException.class, () -> BenchmarkOptions.load(file, NOW), "cudaLibrary required");
        assertInstanceOf(UnrecognizedPropertyException.class, assertThrows(Exception.class, () -> load("\"bogus\":1")));
        assertInstanceOf(
                UnrecognizedPropertyException.class,
                assertThrows(Exception.class, () -> load("\"generation\":{\"beam\":4}")));
        assertEquals("maxContextTokens must be positive", rejection("\"maxContextTokens\":0"));
        assertEquals("prefixCacheBytes must not be negative", rejection("\"prefixCacheBytes\":-1"));
        assertEquals("iterations must be positive", rejection("\"iterations\":0"));
        assertEquals("warmup must not be negative", rejection("\"warmup\":-1"));
        assertEquals("generation.mode must be greedy or sample", rejection("\"generation\":{\"mode\":\"beam\"}"));
        assertEquals(
                "temperature, topK, and topP require mode \"sample\"",
                rejection("\"generation\":{\"temperature\":0.5}"));
        assertEquals("scenario prompt tokens must be positive", rejection("\"scenarios\":[\"prefill:0\"]"));
        assertEquals("decode scenarios need at least two new tokens", rejection("\"scenarios\":[\"decode:32:1\"]"));
        assertEquals("unknown scenario kind: burst:8", rejection("\"scenarios\":[\"burst:8\"]"));
        assertEquals("choose either overwrite or append", rejection("\"overwrite\":true,\"append\":true"));
        assertEquals("append requires JSONL output", rejection("\"output\":\"r.json\",\"append\":true"));
        assertEquals("excludeCpus must not be negative", rejection("\"excludeCpus\":[-1]"));
    }
}
