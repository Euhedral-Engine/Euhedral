package io.euhedral_execution.inference.core.settings;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.state.KvFormat;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SettingsLoaderTest {

    private static final String MODEL = "model:\n  artifact: /m/a.edrl\n  tokenizer: /m/tok\n  name: q\n";
    private static final ModelSettings MODEL_SETTINGS = new ModelSettings(Path.of("/m/a.edrl"), Path.of("/m/tok"), "q");

    @TempDir
    Path dir;

    private Path file(String yaml) throws IOException {
        return Files.writeString(dir.resolve("c.yaml"), yaml);
    }

    private Settings load(String yaml, Map<String, String> env) throws IOException {
        return SettingsLoader.load(Optional.of(file(yaml)), env);
    }

    private SettingsException fails(String yaml, Map<String, String> env) throws IOException {
        return assertThrows(SettingsException.class, () -> load(yaml, env));
    }

    private static Settings defaultsWith(ModelSettings model) {
        return new Settings(
                model,
                ServerSettings.defaults(),
                ContextSettings.defaults(),
                PrefixCacheSettings.defaults(),
                BatchingSettings.defaults(),
                MemorySettings.defaults(),
                HostSettings.defaults(),
                KernelSettings.defaults());
    }

    @Test
    void minimalFile() throws IOException {
        Settings s = load(MODEL, Map.of());
        assertEquals(defaultsWith(MODEL_SETTINGS), s);
        assertEquals(Tuning.defaults(), s.tuning());
    }

    @Test
    void missingRequired() throws IOException {
        String message = String.join("|", fails("", Map.of()).problems());
        assertTrue(message.contains("model.artifact"), message);
        assertTrue(message.contains("model.tokenizer"), message);
        assertTrue(message.contains("model.name"), message);
    }

    @Test
    void variablesSupplyRequired() {
        Settings s = SettingsLoader.load(
                Optional.empty(),
                Map.of(
                        "EUHEDRAL_MODEL_ARTIFACT", "/m/a.edrl",
                        "EUHEDRAL_MODEL_TOKENIZER", "/m/tok",
                        "EUHEDRAL_MODEL_NAME", "q"));
        assertEquals(defaultsWith(MODEL_SETTINGS), s);
    }

    @Test
    void variableOverridesFile() throws IOException {
        Settings s = load(MODEL + "server:\n  port: 8000\n", Map.of("EUHEDRAL_SERVER_PORT", "8001"));
        assertEquals(8001, s.server().port());
    }

    @Test
    void variableCreatesMissingSection() throws IOException {
        Settings s = load(MODEL, Map.of("EUHEDRAL_KERNELS_TRELLIS_TARGET_BLOCKS", "280"));
        assertEquals(280, s.kernels().trellis().targetBlocks().getAsInt());
    }

    @Test
    void variableAutoClearsFileValue() throws IOException {
        Settings s = load(MODEL + "memory:\n  gpu-limit: 1GiB\n", Map.of("EUHEDRAL_MEMORY_GPU_LIMIT", "auto"));
        assertTrue(s.memory().gpuLimit().isEmpty());
    }

    @Test
    void valuesParse() throws IOException {
        Settings s = load(
                MODEL + "context:\n  kv-cache-format: bf16\nmemory:\n  gpu-limit: 1GiB\nhost:\n  cpus: 2-5,8\n"
                        + "  idle-release: 0\nserver:\n  requests:\n    timeout: 90s\n",
                Map.of());
        assertEquals(Optional.of(KvFormat.BF16), s.context().kvCacheFormat());
        assertEquals(1L << 30, s.memory().gpuLimit().getAsLong());
        assertEquals(Optional.of("2-5,8"), s.host().cpus());
        assertEquals(Duration.ZERO, s.host().idleRelease());
        assertEquals(Duration.ofSeconds(90), s.server().requests().timeout());
    }

    @Test
    void unknownKey() throws IOException {
        SettingsException e = fails(MODEL + "memory:\n  gpu-limt: 1GiB\n", Map.of());
        assertTrue(e.getMessage().contains("memory.gpu-limt"), e.getMessage());
    }

    @Test
    void emptyValue() throws IOException {
        SettingsException e = fails(MODEL + "batching:\n  max-sequences:\n", Map.of());
        assertTrue(e.getMessage().contains("batching.max-sequences"), e.getMessage());
    }

    @Test
    void retiredVariable() throws IOException {
        SettingsException e = fails(MODEL, Map.of("EUHEDRAL_INFERENCE_ARTIFACT_PATH", "/x"));
        assertTrue(e.getMessage().contains("renamed to model.artifact (EUHEDRAL_MODEL_ARTIFACT)"), e.getMessage());
        e = fails(MODEL, Map.of("EUHEDRAL_CUDA_GRAPHS", "0"));
        assertTrue(e.getMessage().contains("removed"), e.getMessage());
        assertTrue(RetiredSettings.message("EUHEDRAL_MODEL_ARTIFACT").isEmpty());
    }

    @Test
    void harnessVariablesIgnored() throws IOException {
        Settings s = load(MODEL, Map.of("EUHEDRAL_QWEN4_TRACE", "1", "EUHEDRAL_DEPLOY_HOST", "h"));
        assertEquals(defaultsWith(MODEL_SETTINGS), s);
    }

    @Test
    void portAlias() throws IOException {
        assertEquals(9000, load(MODEL, Map.of("PORT", "9000")).server().port());
        assertEquals(
                9001,
                load(MODEL, Map.of("PORT", "9000", "EUHEDRAL_SERVER_PORT", "9001"))
                        .server()
                        .port());
    }

    @Test
    void zeroPrefixCache() throws IOException {
        assertEquals(
                0,
                load(MODEL + "prefix-cache:\n  size: 0\n", Map.of())
                        .prefixCache()
                        .size());
    }

    @Test
    void autoRejectedWithoutDefault() throws IOException {
        SettingsException e = fails(MODEL + "server:\n  port: auto\n", Map.of());
        assertTrue(e.getMessage().contains("server.port"), e.getMessage());
    }

    @Test
    void checkpointMustBeMultipleOf512() throws IOException {
        SettingsException e = fails(MODEL + "prefix-cache:\n  checkpoint-tokens: 1000\n", Map.of());
        assertTrue(e.getMessage().contains("prefix-cache.checkpoint-tokens"), e.getMessage());
    }

    @Test
    void errorNamesVariable() throws IOException {
        SettingsException e = fails(MODEL, Map.of("EUHEDRAL_SERVER_PORT", "x"));
        assertTrue(e.getMessage().contains("server.port (EUHEDRAL_SERVER_PORT)"), e.getMessage());
    }

    @Test
    void everyProblemReported() throws IOException {
        SettingsException e = fails(MODEL + "server:\n  port: abc\nbatching:\n  max-sequences: -1\n", Map.of());
        assertEquals(2, e.problems().size(), e.getMessage());
        assertTrue(e.problems().get(0).contains("server.port"));
        assertTrue(e.problems().get(1).contains("batching.max-sequences"));
    }

    @Test
    void explicitMissingFile() {
        Path missing = Path.of("/nope");
        Optional<Path> located = SettingsLoader.locate(new String[] {"--config", "/nope"}, Map.of());
        assertEquals(Optional.of(missing), located);
        SettingsException e = assertThrows(SettingsException.class, () -> SettingsLoader.load(located, Map.of()));
        assertTrue(e.getMessage().contains("/nope"), e.getMessage());
    }

    @Test
    void locateForms() {
        assertEquals(Optional.of(Path.of("/a")), SettingsLoader.locate(new String[] {"--config=/a"}, Map.of()));
        assertEquals(Optional.of(Path.of("/b")), SettingsLoader.locate(new String[0], Map.of("EUHEDRAL_CONFIG", "/b")));
        assertEquals(Optional.empty(), SettingsLoader.locate(new String[0], Map.of()));
    }
}
