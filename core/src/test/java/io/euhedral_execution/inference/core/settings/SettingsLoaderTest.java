package io.euhedral_execution.inference.core.settings;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.state.KvFormat;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SettingsLoaderTest {

    private static final String MODEL = "model:\n  artifact: /m/a.edrl\n  name: q\n";
    private static final ModelSettings MODEL_SETTINGS = new ModelSettings(Path.of("/m/a.edrl"), "q");

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
                SamplingSettings.defaults(),
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
        assertTrue(message.contains("model.name"), message);
        assertTrue(!message.contains("model.tokenizer"), message);
    }

    @Test
    void variablesSupplyRequired() {
        Settings s = SettingsLoader.load(
                Optional.empty(),
                Map.of(
                        "EUHEDRAL_MODEL_ARTIFACT", "/m/a.edrl",
                        "EUHEDRAL_MODEL_NAME", "q"));
        assertEquals(defaultsWith(MODEL_SETTINGS), s);
    }

    @Test
    void tokenizerVariableIsRetired() {
        SettingsException e = assertThrows(
                SettingsException.class,
                () -> SettingsLoader.load(
                        Optional.empty(),
                        Map.of(
                                "EUHEDRAL_MODEL_ARTIFACT", "/m/a.edrl",
                                "EUHEDRAL_MODEL_TOKENIZER", "/m/tok",
                                "EUHEDRAL_MODEL_NAME", "q")));
        assertEquals(
                List.of("EUHEDRAL_MODEL_TOKENIZER is retired: removed: the artifact carries its tokenizer"),
                e.problems());
    }

    @Test
    void tokenizerKeyIsRetired() throws IOException {
        SettingsException e = fails(MODEL.replace("  name: q\n", "  tokenizer: /m/tok\n  name: q\n"), Map.of());
        assertEquals(List.of("model.tokenizer is retired: removed: the artifact carries its tokenizer"), e.problems());
    }

    @Test
    void samplingParses() throws IOException {
        Settings s = load(MODEL + "sampling:\n  temperature: 0.7\n  top-k: 0\n  top-p: 1\n", Map.of());
        assertEquals(new SamplingSettings(0.7f, 0, 1.0f), s.sampling());
        assertEquals(
                new SamplingSettings(1.0f, 20, 0.95f), load(MODEL, Map.of()).sampling());
    }

    @Test
    void samplingOutOfRangeNamesTheKey() throws IOException {
        assertEquals(
                List.of("sampling.top-p: must be above 0 and at most 1: 1.5"),
                fails(MODEL + "sampling:\n  top-p: 1.5\n", Map.of()).problems());
        assertEquals(
                List.of("sampling.temperature (EUHEDRAL_SAMPLING_TEMPERATURE): must be 0 or more: -1"),
                fails(MODEL, Map.of("EUHEDRAL_SAMPLING_TEMPERATURE", "-1")).problems());
        assertEquals(
                List.of("sampling.top-k: must be 0 to 2147483647: -2"),
                fails(MODEL + "sampling:\n  top-k: -2\n", Map.of()).problems());
        assertTrue(fails(MODEL + "sampling:\n  top-p: NaN\n", Map.of())
                .getMessage()
                .contains("sampling.top-p"));
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

    /// A host limit of 0 leaves no RAM tier: every expert the device misses is read from the artifact.
    @Test
    void aHostLimitOfZeroIsAccepted() throws IOException {
        Settings s = load(MODEL + "memory:\n  host-limit: 0\n", Map.of());
        assertEquals(0, s.memory().hostLimit().getAsLong());
        assertTrue(String.join(
                        "|",
                        fails(MODEL + "memory:\n  gpu-limit: 0\n", Map.of()).problems())
                .contains("memory.gpu-limit"));
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
    void reversedCpuRangeNamesTheKey() throws IOException {
        SettingsException e = fails(MODEL + "host:\n  cpus: 5-2\n", Map.of());
        assertTrue(e.getMessage().contains("host.cpus"), e.getMessage());
        e = fails(MODEL, Map.of("EUHEDRAL_HOST_CPUS", "5-2"));
        assertTrue(e.getMessage().contains("host.cpus (EUHEDRAL_HOST_CPUS)"), e.getMessage());
    }

    @Test
    void zeroShutdownTimeoutIsRejected() throws IOException {
        SettingsException e = fails(MODEL + "server:\n  shutdown-timeout: 0s\n", Map.of());
        assertTrue(e.getMessage().contains("server.shutdown-timeout"), e.getMessage());
    }

    @Test
    void zeroMaxQueuedIsAccepted() throws IOException {
        assertEquals(
                0,
                load(MODEL + "server:\n  requests:\n    max-queued: 0\n", Map.of())
                        .server()
                        .requests()
                        .maxQueued());
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
        assertTrue(e.getMessage().contains(missing.toString()), e.getMessage());
    }

    @Test
    void locateForms() {
        assertEquals(Optional.of(Path.of("/a")), SettingsLoader.locate(new String[] {"--config=/a"}, Map.of()));
        assertEquals(Optional.of(Path.of("/b")), SettingsLoader.locate(new String[0], Map.of("EUHEDRAL_CONFIG", "/b")));
        assertEquals(Optional.empty(), SettingsLoader.locate(new String[0], Map.of()));
    }

    @Test
    void configWithoutPath() {
        assertThrows(SettingsException.class, () -> SettingsLoader.locate(new String[] {"--config"}, Map.of()));
        assertThrows(SettingsException.class, () -> SettingsLoader.locate(new String[] {"--config="}, Map.of()));
    }

    @Test
    void duplicateKey() throws IOException {
        SettingsException e = fails(MODEL + "server:\n  port: 1\n  port: 2\n", Map.of());
        assertTrue(e.getMessage().contains("port"), e.getMessage());
    }

    @Test
    void blankRequiredVariableReportsOnlyEmpty() throws IOException {
        SettingsException e = fails("model:\n  name: q\n", Map.of("EUHEDRAL_MODEL_ARTIFACT", " "));
        assertEquals(1, e.problems().size(), e.getMessage());
        assertTrue(e.getMessage().contains("empty"), e.getMessage());
    }

    @Test
    void durationDaysRejected() throws IOException {
        SettingsException e = fails(MODEL + "server:\n  shutdown-timeout: 1d\n", Map.of());
        assertTrue(e.getMessage().contains("server.shutdown-timeout"), e.getMessage());
    }

    @Test
    void everyRetiredMessageIsRemovedOrNamesAKey() {
        for (String variable : RetiredSettings.variables()) {
            String message = RetiredSettings.message(variable).orElseThrow();
            boolean ok = message.startsWith("removed")
                    || SettingsLoader.knownKeys().stream().anyMatch(k -> message.startsWith("renamed to " + k + " ("));
            assertTrue(ok, variable + ": " + message);
        }
        assertTrue(RetiredSettings.message("EUHEDRAL_HOST_MEMORY_MIB")
                .orElseThrow()
                .contains("with a unit"));
    }

    /// The experiment switches whose winners the engine hard-wires: setting one fails startup.
    @Test
    void everySettledSwitchIsRemoved() {
        for (String variable : List.of(
                "EUHEDRAL_LAKE_PARTITIONS",
                "EUHEDRAL_CUDA_GRAPHS",
                "EUHEDRAL_CUDA_KV_CODEC",
                "EUHEDRAL_QWEN4_PREFETCH",
                "EUHEDRAL_QWEN4_AHEAD",
                "EUHEDRAL_QWEN4_LINEAR_TC",
                "EUHEDRAL_QWEN4_LINEAR_SPLIT",
                "EUHEDRAL_QWEN4_TIER_POLICY",
                "EUHEDRAL_QWEN4_GPU_POLICY",
                "EUHEDRAL_QWEN4_TIER_PRELOAD",
                "EUHEDRAL_QWEN4_PIN_TIER",
                "EUHEDRAL_QWEN4_TIER_ADMISSION",
                "EUHEDRAL_QWEN4_GROUP",
                "EUHEDRAL_QWEN4_DIRECT_READS"))
            assertEquals(Optional.of("removed: the setting is fixed"), RetiredSettings.message(variable), variable);
    }
}
