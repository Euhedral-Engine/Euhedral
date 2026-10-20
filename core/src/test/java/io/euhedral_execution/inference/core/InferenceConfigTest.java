package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.nativefiles.NativeFiles;
import io.euhedral_execution.inference.core.settings.SettingsLoader;
import io.euhedral_execution.inference.core.settings.Tuning;
import io.euhedral_execution.inference.core.state.KvFormat;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class InferenceConfigTest {
    private static BitSet cpus() {
        BitSet cpus = new BitSet();
        cpus.set(0);
        return cpus;
    }

    private static InferenceConfig withCache(long bytes, int interval) {
        return new InferenceConfig(
                Path.of("a"), Path.of("t"), Path.of("l"), cpus(), 32768, Duration.ofSeconds(1), bytes, interval);
    }

    @Test
    void programmaticConstructorsLeaveThePrefixCacheOff() {
        var config = new InferenceConfig(Path.of("a"), Path.of("t"), Path.of("l"), cpus(), Duration.ofSeconds(1));
        assertEquals(0, config.prefixCacheBytes());
        assertEquals(InferenceConfig.DEFAULT_PREFIX_CACHE_CHECKPOINT_TOKENS, config.prefixCacheCheckpointTokens());
        var sixArguments =
                new InferenceConfig(Path.of("a"), Path.of("t"), Path.of("l"), cpus(), 4096, Duration.ofSeconds(1));
        assertEquals(0, sixArguments.prefixCacheBytes());
    }

    @Test
    void sizesForOneSequenceUnlessToldOtherwise() {
        assertEquals(1, withCache(0, 2048).maxSequences());
        assertEquals(
                1,
                new InferenceConfig(Path.of("a"), Path.of("t"), Path.of("l"), cpus(), Duration.ofSeconds(1))
                        .maxSequences());
        assertEquals(0, withSequences(0).maxSequences());
        assertEquals(6, withSequences(6).maxSequences());
        assertThrows(IllegalArgumentException.class, () -> withSequences(-1));
    }

    private static InferenceConfig withSequences(int sequences) {
        return new InferenceConfig(
                Path.of("a"),
                Path.of("t"),
                Path.of("l"),
                cpus(),
                32768,
                Duration.ofSeconds(1),
                0,
                2048,
                null,
                sequences);
    }

    @Test
    void carriesTheCacheSettings() {
        var config = withCache(4L << 30, 4096);
        assertEquals(4L << 30, config.prefixCacheBytes());
        assertEquals(4096, config.prefixCacheCheckpointTokens());
        assertEquals(0, withCache(0, 2048).prefixCacheBytes());
    }

    @Test
    void rejectsANegativeBudgetAndAnIntervalOffTheChunkGrid() {
        assertThrows(IllegalArgumentException.class, () -> withCache(-1, 2048));
        assertThrows(IllegalArgumentException.class, () -> withCache(1 << 30, 0));
        assertThrows(IllegalArgumentException.class, () -> withCache(1 << 30, 1000));
        assertThrows(IllegalArgumentException.class, () -> withCache(1 << 30, -512));
    }

    @Test
    void carriesTheSelectedCacheFormat() {
        assertNull(withCache(0, 2048).kvCache(), "the artifact's format");
        var selected = new InferenceConfig(
                Path.of("a"), Path.of("t"), Path.of("l"), cpus(), 32768, Duration.ofSeconds(1), 0, 2048, KvFormat.BF16);
        assertEquals(KvFormat.BF16, selected.kvCache());
    }

    private static InferenceConfig fromSettings(Map<String, String> extra) {
        var environment = new HashMap<>(Map.of(
                "EUHEDRAL_MODEL_ARTIFACT", "m.edrl",
                "EUHEDRAL_MODEL_TOKENIZER", "tok",
                "EUHEDRAL_MODEL_NAME", "n",
                "EUHEDRAL_HOST_CPUS", "0"));
        environment.putAll(extra);
        return InferenceConfig.of(
                SettingsLoader.load(Optional.empty(), environment), NativeFiles.at(Path.of("native")));
    }

    @Test
    void settingsBecomeTheEngineInput() {
        var config = fromSettings(Map.of(
                "EUHEDRAL_CONTEXT_MAX_TOKENS", "4096",
                "EUHEDRAL_CONTEXT_KV_CACHE_FORMAT", "bf16",
                "EUHEDRAL_CONTEXT_KV_CACHE_ON_GPU_TOKENS", "8192",
                "EUHEDRAL_SERVER_SHUTDOWN_TIMEOUT", "4s",
                "EUHEDRAL_PREFIX_CACHE_SIZE", "1MiB",
                "EUHEDRAL_PREFIX_CACHE_CHECKPOINT_TOKENS", "1024",
                "EUHEDRAL_BATCHING_MAX_SEQUENCES", "3"));
        assertEquals(Path.of("m.edrl"), config.artifactPath());
        assertEquals(Path.of("tok"), config.tokenizerDirectory());
        assertEquals(NativeFiles.at(Path.of("native")).library(), config.cudaLibraryPath());
        assertEquals(BitSet.valueOf(new long[] {1}), config.workerCpus());
        assertEquals(4096, config.maxContextTokens());
        assertEquals(KvFormat.BF16, config.kvCache());
        assertEquals(8192, config.kvDeviceMinimumTokens());
        assertEquals(Duration.ofSeconds(4), config.shutdownTimeout());
        assertEquals(1L << 20, config.prefixCacheBytes());
        assertEquals(1024, config.prefixCacheCheckpointTokens());
        assertEquals(3, config.maxSequences());
    }

    @Test
    void theSettingsTuneTheEngine() {
        var config = fromSettings(Map.of(
                "EUHEDRAL_BATCHING_PREFILL_CHUNK_TOKENS", "1024",
                "EUHEDRAL_MEMORY_GPU_LIMIT", "8GiB",
                "EUHEDRAL_HOST_TASK_PARALLELISM", "3",
                "EUHEDRAL_KERNELS_TRELLIS_TARGET_BLOCKS", "300"));
        assertEquals(1024, config.tuning().batching().prefillChunkTokens());
        assertEquals(8L << 30, config.tuning().memory().gpuLimit().getAsLong());
        assertEquals(3, config.tuning().host().taskParallelism());
        assertEquals(300, config.tuning().kernels().trellis().targetBlocks().getAsInt());
        var programmatic = new InferenceConfig(Path.of("a"), Path.of("t"), Path.of("l"), cpus(), Duration.ofSeconds(1));
        assertEquals(Tuning.defaults(), programmatic.tuning());
        var tuned = programmatic.withTuning(config.tuning());
        assertEquals(config.tuning(), tuned.tuning());
        assertEquals(programmatic.artifactPath(), tuned.artifactPath());
    }

    @Test
    void defaultsTakeTheArtifactsCacheFormat() {
        var config = fromSettings(Map.of());
        assertNull(config.kvCache());
        assertEquals(InferenceConfig.DEFAULT_MAX_CONTEXT_TOKENS, config.maxContextTokens());
        assertEquals(InferenceConfig.DEFAULT_PREFIX_CACHE_BYTES, config.prefixCacheBytes());
        assertEquals(InferenceConfig.DEFAULT_KV_DEVICE_MINIMUM_TOKENS, config.kvDeviceMinimumTokens());
    }
}
