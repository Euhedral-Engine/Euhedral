package io.euhedral_execution.inference.api.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.ProcessorTopology;
import io.euhedral_execution.inference.core.state.KvFormat;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.springframework.boot.context.properties.bind.BindException;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource;

class InferencePropertiesTest {
    private static final Path PATH = Path.of("/x");

    @Test
    void parsesProcessorIdsAndInclusiveRanges() {
        BitSet expected = new BitSet();
        expected.set(2, 6);
        expected.set(8);
        assertEquals(expected, InferenceProperties.parseCpus(" 2-5, 8 ,"));
        var config = new InferenceProperties(PATH, PATH, PATH, "3", 4096, Duration.ofSeconds(4), "m", 0, 2048, null, 1)
                .toInferenceConfig();
        assertEquals(BitSet.valueOf(new long[] {1L << 3}), config.workerCpus());
        assertEquals(Duration.ofSeconds(4), config.shutdownTimeout());
    }

    @Test
    void rejectsInvalidConfigurationAtBindTime() {
        assertThrows(IllegalArgumentException.class, () -> InferenceProperties.parseCpus("5-2"));
        assertThrows(IllegalArgumentException.class, () -> InferenceProperties.parseCpus("-1"));
        assertThrows(IllegalArgumentException.class, () -> InferenceProperties.parseCpus("a"));
        assertThrows(IllegalArgumentException.class, () -> InferenceProperties.parseCpus(" , "));
        assertThrows(
                IllegalArgumentException.class,
                () -> new InferenceProperties(
                        null, PATH, PATH, "1", 4096, Duration.ofSeconds(1), "m", 0, 2048, null, 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> new InferenceProperties(
                        PATH, PATH, PATH, "1", 4096, Duration.ofSeconds(1), " ", 0, 2048, null, 1));
    }

    private static InferenceProperties bind(Map<String, String> overrides) {
        Map<String, String> values = new HashMap<>(Map.of(
                "euhedral.inference.artifact-path", "/m.edrl",
                "euhedral.inference.tokenizer-directory", "/t",
                "euhedral.inference.cuda-library-path", "/lib.so",
                "euhedral.inference.worker-cpus", "2-3",
                "euhedral.inference.model-id", "m"));
        values.putAll(overrides);
        return new Binder(new MapConfigurationPropertySource(values))
                .bind("euhedral.inference", InferenceProperties.class)
                .get();
    }

    @Test
    void workerCpusDefaultToTheCoresTheProcessHasCpuTimeFor() {
        Map<String, String> values = new HashMap<>(Map.of(
                "euhedral.inference.artifact-path", "/m.edrl",
                "euhedral.inference.tokenizer-directory", "/t",
                "euhedral.inference.cuda-library-path", "/lib.so",
                "euhedral.inference.model-id", "m"));
        InferenceProperties unset = new Binder(new MapConfigurationPropertySource(values))
                .bind("euhedral.inference", InferenceProperties.class)
                .get();
        assertEquals(
                ProcessorTopology.system()
                        .processorsOfCores(Runtime.getRuntime().availableProcessors()),
                unset.toInferenceConfig().workerCpus());
    }

    @Test
    void maxContextDefaultsAndBinds() {
        assertEquals(
                InferenceConfig.DEFAULT_MAX_CONTEXT_TOKENS,
                bind(Map.of()).toInferenceConfig().maxContextTokens());
        assertEquals(
                65536,
                bind(Map.of("euhedral.inference.max-context-tokens", "65536"))
                        .toInferenceConfig()
                        .maxContextTokens());
    }

    @Test
    void maxSequencesDefaultsToOneAndBinds() {
        assertEquals(1, bind(Map.of()).toInferenceConfig().maxSequences());
        assertEquals(
                0,
                bind(Map.of("euhedral.inference.max-sequences", "0"))
                        .toInferenceConfig()
                        .maxSequences());
        assertThrows(Exception.class, () -> bind(Map.of("euhedral.inference.max-sequences", "-1")));
    }

    @Test
    void prefixCacheDefaultsOnAndBinds() {
        assertEquals(4L << 30, bind(Map.of()).toInferenceConfig().prefixCacheBytes());
        assertEquals(2048, bind(Map.of()).toInferenceConfig().prefixCacheCheckpointTokens());
        assertEquals(
                0,
                bind(Map.of("euhedral.inference.prefix-cache-bytes", "0"))
                        .toInferenceConfig()
                        .prefixCacheBytes());
        assertEquals(
                4096,
                bind(Map.of("euhedral.inference.prefix-cache-checkpoint-tokens", "4096"))
                        .toInferenceConfig()
                        .prefixCacheCheckpointTokens());
    }

    @Test
    void theCacheFormatIsTheArtifactsUnlessBound() {
        assertEquals(null, bind(Map.of()).toInferenceConfig().kvCache());
        assertEquals(
                KvFormat.BF16,
                bind(Map.of("euhedral.inference.kv-cache", "bf16"))
                        .toInferenceConfig()
                        .kvCache());
        assertEquals(
                KvFormat.NVFP4,
                bind(Map.of("euhedral.inference.kv-cache", "nvfp4"))
                        .toInferenceConfig()
                        .kvCache());
        assertThrows(BindException.class, () -> bind(Map.of("euhedral.inference.kv-cache", "fp8")));
    }

    @Test
    void rejectsAnInvalidPrefixCacheAtBindTime() {
        assertThrows(BindException.class, () -> bind(Map.of("euhedral.inference.prefix-cache-bytes", "-1")));
        assertThrows(
                BindException.class, () -> bind(Map.of("euhedral.inference.prefix-cache-checkpoint-tokens", "1000")));
        assertThrows(BindException.class, () -> bind(Map.of("euhedral.inference.prefix-cache-checkpoint-tokens", "0")));
    }

    @Test
    void rejectsNonPositiveMaxContextAtBindTime() {
        var failure =
                assertThrows(BindException.class, () -> bind(Map.of("euhedral.inference.max-context-tokens", "0")));
        assertEquals(IllegalArgumentException.class, rootCause(failure).getClass());
    }

    private static Throwable rootCause(Throwable failure) {
        while (failure.getCause() != null) failure = failure.getCause();
        return failure;
    }
}
