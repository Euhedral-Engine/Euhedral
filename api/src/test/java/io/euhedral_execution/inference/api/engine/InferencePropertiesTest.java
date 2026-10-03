package io.euhedral_execution.inference.api.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.InferenceConfig;
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
        var config =
                new InferenceProperties(PATH, PATH, PATH, "3", 4096, Duration.ofSeconds(4), "m").toInferenceConfig();
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
                () -> new InferenceProperties(null, PATH, PATH, "1", 4096, Duration.ofSeconds(1), "m"));
        assertThrows(
                IllegalArgumentException.class,
                () -> new InferenceProperties(PATH, PATH, PATH, "1", 4096, Duration.ofSeconds(1), " "));
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
