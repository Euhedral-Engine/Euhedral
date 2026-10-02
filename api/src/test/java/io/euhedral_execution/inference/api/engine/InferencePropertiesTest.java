package io.euhedral_execution.inference.api.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.InferenceTuning;
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
        var config = new InferenceProperties(PATH, PATH, PATH, "3", Duration.ofSeconds(4), "m", null, null, null, null)
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
                        null, PATH, PATH, "1", Duration.ofSeconds(1), "m", null, null, null, null));
        assertThrows(
                IllegalArgumentException.class,
                () -> new InferenceProperties(
                        PATH, PATH, PATH, "1", Duration.ofSeconds(1), " ", null, null, null, null));
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
    void unsetPrefillChunkKeepsTheCoreDefaultTuning() {
        var config = bind(Map.of()).toInferenceConfig();
        BitSet cpus = new BitSet();
        cpus.set(2, 4);
        assertEquals(InferenceTuning.defaults(cpus), config.tuning());
        assertEquals(
                InferenceTuning.DEFAULT_PREFILL_CHUNK_TOKENS, config.tuning().prefillChunkTokens());
    }

    @Test
    void bindsPrefillChunkOverrideIntoCoreTuning() {
        var config =
                bind(Map.of("euhedral.inference.prefill-chunk-tokens", "256")).toInferenceConfig();
        assertEquals(256, config.tuning().prefillChunkTokens());
        assertEquals(config.workerCpus(), config.tuning().workerProcessorIds());
    }

    @Test
    void rejectsNonPositivePrefillChunkAtBindTime() {
        var failure =
                assertThrows(BindException.class, () -> bind(Map.of("euhedral.inference.prefill-chunk-tokens", "0")));
        assertEquals(IllegalArgumentException.class, rootCause(failure).getClass());
    }

    @Test
    void bindsHostWeightsAndSpeculativeDepthIntoCoreTuning() {
        var tuning = bind(Map.of(
                        "euhedral.inference.host-weight-mib", "1280",
                        "euhedral.inference.staging-slots", "6",
                        "euhedral.inference.speculative-depth", "3"))
                .toInferenceConfig()
                .tuning();
        assertEquals(1280L << 20, tuning.hostWeightBytes());
        assertEquals(6, tuning.stagingSlots());
        assertEquals(3, tuning.speculativeDepth());
        var defaults = bind(Map.of()).toInferenceConfig().tuning();
        assertEquals(0, defaults.hostWeightBytes());
        assertEquals(0, defaults.speculativeDepth());
        for (var invalid : java.util.List.of(
                Map.of("euhedral.inference.host-weight-mib", "-1"),
                Map.of("euhedral.inference.staging-slots", "1"),
                Map.of("euhedral.inference.speculative-depth", "8")))
            assertEquals(
                    IllegalArgumentException.class,
                    rootCause(assertThrows(BindException.class, () -> bind(invalid)))
                            .getClass());
    }

    private static Throwable rootCause(Throwable failure) {
        while (failure.getCause() != null) failure = failure.getCause();
        return failure;
    }
}
