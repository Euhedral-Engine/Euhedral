package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.model.qwen38.EngineExecutionFixture;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.model.qwen38.SequenceStateProbe;
import io.euhedral_execution.inference.core.model.qwen38.Session;
import io.euhedral_execution.inference.core.prefix.PrefixCacheStats;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

/// Engines and prompts shared by the prefix-cache CUDA tests. An engine is kept for the tests that ask for the same
/// artifact and cache size; asking for another closes it, because two engines do not fit the device.
///
/// The checkpoint interval is two prefill chunks, so a prompt of a few thousand tokens has checkpoints that are
/// chunk boundaries and others that are not, which is all the cache's behaviour depends on.
final class PrefixCacheEngines {
    static final long CACHE_BYTES = 4L << 30;
    static final int INTERVAL = 2 * InferenceConfig.PREFILL_CHUNK_TOKENS;

    static final String Q3_PROPERTY = "euhedral.qwen.artifact";
    static final String Q3_DEFAULT = "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl";
    static final String NVFP4_PROPERTY = "euhedral.qwen.nvfp4-artifact";
    static final String NVFP4_DEFAULT = "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_nvfp4.edrl";

    private PrefixCacheEngines() {}

    static SharedEngines.Held q3(long cacheBytes) {
        return held(Q3_PROPERTY, Q3_DEFAULT, cacheBytes);
    }

    static SharedEngines.Held nvfp4(long cacheBytes) {
        return held(NVFP4_PROPERTY, NVFP4_DEFAULT, cacheBytes);
    }

    private static SharedEngines.Held held(String property, String artifact, long cacheBytes) {
        return SharedEngines.get(
                "prefix-cache:" + property + ":" + cacheBytes, () -> config(property, artifact, cacheBytes));
    }

    private static InferenceConfig config(String artifactProperty, String artifactDefault, long cacheBytes) {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)));
        Path artifact = Path.of(System.getProperty(artifactProperty, artifactDefault));
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(artifact) && Files.isRegularFile(tokenizer.resolve("tokenizer.json")));
        BitSet cpus = new BitSet();
        var available = SystemInfo.getPCpuSet();
        for (int cpu = available.nextSetBit(0); cpu >= 0 && cpus.cardinality() < 2; cpu = available.nextSetBit(cpu + 1))
            cpus.set(cpu);
        assumeTrue(cpus.cardinality() == 2);
        return new InferenceConfig(
                artifact, tokenizer, Path.of(library), cpus, 32768, Duration.ofSeconds(10), cacheBytes, INTERVAL);
    }

    /// The first `tokens` tokens of a long text that starts with `lead`, so prompts with different leads share
    /// nothing.
    static int[] prompt(InferenceEngine engine, String lead, int tokens) {
        String text =
                lead + " " + "The quick brown fox jumps over the lazy dog while the engine counts tokens. ".repeat(300);
        int[] ids = engine.tokenizer().encodeWithModelSpecialTokens(text);
        assertTrue(ids.length >= tokens, "the prompt text is too short");
        return Arrays.copyOf(ids, tokens);
    }

    /// Sampling with a fixed seed: equal logits give equal tokens, and the session stays off the speculative
    /// path, which restores through MTP state of its own.
    static GenerationConfig sampling(long seed) {
        return new GenerationConfig(0.7f, 40, 0.9f, seed, false);
    }

    static List<Integer> generate(InferenceEngine engine, int[] prompt, int newTokens, long seed) throws Exception {
        try (Session session = engine.createSession(sampling(seed))) {
            return session.generate(prompt, newTokens, text -> {}, null);
        }
    }

    static List<String> stateAfterPrefill(SharedEngines.Held held, int[] prompt) throws Exception {
        try (Session session = held.engine().createSession(sampling(1))) {
            session.generate(prompt, 0, text -> {}, null);
            return SequenceStateProbe.committedDigests(
                    held.gpu(),
                    EngineExecutionFixture.sequence(session),
                    ((Qwen38Runtime) held.engine().modelRuntime()).config().layerTypes());
        }
    }

    /// The cache's counters now, to be compared with the counters after a step: engines are shared, so a test
    /// reads what it changed rather than a total.
    static Counts counts(InferenceEngine engine) {
        PrefixCacheStats stats = engine.prefixCacheStats();
        return new Counts(stats.captured(), stats.hits(), stats.reusedTokens());
    }

    record Counts(long captured, long hits, long reusedTokens) {
        Counts since(Counts earlier) {
            return new Counts(
                    this.captured - earlier.captured,
                    this.hits - earlier.hits,
                    this.reusedTokens - earlier.reusedTokens);
        }
    }
}
