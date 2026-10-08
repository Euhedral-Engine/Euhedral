package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.EngineExecutionFixture;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.model.qwen38.SequenceStateProbe;
import io.euhedral_execution.inference.core.model.qwen38.Session;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// A DFlash2 generation restored from the prefix cache continues exactly as a cold one: the checkpoint holds the
/// target state and the drafter's context ring, and the rest is prefilled, tapped and projected in the chunks a cold
/// run uses. Skipped unless -Peuhedral.qwen.dflash2-artifact names a DFlash2 artifact.
// Own JVM: needs the DFlash2 artifact, which no group shares.
@ModelGroup.OwnJvm
class DFlash2PrefixCacheCudaIntegrationTest {
    private static final long CACHE_BYTES = 4L << 30;
    private static final int INTERVAL = 2048;

    private static InferenceConfig config(long cacheBytes) {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)));
        Path artifact = Path.of(System.getProperty("euhedral.qwen.dflash2-artifact", ""));
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(artifact), "no DFlash2 artifact: " + artifact);
        BitSet cpus = new BitSet();
        var available = SystemInfo.getPCpuSet();
        for (int cpu = available.nextSetBit(0); cpu >= 0 && cpus.cardinality() < 2; cpu = available.nextSetBit(cpu + 1))
            cpus.set(cpu);
        return new InferenceConfig(
                artifact, tokenizer, Path.of(library), cpus, 8192, Duration.ofSeconds(10), cacheBytes, INTERVAL);
    }

    private static InferenceEngine load(InferenceConfig config, RecordingBootstrap bootstrap) throws Exception {
        InferenceEngine engine = InferenceEngine.load(config, bootstrap);
        if (!"dflash2".equals(engine.description().speculation())) {
            engine.close();
            assumeTrue(false, "not a DFlash2 artifact");
        }
        return engine;
    }

    private static int[] prompt(InferenceEngine engine, String lead, int tokens) {
        String text =
                lead + " " + "The quick brown fox jumps over the lazy dog while the engine counts tokens. ".repeat(900);
        return Arrays.copyOf(engine.tokenizer().encodeWithModelSpecialTokens(text), tokens);
    }

    private static List<Integer> greedy(InferenceEngine engine, int[] prompt, int newTokens) throws Exception {
        try (Session session = engine.createSession(GenerationConfig.greedy(1))) {
            return session.generate(prompt, newTokens, text -> {}, null);
        }
    }

    @Test
    @Timeout(3600)
    void aRestoredDFlash2GenerationProducesTheColdAndUncachedTokens() throws Exception {
        int[] prompt;
        int[] diverging;
        List<Integer> cold;
        List<Integer> restored;
        try (InferenceEngine engine = load(config(CACHE_BYTES), new RecordingBootstrap())) {
            // 5000 tokens: the cold run stores DFlash2 checkpoints at 2048, 4096 and 4608 (the last chunk boundary);
            // the warm run restores 4608, prefills the last 392 tokens and drafts through many verifications.
            prompt = prompt(engine, "Kilo", 5000);
            cold = greedy(engine, prompt, 64);
            assertEquals(3, engine.prefixCacheStats().captured());
            for (int repeat = 0; repeat < 2; repeat++)
                assertEquals(cold, greedy(engine, prompt, 64), "repeat " + repeat);
            assertEquals(2, engine.prefixCacheStats().hits());
            assertEquals(2 * 4608, engine.prefixCacheStats().reusedTokens());
            // A prompt that diverges after 4200 tokens restores the node at 4096, before the prompt's tail.
            diverging = Arrays.copyOf(prompt, 5000);
            for (int i = 4200; i < diverging.length; i++) diverging[i] = prompt[i] ^ 1;
            restored = greedy(engine, diverging, 64);
            assertEquals(2 * 4608 + 4096, engine.prefixCacheStats().reusedTokens());
        }
        try (InferenceEngine reference = load(config(0), new RecordingBootstrap())) {
            assertEquals(cold, greedy(reference, prompt, 64), "the cached engine's cold run");
            assertEquals(greedy(reference, diverging, 64), restored, "the restore at 4096");
        }
    }

    @Test
    @Timeout(3600)
    void aDFlash2RestoreLeavesTheTargetStateAndTheDrafterRingAsAColdRunDoes() throws Exception {
        var bootstrap = new RecordingBootstrap();
        try (InferenceEngine engine = load(config(CACHE_BYTES), bootstrap)) {
            int[] prompt = prompt(engine, "Lima", 5000);
            List<String> cold = digests(engine, bootstrap.gpu, prompt);
            List<String> warm = digests(engine, bootstrap.gpu, prompt);
            assertEquals(1, engine.prefixCacheStats().hits());
            assertEquals(4608, engine.prefixCacheStats().reusedTokens());
            assertTrue(cold.stream().anyMatch(digest -> digest.startsWith("dflash2 layer")), "the sequence drafted");
            assertEquals(cold, warm);
        }
    }

    @Test
    @Timeout(3600)
    void evictionKeepsDFlash2GenerationsExactAndCancellationLeaksNoDeviceMemory() throws Exception {
        // Each 2048-token DFlash2 checkpoint is about 0.15 GiB of GDN state, its KV pages and 40 MiB of drafter
        // ring; 600 MiB holds about three, so four prompts evict.
        List<List<Integer>> expected = new ArrayList<>();
        List<int[]> prompts = new ArrayList<>();
        try (InferenceEngine reference = load(config(0), new RecordingBootstrap())) {
            for (String lead : List.of("Mike", "November", "Oscar", "Papa")) {
                int[] prompt = prompt(reference, lead, 2600);
                prompts.add(prompt);
                expected.add(greedy(reference, prompt, 32));
            }
        }
        try (InferenceEngine engine = load(config(600L << 20), new RecordingBootstrap())) {
            for (int round = 0; round < 2; round++)
                for (int i = 0; i < prompts.size(); i++)
                    assertEquals(
                            expected.get(i), greedy(engine, prompts.get(i), 32), "round " + round + " prompt " + i);
            assertTrue(engine.prefixCacheStats().evictions() > 0, "the small cache evicted");
            long settled = engine.allocatedDeviceBytes();
            // A generation cancelled from its text callback ends with its sequence released.
            try (Session session = engine.createSession(GenerationConfig.greedy(1))) {
                AtomicInteger texts = new AtomicInteger();
                try {
                    session.generate(
                            prompts.get(0),
                            64,
                            text -> {
                                if (texts.incrementAndGet() == 3) session.cancel();
                            },
                            null);
                } catch (Exception cancelled) {
                    // The cancelled generation may report the cancellation.
                }
                assertTrue(session.isCancelled());
            }
            assertEquals(settled, engine.allocatedDeviceBytes(), "the cancelled sequence's device state was released");
        }
    }

    /// The digests of the target state and the drafter's ring after a one-token generation.
    private static List<String> digests(InferenceEngine engine, ExecutionGpu gpu, int[] prompt) throws Exception {
        try (Session session = engine.createSession(GenerationConfig.greedy(1))) {
            session.generate(prompt, 1, text -> {}, null);
            var sequence = EngineExecutionFixture.sequence(session);
            List<String> digests = new ArrayList<>(SequenceStateProbe.committedDigests(
                    gpu,
                    sequence,
                    ((Qwen38Runtime) engine.modelRuntime()).config().layerTypes()));
            digests.addAll(SequenceStateProbe.dflash2Digests(gpu, sequence));
            return digests;
        }
    }

    private static final class RecordingBootstrap extends InferenceEngine.Bootstrap {
        private CudaGpuMemory gpu;

        @Override
        ExecutionGpu openGpu(Path path) {
            ExecutionGpu opened = super.openGpu(path);
            this.gpu = (CudaGpuMemory) opened;
            return opened;
        }
    }
}
