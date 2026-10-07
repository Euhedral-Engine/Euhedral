package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.EngineExecutionFixture;
import io.euhedral_execution.inference.core.model.qwen38.SequenceStateProbe;
import io.euhedral_execution.inference.core.model.qwen38.Session;
import io.euhedral_execution.inference.core.prefix.PrefixCacheStats;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// A restored sequence continues exactly as a cold one: the state at a checkpoint is the state a cold prefill
/// holds there, and the rest is prefilled in the chunks a cold run uses.
class PrefixCacheCudaIntegrationTest {
    private static final long CACHE_BYTES = 4L << 30;
    private static final int INTERVAL = 2048;

    private static InferenceConfig config(long cacheBytes) {
        return config("euhedral.qwen.artifact", "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl", cacheBytes);
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
    private static int[] prompt(InferenceEngine engine, String lead, int tokens) {
        String text =
                lead + " " + "The quick brown fox jumps over the lazy dog while the engine counts tokens. ".repeat(900);
        int[] ids = engine.tokenizer().encodeWithModelSpecialTokens(text);
        assertTrue(ids.length >= tokens, "the prompt text is too short");
        return Arrays.copyOf(ids, tokens);
    }

    /// Sampling with a fixed seed: equal logits give equal tokens, and the session stays off the speculative
    /// path, which restores through MTP state of its own.
    private static GenerationConfig sampling(long seed) {
        return new GenerationConfig(0.7f, 40, 0.9f, seed, false);
    }

    private static List<Integer> generate(InferenceEngine engine, int[] prompt, int newTokens, long seed)
            throws Exception {
        try (Session session = engine.createSession(sampling(seed))) {
            return session.generate(prompt, newTokens, text -> {}, null);
        }
    }

    private static List<String> stateAfterPrefill(InferenceEngine engine, ExecutionGpu gpu, int[] prompt)
            throws Exception {
        try (Session session = engine.createSession(sampling(1))) {
            session.generate(prompt, 0, text -> {}, null);
            return SequenceStateProbe.committedDigests(
                    gpu,
                    EngineExecutionFixture.sequence(session),
                    engine.modelConfig().layerTypes());
        }
    }

    @Test
    @Timeout(1800)
    void aRestoredSessionHoldsTheColdStateBufferForBuffer() throws Exception {
        var bootstrap = new RecordingBootstrap();
        try (InferenceEngine engine = InferenceEngine.load(config(CACHE_BYTES), bootstrap)) {
            // 4352 tokens are 17 full pages. Chunks end at multiples of 512, so the cold run stores
            // checkpoints at 2048 and 4096 and prefills 256 tokens last; the warm run restores 4096 and
            // prefills the same 256 in the same single chunk.
            int[] prompt = prompt(engine, "Alpha", 4352);
            List<String> cold = stateAfterPrefill(engine, bootstrap.gpu, prompt);
            assertEquals(2, engine.prefixCacheStats().captured());
            List<String> warm = stateAfterPrefill(engine, bootstrap.gpu, prompt);
            assertEquals(1, engine.prefixCacheStats().hits());
            assertEquals(4096, engine.prefixCacheStats().reusedTokens());
            assertEquals(cold, warm);
        }
    }

    @Test
    @Timeout(1800)
    void sampledOutputAfterARestoreEqualsTheColdRun() throws Exception {
        try (InferenceEngine engine = InferenceEngine.load(config(CACHE_BYTES), new RecordingBootstrap())) {
            // 5000 tokens: checkpoints at 2048 and 4096, and at 4608, the last chunk boundary before the end.
            int[] prompt = prompt(engine, "Bravo", 5000);
            List<Integer> cold = generate(engine, prompt, 24, 5);
            List<Integer> warm = generate(engine, prompt, 24, 5);
            assertEquals(cold, warm);
            assertEquals(3, engine.prefixCacheStats().captured());
            assertEquals(1, engine.prefixCacheStats().hits());
            assertEquals(4608, engine.prefixCacheStats().reusedTokens());
        }
    }

    @Test
    @Timeout(1800)
    void aPromptThatEndsOnACheckpointStillPrefillsAChunk() throws Exception {
        try (InferenceEngine engine = InferenceEngine.load(config(CACHE_BYTES), new RecordingBootstrap())) {
            int[] prompt = prompt(engine, "Charlie", 4096);
            List<Integer> cold = generate(engine, prompt, 16, 3);
            List<Integer> warm = generate(engine, prompt, 16, 3);
            assertEquals(cold, warm);
            assertEquals(3584, engine.prefixCacheStats().reusedTokens(), "the last boundary below the prompt's length");
        }
    }

    @Test
    @Timeout(3600)
    void aFollowUpThatExtendsTheExchangeEqualsAColdRunOfIt() throws Exception {
        int[] followUp;
        List<Integer> warm;
        long reused;
        try (InferenceEngine engine = InferenceEngine.load(config(CACHE_BYTES), new RecordingBootstrap())) {
            int[] first = prompt(engine, "Delta", 5000);
            List<Integer> fed = new ArrayList<>(generate(engine, first, 24, 7));
            // A sampled terminator is returned but never fed to the model.
            if (!fed.isEmpty() && engine.tokenizer().isGenerationEosToken(fed.getLast())) fed.removeLast();
            int[] extra = engine.tokenizer().encodeText(" And then the second turn asks something else entirely.");
            followUp = new int[first.length + fed.size() + extra.length];
            System.arraycopy(first, 0, followUp, 0, first.length);
            for (int i = 0; i < fed.size(); i++) followUp[first.length + i] = fed.get(i);
            System.arraycopy(extra, 0, followUp, first.length + fed.size(), extra.length);
            long before = engine.prefixCacheStats().reusedTokens();
            warm = generate(engine, followUp, 24, 9);
            reused = engine.prefixCacheStats().reusedTokens() - before;
        }
        assertEquals(4608, reused, "the follow-up restores the first prompt's last chunk boundary");
        try (InferenceEngine reference = InferenceEngine.load(config(0), new RecordingBootstrap())) {
            assertEquals(generate(reference, followUp, 24, 9), warm);
        }
    }

    @Test
    @Timeout(3600)
    void anEvictionRunStaysWithinTheBudgetAndKeepsHitting() throws Exception {
        try (InferenceEngine engine = InferenceEngine.load(config(400L << 20), new RecordingBootstrap())) {
            int[] last = null;
            for (String lead : List.of("Echo", "Foxtrot", "Golf", "Hotel", "India")) {
                last = prompt(engine, lead, 5000);
                generate(engine, last, 4, 1);
            }
            PrefixCacheStats stats = engine.prefixCacheStats();
            assertTrue(stats.evictions() > 0, "five prompts do not fit 400 MiB");
            assertTrue(stats.usedBytes() <= stats.totalBytes());
            long hits = stats.hits();
            generate(engine, last, 4, 1);
            assertEquals(hits + 1, engine.prefixCacheStats().hits(), "the newest prompt is still stored");
        }
    }

    @Test
    @Timeout(3600)
    void aSpeculativePromptRestoresThroughMtpCheckpointsAndGeneratesTheSameTokens() throws Exception {
        String property = "euhedral.qwen.nvfp4-artifact";
        String artifact = "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_nvfp4.edrl";
        int[] diverging;
        List<Integer> restored;
        try (InferenceEngine engine =
                InferenceEngine.load(config(property, artifact, CACHE_BYTES), new RecordingBootstrap())) {
            assumeTrue(
                    engine.profile() != null && engine.profile().speculative(), "needs an artifact with the MTP layer");
            // Greedy and unconstrained: the engine drafts with MTP. The cold run stores MTP checkpoints at 2048,
            // 4096 and 4608; the warm run restores 4608, re-pairs its last MTP row and prefills the rest.
            int[] prompt = prompt(engine, "Kilo", 5000);
            List<Integer> cold = greedy(engine, prompt, 48);
            assertEquals(3, engine.prefixCacheStats().captured());
            List<Integer> warm = greedy(engine, prompt, 48);
            assertEquals(cold, warm);
            assertEquals(1, engine.prefixCacheStats().hits());
            assertEquals(4608, engine.prefixCacheStats().reusedTokens());
            // A prompt that diverges after 4200 tokens restores the node at 4096.
            diverging = Arrays.copyOf(prompt, 5000);
            for (int i = 4200; i < diverging.length; i++) diverging[i] = prompt[i] ^ 1;
            restored = greedy(engine, diverging, 48);
            assertEquals(4096, engine.prefixCacheStats().reusedTokens() - 4608);
        }
        try (InferenceEngine reference =
                InferenceEngine.load(config(property, artifact, 0), new RecordingBootstrap())) {
            assertEquals(greedy(reference, diverging, 48), restored);
        }
    }

    @Test
    @Timeout(3600)
    void aSpeculativeRestoreLeavesTheMtpCacheAsAColdRunDoes() throws Exception {
        var bootstrap = new RecordingBootstrap();
        try (InferenceEngine engine = InferenceEngine.load(
                config(
                        "euhedral.qwen.nvfp4-artifact",
                        "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_nvfp4.edrl",
                        CACHE_BYTES),
                bootstrap)) {
            assumeTrue(
                    engine.profile() != null && engine.profile().speculative(), "needs an artifact with the MTP layer");
            int[] prompt = prompt(engine, "Lima", 5000);
            List<String> cold = mtpAndBaseDigests(engine, bootstrap.gpu, prompt);
            List<String> warm = mtpAndBaseDigests(engine, bootstrap.gpu, prompt);
            assertEquals(1, engine.prefixCacheStats().hits());
            assertEquals(4608, engine.prefixCacheStats().reusedTokens());
            // The base state and every MTP page are the cold run's, except the page that holds MTP row 4607: a
            // restore at 4608 recomputes that row in a one-row quantum, where the cold run computed it inside a
            // chunk's catch-up. It differs in low bits, which can change a draft but never a token.
            String recomputed = "mtp page " + (4607 / 256) + " ";
            assertEquals(without(cold, recomputed), without(warm, recomputed));
            assertEquals(cold.size(), warm.size());
        }
    }

    private static List<String> without(List<String> digests, String prefix) {
        return digests.stream().filter(digest -> !digest.startsWith(prefix)).toList();
    }

    /// The digests of the base state and of the MTP cache after a one-token speculative generation.
    private static List<String> mtpAndBaseDigests(InferenceEngine engine, ExecutionGpu gpu, int[] prompt)
            throws Exception {
        try (Session session = engine.createSession(GenerationConfig.greedy(1))) {
            session.generate(prompt, 1, text -> {}, null);
            var sequence = EngineExecutionFixture.sequence(session);
            List<String> digests = new ArrayList<>(SequenceStateProbe.committedDigests(
                    gpu, sequence, engine.modelConfig().layerTypes()));
            List<String> mtp = SequenceStateProbe.mtpDigests(
                    gpu, sequence, engine.modelConfig().layerTypes(), prompt.length);
            assertTrue(!mtp.isEmpty(), "the speculative sequence has an MTP cache");
            digests.addAll(mtp);
            return digests;
        }
    }

    private static List<Integer> greedy(InferenceEngine engine, int[] prompt, int newTokens) throws Exception {
        try (Session session = engine.createSession(GenerationConfig.greedy(1))) {
            return session.generate(prompt, newTokens, text -> {}, null);
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
