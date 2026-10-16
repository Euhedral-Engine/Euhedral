package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.PrefixCacheEngines.Counts;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.SessionOptions;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.EngineExecutionFixture;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.model.qwen38.SequenceStateProbe;
import io.euhedral_execution.inference.core.model.qwen38.Session;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.MixedArtifacts;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// An engine over a composite artifact, a trellis artifact whose FFN projections are the NVFP4 artifact's (see
/// [MixedArtifacts]), which stores an MTP draft depth of 4: MTP speculation commits the tokens of greedy one-row
/// decode over 64 tokens, and a prompt restored from a prefix-cache checkpoint continues as the uncached one does.
/// The files are `euhedral.test.trellis-artifact` and `euhedral.test.nvfp4-artifact`; unset ones skip the test.
@ModelGroup.OwnJvm
class MixedArtifactEngineCudaIntegrationTest {

    private static final int CONTEXT_TOKENS = 8192;
    private static final long CACHE_BYTES = 2L << 30;
    private static final int NEW_TOKENS = 64;

    /// Reads the composite instead of the artifact file, and keeps the device handle the engine opens.
    private static final class CompositeBootstrap extends InferenceEngine.Bootstrap {
        private final Artifact artifact;
        private volatile CudaGpuMemory gpu;

        CompositeBootstrap(Artifact artifact) {
            this.artifact = artifact;
        }

        @Override
        Artifact readArtifact(Path path) {
            return this.artifact;
        }

        @Override
        ExecutionGpu openGpu(Path path) {
            ExecutionGpu opened = super.openGpu(path);
            this.gpu = (CudaGpuMemory) opened;
            return opened;
        }
    }

    private static InferenceConfig config(Path trellis) {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "unset-euhedral.qwen.tokenizer-dir"));
        assumeTrue(Files.isRegularFile(tokenizer.resolve("tokenizer.json")), "no tokenizer");
        return new InferenceConfig(
                trellis,
                tokenizer,
                Path.of(library),
                SystemInfo.getPCpuSet(),
                CONTEXT_TOKENS,
                Duration.ofSeconds(30),
                CACHE_BYTES,
                PrefixCacheEngines.INTERVAL);
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.MINUTES, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void speculationAndTheCacheOfAMixedArtifactEqualPlainGreedyDecode() throws Exception {
        Path trellis = MixedArtifacts.trellis();
        Artifact mixed = MixedArtifacts.mixed(trellis, MixedArtifacts.nvfp4(), 4);
        var bootstrap = new CompositeBootstrap(mixed);
        try (InferenceEngine engine = InferenceEngine.load(config(trellis), bootstrap)) {
            assertTrue(engine.description().speculative(), "the composite carries the MTP layer");

            // (c) Speculation commits what greedy one-row decode commits.
            int[] question = engine.tokenizer().encodeWithModelSpecialTokens("Explain how a heat pump works.");
            List<Integer> greedy = generate(engine, question, new SessionOptions(false));
            assertEquals(NEW_TOKENS, greedy.size());
            assertEquals(greedy, generate(engine, question, new SessionOptions(true)), "speculative decoding");

            // (e) A prompt restored from a checkpoint continues as the cold one: the same committed state, and the
            // same tokens after it.
            int[] prompt = PrefixCacheEngines.prompt(engine, "Alpha", 2304);
            Counts start = PrefixCacheEngines.counts(engine);
            List<String> cold = stateAfterPrefill(engine, bootstrap.gpu, prompt);
            assertEquals(2, PrefixCacheEngines.counts(engine).since(start).captured());
            List<String> warm = stateAfterPrefill(engine, bootstrap.gpu, prompt);
            Counts change = PrefixCacheEngines.counts(engine).since(start);
            assertEquals(1, change.hits());
            assertEquals(2048, change.reusedTokens());
            assertEquals(cold, warm);

            int[] other = PrefixCacheEngines.prompt(engine, "Bravo", 2700);
            Counts begin = PrefixCacheEngines.counts(engine);
            List<Integer> uncached = PrefixCacheEngines.generate(engine, other, 16, 5);
            List<Integer> restored = PrefixCacheEngines.generate(engine, other, 16, 5);
            assertEquals(uncached, restored, "the tokens after a restore");
            Counts second = PrefixCacheEngines.counts(engine).since(begin);
            assertEquals(1, second.hits());
            assertEquals(2560, second.reusedTokens());
        }
    }

    private static List<String> stateAfterPrefill(InferenceEngine engine, CudaGpuMemory gpu, int[] prompt)
            throws Exception {
        try (Session session = engine.createSession(PrefixCacheEngines.sampling(1))) {
            session.generate(prompt, 0, text -> {}, null);
            return SequenceStateProbe.committedDigests(
                    gpu,
                    EngineExecutionFixture.sequence(session),
                    ((Qwen38Runtime) engine.modelRuntime()).config().layerTypes());
        }
    }

    private static List<Integer> generate(InferenceEngine engine, int[] prompt, SessionOptions options)
            throws Exception {
        GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(7L), options);
        CompletableFuture<List<Integer>> done = session.generateAsync(prompt, NEW_TOKENS, text -> {}, null, null)
                .whenComplete((tokens, failure) -> {
                    try {
                        session.close();
                    } catch (RuntimeException ignored) {
                        // The result carries any failure that matters.
                    }
                });
        return done.get(10, TimeUnit.MINUTES);
    }
}
