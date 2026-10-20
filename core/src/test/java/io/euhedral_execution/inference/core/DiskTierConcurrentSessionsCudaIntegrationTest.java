package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.SessionOptions;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.testing.Tunings;
import io.euhedral_execution.inference.core.testing.WorkerCpus;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Sessions that start together while weights are read from disk finish, and each produces the tokens it produces
/// alone. Their prompts are admitted at once on two workers, fewer than the sessions, so graphs submitting together
/// can fill the device streams and hold the workers in CUDA calls; the device never waits for a read, so the
/// streams drain and the workers are released. This test was not run against the code before that, so it is not
/// shown to fail there; [DiskTierManySequencesCudaIntegrationTest] reproduces the hang. The workers are
/// `-Peuhedral.test.worker-cpus`, or else the first two performance CPUs ([WorkerCpus]).
// Own JVM: loads the engine with residency budgets of its own.
@ModelGroup.OwnJvm
class DiskTierConcurrentSessionsCudaIntegrationTest {

    private static final int CONTEXT_TOKENS = 2048;
    private static final int NEW_TOKENS = 16;
    private static final int SEQUENCES = 4;
    private static final String[] PROMPTS = {
        "The capital of France is",
        "Write a short poem about the sea and the lighthouse keeper.",
        "List the planets of the solar system, from the sun outward, with one fact about each of them.",
        "Explain in a few sentences how a heat pump moves thermal energy from a cold source to a warm sink.",
    };

    @Test
    @Timeout(value = 30, unit = TimeUnit.MINUTES, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void sessionsStartedTogetherOnTheDiskTierMatchRunningAlone() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        String artifact = System.getProperty("euhedral.qwen.nvfp4-artifact");
        assumeTrue(artifact != null && Files.isRegularFile(Path.of(artifact)), "no NVFP4 artifact");
        BitSet workers = WorkerCpus.of(2);
        var config = new InferenceConfig(
                        Path.of(artifact),
                        Path.of(library),
                        workers,
                        CONTEXT_TOKENS,
                        Duration.ofSeconds(30),
                        0,
                        2048,
                        null,
                        SEQUENCES)
                .withTuning(Tunings.limits(OptionalLong.of(8L << 30), OptionalLong.of(2L << 30)));
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            var runtime = (Qwen38Runtime) engine.modelRuntime();
            assertTrue(runtime.diskBackedWeightBytes() > 0, "weights beyond the host budget are read from disk");
            int count = Math.min(engine.maxSequences(), SEQUENCES);
            assertTrue(count >= 2, "the engine admits at least two sequences");
            SessionOptions options = new SessionOptions(false);
            int[][] prompts = new int[count][];
            for (int i = 0; i < count; i++) prompts[i] = engine.tokenizer().encodeWithModelSpecialTokens(PROMPTS[i]);
            List<List<Integer>> alone = new ArrayList<>();
            for (int[] prompt : prompts)
                alone.add(generate(engine, prompt, options).get(10, TimeUnit.MINUTES));
            List<GenerationSession> sessions = new ArrayList<>();
            List<CompletableFuture<List<Integer>>> futures = new ArrayList<>();
            for (int[] prompt : prompts) {
                GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(7L), options);
                sessions.add(session);
                futures.add(session.generateAsync(prompt, NEW_TOKENS, text -> {}, null, null));
            }
            try {
                for (int i = 0; i < count; i++)
                    assertEquals(
                            alone.get(i),
                            futures.get(i).get(10, TimeUnit.MINUTES),
                            "session " + i + " changed when run together");
            } finally {
                for (GenerationSession session : sessions) session.close();
            }
        }
    }

    private static CompletableFuture<List<Integer>> generate(
            InferenceEngine engine, int[] prompt, SessionOptions options) {
        GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(7L), options);
        return session.generateAsync(prompt, NEW_TOKENS, text -> {}, null, null).whenComplete((done, failure) -> {
            try {
                session.close();
            } catch (RuntimeException ignored) {
                // The outcome is the future's.
            }
        });
    }
}
