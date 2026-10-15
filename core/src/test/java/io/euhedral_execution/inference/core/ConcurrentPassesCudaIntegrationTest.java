package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.SessionOptions;
import io.euhedral_execution.inference.core.model.qwen38.Execution.PassStatistics;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.model.qwen38.loader.ResidencyPlanner;
import io.euhedral_execution.inference.core.model.qwen38.loader.SequenceFootprint;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.state.KvFormat;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Four sessions generating at once on the NVFP4 artifact (which has the MTP drafter) share multi-sequence passes and
/// each produces exactly the tokens it produces alone, with speculation off and on. Verification quanta are batched
/// across sessions, and one more open session costs no more device memory than the sequence footprint derives.
// Own JVM: the artifact is chosen per JVM by a system property.
@ModelGroup.OwnJvm
class ConcurrentPassesCudaIntegrationTest {

    private static final int CONTEXT_TOKENS = 4096;
    private static final int NEW_TOKENS = 48;
    private static final int SEQUENCES = 4;
    private static final String[] PROMPTS = {
        "The capital of France is",
        "Write a short poem about the sea and the lighthouse keeper.",
        "List the planets of the solar system, from the sun outward, with one fact about each of them.",
        "Explain in a few sentences how a heat pump moves thermal energy from a cold source to a warm sink.",
    };

    @Test
    @Timeout(value = 45, unit = TimeUnit.MINUTES, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void concurrentSessionsMatchRunningAloneAndFootprintHolds() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        String artifact = System.getProperty("euhedral.qwen.nvfp4-artifact");
        assumeTrue(artifact != null && Files.isRegularFile(Path.of(artifact)), "no NVFP4 artifact");
        String tokenizer = System.getProperty("euhedral.qwen.tokenizer-dir");
        assumeTrue(tokenizer != null, "no tokenizer directory");
        var config = new InferenceConfig(
                Path.of(artifact),
                Path.of(tokenizer),
                Path.of(library),
                SystemInfo.getPCpuSet(),
                CONTEXT_TOKENS,
                Duration.ofSeconds(30),
                0,
                2048,
                null,
                SEQUENCES);
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            Qwen38Runtime runtime = (Qwen38Runtime) engine.modelRuntime();
            int sequences = engine.maxSequences();
            System.out.printf("CONCURRENT_PASSES max sequences %d (requested %d)%n", sequences, SEQUENCES);
            assertTrue(sequences >= 2, "the engine admits at least two sequences");
            int count = Math.min(sequences, SEQUENCES);
            int[][] prompts = new int[count][];
            for (int i = 0; i < count; i++) prompts[i] = engine.tokenizer().encodeWithModelSpecialTokens(PROMPTS[i]);

            for (boolean speculation : new boolean[] {false, true}) {
                SessionOptions options = new SessionOptions(speculation);
                List<List<Integer>> alone = new ArrayList<>();
                for (int[] prompt : prompts)
                    alone.add(generate(engine, prompt, options).get(10, TimeUnit.MINUTES));
                PassStatistics before = runtime.execution().statistics(Quantum.ExecutionKind.VERIFY);
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
                                "session " + i + " changed when run together, speculation " + speculation);
                } finally {
                    for (GenerationSession session : sessions) session.close();
                }
                PassStatistics after = runtime.execution().statistics(Quantum.ExecutionKind.VERIFY);
                System.out.printf(
                        "CONCURRENT_PASSES speculation %b: verify quanta %d, batched %d, segments %d, rows %d%n",
                        speculation,
                        after.quanta() - before.quanta(),
                        after.batchedQuanta() - before.batchedQuanta(),
                        after.segments() - before.segments(),
                        after.rows() - before.rows());
                assertTrue(
                        after.batchedQuanta() - before.batchedQuanta() > 0,
                        "concurrent sessions formed no batched verification pass, speculation " + speculation);
                assertTrue(
                        after.segments() - before.segments() > after.quanta() - before.quanta(),
                        "batched passes carried no more segments than quanta, speculation " + speculation);
            }

            long idle = engine.allocatedDeviceBytes();
            long idleRetained = engine.retainedWorkspaceBytes();
            long opened;
            try (GenerationSession session =
                    engine.createGenerationSession(GenerationConfig.greedy(7L), new SessionOptions(true))) {
                session.generateAsync(prompts[0], NEW_TOKENS, text -> {}, null, null)
                        .get(10, TimeUnit.MINUTES);
                opened = engine.allocatedDeviceBytes();
            }
            long retained = engine.retainedWorkspaceBytes() - idleRetained;
            long footprint = opened - idle - retained;
            var model = runtime.config();
            KvFormat kv = runtime.kvFormat();
            long derived = SequenceFootprint.extraSequenceBytes(
                            model, kv, CONTEXT_TOKENS, 1 + runtime.profile().speculativeDepth(), true)
                    + ResidencyPlanner.kvBytes(model, CONTEXT_TOKENS, kv);
            System.out.printf(
                    "CONCURRENT_PASSES footprint measured %d bytes (%.1f MiB), derived bound %d bytes (%.1f MiB)%n",
                    footprint, footprint / 1048576.0, derived, derived / 1048576.0);
            assertTrue(footprint <= derived, "one open sequence takes " + footprint + " > " + derived);
        }
    }

    private static CompletableFuture<List<Integer>> generate(
            InferenceEngine engine, int[] prompt, SessionOptions options) {
        GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(7L), options);
        return session.generateAsync(prompt, NEW_TOKENS, text -> {}, null, null).whenComplete((done, failure) -> {
            try {
                session.close();
            } catch (RuntimeException ignored) {
                // The result carries any failure that matters.
            }
        });
    }
}
