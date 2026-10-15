package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.GenerationTimingListener;
import io.euhedral_execution.inference.core.generation.SessionOptions;
import io.euhedral_execution.inference.core.model.qwen38.Execution.PassStatistics;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
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

/// Sampled generation with the published settings (temperature 1, top-k 20, top-p 0.95) on the NVFP4 artifact
/// produces the same tokens with speculation off and on, with partially rejected drafts among the verifications,
/// and the same tokens when it shares multi-sequence passes with other sampled and greedy sessions as when it runs
/// alone. The seed is used: another seed gives other tokens.
// Own JVM: the artifact is chosen per JVM by a system property.
@ModelGroup.OwnJvm
class SampledPassesCudaIntegrationTest {

    private static final int CONTEXT_TOKENS = 4096;
    private static final int NEW_TOKENS = 64;
    private static final int SEQUENCES = 4;
    private static final long[] SEEDS = {11L, 12L, 13L};
    private static final String[] PROMPTS = {
        "Write a short poem about the sea and the lighthouse keeper.",
        "List the planets of the solar system, from the sun outward, with one fact about each of them.",
        "Explain in a few sentences how a heat pump moves thermal energy from a cold source to a warm sink.",
        "The capital of France is",
    };

    /// Counts the verifications of one generation by the drafts they accepted.
    private static final class Acceptance implements GenerationTimingListener {
        final long[] accepted = new long[16];
        long verifications;

        @Override
        public void promptEncoded(long nanos, int promptTokens) {}

        @Override
        public void prefillQuantum(long startNanos, long executedNanos, int tokens) {}

        @Override
        public void firstTokenSelected(long nanos, int tokenId) {}

        @Override
        public void decodeQuantum(
                long startNanos, long executedNanos, long selectedNanos, boolean sampled, int selectedTokenId) {}

        @Override
        public synchronized void speculativeStep(long startNanos, long executedNanos, int outputs, int acceptedDrafts) {
            this.accepted[Math.min(acceptedDrafts, this.accepted.length - 1)]++;
            this.verifications++;
        }

        void add(Acceptance other) {
            for (int i = 0; i < this.accepted.length; i++) this.accepted[i] += other.accepted[i];
            this.verifications += other.verifications;
        }

        double mean() {
            long sum = 0;
            for (int a = 0; a < this.accepted.length; a++) sum += a * this.accepted[a];
            return this.verifications == 0 ? 0 : (double) sum / this.verifications;
        }

        long fewerThan(int depth) {
            long n = 0;
            for (int a = 0; a < Math.min(depth, this.accepted.length); a++) n += this.accepted[a];
            return n;
        }

        long some() {
            long n = 0;
            for (int a = 1; a < this.accepted.length; a++) n += this.accepted[a];
            return n;
        }

        long partial(int depth) {
            long n = 0;
            for (int a = 1; a < Math.min(depth, this.accepted.length); a++) n += this.accepted[a];
            return n;
        }

        @Override
        public String toString() {
            return java.util.Arrays.toString(java.util.Arrays.copyOf(this.accepted, 8)) + " of " + this.verifications;
        }
    }

    @Test
    @Timeout(value = 45, unit = TimeUnit.MINUTES, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void sampledTokensDoNotDependOnSpeculationOrOnTheSequencesSharingAPass() throws Exception {
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
            int depth = runtime.profile().speculativeDepth();
            assumeTrue(engine.maxSequences() >= SEQUENCES, "the engine admits fewer than " + SEQUENCES + " sequences");
            int[][] prompts = new int[SEQUENCES][];
            for (int i = 0; i < SEQUENCES; i++)
                prompts[i] = engine.tokenizer().encodeWithModelSpecialTokens(PROMPTS[i]);
            SessionOptions off = new SessionOptions(false);
            SessionOptions on = new SessionOptions(true);

            // (1) Speculation does not change sampled tokens; partial rejections occur.
            Acceptance sampled = new Acceptance();
            List<List<Integer>> sampledAlone = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                List<Integer> plain = generate(engine, prompts[i], published(SEEDS[i]), off, null);
                Acceptance acceptance = new Acceptance();
                List<Integer> speculative = generate(engine, prompts[i], published(SEEDS[i]), on, acceptance);
                sampled.add(acceptance);
                sampledAlone.add(speculative);
                assertEquals(plain.size(), speculative.size(), "prompt " + i + " length");
                for (int k = 0; k < plain.size(); k++)
                    assertEquals(
                            plain.get(k),
                            speculative.get(k),
                            "prompt " + i + " seed " + SEEDS[i] + ": first difference at generated token " + k
                                    + " (position " + (prompts[i].length + k) + "), acceptance " + acceptance);
            }
            Acceptance greedy = new Acceptance();
            List<Integer> greedyPlain = generate(engine, prompts[3], GenerationConfig.greedy(7L), off, null);
            List<Integer> greedyAlone = generate(engine, prompts[3], GenerationConfig.greedy(7L), on, greedy);
            assertEquals(greedyPlain, greedyAlone, "greedy tokens depend on speculation");
            System.out.printf(
                    "SAMPLED_PASSES depth %d: mean accepted drafts sampled %.3f (%s), greedy %.3f (%s)%n",
                    depth, sampled.mean(), sampled, greedy.mean(), greedy);
            assertTrue(sampled.some() > 0, "no sampled verification accepted a draft: " + sampled);
            assertTrue(sampled.fewerThan(depth) > 0, "no sampled verification rejected a draft: " + sampled);
            assertTrue(sampled.partial(depth) > 0, "no sampled verification was partially rejected: " + sampled);
            System.out.printf(
                    "SAMPLED_PASSES verifications accepting some but not all drafts: %d%n", sampled.partial(depth));

            // (3) The seed is used.
            boolean differs = false;
            for (int i = 0; i < 3 && !differs; i++)
                differs = !sampledAlone.get(i).equals(generate(engine, prompts[i], published(99L), on, null));
            assertTrue(differs, "seeds 11/12/13 and 99 gave identical tokens for every prompt");

            // (2) Sharing passes does not change any session's tokens.
            PassStatistics before = runtime.execution().statistics(Quantum.ExecutionKind.VERIFY);
            List<GenerationSession> sessions = new ArrayList<>();
            List<CompletableFuture<List<Integer>>> futures = new ArrayList<>();
            for (int i = 0; i < SEQUENCES; i++) {
                GenerationConfig settings = i < 3 ? published(SEEDS[i]) : GenerationConfig.greedy(7L);
                GenerationSession session = engine.createGenerationSession(settings, on);
                sessions.add(session);
                futures.add(session.generateAsync(prompts[i], NEW_TOKENS, text -> {}, null, null));
            }
            try {
                for (int i = 0; i < SEQUENCES; i++) {
                    List<Integer> together = futures.get(i).get(10, TimeUnit.MINUTES);
                    List<Integer> alone = i < 3 ? sampledAlone.get(i) : greedyAlone;
                    assertEquals(alone.size(), together.size(), "session " + i + " length together");
                    for (int k = 0; k < alone.size(); k++)
                        assertEquals(
                                alone.get(k),
                                together.get(k),
                                "session " + i + " (" + (i < 3 ? "sampled" : "greedy")
                                        + "): first difference at generated token " + k + " (position "
                                        + (prompts[i].length + k) + ")");
                }
            } finally {
                for (GenerationSession session : sessions) session.close();
            }
            PassStatistics after = runtime.execution().statistics(Quantum.ExecutionKind.VERIFY);
            System.out.printf(
                    "SAMPLED_PASSES together: verify quanta %d, batched %d, segments %d, rows %d%n",
                    after.quanta() - before.quanta(),
                    after.batchedQuanta() - before.batchedQuanta(),
                    after.segments() - before.segments(),
                    after.rows() - before.rows());
            assertTrue(after.batchedQuanta() - before.batchedQuanta() > 0, "no batched verification pass formed");
            assertTrue(
                    after.segments() - before.segments() > after.quanta() - before.quanta(),
                    "batched passes carried no more segments than quanta");
            assertNotEquals(0, sampledAlone.get(0).size());
        }
    }

    private static GenerationConfig published(long seed) {
        return new GenerationConfig(1.0f, 20, 0.95f, seed, false);
    }

    private static List<Integer> generate(
            InferenceEngine engine,
            int[] prompt,
            GenerationConfig settings,
            SessionOptions options,
            GenerationTimingListener timing)
            throws Exception {
        try (GenerationSession session = engine.createGenerationSession(settings, options)) {
            return session.generateAsync(prompt, NEW_TOKENS, text -> {}, null, timing)
                    .get(10, TimeUnit.MINUTES);
        }
    }
}
