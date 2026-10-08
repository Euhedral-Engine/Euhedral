package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Two sessions on one runtime share its one workspace: one prefilling a long prompt while the other decodes. Each
/// produces exactly the tokens it produces alone. Runs on one dense artifact (`-Peuhedral.twosessions.artifact=q3`,
/// or `nvfp4` for host-staged weights); skips without one.
// Own JVM: the artifact is chosen per JVM by a system property.
@ModelGroup.OwnJvm
class TwoSessionsCudaIntegrationTest {

    private static final String TOKENIZER =
            System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen");

    private static String longPrompt() {
        StringBuilder prompt = new StringBuilder("Here is a numbered list of facts.\n");
        for (int i = 1; i <= 160; i++)
            prompt.append("Fact ")
                    .append(i)
                    .append(": the square of ")
                    .append(i)
                    .append(" is ")
                    .append(i * i)
                    .append(".\n");
        return prompt.append("Continue the list with the next fact.").toString();
    }

    @Test
    @Timeout(900)
    void concurrentSessionsMatchRunningAlone() throws Exception {
        String name = System.getProperty("euhedral.twosessions.artifact");
        assumeTrue(name != null, "select an artifact with -Peuhedral.twosessions.artifact");
        String artifact = "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_" + name.replace('-', '_') + ".edrl";
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        assumeTrue(Files.isRegularFile(Path.of(artifact)), "no artifact " + artifact);
        var config = new InferenceConfig(
                Path.of(artifact),
                Path.of(TOKENIZER),
                Path.of(library),
                SystemInfo.getPCpuSet(),
                Duration.ofSeconds(30));
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            int[] prefilling = engine.tokenizer().encodeWithModelSpecialTokens(longPrompt());
            int[] decoding = engine.tokenizer().encodeWithModelSpecialTokens("The capital of France is");
            List<Integer> prefillAlone = generate(engine, prefilling, 16).get(10, TimeUnit.MINUTES);
            List<Integer> decodeAlone = generate(engine, decoding, 64).get(10, TimeUnit.MINUTES);

            CompletableFuture<List<Integer>> prefill = generate(engine, prefilling, 16);
            CompletableFuture<List<Integer>> decode = generate(engine, decoding, 64);
            assertEquals(prefillAlone, prefill.get(10, TimeUnit.MINUTES), "the prefilling session changed");
            assertEquals(decodeAlone, decode.get(10, TimeUnit.MINUTES), "the decoding session changed");
        }
    }

    /// The concurrency screen (`-Peuhedral.twosessions.screen=true`): two prompts of 4096 tokens, 32 new tokens each,
    /// one after the other and both at once on one runtime; prints the median wall times of three rounds after a
    /// warm-up. Each prompt's tokens are the same either way.
    @Test
    @Timeout(1800)
    void twoLongPromptsTogetherAndOneAfterTheOther() throws Exception {
        String name = System.getProperty("euhedral.twosessions.artifact");
        assumeTrue(name != null && Boolean.getBoolean("euhedral.twosessions.screen"), "opt-in screen");
        String artifact = "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_" + name.replace('-', '_') + ".edrl";
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        var config = new InferenceConfig(
                Path.of(artifact),
                Path.of(TOKENIZER),
                Path.of(library),
                SystemInfo.getPCpuSet(),
                Duration.ofSeconds(30));
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            int[] first = prompt(engine, "square", 4096);
            int[] second = prompt(engine, "cube", 4096);
            generate(engine, first, 32).get(10, TimeUnit.MINUTES);
            long[] apart = new long[3], together = new long[3];
            List<Integer> firstApart = null, secondApart = null;
            for (int round = 0; round < 3; round++) {
                long started = System.nanoTime();
                firstApart = generate(engine, first, 32).get(10, TimeUnit.MINUTES);
                secondApart = generate(engine, second, 32).get(10, TimeUnit.MINUTES);
                apart[round] = System.nanoTime() - started;
                started = System.nanoTime();
                CompletableFuture<List<Integer>> a = generate(engine, first, 32);
                CompletableFuture<List<Integer>> b = generate(engine, second, 32);
                assertEquals(firstApart, a.get(10, TimeUnit.MINUTES), "the first prompt changed when run together");
                assertEquals(secondApart, b.get(10, TimeUnit.MINUTES), "the second prompt changed when run together");
                together[round] = System.nanoTime() - started;
            }
            java.util.Arrays.sort(apart);
            java.util.Arrays.sort(together);
            System.out.printf(
                    "TWO_SESSIONS_SCREEN %s one-after-the-other %.2f s, together %.2f s (medians of 3; %s)%n",
                    name,
                    apart[1] / 1e9,
                    together[1] / 1e9,
                    java.util.Arrays.toString(java.util.Arrays.stream(together)
                            .mapToObj(t -> String.format("%.2f", t / 1e9))
                            .toArray()));
        }
    }

    /// A prompt of exactly `tokens` tokens: a numbered list of facts about `kind`s of numbers.
    private static int[] prompt(InferenceEngine engine, String kind, int tokens) {
        StringBuilder text = new StringBuilder("Here is a numbered list of facts.\n");
        for (int i = 1; i <= 2000; i++)
            text.append("Fact ")
                    .append(i)
                    .append(": the ")
                    .append(kind)
                    .append(" of ")
                    .append(i)
                    .append(" is ")
                    .append(kind.equals("cube") ? (long) i * i * i : (long) i * i)
                    .append(".\n");
        int[] encoded = engine.tokenizer().encodeWithModelSpecialTokens(text.toString());
        if (encoded.length < tokens) throw new IllegalStateException("the prompt text is too short");
        return java.util.Arrays.copyOf(encoded, tokens);
    }

    private static CompletableFuture<List<Integer>> generate(InferenceEngine engine, int[] prompt, int tokens) {
        GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(7L));
        return session.generateAsync(prompt, tokens, text -> {}, null, null).whenComplete((done, failure) -> {
            try {
                session.close();
            } catch (RuntimeException ignored) {
                // The result carries any failure that matters.
            }
        });
    }
}
