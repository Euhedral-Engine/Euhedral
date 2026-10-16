package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.SessionOptions;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/// A trellis artifact in the engine end to end: a greedy continuation answers the capital of France; MTP speculative
/// decoding (row-exact batched verification over the trellis MTP layer) gives the tokens of plain greedy decoding;
/// two concurrent sessions give what each gives alone; and with payloads in host memory (more sequences than the
/// device holds weights beside) the tokens are those of the resident run. The artifacts are
/// `euhedral.qwen.trellis30-artifact` and `euhedral.qwen.trellis35-artifact`; an unset one is skipped.
// Own JVM: two artifacts load one after the other.
@ModelGroup.OwnJvm
class TrellisEngineCudaIntegrationTest {

    private static final int CONTEXT_TOKENS = 4096;
    private static final int NEW_TOKENS = 48;
    private static final String[] PROMPTS = {
        "The capital of France is",
        "Explain in a few sentences how a heat pump moves thermal energy from a cold source to a warm sink.",
    };

    static Stream<String> artifacts() {
        return Stream.of("euhedral.qwen.trellis30-artifact", "euhedral.qwen.trellis35-artifact");
    }

    @ParameterizedTest
    @MethodSource("artifacts")
    @Timeout(value = 60, unit = TimeUnit.MINUTES, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void generatesConsistentlyOnEveryRoute(String property) throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        String artifact = System.getProperty(property);
        assumeTrue(artifact != null && Files.isRegularFile(Path.of(artifact)), "no artifact for " + property);
        String tokenizer = System.getProperty("euhedral.qwen.tokenizer-dir");
        assumeTrue(tokenizer != null, "no tokenizer directory");

        List<List<Integer>> greedy = new ArrayList<>();
        try (InferenceEngine engine = InferenceEngine.load(config(artifact, tokenizer, library, 2))) {
            int[][] prompts = encode(engine);
            for (int[] prompt : prompts)
                greedy.add(generate(engine, prompt, new SessionOptions(false)).get(10, TimeUnit.MINUTES));
            String answer = engine.tokenizer()
                    .decode(greedy.getFirst().stream()
                            .mapToInt(Integer::intValue)
                            .toArray());
            System.out.printf("TRELLIS %s greedy: %s%n", property, answer.replace('\n', ' '));
            assertTrue(answer.contains("Paris"), "the continuation does not name Paris: " + answer);

            for (int i = 0; i < prompts.length; i++)
                assertEquals(
                        greedy.get(i),
                        generate(engine, prompts[i], new SessionOptions(true)).get(10, TimeUnit.MINUTES),
                        "speculative decoding changed prompt " + i);

            List<CompletableFuture<List<Integer>>> together = new ArrayList<>();
            for (int[] prompt : prompts) together.add(generate(engine, prompt, new SessionOptions(true)));
            for (int i = 0; i < prompts.length; i++)
                assertEquals(greedy.get(i), together.get(i).get(10, TimeUnit.MINUTES), "prompt " + i + " run together");
        }

        // Enough sequences that the plan moves trellis payloads to host memory.
        int crowded = Integer.getInteger("euhedral.trellis.test.host-sequences", 48);
        try (InferenceEngine engine = InferenceEngine.load(config(artifact, tokenizer, library, crowded))) {
            long hostBytes = engine.hostBackedWeightBytes();
            System.out.printf(
                    "TRELLIS %s with %d sequences: %d MiB of weights in host memory%n",
                    property, crowded, hostBytes >> 20);
            assertTrue(hostBytes > 0, "no weights moved to host memory with " + crowded + " sequences");
            int[][] prompts = encode(engine);
            for (int i = 0; i < prompts.length; i++)
                assertEquals(
                        greedy.get(i),
                        generate(engine, prompts[i], new SessionOptions(true)).get(10, TimeUnit.MINUTES),
                        "host-backed payloads changed prompt " + i);
        }
    }

    private static InferenceConfig config(String artifact, String tokenizer, String library, int sequences) {
        return new InferenceConfig(
                Path.of(artifact),
                Path.of(tokenizer),
                Path.of(library),
                SystemInfo.getPCpuSet(),
                CONTEXT_TOKENS,
                Duration.ofSeconds(30),
                0,
                2048,
                null,
                sequences);
    }

    private static int[][] encode(InferenceEngine engine) {
        int[][] prompts = new int[PROMPTS.length][];
        for (int i = 0; i < PROMPTS.length; i++)
            prompts[i] = engine.tokenizer().encodeWithModelSpecialTokens(PROMPTS[i]);
        return prompts;
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
