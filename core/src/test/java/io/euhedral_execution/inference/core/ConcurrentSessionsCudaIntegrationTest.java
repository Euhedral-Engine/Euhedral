package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Two Flash-Next sessions generating at once, from different prompts, each produce exactly the tokens they produce
/// alone: their quanta share the plan's one workspace, and nothing one of them stages (its tokens, its n-gram rows,
/// its expert leases) is the other's. Runs on `-Peuhedral.qwen4.artifact`.
class ConcurrentSessionsCudaIntegrationTest {

    private static final int NEW_TOKENS = 24;

    @Test
    @Timeout(1800)
    void twoFlashNextSessionsAtOnceGenerateWhatEachGeneratesAlone() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        Path artifact = Path.of(System.getProperty(
                "euhedral.qwen4.artifact", "/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl"));
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen4.tokenizer-dir", "/mnt/shared/qwen38-flash-next/nvfp4"));
        assumeTrue(Files.isRegularFile(artifact), "no artifact " + artifact);
        assumeTrue(Files.isRegularFile(tokenizer.resolve("tokenizer.json")), "no tokenizer in " + tokenizer);
        var config = new InferenceConfig(
                artifact, tokenizer, Path.of(library), SystemInfo.getPCpuSet(), Duration.ofSeconds(30));
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            int[] first = engine.tokenizer()
                    .encodeWithModelSpecialTokens("Write a short poem about the sea and the lighthouse keeper.");
            int[] second = engine.tokenizer()
                    .encodeWithModelSpecialTokens("List the planets of the solar system, from the sun outward, "
                            + "with one fact about each of them.");
            List<Integer> firstAlone = alone(engine, first);
            List<Integer> secondAlone = alone(engine, second);
            for (int round = 0; round < 3; round++) {
                try (GenerationSession a = engine.createGenerationSession(GenerationConfig.greedy(7L));
                        GenerationSession b = engine.createGenerationSession(GenerationConfig.greedy(7L))) {
                    CompletableFuture<List<Integer>> fromA = a.generateAsync(first, NEW_TOKENS, text -> {}, null, null);
                    CompletableFuture<List<Integer>> fromB =
                            b.generateAsync(second, NEW_TOKENS, text -> {}, null, null);
                    assertEquals(firstAlone, fromA.get(10, TimeUnit.MINUTES), "the first session, round " + round);
                    assertEquals(secondAlone, fromB.get(10, TimeUnit.MINUTES), "the second session, round " + round);
                }
            }
        }
    }

    private static List<Integer> alone(InferenceEngine engine, int[] prompt) throws Exception {
        try (GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(7L))) {
            return session.generateAsync(prompt, NEW_TOKENS, text -> {}, null, null)
                    .get(10, TimeUnit.MINUTES);
        }
    }
}
