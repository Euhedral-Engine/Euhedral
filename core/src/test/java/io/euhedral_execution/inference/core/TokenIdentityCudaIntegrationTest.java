package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/// The refactor's identity gate: greedy generation through the public engine surface produces exactly the tokens
/// recorded on the base commit, for every artifact. Prompts cross many 512-token prefill chunks. Run with
/// `-Peuhedral.identity.record=true` to record instead of compare.
class TokenIdentityCudaIntegrationTest {

    private static final int NEW_TOKENS = 32;
    private static final Path GOLDENS = Path.of("src/test/resources/identity");
    private static final String QWEN38_TOKENIZER =
            System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen");

    /// Flash-Next first: it pins host memory from the process's cgroup headroom, which the page cache of the dense
    /// artifacts read later would take.
    static Stream<Arguments> artifacts() {
        return Stream.of(
                Arguments.of(
                        "flash-next",
                        System.getProperty(
                                "euhedral.qwen4.artifact",
                                "/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl"),
                        System.getProperty("euhedral.qwen4.tokenizer-dir", "/mnt/shared/qwen38-flash-next/nvfp4")),
                Arguments.of(
                        "q3",
                        System.getProperty(
                                "euhedral.qwen.artifact", "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl"),
                        QWEN38_TOKENIZER),
                Arguments.of(
                        "nvfp4",
                        System.getProperty(
                                "euhedral.qwen.nvfp4-artifact",
                                "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_nvfp4.edrl"),
                        QWEN38_TOKENIZER),
                Arguments.of(
                        "nvfp4-compressed",
                        System.getProperty(
                                "euhedral.qwen.nvfp4-compressed-artifact",
                                "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_nvfp4_compressed.edrl"),
                        QWEN38_TOKENIZER));
    }

    /// A short prompt, and two numbered lists of about 3,000 and 8,500 tokens (6 and 17 prefill chunks).
    static List<String> prompts() {
        StringBuilder medium = new StringBuilder("Here is a numbered list of facts.\n");
        for (int i = 1; i <= 220; i++)
            medium.append("Fact ")
                    .append(i)
                    .append(": the square of ")
                    .append(i)
                    .append(" is ")
                    .append(i * i)
                    .append(".\n");
        StringBuilder longer = new StringBuilder(medium);
        for (int i = 221; i <= 600; i++)
            longer.append("Fact ")
                    .append(i)
                    .append(": the square of ")
                    .append(i)
                    .append(" is ")
                    .append(i * i)
                    .append(".\n");
        return List.of(
                "The capital of France is",
                medium + "Continue the list with the next fact.",
                longer + "Continue the list with the next fact.");
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("artifacts")
    @Timeout(3600)
    void greedyTokensMatchTheRecordedBaseline(String name, String artifact, String tokenizer) throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        assumeTrue(Files.isRegularFile(Path.of(artifact)), "no artifact " + artifact);
        assumeTrue(Files.isRegularFile(Path.of(tokenizer, "tokenizer.json")), "no tokenizer in " + tokenizer);
        var config = new InferenceConfig(
                Path.of(artifact),
                Path.of(tokenizer),
                Path.of(library),
                SystemInfo.getPCpuSet(),
                Duration.ofSeconds(30));
        List<String> lines = new ArrayList<>();
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            for (String prompt : prompts()) {
                try (GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(7L))) {
                    int[] ids = engine.tokenizer().encodeWithModelSpecialTokens(prompt);
                    List<Integer> tokens = session.generateAsync(ids, NEW_TOKENS, text -> {}, null, null)
                            .get(30, TimeUnit.MINUTES);
                    lines.add(tokens.stream().map(String::valueOf).collect(Collectors.joining(",")));
                }
            }
        }
        Path golden = GOLDENS.resolve(name + ".txt");
        if (Boolean.parseBoolean(System.getProperty("euhedral.identity.record"))) {
            Files.createDirectories(GOLDENS);
            Files.write(golden, lines);
            return;
        }
        assumeTrue(Files.isRegularFile(golden), "no recorded baseline for " + name);
        assertEquals(Files.readAllLines(golden), lines, name + " generated different tokens than the baseline");
    }
}
