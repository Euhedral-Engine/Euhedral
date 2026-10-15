package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.SessionOptions;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The `bf16` artifact loaded and closed several times in one JVM with a 24 GiB pinned host budget, each engine
/// staging host-backed and disk-backed weights for a speculative decode or a 2048-token prefill: every reload runs as
/// the first load did. A run that does not finish within its bound ends the JVM, whose device
/// work could not be released otherwise. Skipped unless -Peuhedral.qwen.bf16-artifact names the artifact and
/// -Peuhedral.qwen.tokenizer-dir its tokenizer.
// Own JVM: sets the global residency budget property and loads the engine several times.
@ModelGroup.OwnJvm
class HostWeightsReloadCudaIntegrationTest {

    private static final int PROMPT_TOKENS = 2048;
    private static final int DECODE_TOKENS = 64;
    private static final String[] MODES = System.getProperty("euhedral.reload.modes", "decode-speculative,prefill")
            .split(",");
    private static final String SENTENCE =
            "The heat pump moves thermal energy between a cold source and a warm sink by cycling a refrigerant"
                    + " through an evaporator, a compressor, a condenser and an expansion valve. ";

    @Test
    @Timeout(value = 3600, unit = TimeUnit.SECONDS)
    void everyReloadPrefillsLikeTheFirst() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        String artifactProperty = System.getProperty("euhedral.qwen.bf16-artifact");
        assumeTrue(artifactProperty != null && Files.isRegularFile(Path.of(artifactProperty)), "no BF16 artifact");
        Path artifact = Path.of(artifactProperty);
        Path tokenizer = Path.of(System.getProperty("euhedral.qwen.tokenizer-dir"));
        int cycles = Integer.getInteger("euhedral.reload.cycles", 4);
        long minutes = Long.getLong("euhedral.reload.prefill-minutes", 10L);
        System.out.printf(
                "[reload] pid %d, %d cycles%n", ProcessHandle.current().pid(), cycles);
        System.setProperty("euhedral.residency.host-budget-bytes", Long.toString(24L << 30));
        try {
            for (int cycle = 1; cycle <= cycles; cycle++) {
                var config = new InferenceConfig(
                        artifact,
                        tokenizer,
                        Path.of(library),
                        SystemInfo.getPCpuSet(),
                        4096,
                        Duration.ofSeconds(60),
                        0L,
                        2048);
                long loadStart = System.nanoTime();
                try (InferenceEngine engine = InferenceEngine.load(config)) {
                    var dense = (Qwen38Runtime) engine.modelRuntime();
                    assertTrue(dense.diskBackedWeightBytes() > 0, "part of the model is read from disk");
                    System.out.printf(
                            "[reload] cycle %d loaded in %.1f s, %d MiB from disk%n",
                            cycle, (System.nanoTime() - loadStart) / 1e9, dense.diskBackedWeightBytes() >> 20);
                    // Cycles follow decode without speculation, decode with it, then a long prefill.
                    String mode = MODES[(cycle - 1) % MODES.length];
                    boolean speculation = mode.equals("decode-speculative");
                    int[] prompt = mode.equals("prefill") ? prompt(engine) : chat(engine);
                    int newTokens = mode.equals("prefill") ? 1 : DECODE_TOKENS;
                    long start = System.nanoTime();
                    try (GenerationSession session = engine.createGenerationSession(
                            GenerationConfig.greedy(91L), new SessionOptions(speculation))) {
                        try {
                            session.generateAsync(prompt, newTokens, text -> {}, null, null)
                                    .get(minutes, TimeUnit.MINUTES);
                        } catch (TimeoutException hung) {
                            // A thread dump would wait for a safepoint behind the stuck worker; take one with jcmd. The
                            // engine cannot close while its device work is stuck: end the JVM rather than hang.
                            System.out.printf(
                                    "[reload] cycle %d (%s) did not finish in %d minutes%n", cycle, mode, minutes);
                            System.out.flush();
                            Runtime.getRuntime().halt(97);
                        }
                    }
                    System.out.printf(
                            "[reload] cycle %d: %s, %d-token prompt, %d tokens in %.1f s%n",
                            cycle, mode, prompt.length, newTokens, (System.nanoTime() - start) / 1e9);
                }
            }
        } finally {
            System.clearProperty("euhedral.residency.host-budget-bytes");
        }
    }

    private static int[] prompt(InferenceEngine engine) {
        int[] ids = engine.tokenizer().encodeText(SENTENCE.repeat(PROMPT_TOKENS / 16 + 8));
        if (ids.length < PROMPT_TOKENS) fail("the filler text encodes to fewer than " + PROMPT_TOKENS + " tokens");
        return Arrays.copyOf(ids, PROMPT_TOKENS);
    }

    private static int[] chat(InferenceEngine engine) {
        return engine.tokenizer()
                .encodeWithModelSpecialTokens("<|im_start|>user\nExplain how a heat pump works.<|im_end|>\n"
                        + "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
}
