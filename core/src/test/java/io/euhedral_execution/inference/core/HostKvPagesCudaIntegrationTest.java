package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.SessionOptions;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.ResidencyPlanner;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.state.KvFormat;
import io.euhedral_execution.inference.core.state.KvHostPages;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.testing.Tunings;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Attention cache pages past the device share live in pinned host memory and are copied in for each layer: greedy
/// generation over a prompt that reaches far into the host pages (prefill chunks, decode, MTP verification and
/// drafting) produces exactly the tokens of the same engine with every page on the device. The host-page engine is
/// given a device budget that cannot hold the whole cache, so the planner moves the pages past 1024 tokens to the
/// host before it moves any weight.
// Own JVM: loads the artifact twice, with a device budget set per load.
@ModelGroup.OwnJvm
class HostKvPagesCudaIntegrationTest {

    private static final int CONTEXT_TOKENS = 8192;
    private static final int DEVICE_TOKENS = 1024;
    private static final int PROMPT_TOKENS = 3000;
    private static final int NEW_TOKENS = 48;

    @Test
    @Timeout(value = 30, unit = TimeUnit.MINUTES, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void generationWithHostPagesEqualsEveryPageOnTheDevice() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        String artifact = System.getProperty("euhedral.qwen.nvfp4-artifact");
        assumeTrue(artifact != null && Files.isRegularFile(Path.of(artifact)), "no NVFP4 artifact");

        var data = ArtifactReader.read(Path.of(artifact));
        var profile = ArtifactProfile.of(data);
        long whole = ResidencyPlanner.plan(
                        data, profile, Long.MAX_VALUE, CONTEXT_TOKENS, Long.MAX_VALUE, KvFormat.NVFP4, 1)
                .deviceBytes();

        Runs device = generate(artifact, library, Integer.MAX_VALUE, null);
        // One byte short of the whole cache: the pages past the device share move to host memory.
        Runs host = generate(artifact, library, DEVICE_TOKENS, whole - 1);
        // Host pages leave the device before weights: never more weights in host memory than with every page on it.
        assertTrue(
                host.hostWeightBytes() <= device.hostWeightBytes(),
                host.hostWeightBytes() + " weight bytes in host memory, " + device.hostWeightBytes()
                        + " without host pages");
        for (int run = 0; run < device.tokens().size(); run++)
            assertEquals(
                    device.tokens().get(run),
                    host.tokens().get(run),
                    "run " + run + " changed with host attention pages");
    }

    private record Runs(List<List<Integer>> tokens, long hostWeightBytes) {}

    /// A long document of the repository (the tests run from a module directory).
    private static Path document() {
        for (Path directory = Path.of("").toAbsolutePath(); directory != null; directory = directory.getParent()) {
            Path document = directory.resolve("docs/FRAME_MODEL.md");
            if (Files.isRegularFile(document)) return document;
        }
        throw new IllegalStateException("no docs/FRAME_MODEL.md above the working directory");
    }

    /// Greedy tokens of two sessions over the same long prompt, speculation off then on, and the weight bytes the
    /// engine keeps in host memory.
    private static Runs generate(String artifact, String library, int deviceTokens, Long budget) throws Exception {
        var config = new InferenceConfig(
                        Path.of(artifact),
                        Path.of(library),
                        SystemInfo.getPCpuSet(),
                        CONTEXT_TOKENS,
                        Duration.ofSeconds(30),
                        0,
                        2048,
                        null,
                        1,
                        deviceTokens)
                .withTuning(Tunings.limits(
                        budget == null ? OptionalLong.empty() : OptionalLong.of(budget), OptionalLong.empty()));
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            KvHostPages pages = ((Qwen38Runtime) engine.modelRuntime()).kvHostPages();
            if (budget == null) {
                assertNull(pages, "every page fits on the device");
            } else {
                assertNotNull(pages, "the budget moves cache pages to host memory");
                assertEquals(DEVICE_TOKENS / 256, pages.devicePages());
            }
            int[] text = engine.tokenizer().encodeText(Files.readString(document()));
            int[] prompt = Arrays.copyOf(text, PROMPT_TOKENS);
            List<List<Integer>> runs = new ArrayList<>();
            for (boolean speculation : new boolean[] {false, true}) {
                try (GenerationSession session =
                        engine.createGenerationSession(GenerationConfig.greedy(7L), new SessionOptions(speculation))) {
                    runs.add(session.generateAsync(prompt, NEW_TOKENS, out -> {}, null, null)
                            .get(10, TimeUnit.MINUTES));
                }
            }
            return new Runs(runs, engine.hostBackedWeightBytes());
        }
    }
}
