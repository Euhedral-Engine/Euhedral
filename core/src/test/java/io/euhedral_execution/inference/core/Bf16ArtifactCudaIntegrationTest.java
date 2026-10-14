package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.model.qwen38.Session;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The unquantized Qwen3.8 model (a `bf16` artifact, 54 GB) on one 16 GB card: its weights split between the device,
/// pinned host memory up to the host budget and the artifact file, its attention caches in BF16. It answers a
/// factual prompt greedily, with the same tokens whatever share of its weights is read from disk. Skipped unless
/// -Peuhedral.qwen.bf16-artifact names the artifact (local disk) and -Peuhedral.qwen.tokenizer-dir its tokenizer.
// Own JVM: sets the global residency budget properties.
@ModelGroup.OwnJvm
class Bf16ArtifactCudaIntegrationTest {

    @Test
    @Timeout(value = 3600, unit = TimeUnit.SECONDS)
    void theUnquantizedModelAnswersTheSameWhereverItsWeightsLive() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        String artifactProperty = System.getProperty("euhedral.qwen.bf16-artifact");
        assumeTrue(artifactProperty != null && Files.isRegularFile(Path.of(artifactProperty)), "no BF16 artifact");
        Path artifact = Path.of(artifactProperty);
        Path tokenizer = Path.of(System.getProperty("euhedral.qwen.tokenizer-dir"));
        List<Integer> roomy = generate(artifact, tokenizer, Path.of(library), 24L << 30, "host budget 24 GiB");
        List<Integer> tight = generate(artifact, tokenizer, Path.of(library), 8L << 30, "host budget 8 GiB");
        assertEquals(roomy, tight);
    }

    private static List<Integer> generate(Path artifact, Path tokenizer, Path library, long hostBudget, String label)
            throws Exception {
        var config = new InferenceConfig(
                artifact, tokenizer, library, SystemInfo.getPCpuSet(), 4096, Duration.ofSeconds(60), 0L, 2048);
        System.setProperty("euhedral.residency.host-budget-bytes", Long.toString(hostBudget));
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            var dense = (Qwen38Runtime) engine.modelRuntime();
            assertTrue(dense.diskBackedWeightBytes() > 0, "the BF16 model reads part of its weights from disk");
            StringBuilder output = new StringBuilder();
            long start = System.nanoTime();
            List<Integer> tokens;
            try (Session session = engine.createSession(GenerationConfig.greedy(91L))) {
                tokens = session.generate("The capital of France is", 24, output::append);
            }
            double seconds = (System.nanoTime() - start) / 1e9;
            System.out.printf(
                    "%s: %d MiB of weights read from disk; %d tokens in %.1f s (%.2f tok/s): %s%n",
                    label,
                    dense.diskBackedWeightBytes() >> 20,
                    tokens.size(),
                    seconds,
                    tokens.size() / seconds,
                    output);
            assertTrue(output.toString().contains("Paris"), output.toString());
            return tokens;
        } finally {
            System.clearProperty("euhedral.residency.host-budget-bytes");
        }
    }
}
