package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The engine plans the three tiers from its budgets: with a device budget below the Q3 model and a host budget below
/// what leaves the device, weights are read from the artifact, and the engine loads and closes.
// Own JVM: sets the global residency budget properties.
@ModelGroup.OwnJvm
class DiskTierEngineCudaIntegrationTest {

    @Test
    @Timeout(value = 900, unit = TimeUnit.SECONDS)
    void theEngineReadsWeightsBeyondItsHostBudgetFromDisk() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        Path artifact = Path.of(System.getProperty("euhedral.qwen.artifact", "unset-euhedral.qwen.artifact"));
        assumeTrue(Files.isRegularFile(artifact), "no artifact " + artifact);
        var config = new InferenceConfig(
                artifact,
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "unset-euhedral.qwen.tokenizer-dir")),
                Path.of(library),
                SystemInfo.getPCpuSet(),
                Duration.ofSeconds(30));
        System.setProperty("euhedral.residency.device-budget-bytes", Long.toString(8L << 30));
        System.setProperty("euhedral.residency.host-budget-bytes", Long.toString(2L << 30));
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            var dense = (Qwen38Runtime) engine.modelRuntime();
            assertTrue(dense.diskBackedWeightBytes() > 0, "weights beyond the host budget are read from disk");
        } finally {
            System.clearProperty("euhedral.residency.device-budget-bytes");
            System.clearProperty("euhedral.residency.host-budget-bytes");
        }
    }
}
