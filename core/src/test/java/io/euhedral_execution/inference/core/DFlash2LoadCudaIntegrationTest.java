package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// An artifact that holds the DFlash2 drafter loads it instead of the MTP layer: the drafter's objects are typed
/// weights, its selector codebooks stay in mapped host memory, and the MTP layer and draft head are not loaded.
/// Skipped unless -Peuhedral.qwen.dflash2-artifact names such an artifact.
// Own JVM: needs the DFlash2 artifact, which no group shares.
@ModelGroup.OwnJvm
class DFlash2LoadCudaIntegrationTest {

    @Test
    @Timeout(value = 900, unit = TimeUnit.SECONDS)
    void aDFlash2ArtifactLoadsTheDrafterInsteadOfMtp() throws Exception {
        Path artifact = Path.of(System.getProperty("euhedral.qwen.dflash2-artifact", ""));
        assumeTrue(Files.isRegularFile(artifact), "no DFlash2 artifact: " + artifact);
        BitSet cpus = new BitSet();
        cpus.set(0, Math.min(8, Runtime.getRuntime().availableProcessors()));
        var config = new InferenceConfig(
                artifact,
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen")),
                Path.of(System.getProperty("euhedral.cuda.library")),
                cpus,
                4096,
                Duration.ofSeconds(10));
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            assertEquals("dflash2", engine.description().speculation());
            var weights = ((Qwen38Runtime) engine.modelRuntime()).plan().weights();
            assertNull(weights.mtp());
            assertTrue(weights.runtimeObjects().keySet().stream().noneMatch(name -> name.startsWith("mtp/")));
            var drafter = assertNotNull(weights.dflash2());
            assertEquals(5, drafter.layers().length);
            assertEquals(8, drafter.config().blockSize());
            assertEquals(248070, drafter.config().maskToken());
            assertTrue(drafter.predecessorCodebook().hostMapped(), "the codebooks are read in place");
            assertTrue(drafter.successorCodebook().hostMapped());
            System.out.println("DFLASH2_LOAD PASS " + drafter.config() + " host_backed_bytes="
                    + engine.hostBackedWeightBytes() + " allocated=" + engine.allocatedDeviceBytes());
        }
    }

    private static <T> T assertNotNull(T value) {
        org.junit.jupiter.api.Assertions.assertNotNull(value);
        return value;
    }
}
