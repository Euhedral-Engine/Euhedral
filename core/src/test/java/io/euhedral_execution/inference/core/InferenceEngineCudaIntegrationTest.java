package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.Session;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class InferenceEngineCudaIntegrationTest {
    @Test
    @Timeout(1800)
    void highLevelGenerationRestoresSessionAndEngineDeviceMemory() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)));
        Path artifact = Path.of(
                System.getProperty("euhedral.qwen.artifact", "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl"));
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(artifact) && Files.isRegularFile(tokenizer.resolve("tokenizer.json")));
        BitSet cpus = new BitSet();
        var available = SystemInfo.getPCpuSet();
        for (int cpu = available.nextSetBit(0); cpu >= 0 && cpus.cardinality() < 2; cpu = available.nextSetBit(cpu + 1))
            cpus.set(cpu);
        assumeTrue(cpus.cardinality() == 2);
        var config = new InferenceConfig(artifact, tokenizer, Path.of(library), cpus, Duration.ofSeconds(10));
        // Only records the engine's CUDA binding, whose allocations stay readable after the engine closed;
        // inference uses exclusively the public engine/session surface.
        var bootstrap = new RecordingBootstrap();
        try (InferenceEngine engine = InferenceEngine.load(config, bootstrap)) {
            long loaded = engine.allocatedDeviceBytes();
            assertTrue(loaded > (1L << 30), "real model was not resident");
            // The runtime's workspace is allocated at load, inside `loaded`.
            long atLoad = engine.retainedWorkspaceBytes();
            assertTrue(atLoad > 0, "the workspace is allocated at load");
            StringBuilder output = new StringBuilder();
            try (Session session = engine.createSession(GenerationConfig.greedy(91L))) {
                var tokens = session.generate("The capital of France is", 5, output::append);
                assertEquals(5, tokens.size(), "expected multiple real decode quanta for the fixed prompt");
                assertFalse(output.isEmpty());
                var visible = tokens.stream()
                        .filter(id -> !engine.tokenizer().isGenerationEosToken(id))
                        .mapToInt(Integer::intValue)
                        .toArray();
                assertEquals(engine.tokenizer().decode(visible), output.toString());
                assertEquals(
                        engine.tokenizer().encodeWithModelSpecialTokens("The capital of France is").length
                                + visible.length,
                        session.currentTokenPosition());
                assertTrue(engine.allocatedDeviceBytes() > loaded, "sequence did not retain device state");
            }
            // Execution graphs keep their own storage; everything the session owned is released.
            assertEquals(
                    loaded + engine.retainedWorkspaceBytes() - atLoad,
                    engine.allocatedDeviceBytes(),
                    "session device memory was not released");
            System.out.println("Generated text: " + output);
            System.out.println("Allocated device bytes with the model loaded: " + loaded);
        }
        assertEquals(0, bootstrap.gpu.allocatedBytes(), "engine close did not free every device allocation");
    }

    private static final class RecordingBootstrap extends InferenceEngine.Bootstrap {
        private CudaGpuMemory gpu;

        @Override
        ExecutionGpu openGpu(Path path) {
            ExecutionGpu opened = super.openGpu(path);
            this.gpu = (CudaGpuMemory) opened;
            return opened;
        }
    }
}
