package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Every scratch use of a prefill chunk and of decoding finds the workspace's scratch bound, sized at load: the
/// stages that take scratch are the ones their shapes declare, and the sizes match the routes. One artifact per JVM
/// (`-Peuhedral.scratch.artifact=q3` or `nvfp4`); skips without one.
class ScratchCudaIntegrationTest {

    private static final String TOKENIZER =
            System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen");

    @Test
    @Timeout(900)
    void prefillAndDecodeTakeOnlyTheBoundScratch() throws Exception {
        String name = System.getProperty("euhedral.scratch.artifact");
        assumeTrue(name != null, "select an artifact with -Peuhedral.scratch.artifact");
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
            var gpu = (CudaGpuMemory) ((Qwen38Runtime) engine.modelRuntime()).gpu();
            StringBuilder prompt = new StringBuilder();
            for (int i = 1; i <= 120; i++)
                prompt.append("Fact ").append(i).append(": ").append(i * i).append(".\n");
            int[] ids = engine.tokenizer().encodeWithModelSpecialTokens(prompt.toString());
            long before = gpu.scratchFallbacks();
            try (GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(7L))) {
                session.generateAsync(ids, 8, text -> {}, null, null).get(10, TimeUnit.MINUTES);
            }
            assertEquals(0, gpu.scratchFallbacks() - before, "a scratch use found no bound region");
        }
    }
}
