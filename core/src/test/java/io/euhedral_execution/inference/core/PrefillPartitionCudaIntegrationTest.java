package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.SystemInfo;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.EngineExecutionFixture;
import io.euhedral_execution.inference.core.scheduling.QwenGenerationSession;
import io.euhedral_execution.inference.core.scheduling.SequenceStateProbe;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class PrefillPartitionCudaIntegrationTest {
    private static final int PROMPT_TOKENS = 1536;

    /// Prefills the same 1536 tokens split into chunks of several sizes and reports how the resulting
    /// sequence state differs from the 512-token partition a cold run uses. 1536 is a multiple of the
    /// 256-token page, so every KV row is in a full page and is compared.
    @Test
    @Timeout(1800)
    void reportsWhetherPrefillStateDependsOnTheChunkPartition() throws Exception {
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
        var bootstrap = new RecordingBootstrap();
        try (InferenceEngine engine = InferenceEngine.load(config, bootstrap)) {
            String text = "The quick brown fox jumps over the lazy dog while the engine counts tokens. ".repeat(400);
            int[] encoded = engine.tokenizer().encodeWithModelSpecialTokens(text);
            assertTrue(encoded.length >= PROMPT_TOKENS, "the prompt text is too short");
            int[] prompt = Arrays.copyOf(encoded, PROMPT_TOKENS);
            List<String> baseline = stateAfterPrefill(engine, bootstrap.gpu, prompt, 512);
            System.out.println("prefill partition 512 (baseline): " + baseline.size() + " buffers compared");
            for (int chunk : new int[] {512, 1024, 256, 700, 1000, 1536}) {
                List<String> other = stateAfterPrefill(engine, bootstrap.gpu, prompt, chunk);
                System.out.println("prefill partition " + chunk + ": " + describe(baseline, other));
            }
        }
    }

    private static List<String> stateAfterPrefill(InferenceEngine engine, ExecutionGpu gpu, int[] prompt, int chunk)
            throws Exception {
        try (QwenGenerationSession session = engine.createSession(GenerationConfig.greedy(1L), chunk)) {
            session.generate(prompt, 0, text -> {}, null);
            return SequenceStateProbe.committedDigests(
                    gpu,
                    EngineExecutionFixture.sequence(session),
                    engine.modelConfig().layerTypes());
        }
    }

    private static String describe(List<String> expected, List<String> actual) {
        if (expected.size() != actual.size())
            return "different buffer counts " + expected.size() + " vs " + actual.size();
        List<String> differing = new java.util.ArrayList<>();
        for (int i = 0; i < expected.size(); i++)
            if (!expected.get(i).equals(actual.get(i)))
                differing.add(expected.get(i).replaceAll(" [0-9a-f]{16}$", ""));
        if (differing.isEmpty()) return "identical (" + expected.size() + " buffers)";
        return differing.size() + " of " + expected.size() + " buffers differ, first: "
                + differing.subList(0, Math.min(3, differing.size()));
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
