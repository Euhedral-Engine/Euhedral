package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.SystemInfo;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.SynchronousReferenceGpu;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Stream-ordered generation against the synchronous reference: every kernel of the reference engine
/// runs on no selected stream and synchronizes, which is the retired synchronous execution mode.
class StreamOrderedEngineCudaIntegrationTest {
    private static final String PROMPT = "The capital of France is";

    @Test
    @Timeout(value = 1800, unit = TimeUnit.SECONDS)
    void streamOrderedGenerationMatchesTheSynchronousReferenceAndReleasesState() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "CUDA library is required");
        Path artifact = Path.of(System.getProperty(
                "euhedral.qwen.artifact", "/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl"));
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(artifact) && Files.isRegularFile(tokenizer.resolve("tokenizer.json")));
        BitSet cpus = new BitSet();
        var available = SystemInfo.getPCpuSet();
        for (int cpu = available.nextSetBit(0); cpu >= 0 && cpus.cardinality() < 2; cpu = available.nextSetBit(cpu + 1))
            cpus.set(cpu);
        assumeTrue(cpus.cardinality() == 2);

        try (var observer = new CudaGpuMemory(Path.of(library))) {
            long before = observer.deviceMemoryInfo().freeBytes();
            Result reference = generate(artifact, tokenizer, Path.of(library), cpus, new ReferenceBootstrap());
            Result streamOrdered =
                    generate(artifact, tokenizer, Path.of(library), cpus, new InferenceEngine.Bootstrap());
            assertEquals(reference.tokens(), streamOrdered.tokens(), "stream ordering changed committed token IDs");
            assertEquals(reference.position(), streamOrdered.position(), "KV/GDN sequence advancement differs");
            assertEquals(reference.output(), streamOrdered.output(), "incremental text differs");
            assertTrue(streamOrdered.tokens().size() >= 4, "expected several decode quanta");
            assertTrue(
                    observer.deviceMemoryInfo().freeBytes() >= before - (128L << 20),
                    "the engine did not release model and persistent sequence allocations");
        }
    }

    private static Result generate(
            Path artifact, Path tokenizer, Path library, BitSet cpus, InferenceEngine.Bootstrap bootstrap)
            throws Exception {
        var config =
                new InferenceConfig(artifact, tokenizer, library, new InferenceTuning(cpus, 4), Duration.ofSeconds(10));
        try (var engine = InferenceEngine.load(config, bootstrap)) {
            assertTrue(
                    engine.tokenizer().encodeWithModelSpecialTokens(PROMPT).length > 4,
                    "test prompt must exercise multiple prefill quanta");
            Result first = generateSession(engine);
            long afterFirst = engine.deviceMemoryInfo().freeBytes();
            Result second = generateSession(engine);
            long afterSecond = engine.deviceMemoryInfo().freeBytes();
            Result third = generateSession(engine);
            long afterThird = engine.deviceMemoryInfo().freeBytes();
            assertEquals(first, second, "a new session did not reproduce the same sequence");
            assertEquals(second, third, "a later session did not reproduce the same sequence");
            // A first-use CUDA module may become resident between the first two sessions;
            // compare warmed sessions to detect persistent per-session allocation growth.
            assertTrue(
                    afterThird >= afterSecond - (16L << 20),
                    "session state accumulated after warmup: after first=" + afterFirst + ", after second="
                            + afterSecond + ", after third=" + afterThird);
            return first;
        }
    }

    private static Result generateSession(InferenceEngine engine) throws Exception {
        try (var session = engine.createSession(GenerationConfig.greedy(91L))) {
            StringBuilder output = new StringBuilder();
            List<Integer> tokens = session.generate(PROMPT, 300, output::append);
            long position = session.currentTokenPosition();
            long visible = tokens.stream()
                    .filter(id -> !engine.tokenizer().isGenerationEosToken(id))
                    .count();
            assertEquals(engine.tokenizer().encodeWithModelSpecialTokens(PROMPT).length + visible, position);
            return new Result(tokens, position, output.toString());
        }
    }

    private record Result(List<Integer> tokens, long position, String output) {}

    /// Loads the model on the synchronous reference GPU fixture.
    private static final class ReferenceBootstrap extends InferenceEngine.Bootstrap {
        @Override
        ExecutionGpu openGpu(Path path, InferenceTuning tuning) {
            return new SynchronousReferenceGpu(path);
        }

        @Override
        void closeGpu(ExecutionGpu gpu) {
            ((SynchronousReferenceGpu) gpu).close();
        }

        @Override
        CudaGpuMemory.DeviceMemoryInfo memoryInfo(ExecutionGpu gpu) {
            return ((SynchronousReferenceGpu) gpu).deviceMemoryInfo();
        }
    }
}
