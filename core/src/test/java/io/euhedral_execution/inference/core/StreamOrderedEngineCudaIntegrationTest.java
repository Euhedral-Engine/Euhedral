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

        Result reference = generate(artifact, tokenizer, Path.of(library), cpus, new ReferenceBootstrap());
        Result streamOrdered = generate(artifact, tokenizer, Path.of(library), cpus, new RecordingBootstrap());
        assertEquals(reference.tokens(), streamOrdered.tokens(), "stream ordering changed committed token IDs");
        assertEquals(reference.position(), streamOrdered.position(), "KV/GDN sequence advancement differs");
        assertEquals(reference.output(), streamOrdered.output(), "incremental text differs");
        assertTrue(streamOrdered.tokens().size() >= 4, "expected several decode quanta");
    }

    private static Result generate(
            Path artifact, Path tokenizer, Path library, BitSet cpus, RecordingBootstrap bootstrap) throws Exception {
        var config = new InferenceConfig(artifact, tokenizer, library, cpus, Duration.ofSeconds(10));
        Result first;
        try (var engine = InferenceEngine.load(config, bootstrap)) {
            assertTrue(
                    engine.tokenizer().encodeWithModelSpecialTokens(PROMPT).length > 4,
                    "test prompt must exercise multiple prefill quanta");
            // A session owns its KV/GDN state, sampled logits, and decode scratch; the model owns the rest,
            // and the execution graphs keep their workspace storage for later quanta.
            long loaded = engine.allocatedDeviceBytes();
            assertEquals(0, engine.retainedWorkspaceBytes());
            first = generateSession(engine);
            long retained = engine.retainedWorkspaceBytes();
            assertTrue(retained > 0, "the graphs retained no workspace storage");
            assertEquals(loaded + retained, engine.allocatedDeviceBytes(), "the first session kept device allocations");
            Result second = generateSession(engine);
            assertEquals(retained, engine.retainedWorkspaceBytes(), "an equal session grew the retained storage");
            assertEquals(
                    loaded + retained, engine.allocatedDeviceBytes(), "the second session kept device allocations");
            Result third = generateSession(engine);
            assertEquals(loaded + retained, engine.allocatedDeviceBytes(), "the third session kept device allocations");
            assertEquals(first, second, "a new session did not reproduce the same sequence");
            assertEquals(second, third, "a later session did not reproduce the same sequence");
        }
        assertEquals(0, bootstrap.heldBytes(), "the engine did not release model and persistent sequence allocations");
        return first;
    }

    private static Result generateSession(InferenceEngine engine) throws Exception {
        try (var session = engine.createSession(GenerationConfig.greedy(91L), 4)) {
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

    /// Keeps the engine's GPU, whose allocation count stays readable after the engine closed.
    private static class RecordingBootstrap extends InferenceEngine.Bootstrap {
        private ExecutionGpu gpu;

        /// The stream-ordered CUDA binding unless a subclass supplies another GPU.
        ExecutionGpu create(Path path) {
            return super.openGpu(path);
        }

        @Override
        final ExecutionGpu openGpu(Path path) {
            this.gpu = create(path);
            return this.gpu;
        }

        long heldBytes() {
            return allocatedBytes(this.gpu);
        }
    }

    /// Loads the model on the synchronous reference GPU fixture.
    private static final class ReferenceBootstrap extends RecordingBootstrap {
        @Override
        ExecutionGpu create(Path path) {
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

        @Override
        long allocatedBytes(ExecutionGpu gpu) {
            return ((SynchronousReferenceGpu) gpu).allocatedBytes();
        }
    }
}
