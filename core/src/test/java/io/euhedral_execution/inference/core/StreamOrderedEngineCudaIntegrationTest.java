package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.SynchronousReferenceGpu;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
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
///
/// The long generation (300 tokens, which crosses the 256-token KV page) runs once on each engine and the outputs are
/// compared; the repeated sessions that check the engine takes no device memory from a session, and reproduces the
/// sequence, are short and compare with the start of the long one.
@ModelGroup.CompactQ3Engine
class StreamOrderedEngineCudaIntegrationTest {
    private static final String PROMPT = "The capital of France is";
    private static final int LONG = 300;
    private static final int SHORT = 24;

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
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

        // The reference engine needs the device to itself: release the shared engine first.
        CoreEngines.releaseDevice();
        Result reference = generateOnReference(artifact, tokenizer, Path.of(library), cpus);
        Result streamOrdered = generateStreamOrdered();
        assertEquals(reference.tokens(), streamOrdered.tokens(), "stream ordering changed committed token IDs");
        assertEquals(reference.position(), streamOrdered.position(), "KV/GDN sequence advancement differs");
        assertEquals(reference.output(), streamOrdered.output(), "incremental text differs");
        assertTrue(streamOrdered.tokens().size() >= 4, "expected several decode quanta");
    }

    private static Result generateOnReference(Path artifact, Path tokenizer, Path library, BitSet cpus)
            throws Exception {
        var config = new InferenceConfig(artifact, tokenizer, library, cpus, Duration.ofSeconds(10));
        var bootstrap = new ReferenceBootstrap();
        Result result;
        try (var engine = InferenceEngine.load(config, bootstrap)) {
            result = generateSession(engine, LONG);
        }
        assertEquals(0, bootstrap.heldBytes(), "the engine did not release model and persistent sequence allocations");
        return result;
    }

    private static Result generateStreamOrdered() throws Exception {
        var engine = CoreEngines.q3();
        assertTrue(
                engine.tokenizer().encodeWithModelSpecialTokens(PROMPT).length > 4,
                "test prompt must exercise multiple prefill quanta");
        // A session owns its KV/GDN state, sampled logits, and decode scratch; the model owns the rest, the
        // runtime's workspace is allocated at load (inside `loaded`), and the execution graphs keep their own
        // storage for later quanta. The engine may have served sessions before: measure from here.
        long loaded = engine.allocatedDeviceBytes();
        long atLoad = engine.retainedWorkspaceBytes();
        assertTrue(atLoad > 0, "the workspace is allocated at load");
        Result first = generateSession(engine, LONG);
        long retained = engine.retainedWorkspaceBytes();
        assertTrue(retained >= atLoad, "the retained storage shrank");
        long kept = loaded + retained - atLoad;
        assertEquals(kept, engine.allocatedDeviceBytes(), "the first session kept device allocations");
        Result second = generateSession(engine, SHORT);
        assertEquals(retained, engine.retainedWorkspaceBytes(), "an equal session grew the retained storage");
        assertEquals(kept, engine.allocatedDeviceBytes(), "the second session kept device allocations");
        Result third = generateSession(engine, SHORT);
        assertEquals(kept, engine.allocatedDeviceBytes(), "the third session kept device allocations");
        List<Integer> start =
                first.tokens().subList(0, Math.min(SHORT, first.tokens().size()));
        assertEquals(start, second.tokens().subList(0, start.size()), "a new session did not reproduce the sequence");
        assertEquals(second, third, "a later session did not reproduce the same sequence");
        return first;
    }

    private static Result generateSession(InferenceEngine engine, int newTokens) throws Exception {
        try (var session = engine.createSession(GenerationConfig.greedy(91L), 4)) {
            StringBuilder output = new StringBuilder();
            List<Integer> tokens = session.generate(PROMPT, newTokens, output::append);
            long position = session.currentTokenPosition();
            long visible = tokens.stream()
                    .filter(id -> !engine.tokenizer().isGenerationEosToken(id))
                    .count();
            assertEquals(engine.tokenizer().encodeWithModelSpecialTokens(PROMPT).length + visible, position);
            return new Result(tokens, position, output.toString());
        }
    }

    private record Result(List<Integer> tokens, long position, String output) {}

    /// Loads the model on the synchronous reference GPU fixture, whose allocation count stays readable after the
    /// engine closed.
    private static final class ReferenceBootstrap extends InferenceEngine.Bootstrap {
        private SynchronousReferenceGpu gpu;

        @Override
        ExecutionGpu openGpu(Path path) {
            this.gpu = new SynchronousReferenceGpu(path);
            return this.gpu;
        }

        long heldBytes() {
            return this.gpu.allocatedBytes();
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
