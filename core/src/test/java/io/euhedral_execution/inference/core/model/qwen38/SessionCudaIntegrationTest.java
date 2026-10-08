package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.core.config.LatticeConfig;
import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.control_plane.ControlPlaneShard;
import io.euhedral_execution.core.flow_control.LatticeEdge;
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.loader.Weights;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.testing.SharedQwen38;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.ExecutionMode;

@org.junit.jupiter.api.parallel.Execution(ExecutionMode.SAME_THREAD)
@ModelGroup.CompactQ3
class SessionCudaIntegrationTest {

    private static final Path DEFAULT_TOKENIZER = Path.of("/mnt/shared/qwen38-quant/source/qwen");
    private static final AtomicLong LATTICE_ID = new AtomicLong();

    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void plainPromptGeneratesThroughRealLatticeLogitsAndPersistentSequenceState() throws Exception {
        Path tokenizerPath = Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", DEFAULT_TOKENIZER.toString()));
        assumeTrue(Files.isRegularFile(tokenizerPath.resolve("tokenizer.json")));
        BitSet cpus = twoWorkerCpus();
        assumeTrue(cpus.cardinality() >= 2, "requires two available physical CPUs for the real lattice");

        QwenTokenizer tokenizer = QwenTokenizer.load(tokenizerPath);
        var loaded = SharedQwen38.q3();
        CudaGpuMemory gpu = loaded.gpu();
        ControlPlaneLattice lattice = createLattice(cpus);
        try {
            // The model and the retained scratch of earlier classes are already on the shared device; measure
            // this session's allocations against them.
            long scratchBefore = gpu.retainedScratchBytes();
            long allocatedAfterWeights = gpu.allocatedBytes() - scratchBefore;
            Weights weights = loaded.model().weights();
            Throwable failure = null;
            try {
                ExecutionPlan plan = new ExecutionPlan(weights);
                Execution runtime = new Execution(lattice, plan, gpu);
                Session session = new Session(tokenizer, plan, runtime, gpu, 901, GenerationConfig.greedy(91L));
                try {
                    LatticeEdge registrationProbe = new LatticeEdge(new AtomicBoolean());
                    int registrationsBeforeStart = registrationProbe.getThreadCount();
                    lattice.start();
                    awaitWorkers(lattice, cpus.cardinality(), registrationProbe, registrationsBeforeStart);
                    String prompt = "The capital of France is";
                    int promptTokenCount = tokenizer.encodeWithModelSpecialTokens(prompt).length;
                    StringBuilder output = new StringBuilder();
                    List<Long> callbackPositions = new ArrayList<>();
                    List<Long> decodeCallbackAllocated = new ArrayList<>();
                    AtomicReference<Object> recurrentState = new AtomicReference<>();
                    AtomicReference<Object> kvState = new AtomicReference<>();

                    // The asynchronous form's text sink runs before the next quantum is admitted; the blocking
                    // form hands text to the caller while later quanta run.
                    List<Integer> generated = session.generateAsync(
                                    prompt,
                                    5,
                                    text -> {
                                        output.append(text);
                                        callbackPositions.add(session.currentTokenPosition());
                                        assertEquals(
                                                0,
                                                runtime.activeQuanta(),
                                                "a token is emitted only after its quantum retired");
                                        Object currentRecurrent =
                                                session.sequenceState().recurrentState();
                                        Object currentKv =
                                                session.sequenceState().kvCacheState();
                                        if (recurrentState.compareAndSet(null, currentRecurrent)) {
                                            assertTrue(currentRecurrent instanceof GdnStates);
                                        } else {
                                            assertSame(recurrentState.get(), currentRecurrent);
                                        }
                                        if (kvState.compareAndSet(null, currentKv)) {
                                            assertTrue(currentKv instanceof AttentionStates);
                                        } else {
                                            assertSame(kvState.get(), currentKv);
                                        }
                                        long generatedCount =
                                                session.generatedTokenIds().size();
                                        assertTrue(session.currentTokenPosition()
                                                >= promptTokenCount + generatedCount - 1L);
                                        assertTrue(session.currentTokenPosition() <= promptTokenCount + generatedCount);
                                        // The first text callback follows prefill, before lazy split-KV
                                        // scratch reservation. Compare steady-state decode callbacks only;
                                        // scratch address stability and ownership have separate exact tests.
                                        if (session.currentTokenPosition() > promptTokenCount) {
                                            decodeCallbackAllocated.add(gpu.allocatedBytes());
                                        }
                                    },
                                    null,
                                    null)
                            .join();

                    assertTrue(generated.size() >= 4, "model stopped before several decode quanta");
                    List<Integer> visibleTokens = generated.stream()
                            .filter(tokenId -> !tokenizer.isGenerationEosToken(tokenId))
                            .toList();
                    int decodeQuanta = visibleTokens.size();
                    assertTrue(decodeQuanta >= 3, "generation did not execute several decode quanta");
                    assertEquals(
                            tokenizer.decode(visibleTokens.stream()
                                    .mapToInt(Integer::intValue)
                                    .toArray()),
                            output.toString());
                    assertEquals(promptTokenCount + visibleTokens.size(), session.currentTokenPosition());
                    assertEquals(
                            session.currentTokenPosition(),
                            ((AttentionStates) kvState.get()).forLayer(3).length());
                    assertTrue(callbackPositions.size() >= 3, "incremental decoder did not emit token text");
                    assertTrue(decodeCallbackAllocated.size() >= 3);
                    assertEquals(
                            1,
                            decodeCallbackAllocated.stream().distinct().count(),
                            "device allocations changed between steady decode tokens: " + decodeCallbackAllocated);
                    assertEquals(0, runtime.activeQuanta());
                    assertTrue(awaitDrained(lattice), "the lattice kept work after the generation retired");

                    GdnStates recurrent = (GdnStates) recurrentState.get();
                    AttentionStates attention = (AttentionStates) kvState.get();
                    session.close();
                    assertEquals(
                            Sequence.TerminalState.COMPLETED,
                            session.sequenceState().terminalState());
                    assertThrows(IllegalStateException.class, () -> recurrent.forLayer(0));
                    assertThrows(IllegalStateException.class, () -> attention.forLayer(3));
                    // The runtime's graphs keep their workspace storage for later quanta.
                    assertTrue(runtime.retainedWorkspaceBytes() > 0);
                    assertEquals(
                            allocatedAfterWeights + runtime.retainedWorkspaceBytes() + gpu.retainedScratchBytes(),
                            gpu.allocatedBytes(),
                            "session close did not release its persistent KV/GDN state and sampled logits");
                } finally {
                    session.close();
                    runtime.close();
                }
            } catch (Throwable executionFailure) {
                failure = executionFailure;
            }
            long allocatedAfterRelease = gpu.allocatedBytes() - gpu.retainedScratchBytes();
            if (allocatedAfterRelease != allocatedAfterWeights) {
                IllegalStateException cleanupFailure =
                        new IllegalStateException("generation integration leaked device memory: allocated before="
                                + allocatedAfterWeights + ", after=" + allocatedAfterRelease);
                if (failure == null) failure = cleanupFailure;
                else failure.addSuppressed(cleanupFailure);
            }
            if (failure instanceof Exception exception) throw exception;
            if (failure instanceof Error error) throw error;
            if (failure != null) throw new IllegalStateException(failure);
        } finally {
            lattice.close();
        }
    }

    /// The retirement frame publishes its quantum's outcome from inside its own execution, so its worker
    /// may still be finishing that frame when `generate` returns.
    private static boolean awaitDrained(ControlPlaneLattice lattice) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (!lattice.isDrained()) {
            if (System.nanoTime() > deadline) return false;
            Thread.sleep(1);
        }
        return true;
    }

    private static ControlPlaneLattice createLattice(BitSet cpus) {
        var workers = new BaseCloneableObject(new DefaultExecutor());
        var shard = ControlPlaneShard.createBaseShard("QwenGenerationCudaTestShard", workers);
        return ControlPlaneLattice.getOrCreate(new LatticeConfig(
                "QwenGenerationCudaTestLattice-" + LATTICE_ID.incrementAndGet(), cpus, Duration.ofSeconds(10), shard));
    }

    private static BitSet twoWorkerCpus() {
        BitSet cpus = new BitSet();
        BitSet selectedCores = new BitSet();
        BitSet physicalCpus = SystemInfo.getPCpuSet();
        int selected = 0;
        for (int cpu = physicalCpus.nextSetBit(0); cpu >= 0 && selected < 2; cpu = physicalCpus.nextSetBit(cpu + 1)) {
            var info = SystemInfo.getCpuInfo(cpu);
            if (info == null || selectedCores.get(info.core())) continue;
            selectedCores.set(info.core());
            cpus.set(cpu);
            selected++;
        }
        return cpus;
    }

    private static void awaitWorkers(
            ControlPlaneLattice lattice, int expected, LatticeEdge registrationProbe, int registrationsBeforeStart)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while ((lattice.getActiveWorkers() < expected
                        || registrationProbe.getThreadCount() < registrationsBeforeStart + expected)
                && System.nanoTime() < deadline) Thread.onSpinWait();
        assertEquals(expected, lattice.getActiveWorkers(), "real ControlPlaneLattice workers did not start");
        assertEquals(
                registrationsBeforeStart + expected,
                registrationProbe.getThreadCount(),
                "lattice workers did not register before inference");
    }
}
