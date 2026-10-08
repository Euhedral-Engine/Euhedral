package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.runtime.graph.WorkspaceUse;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// One workspace per runtime, sized at load: every graph binds the same buffers, nothing allocates per quantum, and
/// a quantum larger than the workspace fails at admission.
@Timeout(20)
class SharedWorkspaceTest {

    private static Quantum prefill(ExecutionPlan plan, long sequence, int rows) {
        return new Quantum(
                plan, new Sequence(sequence), Quantum.ExecutionKind.PREFILL, 0, new int[rows], LogitsRequirement.NONE);
    }

    @Test
    void everyGraphBindsTheSameSharedBuffersAndItsOwnInputRecord() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var plan = new ExecutionPlan(weights);
        var stream = new ExecutionFixtures.HoldingStream();
        var gpu = new EngineExecutionFixture.SamplingGpu(weights.config().vocabSize()) {
            @Override
            public GpuStream openStream() {
                return stream;
            }
        };
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, gpu);
        List<long[]> seen = new ArrayList<>();
        try {
            // The first quantum's retirement is held, so the second needs a second graph.
            var first = runtime.submit(
                    prefill(plan, 1, 64),
                    quantum -> seen.add(new long[] {
                        quantum.workspace().address(ExecutionPlan.Buffer.HIDDEN_STATE),
                        quantum.workspace().tokenIdsAddress()
                    }));
            lattice.drive();
            var second = runtime.submit(
                    prefill(plan, 2, 64),
                    quantum -> seen.add(new long[] {
                        quantum.workspace().address(ExecutionPlan.Buffer.HIDDEN_STATE),
                        quantum.workspace().tokenIdsAddress()
                    }));
            lattice.drive();
            assertEquals(2, stream.held(), "both quanta run at once, on two graphs");
            while (stream.held() > 0) stream.release(null);
            lattice.drive();
            assertEquals(Quantum.Status.SUCCESS, first.get(2, TimeUnit.SECONDS).status());
            assertEquals(Quantum.Status.SUCCESS, second.get(2, TimeUnit.SECONDS).status());
            assertEquals(seen.get(0)[0], seen.get(1)[0], "one copy of each workspace buffer");
            assertNotEquals(seen.get(0)[1], seen.get(1)[1], "each graph keeps its own input record");
        } finally {
            while (stream.held() > 0) stream.release(null);
            lattice.drive();
            runtime.close();
        }
    }

    @Test
    void aQuantumLargerThanTheWorkspaceFailsAtAdmission() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var plan = new ExecutionPlan(weights);
        var runtime = new Execution(
                ExecutionFixtures.inlineLattice(),
                plan,
                new EngineExecutionFixture.SamplingGpu(weights.config().vocabSize()),
                2,
                false,
                64);
        try {
            var outcome = runtime.submit(prefill(plan, 3, 65)).get(5, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.FAILED, outcome.status());
            assertInstanceOf(IllegalArgumentException.class, outcome.failure());
            assertTrue(
                    outcome.failure().getMessage().contains("exceeds the workspace"),
                    outcome.failure().getMessage());
        } finally {
            runtime.close();
        }
    }

    @Test
    void theWorkspaceAllocatesOnceAtLoad() throws Exception {
        // Embedding-only weights: no sequence state, so every allocation is the workspace's or a graph's.
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new ExecutionFixtures.RecordingGpu();
        var runtime = new Execution(ExecutionFixtures.inlineLattice(), plan, gpu, 2, false, 128);
        try {
            assertEquals(
                    Quantum.Status.SUCCESS,
                    runtime.submit(prefill(plan, 4, 128))
                            .get(5, TimeUnit.SECONDS)
                            .status());
            int allocations = gpu.allocations.size();
            for (int i = 0; i < 20; i++) {
                int rows = 1 + (i * 37) % 128;
                assertEquals(
                        Quantum.Status.SUCCESS,
                        runtime.submit(prefill(plan, 10 + i, rows))
                                .get(5, TimeUnit.SECONDS)
                                .status());
            }
            assertEquals(allocations, gpu.allocations.size(), "no quantum allocated workspace memory");
        } finally {
            runtime.close();
        }
    }

    @Test
    void theDecodeViewDeclaresTheBuffersEachStageTouches() {
        var plan = new ExecutionPlan(EngineExecutionFixture.weights());
        Shape decode = plan.forExecution(Quantum.ExecutionKind.DECODE, 1);
        assertTrue(decode.workspaceBufferCount() > ExecutionPlan.Buffer.values().length);
        var use = WorkspaceUse.of(decode);
        int hidden = ExecutionPlan.Buffer.HIDDEN_STATE.ordinal();
        assertArrayEquals(new int[] {0}, use.entries(hidden), "the embedding writes the hidden state first");
        int logits = ExecutionPlan.Buffer.LOGITS.ordinal();
        assertEquals(0, use.entries(logits).length, "logits are graph-owned, not a workspace buffer");
    }
}
