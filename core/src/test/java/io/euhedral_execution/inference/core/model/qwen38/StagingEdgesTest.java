package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.WeightStaging;
import io.euhedral_execution.inference.core.runtime.graph.StageFrame;
import io.euhedral_execution.inference.core.runtime.graph.WorkspaceOwner;
import io.euhedral_execution.inference.core.runtime.graph.WorkspaceUse;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Staging slots are workspace buffers: a transfer into a slot follows the slot's last reader, within a graph and
/// across graphs, and nothing holds the ring. The preloaded decode view runs only behind its own sequence's
/// successful prefetch.
class StagingEdgesTest {

    private static final int SLOTS = 3;

    private static ExecutionPlan stagedPlan() {
        var weights = FullModelExecutionPlanTest.fullModelWeights(
                name -> name.endsWith("/gdn/output") || name.endsWith("/mlp/down"));
        long largest = 5120L * 17408;
        return new ExecutionPlan(weights, new WeightStaging(1L << 40, WeightStaging.slotBytesFor(largest), SLOTS));
    }

    @Test
    void eachSlotIsAWorkspaceBufferWhoseFirstAccessorIsATransferAndLastAReader() {
        ExecutionPlan plan = stagedPlan();
        Shape decode = plan.forExecution(Quantum.ExecutionKind.DECODE, 1);
        var use = WorkspaceUse.of(decode);
        for (int slot = 0; slot < SLOTS; slot++) {
            int buffer = SharedWorkspace.stagingBuffer(plan, slot);
            int[] entries = use.entries(buffer);
            int[] exits = use.exits(buffer);
            assertEquals(1, entries.length, "slot " + slot);
            assertEquals(
                    ExecutionPlan.Kind.WEIGHT_TRANSFER,
                    decode.instructions().get(entries[0]).kind(),
                    "a slot's first accessor in a quantum fills it");
            assertEquals(1, exits.length);
            assertTrue(
                    decode.instructions().get(exits[0]).kind() != ExecutionPlan.Kind.WEIGHT_TRANSFER,
                    "a slot's last accessor reads it");
        }
        assertTrue(decode.workspaceBufferCount() > SharedWorkspace.stagingBuffer(plan, SLOTS - 1));
    }

    @Test
    void anUnstagedViewNamesNoSlot() {
        ExecutionPlan plan = new ExecutionPlan(EngineExecutionFixture.weights());
        Shape decode = plan.forExecution(Quantum.ExecutionKind.DECODE, 1);
        assertEquals(SharedWorkspace.scratchBuffer(plan) + 1, decode.workspaceBufferCount());
    }

    @Test
    @Timeout(20)
    void thePreloadedVariantNeedsTheBlocksOwnSuccessfulPrefetch() throws Exception {
        ExecutionPlan plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new ExecutionFixtures.RecordingGpu();
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        try {
            var sequence = new Sequence(61);
            var other = new Sequence(62);
            var block = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});
            assertEquals(
                    Quantum.Status.SUCCESS,
                    runtime.submit(block).get(5, TimeUnit.SECONDS).status());
            gpu.linearFailure = null;
            var failed = new Quantum(
                    plan, sequence, Quantum.ExecutionKind.DECODE, 1, new int[] {ExecutionFixtures.VOCABULARY});
            assertEquals(
                    Quantum.Status.FAILED,
                    runtime.submit(failed).get(5, TimeUnit.SECONDS).status());
            Shape blockShape = plan.shape();
            var held = new WorkspaceOwner.Last(blockShape, block, new StageFrame[0]);
            List<WorkspaceOwner.Last> slots = new ArrayList<>(List.of(held, held));
            assertTrue(Execution.preloadHolds(blockShape, sequence, slots.size(), slots::get));
            assertFalse(Execution.preloadHolds(blockShape, other, slots.size(), slots::get), "another sequence's");
            assertFalse(
                    Execution.preloadHolds(
                            new ExecutionPlan(ExecutionFixtures.weights()).shape(), sequence, 2, slots::get),
                    "not written by a prefetching block");
            slots.set(1, new WorkspaceOwner.Last(blockShape, failed, new StageFrame[0]));
            assertFalse(Execution.preloadHolds(blockShape, sequence, slots.size(), slots::get), "a failed quantum's");
            slots.set(1, null);
            assertFalse(Execution.preloadHolds(blockShape, sequence, slots.size(), slots::get), "never written");
        } finally {
            runtime.close();
        }
    }
}
