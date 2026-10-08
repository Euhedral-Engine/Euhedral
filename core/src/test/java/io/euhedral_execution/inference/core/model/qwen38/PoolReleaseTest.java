package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// A shape the runtime stops pooling (an evicted prompt shape) closes its graphs, and their storage, once idle.
@Timeout(20)
class PoolReleaseTest {

    @Test
    void aReleasedShapesIdleGraphsCloseWithTheirStorage() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new ExecutionFixtures.RecordingGpu();
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        try {
            var quantum = new Quantum(plan, new Sequence(1), Quantum.ExecutionKind.PREFILL, 0, new int[] {1, 2});
            assertEquals(
                    Quantum.Status.SUCCESS,
                    runtime.submit(quantum).get(5, TimeUnit.SECONDS).status());
            assertTrue(runtime.runtime().retainedWorkspaceBytes() > 0, "the graph keeps its own storage");
            int frees = gpu.frees.size();
            runtime.runtime().release(quantum.shape());
            assertEquals(0, runtime.runtime().retainedWorkspaceBytes(), "the idle graph closed");
            assertTrue(gpu.frees.size() > frees, "with its storage");
            var again = new Quantum(plan, new Sequence(2), Quantum.ExecutionKind.PREFILL, 0, new int[] {1, 2});
            assertEquals(
                    Quantum.Status.SUCCESS,
                    runtime.submit(again).get(5, TimeUnit.SECONDS).status());
        } finally {
            runtime.close();
        }
    }

    @Test
    void aGraphBusyWhenItsShapeIsReleasedClosesWhenItsQuantumRetires() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var stream = new ExecutionFixtures.HoldingStream();
        var gpu = new ExecutionFixtures.RecordingGpu() {
            @Override
            public io.euhedral_execution.inference.core.gpu.GpuStream openStream() {
                return stream;
            }
        };
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, gpu);
        try {
            var quantum = new Quantum(plan, new Sequence(3), Quantum.ExecutionKind.PREFILL, 0, new int[] {1, 2});
            var outcome = runtime.submit(quantum);
            lattice.drive();
            assertEquals(1, stream.held());
            int frees = gpu.frees.size();
            runtime.runtime().release(quantum.shape());
            assertEquals(frees, gpu.frees.size(), "a busy graph's storage stays until its quantum retires");
            stream.release(null);
            lattice.drive();
            assertEquals(
                    Quantum.Status.SUCCESS, outcome.get(5, TimeUnit.SECONDS).status());
            assertTrue(gpu.frees.size() > frees, "the graph's storage was freed when it went idle");
        } finally {
            while (stream.held() > 0) stream.release(null);
            lattice.drive();
            runtime.close();
        }
    }
}
