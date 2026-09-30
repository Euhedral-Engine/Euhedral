package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.Test;

/// Workspace storage belongs to a reusable graph: consecutive quanta on the graph bind the same device
/// allocations, and storage passes to another quantum only after the one using it retired.
class QwenWorkspaceReuseTest {

    private static QwenExecutionPlan slicePlan() {
        return new QwenExecutionPlan(
                QwenExecutionFixtures.weights(),
                QwenExecutionFixtures.norm(),
                List.of(QwenExecutionFixtures.q3("projection", 64, 201)));
    }

    private static List<Long> bound(QwenExecutionContext context) {
        var workspace = context.workspace();
        return List.of(
                workspace.hiddenStateAddress(), workspace.normalizedStateAddress(), workspace.projectionAddress(0));
    }

    @Test
    void consecutiveQuantaBindTheSameStorageWithoutAllocatingOrFreeing() throws Exception {
        var plan = slicePlan();
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);
        List<List<Long>> addresses = new ArrayList<>();
        long retained = -1;
        for (int quantum = 0; quantum < 4; quantum++) {
            var context = new QwenExecutionContext(
                    plan,
                    new QwenSequenceState(700 + quantum),
                    QwenExecutionContext.ExecutionKind.DECODE,
                    0,
                    new int[] {1});
            var outcome = runtime.submit(context, done -> addresses.add(bound(done)));
            assertEquals(QwenExecutionContext.Status.SUCCESS, outcome.join().status());
            if (quantum == 0) retained = runtime.retainedWorkspaceBytes();
            assertEquals(retained, runtime.retainedWorkspaceBytes(), "retained storage must not grow per quantum");
        }
        assertEquals(1, new HashSet<>(addresses).size(), "every quantum bound the first quantum's buffers");
        assertEquals(4, gpu.allocations.size(), "hidden, normalized, projection and token IDs, once");
        assertTrue(gpu.frees.isEmpty(), "retirement released bindings, not device storage");
        assertEquals((3L * 64 * Short.BYTES) + Integer.BYTES, retained);
        runtime.close();
        QwenExecutionFixtures.assertEachAllocationFreedOnce(gpu);
        assertEquals(0, runtime.retainedWorkspaceBytes());
    }

    @Test
    void storageOfAnUnretiredQuantumIsNeverLentAndConcurrentQuantaNeverShareIt() {
        var plan = slicePlan();
        var stream = new QwenExecutionFixtures.HoldingStream();
        var gpu = new QwenExecutionFixtures.RecordingGpu() {
            @Override
            public GpuStream openStream() {
                return stream;
            }
        };
        var lattice = new QwenExecutionFixtures.ManualLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        List<List<Long>> first = new ArrayList<>();
        List<List<Long>> second = new ArrayList<>();
        List<List<Long>> third = new ArrayList<>();

        var held = submit(runtime, plan, 710, first);
        lattice.drive();
        var concurrent = submit(runtime, plan, 711, second);
        lattice.drive();
        assertFalse(held.isDone(), "the first quantum's device work has not retired");
        assertEquals(2, stream.held());
        assertTrue(gpu.frees.isEmpty());

        stream.release(null);
        lattice.drive();
        stream.release(null);
        lattice.drive();
        assertEquals(QwenExecutionContext.Status.SUCCESS, held.join().status());
        assertEquals(QwenExecutionContext.Status.SUCCESS, concurrent.join().status());
        assertTrue(
                Collections.disjoint(first.getFirst(), second.getFirst()),
                "a quantum admitted while another was in flight got storage of its own");

        int allocated = gpu.allocations.size();
        var reused = submit(runtime, plan, 712, third);
        lattice.drive();
        stream.release(null);
        lattice.drive();
        assertEquals(QwenExecutionContext.Status.SUCCESS, reused.join().status());
        assertEquals(allocated, gpu.allocations.size(), "a retired graph's storage served the next quantum");
        assertTrue(third.getFirst().equals(first.getFirst()) || third.getFirst().equals(second.getFirst()));
        runtime.close();
        QwenExecutionFixtures.assertEachAllocationFreedOnce(gpu);
    }

    @Test
    void largerQuantaGrowOnlyUndersizedSlotsAndSmallerQuantaReuseThem() {
        var plan = slicePlan();
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);
        List<List<Long>> addresses = new ArrayList<>();
        long[] retained = new long[3];
        int[] rows = {2, 5, 3};
        for (int index = 0; index < rows.length; index++) {
            var context = new QwenExecutionContext(
                    plan,
                    new QwenSequenceState(720 + index),
                    QwenExecutionContext.ExecutionKind.PREFILL,
                    0,
                    new int[rows[index]]);
            var outcome = runtime.submit(context, done -> addresses.add(bound(done)));
            assertEquals(QwenExecutionContext.Status.SUCCESS, outcome.join().status());
            retained[index] = runtime.retainedWorkspaceBytes();
        }
        long perRow = 3L * 64 * Short.BYTES + Integer.BYTES;
        assertEquals(2 * perRow, retained[0]);
        assertEquals(5 * perRow, retained[1], "the five-row quantum replaced each undersized slot");
        assertEquals(5 * perRow, retained[2], "a smaller quantum neither grows nor shrinks the storage");
        assertEquals(gpu.allocations.subList(0, 4), gpu.frees, "only the outgrown two-row buffers were freed");
        assertEquals(addresses.get(1), addresses.get(2));
        assertTrue(Collections.disjoint(addresses.get(0), addresses.get(1)));
        runtime.close();
        QwenExecutionFixtures.assertEachAllocationFreedOnce(gpu);
    }

    @Test
    void detachedDeviceLogitsStayWithTheCallerAndTheNextQuantumGetsItsOwn() throws Exception {
        int vocabulary = 1009;
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.statefulCompactWeights(vocabulary));
        var gpu = new EngineExecutionFixture.SamplingGpu(vocabulary);
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);
        var sequence = new QwenSequenceState(730);
        List<Long> logits = new ArrayList<>();
        int allocatedBeforeSecond = 0;
        try {
            for (int position = 0; position < 2; position++) {
                var context = new QwenExecutionContext(
                        plan,
                        sequence,
                        QwenExecutionContext.ExecutionKind.DECODE,
                        position,
                        new int[] {1},
                        QwenLogitsRequirement.LAST_TOKEN);
                if (position == 1) allocatedBeforeSecond = gpu.allocated().size();
                var outcome = runtime.execute(List.of(context)).getFirst();
                assertEquals(QwenExecutionContext.Status.SUCCESS, outcome.status(), () -> "" + outcome.failure());
                try (var output = context.logitsOutput().orElseThrow()) {
                    logits.add(output.deviceAddress());
                }
            }
            assertNotEquals(logits.get(0), logits.get(1));
            assertEquals(
                    List.of(logits.get(1)),
                    gpu.allocated()
                            .subList(allocatedBeforeSecond, gpu.allocated().size()),
                    "the second quantum allocated only a replacement for the detached logits");
            assertTrue(gpu.freed().containsAll(logits), "each caller released its logits");
        } finally {
            sequence.complete();
            runtime.close();
        }
        assertEquals(1, Collections.frequency(gpu.freed(), logits.get(0)), "the storage never freed caller logits");
    }

    private static CompletableFuture<QwenExecutionContext.Outcome> submit(
            EuhedralInferenceRuntime runtime, QwenExecutionPlan plan, long sequenceId, List<List<Long>> addresses) {
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(sequenceId), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        return runtime.submit(context, done -> addresses.add(bound(done)));
    }
}
