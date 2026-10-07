package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.generation.LogitsSampler;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.Test;

/// Workspace storage belongs to a reusable graph: consecutive quanta on the graph bind the same device
/// allocations, and storage passes to another quantum only after the one using it retired.
class WorkspaceReuseTest {

    private static ExecutionPlan slicePlan() {
        return new ExecutionPlan(
                ExecutionFixtures.weights(),
                ExecutionFixtures.norm(),
                List.of(ExecutionFixtures.q3("projection", 64, 201)));
    }

    /// Token IDs padded to 8 bytes, then the 64-bit start position.
    private static long inputRecordBytes(int rows) {
        return ((long) rows * Integer.BYTES + 7) / 8 * 8 + Long.BYTES;
    }

    private static List<Long> bound(Quantum context) {
        var workspace = context.workspace();
        return List.of(
                workspace.hiddenStateAddress(), workspace.normalizedStateAddress(), workspace.projectionAddress(0));
    }

    @Test
    void consecutiveQuantaBindTheSameStorageWithoutAllocatingOrFreeing() throws Exception {
        var plan = slicePlan();
        var gpu = new ExecutionFixtures.RecordingGpu();
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        List<List<Long>> addresses = new ArrayList<>();
        long retained = -1;
        for (int quantum = 0; quantum < 4; quantum++) {
            var context =
                    new Quantum(plan, new Sequence(700 + quantum), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
            var outcome = runtime.submit(context, done -> addresses.add(bound(done)));
            assertEquals(Quantum.Status.SUCCESS, outcome.join().status());
            if (quantum == 0) retained = runtime.retainedWorkspaceBytes();
            assertEquals(retained, runtime.retainedWorkspaceBytes(), "retained storage must not grow per quantum");
        }
        assertEquals(1, new HashSet<>(addresses).size(), "every quantum bound the first quantum's buffers");
        assertEquals(4, gpu.allocations.size(), "hidden, normalized, projection and input record, once");
        assertTrue(gpu.frees.isEmpty(), "retirement released bindings, not device storage");
        assertEquals((3L * 64 * Short.BYTES) + inputRecordBytes(1), retained);
        runtime.close();
        ExecutionFixtures.assertEachAllocationFreedOnce(gpu);
        assertEquals(0, runtime.retainedWorkspaceBytes());
    }

    @Test
    void storageOfAnUnretiredQuantumIsNeverLentAndConcurrentQuantaNeverShareIt() {
        var plan = slicePlan();
        var stream = new ExecutionFixtures.HoldingStream();
        var gpu = new ExecutionFixtures.RecordingGpu() {
            @Override
            public GpuStream openStream() {
                return stream;
            }
        };
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, gpu);
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
        assertEquals(Quantum.Status.SUCCESS, held.join().status());
        assertEquals(Quantum.Status.SUCCESS, concurrent.join().status());
        assertTrue(
                Collections.disjoint(first.getFirst(), second.getFirst()),
                "a quantum admitted while another was in flight got storage of its own");

        int allocated = gpu.allocations.size();
        var reused = submit(runtime, plan, 712, third);
        lattice.drive();
        stream.release(null);
        lattice.drive();
        assertEquals(Quantum.Status.SUCCESS, reused.join().status());
        assertEquals(allocated, gpu.allocations.size(), "a retired graph's storage served the next quantum");
        assertTrue(third.getFirst().equals(first.getFirst()) || third.getFirst().equals(second.getFirst()));
        runtime.close();
        ExecutionFixtures.assertEachAllocationFreedOnce(gpu);
    }

    @Test
    void largerQuantaGrowOnlyUndersizedSlotsAndSmallerQuantaReuseThem() {
        var plan = slicePlan();
        var gpu = new ExecutionFixtures.RecordingGpu();
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        List<List<Long>> addresses = new ArrayList<>();
        long[] retained = new long[3];
        int[] rows = {2, 5, 3};
        for (int index = 0; index < rows.length; index++) {
            var context = new Quantum(
                    plan, new Sequence(720 + index), Quantum.ExecutionKind.PREFILL, 0, new int[rows[index]]);
            var outcome = runtime.submit(context, done -> addresses.add(bound(done)));
            assertEquals(Quantum.Status.SUCCESS, outcome.join().status());
            retained[index] = runtime.retainedWorkspaceBytes();
        }
        long perRow = 3L * 64 * Short.BYTES;
        assertEquals(2 * perRow + inputRecordBytes(2), retained[0]);
        assertEquals(
                5 * perRow + inputRecordBytes(5), retained[1], "the five-row quantum replaced each undersized slot");
        assertEquals(
                5 * perRow + inputRecordBytes(5),
                retained[2],
                "a smaller quantum neither grows nor shrinks the storage");
        assertEquals(gpu.allocations.subList(0, 4), gpu.frees, "only the outgrown two-row buffers were freed");
        assertEquals(addresses.get(1), addresses.get(2));
        assertTrue(Collections.disjoint(addresses.get(0), addresses.get(1)));
        runtime.close();
        ExecutionFixtures.assertEachAllocationFreedOnce(gpu);
    }

    @Test
    void detachedDeviceLogitsStayWithTheCallerAndTheNextQuantumGetsItsOwn() throws Exception {
        int vocabulary = 1009;
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(vocabulary));
        var gpu = new EngineExecutionFixture.SamplingGpu(vocabulary);
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var sequence = new Sequence(730);
        List<Long> logits = new ArrayList<>();
        int allocatedBeforeSecond = 0;
        try {
            for (int position = 0; position < 2; position++) {
                var context = new Quantum(
                        plan,
                        sequence,
                        Quantum.ExecutionKind.DECODE,
                        position,
                        new int[] {1},
                        LogitsRequirement.LAST_TOKEN);
                if (position == 1) allocatedBeforeSecond = gpu.allocated().size();
                var outcome = runtime.execute(List.of(context)).getFirst();
                assertEquals(Quantum.Status.SUCCESS, outcome.status(), () -> "" + outcome.failure());
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

    @Test
    void hostSamplingQuantaKeepDeviceLogitsInTheirGraphAndExposeOnlyRetiredRows() throws Exception {
        int vocabulary = 1009;
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(vocabulary));
        var gpu = new EngineExecutionFixture.SamplingGpu(vocabulary);
        gpu.selectTokens(7, 9);
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var sequence = new Sequence(740);
        var hostLogits = new HostLogits(gpu, vocabulary);
        var sampler = new LogitsSampler(GenerationConfig.greedy(1L), vocabulary);
        try {
            List<Integer> selected = new ArrayList<>();
            for (int position = 0; position < 2; position++) {
                var context = new Quantum(
                        plan,
                        sequence,
                        Quantum.ExecutionKind.DECODE,
                        position,
                        new int[] {1},
                        LogitsRequirement.LAST_TOKEN,
                        hostLogits);
                var outcome = runtime.execute(List.of(context)).getFirst();
                assertEquals(Quantum.Status.SUCCESS, outcome.status(), () -> "" + outcome.failure());
                assertTrue(context.logitsOutput().isEmpty(), "host sampling detaches no device logits");
                selected.add(sampler.selectToken(hostLogits, null));
            }
            assertEquals(List.of(7, 9), selected);
            assertEquals(1, gpu.allocatedLogits.size(), "the decode graph kept one logits buffer");
            assertTrue(Collections.disjoint(gpu.freed(), gpu.allocatedLogits));

            gpu.linearFailure = new IllegalStateException("injected projection failure");
            var failing = new Quantum(
                    plan,
                    new Sequence(741),
                    Quantum.ExecutionKind.DECODE,
                    0,
                    new int[] {1},
                    LogitsRequirement.LAST_TOKEN,
                    hostLogits);
            assertEquals(
                    Quantum.Status.FAILED,
                    runtime.execute(List.of(failing)).getFirst().status());
            assertThrows(IllegalStateException.class, hostLogits::row, "a failed quantum exposes no row");
        } finally {
            sequence.complete();
            runtime.close();
            hostLogits.close();
        }
        assertTrue(gpu.freed().containsAll(gpu.allocatedLogits));
    }

    private static CompletableFuture<Quantum.Outcome> submit(
            Execution runtime, ExecutionPlan plan, long sequenceId, List<List<Long>> addresses) {
        var context = new Quantum(plan, new Sequence(sequenceId), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        return runtime.submit(context, done -> addresses.add(bound(done)));
    }
}
