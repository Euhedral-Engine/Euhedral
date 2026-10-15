package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.model.qwen38.speculative.SpeculativeAcceptance;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Each sequence of a batched quantum keeps exactly its own accepted rows, and the quantum is cancelled only when
/// every sequence is.
class BatchedSegmentsTest {

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void eachSegmentCommitsOnlyItsOwnRows() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var execution = new Execution(new PullingLattice(), plan, new EngineExecutionFixture.SamplingGpu(8));
        var a = new Sequence(1);
        var b = new Sequence(2);
        try {
            var parts = List.of(
                    new Quantum.Part(a, 0, new int[] {1, 2, 3}).withCommittedRows(2),
                    new Quantum.Part(b, 0, new int[] {4, 5, 6}));
            var outcome = execution
                    .submit(Quantum.batch(plan, Quantum.ExecutionKind.VERIFY, parts, LogitsRequirement.NONE))
                    .get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.SUCCESS, outcome.status());
            assertEquals(2, a.currentTokenPosition());
            assertEquals(3, b.currentTokenPosition());
            var layers = plan.weights().config().layerTypes();
            for (int l = 0; l < layers.length; l++) {
                if (layers[l] == LayerType.GATED_DELTA_NET) {
                    assertEquals(2, ((GdnStates) a.recurrentState()).forLayer(l).pendingReplayRows());
                    assertEquals(0, ((GdnStates) b.recurrentState()).forLayer(l).pendingReplayRows());
                } else {
                    assertEquals(
                            2, ((AttentionStates) a.kvCacheState()).forLayer(l).length());
                    assertEquals(
                            3, ((AttentionStates) b.kvCacheState()).forLayer(l).length());
                }
            }
        } finally {
            a.complete();
            b.complete();
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void oneCancelledSegmentLeavesTheOthersSucceeded() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var execution = new Execution(new PullingLattice(), plan, new EngineExecutionFixture.SamplingGpu(8));
        var a = new Sequence(1);
        var b = new Sequence(2);
        try {
            var quantum = Quantum.batch(
                    plan,
                    Quantum.ExecutionKind.VERIFY,
                    List.of(new Quantum.Part(a, 0, new int[] {1}), new Quantum.Part(b, 0, new int[] {2})),
                    LogitsRequirement.NONE);
            // The terminal consumer runs at commit, after the device work: a cancellation then must not undo b.
            execution.submit(quantum, done -> a.cancel()).get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.CANCELLED, quantum.conclusionFor(a).status());
            assertEquals(Quantum.Status.SUCCESS, quantum.conclusionFor(b).status());
            assertEquals(1, b.currentTokenPosition());
            assertEquals(Sequence.TerminalState.CANCELLED, a.terminalState());
        } finally {
            a.complete();
            b.complete();
            execution.close();
        }
    }

    @Test
    void aBatchIsCancelledOnlyWhenEverySequenceIs() {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var a = new Sequence(1);
        var b = new Sequence(2);
        var quantum = Quantum.batch(
                plan,
                Quantum.ExecutionKind.VERIFY,
                List.of(new Quantum.Part(a, 0, new int[] {1}), new Quantum.Part(b, 0, new int[] {2})),
                LogitsRequirement.NONE);
        a.cancel();
        assertFalse(quantum.hasFailureOrCancellation());
        b.cancel();
        assertTrue(quantum.hasFailureOrCancellation());
    }

    /// Records which host logits a selector was asked about and answers with fixed tokens.
    private static final class FixedSelector implements SpeculativeAcceptance.Selector {
        final int[] tokens;
        final List<HostLogits> asked = new ArrayList<>();

        FixedSelector(int... tokens) {
            this.tokens = tokens;
        }

        @Override
        public int[] select(HostLogits rows) {
            this.asked.add(rows);
            return this.tokens;
        }
    }

    @Test
    void aSelectorOtherThanGreedyDrivesASegmentsCommittedRows() {
        var gpu = new ExecutionFixtures.RecordingGpu();
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var logitsA = new HostLogits(gpu, 8);
        var logitsB = new HostLogits(gpu, 8);
        var logitsC = new HostLogits(gpu, 8);
        // Rows [t0, d1, d2]: every draft matches the selection, so all 3 rows commit.
        var acceptA = new SpeculativeAcceptance(new int[] {1, 2, 3}, t -> false, 8);
        var selectA = new FixedSelector(2, 3, 7);
        // No draft matches: only row 0 commits.
        var acceptB = new SpeculativeAcceptance(new int[] {4, 5, 6}, t -> false, 8);
        var selectB = new FixedSelector(0, 0, 0);
        var parts = List.of(
                new Quantum.Part(new Sequence(1), 0, new int[] {1, 2, 3}).verifying(logitsA, acceptA, selectA),
                new Quantum.Part(new Sequence(2), 0, new int[] {4, 5, 6}).verifying(logitsB, acceptB, selectB),
                new Quantum.Part(new Sequence(3), 0, new int[] {7, 7, 7}).withCommittedRows(2),
                new Quantum.Part(new Sequence(4), 0, new int[] {1, 2}).sampling(logitsC));
        var quantum = Quantum.batch(plan, Quantum.ExecutionKind.VERIFY, parts, LogitsRequirement.ALL_TOKENS);
        assertEquals(3, quantum.committedRows(0));
        assertEquals(1, quantum.committedRows(1));
        assertEquals(2, quantum.committedRows(2));
        assertEquals(2, quantum.committedRows(3));
        // Resolved once, from the part's own host logits.
        assertEquals(3, quantum.committedRows(0));
        assertEquals(List.of(logitsA), selectA.asked);
        assertEquals(List.of(logitsB), selectB.asked);
        assertTrue(acceptA.resolved());
    }

    @Test
    void anUnbatchedQuantumHasOneSegment() {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var quantum = new Quantum(plan, new Sequence(1), Quantum.ExecutionKind.VERIFY, 0, new int[] {1, 2, 3})
                .withCommittedRows(2);
        assertEquals(2, quantum.committedRows(0));
        assertThrows(IllegalArgumentException.class, () -> quantum.committedRows(1));
    }

    @Test
    void aBatchRefusesHostLogitsWithoutLogitsRows() {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var logits = new HostLogits(new ExecutionFixtures.RecordingGpu(), 8);
        var part = new Quantum.Part(new Sequence(1), 0, new int[] {1}).sampling(logits);
        assertThrows(
                IllegalArgumentException.class,
                () -> Quantum.batch(plan, Quantum.ExecutionKind.DECODE, List.of(part), LogitsRequirement.NONE));
    }

    @Test
    void aBatchRefusesAcceptanceOnADecode() {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var part = new Quantum.Part(new Sequence(1), 0, new int[] {1}).withCommittedRows(1);
        assertThrows(
                IllegalArgumentException.class,
                () -> Quantum.batch(plan, Quantum.ExecutionKind.DECODE, List.of(part), LogitsRequirement.NONE));
    }

    @Test
    void aBatchSeedsTheDraftWhenAnyPartDoes() {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var none = Quantum.batch(
                plan,
                Quantum.ExecutionKind.DECODE,
                List.of(new Quantum.Part(new Sequence(1), 0, new int[] {1})),
                LogitsRequirement.NONE);
        assertFalse(none.seedsDraft());
        var some = Quantum.batch(
                plan,
                Quantum.ExecutionKind.DECODE,
                List.of(
                        new Quantum.Part(new Sequence(1), 0, new int[] {1}),
                        new Quantum.Part(new Sequence(2), 0, new int[] {2}).seedingDraft()),
                LogitsRequirement.NONE);
        assertTrue(some.seedsDraft());
        assertSame(some.segments()[1].part().sequence(), some.segments()[1].sequence());
    }

    /// Records the source and size of every device-to-host copy.
    private static final class CopyRecordingGpu extends EngineExecutionFixture.SamplingGpu {
        final List<long[]> copies = new CopyOnWriteArrayList<>();

        CopyRecordingGpu() {
            super(8);
        }

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            this.copies.add(new long[] {source, byteSize});
            super.copyDeviceToHost(destination, source, byteSize);
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aVerifyingPartWithoutDeviceSelectionCopiesEveryOfItsRows() throws Exception {
        var gpu = new CopyRecordingGpu();
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var execution = new Execution(new PullingLattice(), plan, gpu);
        var a = new Sequence(1);
        var b = new Sequence(2);
        try {
            var logits = new HostLogits(gpu, 8);
            var logitsB = new HostLogits(gpu, 8);
            var acceptance = new SpeculativeAcceptance(new int[] {1, 2}, t -> false, 8);
            var parts = List.of(
                    new Quantum.Part(a, 0, new int[] {1, 2}).verifying(logits, acceptance, h -> {
                        // Acceptance reads the rows at stage commit, before the quantum retires.
                        assertEquals(2, h.rowCount());
                        return new int[] {2, 0};
                    }),
                    new Quantum.Part(b, 0, new int[] {3, 4}).sampling(logitsB));
            var outcome = execution
                    .submit(Quantum.batch(plan, Quantum.ExecutionKind.VERIFY, parts, LogitsRequirement.ALL_TOKENS))
                    .get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.SUCCESS, outcome.status(), String.valueOf(outcome.failure()));
            long row = 8L * Short.BYTES;
            var rowsCopy = gpu.copies.stream()
                    .filter(copy -> copy[1] == 2 * row)
                    .findFirst()
                    .orElseThrow(() -> new AssertionError("both rows cross to the host in one copy"));
            assertTrue(
                    gpu.copies.stream().anyMatch(copy -> copy[1] == row && copy[0] == rowsCopy[0] + 3 * row),
                    "the sampling part copies only its final row, after the verifying part's two rows");
            assertEquals(2, logits.rowCount());
            assertEquals(2, a.currentTokenPosition());
        } finally {
            a.complete();
            b.complete();
            execution.close();
        }
    }

    /// Records the rows each device selection reads and where every norm writes.
    private static final class RecordingSelectionGpu extends EngineExecutionFixture.SamplingGpu {
        final List<Long> argmaxAddresses = new CopyOnWriteArrayList<>();
        final List<Long> normOutputs = new CopyOnWriteArrayList<>();

        RecordingSelectionGpu() {
            super(8);
        }

        @Override
        public boolean argmaxBf16(long logitsAddress, int count, long resultAddress) {
            this.argmaxAddresses.add(logitsAddress);
            return true;
        }

        @Override
        public void rmsNormUnitOffsetBf16(
                long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
            this.normOutputs.add(outputAddress);
        }

        @Override
        public void rmsNormBf16(
                long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
            this.normOutputs.add(outputAddress);
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void eachPartReadsItsOwnLogitsRowsAndAnAcceptanceDrivesItsCommit() throws Exception {
        var gpu = new RecordingSelectionGpu();
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var execution = new Execution(new PullingLattice(), plan, gpu);
        var a = new Sequence(1);
        var b = new Sequence(2);
        var c = new Sequence(3);
        var logitsA = new HostLogits(gpu, 8);
        var logitsB = new HostLogits(gpu, 8);
        var logitsC = new HostLogits(gpu, 8);
        for (var logits : List.of(logitsA, logitsB, logitsC)) logits.selectOnDevice(true);
        try {
            // Rows [1, 2, 3] with selections [2, 0, 0]: one draft matches, so 2 of 3 rows commit.
            var acceptA = new SpeculativeAcceptance(new int[] {1, 2, 3}, t -> false, 8);
            var acceptC = new SpeculativeAcceptance(new int[] {4, 5}, t -> false, 8);
            var parts = List.of(
                    new Quantum.Part(a, 0, new int[] {1, 2, 3}).verifying(logitsA, acceptA, new FixedSelector(2, 0, 0)),
                    new Quantum.Part(b, 0, new int[] {6, 7}).sampling(logitsB),
                    new Quantum.Part(c, 0, new int[] {4, 5}).verifying(logitsC, acceptC, new FixedSelector(5, 1)));
            var outcome = execution
                    .submit(Quantum.batch(plan, Quantum.ExecutionKind.VERIFY, parts, LogitsRequirement.ALL_TOKENS))
                    .get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.SUCCESS, outcome.status(), String.valueOf(outcome.failure()));
            long row = 8L * Short.BYTES;
            long base = gpu.argmaxAddresses.get(0);
            // A: rows 0-2 from offset 0; B: its final row (1 of 2) at offset 3; C: rows 0-1 at offset 5.
            assertEquals(
                    List.of(base, base + row, base + 2 * row, base + 4 * row, base + 5 * row, base + 6 * row),
                    gpu.argmaxAddresses);
            assertEquals(2, a.currentTokenPosition());
            assertEquals(2, b.currentTokenPosition());
            assertEquals(2, c.currentTokenPosition());
            var layers = plan.weights().config().layerTypes();
            for (int l = 0; l < layers.length; l++) {
                if (layers[l] == LayerType.GATED_DELTA_NET) {
                    assertEquals(2, ((GdnStates) a.recurrentState()).forLayer(l).pendingReplayRows());
                    assertEquals(0, ((GdnStates) b.recurrentState()).forLayer(l).pendingReplayRows());
                    assertEquals(0, ((GdnStates) c.recurrentState()).forLayer(l).pendingReplayRows());
                } else {
                    assertEquals(
                            2, ((AttentionStates) a.kvCacheState()).forLayer(l).length());
                    assertEquals(
                            2, ((AttentionStates) c.kvCacheState()).forLayer(l).length());
                }
            }
        } finally {
            a.complete();
            b.complete();
            c.complete();
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void seedingDraftOnABatchSeedsEveryPart() throws Exception {
        var gpu = new RecordingSelectionGpu();
        var plan = new ExecutionPlan(ExecutionFixtures.mtpCompactWeights(8));
        var execution = new Execution(new PullingLattice(), plan, gpu);
        var a = new Sequence(1);
        var b = new Sequence(2);
        try {
            var quantum = Quantum.batch(
                            plan,
                            Quantum.ExecutionKind.VERIFY,
                            List.of(new Quantum.Part(a, 0, new int[] {1, 2}), new Quantum.Part(b, 0, new int[] {3})),
                            LogitsRequirement.NONE)
                    .seedingDraft();
            var outcome = execution.submit(quantum).get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.SUCCESS, outcome.status(), String.valueOf(outcome.failure()));
            assertTrue(gpu.normOutputs.contains(((AttentionStates) a.kvCacheState()).seedRows()));
            assertTrue(gpu.normOutputs.contains(((AttentionStates) b.kvCacheState()).seedRows()));
        } finally {
            a.complete();
            b.complete();
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aSequenceCancelledAfterFormationEndsCancelledAndTheOthersSucceed() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var execution = new Execution(new PullingLattice(), plan, new EngineExecutionFixture.SamplingGpu(8));
        var a = new Sequence(1);
        var b = new Sequence(2);
        try {
            var quantum = Quantum.batch(
                    plan,
                    Quantum.ExecutionKind.VERIFY,
                    List.of(new Quantum.Part(a, 0, new int[] {1}), new Quantum.Part(b, 0, new int[] {2})),
                    LogitsRequirement.NONE);
            a.cancel();
            var outcome = execution.submit(quantum).get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.SUCCESS, outcome.status(), String.valueOf(outcome.failure()));
            assertEquals(Quantum.Status.CANCELLED, quantum.conclusionFor(a).status());
            assertEquals(Quantum.Status.SUCCESS, quantum.conclusionFor(b).status());
            assertEquals(Sequence.TerminalState.CANCELLED, a.terminalState());
            assertEquals(Sequence.TerminalState.ACTIVE, b.terminalState());
            assertEquals(1, a.currentTokenPosition());
            assertEquals(1, b.currentTokenPosition());
        } finally {
            a.complete();
            b.complete();
            execution.close();
        }
    }

    @Test
    void conclusionForRejectsASequenceOutsideTheBatch() {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var quantum = Quantum.batch(
                plan,
                Quantum.ExecutionKind.VERIFY,
                List.of(new Quantum.Part(new Sequence(1), 0, new int[] {1})),
                LogitsRequirement.NONE);
        assertThrows(IllegalArgumentException.class, () -> quantum.conclusionFor(new Sequence(9)));
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aSequenceCancelledAndCompletedBeforeAdmissionRefusesTheBatch() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var execution = new Execution(new PullingLattice(), plan, new EngineExecutionFixture.SamplingGpu(8));
        var a = new Sequence(1);
        var b = new Sequence(2);
        try {
            var quantum = Quantum.batch(
                    plan,
                    Quantum.ExecutionKind.VERIFY,
                    List.of(new Quantum.Part(a, 0, new int[] {1}), new Quantum.Part(b, 0, new int[] {2})),
                    LogitsRequirement.NONE);
            a.cancel();
            a.complete();
            var outcome = execution.submit(quantum).get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.FAILED, outcome.status());
            assertEquals(0, b.currentTokenPosition());
        } finally {
            b.complete();
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aClosedSecondSequenceRefusesTheBatchWithoutTouchingTheFirst() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var execution = new Execution(new PullingLattice(), plan, new EngineExecutionFixture.SamplingGpu(8));
        var a = new Sequence(1);
        var b = new Sequence(2);
        try {
            var quantum = Quantum.batch(
                    plan,
                    Quantum.ExecutionKind.VERIFY,
                    List.of(new Quantum.Part(a, 0, new int[] {1}), new Quantum.Part(b, 0, new int[] {2})),
                    LogitsRequirement.NONE);
            b.cancel();
            b.complete();
            var outcome = execution.submit(quantum).get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.FAILED, outcome.status());
            assertEquals(Sequence.TerminalState.ACTIVE, a.terminalState());
            assertEquals(0, a.currentTokenPosition());
            assertFalse(a.inFlight());
        } finally {
            a.complete();
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aSequenceCancelledBeforeTheBatchFormedCommitsAndEndsCancelled() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var execution = new Execution(new PullingLattice(), plan, new EngineExecutionFixture.SamplingGpu(8));
        var a = new Sequence(1);
        var b = new Sequence(2);
        try {
            b.cancel();
            var quantum = Quantum.batch(
                    plan,
                    Quantum.ExecutionKind.VERIFY,
                    List.of(new Quantum.Part(a, 0, new int[] {1}), new Quantum.Part(b, 0, new int[] {2})),
                    LogitsRequirement.NONE);
            var outcome = execution.submit(quantum).get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.SUCCESS, outcome.status(), String.valueOf(outcome.failure()));
            assertEquals(Sequence.TerminalState.ACTIVE, a.terminalState());
            assertEquals(1, a.currentTokenPosition());
            assertEquals(Sequence.TerminalState.CANCELLED, b.terminalState());
            assertEquals(1, b.currentTokenPosition());
            assertEquals(Quantum.Status.CANCELLED, quantum.conclusionFor(b).status());
        } finally {
            a.complete();
            b.complete();
            execution.close();
        }
    }
}
