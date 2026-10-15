package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.model.qwen38.speculative.SpeculativeAcceptance;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import java.util.ArrayList;
import java.util.List;
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

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aVerifyingPartWithoutDeviceSelectionFailsTheQuantumForNow() throws Exception {
        // TEMPORARY: replaced when a batched verification copies every row's logits to the host.
        var gpu = new EngineExecutionFixture.SamplingGpu(8);
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var execution = new Execution(new PullingLattice(), plan, gpu);
        var a = new Sequence(1);
        try {
            var logits = new HostLogits(gpu, 8);
            var acceptance = new SpeculativeAcceptance(new int[] {1, 2}, t -> false, 8);
            var part = new Quantum.Part(a, 0, new int[] {1, 2})
                    .verifying(logits, acceptance, SpeculativeAcceptance.Selector.GREEDY);
            var outcome = execution
                    .submit(Quantum.batch(
                            plan, Quantum.ExecutionKind.VERIFY, List.of(part), LogitsRequirement.ALL_TOKENS))
                    .get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.FAILED, outcome.status());
            assertInstanceOf(UnsupportedOperationException.class, outcome.failure());
        } finally {
            a.complete();
            execution.close();
        }
    }
}
