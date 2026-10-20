package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import io.euhedral_execution.inference.core.runtime.graph.FrameSeeds;
import io.euhedral_execution.inference.core.testing.Tunings;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Offered steps of concurrent sequences form passes on the workspace's owner: a lone step runs alone, steps that
/// are ready together share a quantum, a leaving or cancelled sequence holds nothing back, and a step that cannot
/// be admitted concludes failed without stopping later passes.
class ExecutionPassTest {

    /// A select that records its quantum and then runs `next`.
    private static class Told extends AbstractFrame implements AbstractQuantum.Continuation {
        final CompletableFuture<Quantum> seen = new CompletableFuture<>();
        private final Consumer<Quantum> next;
        private Quantum quantum;

        Told(Consumer<Quantum> next) {
            super(FrameSeeds.ID_HASH);
            randomizeHash(FrameSeeds.forHostWork().next());
            this.next = next;
        }

        Told() {
            this(quantum -> {});
        }

        @Override
        public void concluded(AbstractQuantum quantum) {
            this.quantum = (Quantum) quantum;
        }

        @Override
        public void execute() {
            try {
                this.next.accept(this.quantum);
            } finally {
                this.seen.complete(this.quantum);
            }
        }
    }

    private static Execution execution(ExecutionPlan plan) {
        return new Execution(new PullingLattice(), plan, new EngineExecutionFixture.SamplingGpu(8));
    }

    private static ExecutionPlan plan() {
        return new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
    }

    private static Quantum.Part part(Sequence sequence, long position, int token) {
        return new Quantum.Part(sequence, position, new int[] {token});
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aSingleOfferRunsAsItsOwnQuantum() throws Exception {
        var plan = plan();
        var execution = execution(plan);
        var a = new Sequence(1);
        try {
            var told = new Told();
            execution.runtime().publishOnOwner(() -> execution.offer(part(a, 0, 1), told));
            Quantum quantum = told.seen.get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.SUCCESS, quantum.conclusion().status());
            assertFalse(quantum.batched());
            assertEquals(1, execution.statistics(Quantum.ExecutionKind.DECODE).quanta());
        } finally {
            a.complete();
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aSecondSequenceJoinsTheFirstOnceItReOffers() throws Exception {
        var plan = plan();
        var execution = execution(plan);
        var a = new Sequence(1);
        var b = new Sequence(2);
        try {
            var toldA2 = new Told();
            var toldA1 =
                    new Told(q -> execution.runtime().publishOnOwner(() -> execution.offer(part(a, 1, 2), toldA2)));
            var toldB = new Told();
            execution.runtime().publishOnOwner(() -> {
                execution.offer(part(a, 0, 1), toldA1);
                execution.offer(part(b, 0, 5), toldB);
            });
            Quantum joint = toldA2.seen.get(10, TimeUnit.SECONDS);
            assertSame(joint, toldB.seen.get(10, TimeUnit.SECONDS));
            assertTrue(joint.batched());
            assertEquals(1, execution.statistics(Quantum.ExecutionKind.VERIFY).batchedQuanta());
            assertEquals(2, a.currentTokenPosition());
            assertEquals(1, b.currentTokenPosition());
        } finally {
            a.complete();
            b.complete();
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aLeavingMemberReleasesThePendingOffer() throws Exception {
        var plan = plan();
        var execution = execution(plan);
        var a = new Sequence(1);
        var b = new Sequence(2);
        try {
            var toldA1 = new Told(q -> execution.leave(a));
            var toldB = new Told();
            execution.runtime().publishOnOwner(() -> {
                execution.offer(part(a, 0, 1), toldA1);
                execution.offer(part(b, 0, 5), toldB);
            });
            Quantum quantum = toldB.seen.get(10, TimeUnit.SECONDS);
            assertFalse(quantum.batched());
            assertNotSame(toldA1.seen.get(10, TimeUnit.SECONDS), quantum);
        } finally {
            a.complete();
            b.complete();
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aCancelledPendingOfferConcludesCancelledAtOnce() throws Exception {
        var plan = plan();
        var execution = execution(plan);
        var a = new Sequence(1);
        var b = new Sequence(2);
        try {
            var toldA = new Told();
            var toldB = new Told();
            execution.runtime().publishOnOwner(() -> {
                execution.offer(part(a, 0, 1), toldA);
                execution.offer(part(b, 0, 5), toldB);
            });
            toldA.seen.get(10, TimeUnit.SECONDS);
            b.cancel();
            execution.withdrawCancelled();
            Quantum quantum = toldB.seen.get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.CANCELLED, quantum.conclusion().status());
        } finally {
            a.complete();
            b.complete();
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aStepThatCannotBeAdmittedConcludesFailedAndLaterPassesStillForm() throws Exception {
        var plan = plan();
        var gpu = new EngineExecutionFixture.SamplingGpu(8);
        var execution = new Execution(new PullingLattice(), plan, gpu);
        var a = new Sequence(1);
        var b = new Sequence(2);
        var c = new Sequence(3);
        var d = new Sequence(4);
        try {
            // a is in flight while b and c are pending, so b and c form one pass.
            var toldA = new Told();
            var toldB = new Told();
            var toldC = new Told();
            var toldD = new Told();
            var wrongVocabulary = new HostLogits(gpu, 9);
            execution.runtime().publishOnOwner(() -> {
                execution.offer(part(a, 0, 1), toldA);
                execution.offer(part(b, 0, 5).sampling(wrongVocabulary), toldB);
                execution.offer(part(c, 0, 6), toldC);
            });
            toldA.seen.get(10, TimeUnit.SECONDS);
            execution.leave(a);
            Quantum failed = toldB.seen.get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.FAILED, failed.conclusion().status());
            assertEquals(
                    Quantum.Status.SUCCESS,
                    toldC.seen.get(10, TimeUnit.SECONDS).conclusion().status());
            assertEquals(1, c.currentTokenPosition());
            assertEquals(0, b.currentTokenPosition());
            // b's step never ran, so b is not awaited: d's offer forms a pass once c, which ran, has left.
            execution.leave(c);
            execution.runtime().publishOnOwner(() -> execution.offer(part(d, 0, 7), toldD));
            assertEquals(
                    Quantum.Status.SUCCESS,
                    toldD.seen.get(10, TimeUnit.SECONDS).conclusion().status());
        } finally {
            for (var s : List.of(a, b, c, d)) s.complete();
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aPassWhoseBatchCannotBeBuiltRunsItsStepsAlone() throws Exception {
        var plan = plan();
        var gpu = new EngineExecutionFixture.SamplingGpu(8);
        var execution = new Execution(new PullingLattice(), plan, gpu);
        var a = new Sequence(1);
        var b = new Sequence(2);
        var c = new Sequence(3);
        try {
            var shared = new HostLogits(gpu, 8);
            var toldA = new Told();
            var toldB = new Told();
            var toldC = new Told();
            execution.runtime().publishOnOwner(() -> {
                execution.offer(part(c, 0, 1), toldC);
                // b and a take the same host logits, so a batch of them is refused; each runs alone.
                execution.offer(part(a, 0, 2).sampling(shared), toldA);
                execution.offer(part(b, 0, 3).sampling(shared), toldB);
            });
            toldC.seen.get(10, TimeUnit.SECONDS);
            execution.leave(c);
            Quantum qa = toldA.seen.get(10, TimeUnit.SECONDS);
            Quantum qb = toldB.seen.get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.SUCCESS, qa.conclusion().status());
            assertEquals(Quantum.Status.SUCCESS, qb.conclusion().status());
            assertNotSame(qa, qb);
            assertNull(qa.conclusion().failure());
        } finally {
            for (var s : List.of(a, b, c)) s.complete();
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aDuplicatePendingOfferConcludesFailedAndTheOthersAreUnaffected() throws Exception {
        var plan = plan();
        var execution = execution(plan);
        var a = new Sequence(1);
        var b = new Sequence(2);
        try {
            var toldA = new Told(q -> execution.leave(a));
            var toldB = new Told();
            var toldDuplicate = new Told();
            execution.runtime().publishOnOwner(() -> {
                execution.offer(part(a, 0, 1), toldA);
                execution.offer(part(b, 0, 5), toldB);
                execution.offer(part(b, 0, 6), toldDuplicate);
            });
            assertEquals(
                    Quantum.Status.FAILED,
                    toldDuplicate.seen.get(10, TimeUnit.SECONDS).conclusion().status());
            assertEquals(
                    Quantum.Status.SUCCESS,
                    toldB.seen.get(10, TimeUnit.SECONDS).conclusion().status());
            assertEquals(1, b.currentTokenPosition());
        } finally {
            a.complete();
            b.complete();
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aStepOverThePassRowCapRunsAloneAndWedgesNothing() throws Exception {
        var plan = plan();
        var execution = new Execution(
                new PullingLattice(),
                plan,
                new EngineExecutionFixture.SamplingGpu(8),
                1,
                Tunings.prefillChunkTokens(4));
        var a = new Sequence(1);
        var b = new Sequence(2);
        try {
            var toldBig = new Told(q -> execution.leave(a));
            var toldB = new Told();
            execution.runtime().publishOnOwner(() -> {
                execution.offer(new Quantum.Part(a, 0, new int[] {1, 2, 3, 4, 5, 6}), toldBig);
                execution.offer(part(b, 0, 5), toldB);
            });
            assertTrue(toldBig.seen.get(10, TimeUnit.SECONDS) != null);
            assertEquals(
                    Quantum.Status.SUCCESS,
                    toldB.seen.get(10, TimeUnit.SECONDS).conclusion().status());
        } finally {
            a.complete();
            b.complete();
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void closingConcludesTheOffersStillWaitingForAPass() throws Exception {
        var plan = plan();
        var execution = execution(plan);
        var a = new Sequence(1);
        var b = new Sequence(2);
        var toldA = new Told();
        var toldB = new Told();
        try {
            execution.runtime().publishOnOwner(() -> {
                execution.offer(part(a, 0, 1), toldA);
                execution.offer(part(b, 0, 5), toldB);
            });
            toldA.seen.get(10, TimeUnit.SECONDS);
        } finally {
            execution.close();
        }
        // a never reported, so b's offer was still pending when the execution closed.
        assertEquals(
                Quantum.Status.FAILED,
                toldB.seen.get(10, TimeUnit.SECONDS).conclusion().status());
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aSelectThatFailsToBeToldDoesNotKeepTheOthersFromRunning() {
        var plan = plan();
        var execution = execution(plan);
        try {
            var first = new Told() {
                @Override
                public void concluded(AbstractQuantum quantum) {
                    throw new IllegalStateException("not told");
                }
            };
            var second = new Told();
            var retirement = new PassRetirement(execution, execution.lake(), List.of(), List.of(first, second), false);
            retirement.concluded(new Quantum(plan, new Sequence(9), Quantum.ExecutionKind.DECODE, 0, new int[] {1}));
            assertThrows(IllegalStateException.class, retirement::execute);
            assertTrue(second.seen.orTimeout(10, TimeUnit.SECONDS).join() != null);
        } finally {
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aRefusedStepWhoseSelectFailsAtItsEndRunsItsSelectOnce() throws Exception {
        var plan = plan();
        var execution = execution(plan);
        var a = new Sequence(1);
        var b = new Sequence(2);
        var runs = new java.util.concurrent.atomic.AtomicInteger();
        var toldA = new Told();
        var failing = new Told(q -> runs.incrementAndGet()) {
            @Override
            public void doFinally() {
                throw new IllegalStateException("fails at its end");
            }
        };
        try {
            execution.runtime().publishOnOwner(() -> {
                execution.offer(part(a, 0, 1), toldA);
                execution.offer(part(b, 0, 5), failing);
            });
            toldA.seen.get(10, TimeUnit.SECONDS);
        } finally {
            // Closing refuses b's pending step, running its select in place.
            execution.close();
        }
        assertEquals(1, runs.get());
    }

    @Test
    void aSecondCompleteKeepsTheSequenceClosed() {
        var sequence = new Sequence(1);
        sequence.complete();
        sequence.complete();
        assertThrows(IllegalStateException.class, () -> sequence.requireSegmentAdmissible());
    }
}
