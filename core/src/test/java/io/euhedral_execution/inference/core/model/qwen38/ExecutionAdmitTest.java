package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import io.euhedral_execution.inference.core.runtime.graph.FrameSeeds;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The generation path admits a quantum with a continuation: when its outcome is published the quantum throws the
/// frame, told which quantum concluded, into the lake.
class ExecutionAdmitTest {

    private static final class Told extends AbstractFrame implements AbstractQuantum.Continuation {
        final CompletableFuture<Quantum.Outcome> seen = new CompletableFuture<>();
        private Quantum quantum;

        Told() {
            super(FrameSeeds.ID_HASH);
            randomizeHash(FrameSeeds.forHostWork().next());
        }

        @Override
        public void concluded(AbstractQuantum quantum) {
            this.quantum = (Quantum) quantum;
        }

        @Override
        public void execute() {
            this.seen.complete(this.quantum.conclusion());
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void anAdmittedQuantumThrowsItsContinuationWithItsConclusion() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var execution = new Execution(new PullingLattice(), plan, new ExecutionFixtures.RecordingGpu());
        try {
            var sequence = new Sequence(901);
            var quantum = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});
            var told = new Told();
            execution.admit(quantum, told);
            Quantum.Outcome outcome = told.seen.get(10, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.SUCCESS, outcome.status());
            assertSame(outcome, quantum.conclusion());
            sequence.complete();
        } finally {
            execution.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aDuplicateAdmissionThrowsAndThrowsNoContinuation() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var execution = new Execution(new PullingLattice(), plan, new ExecutionFixtures.RecordingGpu());
        try {
            var sequence = new Sequence(902);
            var quantum = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});
            var first = new Told();
            execution.admit(quantum, first);
            assertEquals(
                    Quantum.Status.SUCCESS, first.seen.get(10, TimeUnit.SECONDS).status());
            var second = new Told();
            assertThrows(IllegalStateException.class, () -> execution.admit(quantum, second));
            assertFalse(second.seen.isDone(), "a quantum that never reached the runtime throws no continuation");
            sequence.complete();
        } finally {
            execution.close();
        }
    }
}
