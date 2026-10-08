package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// A later quantum of a sequence may be admitted while an earlier one is in flight when the state they share is
/// carried state, ordered by edges between their graphs: it concludes after the earlier one, and both commit.
@Timeout(20)
class SequenceInFlightTest {

    @Test
    void aLaterPrefillIsAdmittedWhileTheEarlierOneRunsAndConcludesAfterIt() throws Exception {
        var plan = new ExecutionPlan(EngineExecutionFixture.weights());
        var stream = new ExecutionFixtures.HoldingStream();
        var gpu = new EngineExecutionFixture.SamplingGpu(8) {
            @Override
            public io.euhedral_execution.inference.core.gpu.GpuStream openStream() {
                return stream;
            }
        };
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, gpu);
        try {
            var sequence = new Sequence(1);
            var first = runtime.submit(new Quantum(
                    plan, sequence, Quantum.ExecutionKind.PREFILL, 0, new int[] {1, 2}, LogitsRequirement.NONE));
            lattice.drive();
            var second = runtime.submit(new Quantum(
                    plan, sequence, Quantum.ExecutionKind.PREFILL, 2, new int[] {3, 4, 5}, LogitsRequirement.NONE));
            lattice.drive();
            assertTrue(sequence.inFlight());
            assertFalse(second.isDone(), "admitted, not refused");
            while (stream.held() > 0) {
                stream.releaseNewest(null);
                lattice.drive();
            }
            assertEquals(Quantum.Status.SUCCESS, first.get(5, TimeUnit.SECONDS).status());
            assertEquals(Quantum.Status.SUCCESS, second.get(5, TimeUnit.SECONDS).status());
            assertEquals(5, sequence.currentTokenPosition(), "both committed, in order");
        } finally {
            while (stream.held() > 0) stream.release(null);
            lattice.drive();
            runtime.close();
        }
    }
}
