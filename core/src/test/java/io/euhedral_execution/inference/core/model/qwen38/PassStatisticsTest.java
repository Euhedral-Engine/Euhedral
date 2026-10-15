package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.runtime.PullingLattice;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class PassStatisticsTest {

    @Test
    void anUnstagedViewMovesNoWeightBytes() {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        Shape decode = plan.forExecution(Quantum.ExecutionKind.DECODE, 1);
        assertEquals(0, decode.diskReadBytes());
        assertEquals(0, decode.transferBytes());
    }

    @Test
    void aStagedViewTransfersEachStagedWeightOnce() {
        ExecutionPlan plan = StagingEdgesTest.stagedPlan();
        Shape decode = plan.forExecution(Quantum.ExecutionKind.DECODE, 1);
        long expected = decode.instructions().stream()
                .filter(i -> i.kind() == ExecutionPlan.Kind.WEIGHT_TRANSFER)
                .mapToLong(i -> i.weight().byteSize())
                .sum();
        assertTrue(expected > 0);
        assertEquals(expected, decode.transferBytes());
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void everyAdmittedQuantumIsCountedByKind() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8));
        var execution = new Execution(new PullingLattice(), plan, new ExecutionFixtures.RecordingGpu());
        try {
            var sequence = new Sequence(7);
            execution
                    .submit(new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1}))
                    .get(10, TimeUnit.SECONDS);
            var decode = execution.statistics(Quantum.ExecutionKind.DECODE);
            assertEquals(1, decode.quanta());
            assertEquals(1, decode.rows());
            assertEquals(1, decode.segments());
            assertEquals(0, decode.batchedQuanta());
            sequence.complete();
        } finally {
            execution.close();
        }
    }
}
