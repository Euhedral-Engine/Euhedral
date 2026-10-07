package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class ExecutionPlanTest {
    @Test
    void instructionExposesBorrowedWeightAddressAndSizeWithoutReturningAHandle() {
        var projection = ExecutionFixtures.q3("projection", 64, 201);
        var plan = new ExecutionPlan(ExecutionFixtures.weights(), ExecutionFixtures.norm(), List.of(projection));
        var instruction = plan.instructions().get(2);

        assertEquals(201, instruction.weightAddress());
        assertEquals(projection.byteSize(), instruction.weightByteSize());
    }

    @Test
    void publishedTopologyAndWeightShapeCannotBeMutatedByCallers() {
        var original = ExecutionFixtures.q3("projection", 64, 201);
        var plan = new ExecutionPlan(ExecutionFixtures.weights(), ExecutionFixtures.norm(), List.of(original));
        original.shape()[0] = 128;
        plan.instructions().get(2).weight().shape()[0] = 128;
        assertEquals(64, plan.instructions().get(2).weight().shape()[0]);
        assertThrows(
                UnsupportedOperationException.class, () -> plan.instructions().clear());
        assertThrows(
                UnsupportedOperationException.class, () -> plan.successors(1).clear());
        assertEquals(List.of(2), plan.successors(1));
    }
}
