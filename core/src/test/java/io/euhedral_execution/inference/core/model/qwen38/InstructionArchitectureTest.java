package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertFalse;

import io.euhedral_execution.core.frames.PipelineFrame;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class InstructionArchitectureTest {

    @Test
    void executionPlanIsNotAPreconstructedPipeline() {
        assertFalse(Arrays.stream(ExecutionPlan.class.getDeclaredFields())
                .anyMatch(field -> PipelineFrame.Builder.class.isAssignableFrom(field.getType())));
    }
}
