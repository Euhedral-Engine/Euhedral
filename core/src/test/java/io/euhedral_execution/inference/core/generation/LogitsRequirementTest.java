package io.euhedral_execution.inference.core.generation;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.inference.core.model.qwen38.EngineExecutionFixture;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionFixtures;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class LogitsRequirementTest {
    @ParameterizedTest
    @EnumSource(LogitsRequirement.class)
    void outputRequirementControlsAllocationNormalizationAndProjection(LogitsRequirement requirement) throws Exception {
        // A prime vocabulary size distinguishes logits bytes from power-of-two state buffers.
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(1009));
        var gpu = new OutputGpu();
        var sequence = new Sequence(71);
        var context = new Quantum(plan, sequence, Quantum.ExecutionKind.PREFILL, 0, new int[] {1, 2, 3}, requirement);
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        int rows = requirement.outputRows(3);
        long[] finalBuffers = new long[3];
        var norm =
                context.plan().instructions().get(context.plan().instructions().size() - 2);
        try {
            var outcome = runtime.execute(List.of(context), live -> {
                assertEquals(requirement, live.logitsRequirement());
                var workspace = live.workspace();
                finalBuffers[0] = workspace.address(ExecutionPlan.Buffer.FINAL_HIDDEN_STATE);
                assertEquals(rows != 0, workspace.hasBuffer(ExecutionPlan.Buffer.LOGITS));
                assertEquals(rows != 0, workspace.hasBuffer(ExecutionPlan.Buffer.FINAL_NORMALIZED));
                if (rows != 0) {
                    finalBuffers[1] = workspace.address(ExecutionPlan.Buffer.FINAL_NORMALIZED);
                    // Retirement hands the logits rows to the caller before the terminal consumer runs.
                    finalBuffers[2] = live.logitsOutput().orElseThrow().deviceAddress();
                    assertEquals(
                            (long) rows * 1009 * Short.BYTES, workspace.bufferByteSize(ExecutionPlan.Buffer.LOGITS));
                }
            });
            assertEquals(
                    Quantum.Status.SUCCESS,
                    outcome.getFirst().status(),
                    () -> String.valueOf(outcome.getFirst().failure()));
            List<long[]> finalNorms = rows == 0
                    ? List.of()
                    : gpu.norms.stream()
                            .filter(call -> call[1] == finalBuffers[1])
                            .toList();
            List<long[]> projections = rows == 0
                    ? List.of()
                    : gpu.projections.stream()
                            .filter(call -> call[1] == finalBuffers[2])
                            .toList();
            assertEquals(
                    rows == 0 ? List.of() : List.of((long) rows),
                    finalNorms.stream().map(call -> call[2]).toList());
            assertEquals(
                    rows == 0 ? List.of() : List.of((long) rows),
                    projections.stream().map(call -> call[2]).toList());
            if (rows != 0) {
                assertEquals(
                        finalBuffers[0]
                                + (requirement == LogitsRequirement.LAST_TOKEN
                                        ? 2L * norm.inputWidth() * Short.BYTES
                                        : 0),
                        finalNorms.getFirst()[0]);
                assertEquals(finalBuffers[1], projections.getFirst()[0]);
            }
            assertEquals(
                    requirement == LogitsRequirement.LAST_TOKEN ? 1 : 0,
                    java.util.Collections.frequency(gpu.allocationSizes, 1009L * Short.BYTES));
            assertEquals(
                    requirement == LogitsRequirement.ALL_TOKENS ? 1 : 0,
                    java.util.Collections.frequency(gpu.allocationSizes, 3L * 1009 * Short.BYTES));
            assertEquals(3, sequence.currentTokenPosition());
            assertEquals(rows != 0, context.logitsOutput().isPresent());
            context.logitsOutput().ifPresent(logits -> {
                assertEquals(rows, logits.tokenCount());
                logits.close();
            });
        } finally {
            sequence.complete();
            runtime.close();
        }
    }

    private static class OutputGpu extends EngineExecutionFixture.SamplingGpu {
        OutputGpu() {
            super(1009);
        }

        /// Each call as {input, output, rows}.
        final List<long[]> norms = new ArrayList<>();
        final List<long[]> projections = new ArrayList<>();
        final List<Long> allocationSizes = new ArrayList<>();

        @Override
        public long allocate(long bytes) {
            allocationSizes.add(bytes);
            return super.allocate(bytes);
        }

        @Override
        public void zeroDeviceMemory(long address, long bytes) {}

        @Override
        public void rmsNormUnitOffsetBf16(long input, long weight, long output, int rows, int width, float epsilon) {
            norms.add(new long[] {input, output, rows});
        }

        @Override
        public void linearQ3Bf16(
                long input, long weights, long output, int rows, int inFeatures, int outFeatures, long bytes) {
            projections.add(new long[] {input, output, rows});
        }
    }
}
