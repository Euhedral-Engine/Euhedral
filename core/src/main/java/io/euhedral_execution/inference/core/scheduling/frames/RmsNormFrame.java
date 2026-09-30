package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraph;

/// Runs the standalone BF16 RMSNorm instruction.
public final class RmsNormFrame extends QwenStageFrame {

    RmsNormFrame(StageGraph graph, QwenExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        super(graph, instruction, gpu);
    }

    @Override
    protected void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        boolean finalNorm = instruction.outputBuffers().contains(QwenExecutionPlan.Buffer.FINAL_NORMALIZED);
        int rows = finalNorm ? context.logitsRowCount() : context.inputTokenCount();
        if (rows == 0) return;
        long input = context.plan().hasFirstLayer()
                ? context.workspace().address(instruction.inputBuffers().getFirst())
                : context.workspace().hiddenStateAddress();
        if (finalNorm && rows != context.inputTokenCount()) {
            input += (long) (context.inputTokenCount() - 1) * instruction.inputWidth() * Short.BYTES;
        }
        long output = context.plan().hasFirstLayer()
                ? context.workspace().address(instruction.outputBuffers().getFirst())
                : context.workspace().normalizedStateAddress();
        float epsilon = (float) context.plan().weights().config().rmsNormEpsilon();
        if (instruction.kind() == QwenExecutionPlan.Kind.RMS_NORM_UNIT_OFFSET) {
            gpu().rmsNormUnitOffsetBf16(
                            input, instruction.weightAddress(), output, rows, instruction.outputWidth(), epsilon);
        } else {
            gpu().rmsNormBf16(input, instruction.weightAddress(), output, rows, instruction.outputWidth(), epsilon);
        }
    }
}
