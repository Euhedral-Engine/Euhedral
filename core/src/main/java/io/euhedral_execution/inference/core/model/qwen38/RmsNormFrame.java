package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;

/// Runs the standalone BF16 RMSNorm instruction.
public final class RmsNormFrame extends InstructionFrame {

    RmsNormFrame(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        super(graph, instruction, gpu);
    }

    @Override
    protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
        boolean finalNorm = instruction.outputBuffers().contains(ExecutionPlan.Buffer.FINAL_NORMALIZED);
        int rows = finalNorm ? context.logitsRowCount() : context.inputTokenCount();
        if (finalNorm && context.seedsDraft() && context.plan().drafts()) {
            // Speculative drafting reads every row's post-final-norm hidden (docs/MTP_CONTRACT.md §2).
            var states = (io.euhedral_execution.inference.core.model.qwen38.AttentionStates)
                    context.sequenceState().kvCacheState();
            int all = context.inputTokenCount();
            normalize(
                    instruction,
                    context.workspace().address(instruction.inputBuffers().getFirst()),
                    states.draftSeedRows(all, instruction.outputWidth()),
                    all,
                    (float) context.plan().weights().config().rmsNormEpsilon());
        }
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
        normalize(instruction, input, output, rows, epsilon);
        if (finalNorm && context.kind() == Quantum.ExecutionKind.DRAFT) {
            // The last MTP row's post-mtp.norm hidden seeds the next recursive draft row.
            var states = (io.euhedral_execution.inference.core.model.qwen38.AttentionStates)
                    context.sequenceState().kvCacheState();
            gpu().copyDeviceToDevice(
                            states.draftRecursionHidden(instruction.outputWidth()),
                            output + (long) (rows - 1) * instruction.outputWidth() * Short.BYTES,
                            (long) instruction.outputWidth() * Short.BYTES);
        }
    }

    private void normalize(ExecutionPlan.Instruction instruction, long input, long output, int rows, float epsilon) {
        if (instruction.kind() == ExecutionPlan.Kind.RMS_NORM_UNIT_OFFSET) {
            gpu().rmsNormUnitOffsetBf16(
                            input, instruction.weightAddress(), output, rows, instruction.outputWidth(), epsilon);
        } else {
            gpu().rmsNormBf16(input, instruction.weightAddress(), output, rows, instruction.outputWidth(), epsilon);
        }
    }
}
