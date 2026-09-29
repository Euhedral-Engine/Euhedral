package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenWorkGenerator;

/// Runs one independent quantized or BF16 projection instruction.
public final class LinearFrame extends QwenInstructionFrame {

    public LinearFrame(
            long idHash,
            FrameManager<QwenExecutionContext, LinearFrame> recycler,
            QwenExecutionContext context,
            QwenExecutionPlan.Instruction instruction,
            ExecutionGpu gpu,
            QwenWorkGenerator generator) {
        super(idHash, recycler, context, instruction, gpu, generator);
    }

    @Override
    protected void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        int rows = instruction.outputBuffers().contains(QwenExecutionPlan.Buffer.LOGITS)
                ? context.logitsRowCount()
                : context.inputTokenCount();
        if (rows == 0) return;
        long input = context.plan().hasFirstLayer()
                ? context.workspace().address(instruction.inputBuffers().getFirst())
                : context.workspace().normalizedStateAddress();
        long output = context.plan().hasFirstLayer()
                ? context.workspace().address(instruction.outputBuffers().getFirst())
                : context.workspace().projectionAddress(instruction.outputBufferIndex());
        switch (instruction.kind()) {
            case Q3_FFN_DOWN ->
                gpu().q3FfnDownBf16(
                                input,
                                instruction.weightAddress(),
                                output,
                                rows,
                                instruction.inputWidth(),
                                instruction.outputWidth(),
                                instruction.weightByteSize());
            case Q3_LINEAR ->
                gpu().linearQ3Bf16(
                                input,
                                instruction.weightAddress(),
                                output,
                                rows,
                                instruction.inputWidth(),
                                instruction.outputWidth(),
                                instruction.weightByteSize());
            case Q4_LINEAR ->
                gpu().linearQ4Bf16(
                                input,
                                instruction.weightAddress(),
                                output,
                                rows,
                                instruction.inputWidth(),
                                instruction.outputWidth(),
                                instruction.weightByteSize());
            case Q5_LINEAR ->
                gpu().linearQ5Bf16(
                                input,
                                instruction.weightAddress(),
                                output,
                                rows,
                                instruction.inputWidth(),
                                instruction.outputWidth(),
                                instruction.weightByteSize());
            case BF16_LINEAR ->
                gpu().linearBf16ToFloat(
                                input,
                                instruction.weightAddress(),
                                output,
                                rows,
                                instruction.inputWidth(),
                                instruction.outputWidth());
            default ->
                throw new IllegalArgumentException(
                        "linear frame received non-linear instruction: " + instruction.kind());
        }
    }
}
