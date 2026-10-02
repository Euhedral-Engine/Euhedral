package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraph;

/// Runs one independent quantized or BF16 projection instruction.
public final class LinearFrame extends QwenStageFrame {

    LinearFrame(StageGraph graph, QwenExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        super(graph, instruction, gpu);
    }

    @Override
    protected void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        boolean logits = instruction.outputBuffers().contains(QwenExecutionPlan.Buffer.LOGITS);
        int rows = logits ? context.logitsRowCount() : context.inputTokenCount();
        if (rows == 0) return;
        long input = context.plan().hasFirstLayer()
                ? context.workspace().address(instruction.inputBuffers().getFirst())
                : context.workspace().normalizedStateAddress();
        long output = context.plan().hasFirstLayer()
                ? context.workspace().address(instruction.outputBuffers().getFirst())
                : context.workspace().projectionAddress(instruction.outputBufferIndex());
        if (instruction.weights().size() == 1
                && instruction.weightFormat() == WeightFormat.NVFP4
                && (instruction.kind() == QwenExecutionPlan.Kind.Q3_FFN_DOWN
                        || instruction.kind() == QwenExecutionPlan.Kind.Q3_LINEAR
                        || instruction.kind() == QwenExecutionPlan.Kind.Q4_LINEAR
                        || instruction.kind() == QwenExecutionPlan.Kind.Q5_LINEAR)) {
            gpu().linearNvfp4Bf16(
                            input,
                            instruction.weightAddress(),
                            output,
                            rows,
                            instruction.inputWidth(),
                            instruction.outputWidth(),
                            instruction.weightByteSize());
            if (logits) context.logitsProduced(output);
            return;
        }
        boolean quantized = instruction.kind() == QwenExecutionPlan.Kind.Q3_LINEAR
                || instruction.kind() == QwenExecutionPlan.Kind.Q4_LINEAR
                || instruction.kind() == QwenExecutionPlan.Kind.Q5_LINEAR;
        if (context.kind() == QwenExecutionContext.ExecutionKind.VERIFY
                && rows > 1
                && !(quantized && gpu().rowExactQuantizedLinears())) {
            // Row-exact verification: each row takes exactly the route and kernel one-row decode does.
            int outputBytes = instruction.kind() == QwenExecutionPlan.Kind.BF16_LINEAR ? Float.BYTES : Short.BYTES;
            for (int row = 0; row < rows; row++) {
                launch(
                        context,
                        instruction,
                        input + (long) row * instruction.inputWidth() * Short.BYTES,
                        output + (long) row * instruction.outputWidth() * outputBytes,
                        1);
            }
        } else launch(context, instruction, input, output, rows);
        if (logits) context.logitsProduced(output);
    }

    private void launch(
            QwenExecutionContext context,
            QwenExecutionPlan.Instruction instruction,
            long input,
            long output,
            int rows) {
        switch (instruction.kind()) {
            case Q3_FFN_DOWN -> {
                if (instruction.outputBuffers().contains(QwenExecutionPlan.Buffer.FFN_PARTIALS))
                    gpu().q3FfnDownSplitBf16(
                                    input,
                                    instruction.weightAddress(),
                                    output,
                                    context.workspace().address(QwenExecutionPlan.Buffer.FFN_PARTIALS),
                                    rows,
                                    instruction.inputWidth(),
                                    instruction.outputWidth(),
                                    instruction.weightByteSize(),
                                    instruction.weightLayout());
                else
                    gpu().q3FfnDownBf16(
                                    input,
                                    instruction.weightAddress(),
                                    output,
                                    rows,
                                    instruction.inputWidth(),
                                    instruction.outputWidth(),
                                    instruction.weightByteSize(),
                                    instruction.weightLayout());
            }
            case Q3_LINEAR ->
                gpu().linearQ3Bf16(
                                input,
                                instruction.weightAddress(),
                                output,
                                rows,
                                instruction.inputWidth(),
                                instruction.outputWidth(),
                                instruction.weightByteSize(),
                                instruction.weightLayout());
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
