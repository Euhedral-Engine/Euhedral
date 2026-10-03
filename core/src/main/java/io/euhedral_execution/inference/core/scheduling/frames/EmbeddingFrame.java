package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraph;

/// Runs the embedding lookup over the token IDs the quantum uploaded at admission.
public final class EmbeddingFrame extends QwenStageFrame {

    EmbeddingFrame(StageGraph graph, QwenExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        super(graph, instruction, gpu);
    }

    @Override
    protected void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        gpu().embedQ3(
                        context.workspace().tokenIdsAddress(),
                        instruction.weightAddress(),
                        instruction.weightByteSize(),
                        context.workspace().hiddenStateAddress(),
                        context.inputTokenCount(),
                        context.plan().weights().config().vocabSize(),
                        instruction.outputWidth(),
                        instruction.weightLayout());
    }
}
