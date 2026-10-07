package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;

/// Runs the embedding lookup over the token IDs the quantum uploaded at admission.
public final class EmbeddingFrame extends InstructionFrame {

    EmbeddingFrame(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        super(graph, instruction, gpu);
    }

    @Override
    protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
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
