package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.graph.StageFrame;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraph;
import java.util.Objects;

/// A Qwen plan instruction as an independently schedulable stage.
///
/// The frame is built once per reusable graph with an immutable instruction and weight binding. Each
/// quantum rebinds only the graph's context; the frame launches its operation on the quantum's stream.
public abstract class QwenStageFrame extends StageFrame {

    private final QwenExecutionPlan.Instruction instruction;
    private final ExecutionGpu gpu;

    protected QwenStageFrame(StageGraph graph, QwenExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        super(graph, Objects.requireNonNull(instruction, "instruction").id());
        this.instruction = instruction;
        this.gpu = Objects.requireNonNull(gpu, "gpu");
    }

    /// Creates the stage frame for one plan instruction.
    public static QwenStageFrame create(StageGraph graph, QwenExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        return switch (instruction.kind()) {
            case EMBEDDING -> new EmbeddingFrame(graph, instruction, gpu);
            case RMS_NORM, RMS_NORM_UNIT_OFFSET -> new RmsNormFrame(graph, instruction, gpu);
            case Q3_LINEAR, Q3_FFN_DOWN, Q4_LINEAR, Q5_LINEAR, BF16_LINEAR -> new LinearFrame(graph, instruction, gpu);
            case GDN_CONTROL,
                    GDN_CONVOLUTION,
                    GDN_RECURRENCE,
                    GDN_GATED_RMS_NORM,
                    FFN_STREAMED,
                    Q3_GATE_UP_SWIGLU,
                    RESIDUAL_RMS_NORM,
                    GDN_PROJECT_CONTROL,
                    GDN_PROJECTIONS,
                    RESIDUAL_ADD,
                    SWIGLU,
                    ATTENTION_QK_NORM_ROPE,
                    ATTENTION_PRODUCERS,
                    ATTENTION_KV_APPEND,
                    ATTENTION_CAUSAL -> new QwenGpuOperationFrame(graph, instruction, gpu);
        };
    }

    /// Launches this instruction's operation for the bound quantum.
    protected abstract void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction);

    /// Publishes externally visible state that this stage produced; runs only for a quantum whose
    /// device work retired successfully.
    protected void commit() {}

    /// Releases resources this stage acquired for the quantum; runs after its device work retired.
    protected void releaseTemporary(QwenExecutionContext context) {}

    @Override
    protected final void submit() {
        perform(context(), this.instruction);
    }

    @Override
    protected final void retired(boolean committed) {
        QwenExecutionContext context = context();
        try {
            if (committed) commit();
        } finally {
            releaseTemporary(context);
        }
    }

    public final QwenExecutionPlan.Instruction instruction() {
        return this.instruction;
    }

    protected final ExecutionGpu gpu() {
        return this.gpu;
    }

    protected final QwenExecutionContext context() {
        return (QwenExecutionContext) graph().quantum();
    }
}
