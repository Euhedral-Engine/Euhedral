package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.StageFrame;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;
import java.util.Objects;

/// A Qwen plan instruction as an independent stage frame.
///
/// The frame is built once per reusable graph with an immutable instruction and weight binding. Each
/// quantum rebinds only the graph's context; the frame launches its operation on the quantum's stream.
public abstract class InstructionFrame extends StageFrame {

    private final ExecutionPlan.Instruction instruction;
    private final ExecutionGpu gpu;

    protected InstructionFrame(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        super(graph, Objects.requireNonNull(instruction, "instruction").id());
        this.instruction = instruction;
        this.gpu = Objects.requireNonNull(gpu, "gpu");
    }

    /// Creates the stage frame for one plan instruction.
    public static InstructionFrame create(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        return switch (instruction.kind()) {
            case EMBEDDING -> new EmbeddingFrame(graph, instruction, gpu);
            case WEIGHT_TRANSFER -> new WeightTransferFrame(graph, instruction, gpu);
            case RMS_NORM, RMS_NORM_UNIT_OFFSET -> new RmsNormFrame(graph, instruction, gpu);
            case Q3_LINEAR, Q4_LINEAR, Q5_LINEAR, BF16_LINEAR -> new LinearFrame(graph, instruction, gpu);
            case GDN_CONTROL,
                    GDN_CONVOLUTION,
                    GDN_RECURRENCE,
                    GDN_GATED_RMS_NORM,
                    Q3_GATE_UP_SWIGLU,
                    RESIDUAL_RMS_NORM,
                    GDN_PROJECT_CONTROL,
                    RESIDUAL_ADD,
                    SWIGLU,
                    ATTENTION_QK_NORM_ROPE,
                    ATTENTION_KV_APPEND,
                    ATTENTION_CAUSAL,
                    MTP_STEM -> new OperationFrame(graph, instruction, gpu);
            case DFLASH_TAP,
                    DFLASH_LINEAR,
                    DFLASH_RMS_NORM,
                    DFLASH_CONV,
                    DFLASH_CONTEXT_KV,
                    DFLASH_BLOCK_QK,
                    DFLASH_ATTENTION,
                    DFLASH_SWIGLU,
                    DFLASH_LM_HEAD,
                    DFLASH_TOPK,
                    DFLASH_SELECT -> new DFlash2Frame(graph, instruction, gpu);
        };
    }

    /// Launches this instruction's operation for the bound quantum.
    protected abstract void perform(Quantum context, ExecutionPlan.Instruction instruction);

    /// Publishes externally visible state that this stage produced; runs only for a quantum whose
    /// device work retired successfully.
    protected void commit() {}

    /// Releases resources this stage acquired for the quantum; runs after its device work retired.
    protected void releaseTemporary(Quantum context) {}

    @Override
    protected final void submit() {
        Quantum context = context();
        if (context.kind() != Quantum.ExecutionKind.VERIFY) {
            perform(context, this.instruction);
            return;
        }
        // Verification rows must be bit for bit one-row decode; the selection is per submitting thread.
        this.gpu.selectRowExact(true);
        try {
            perform(context, this.instruction);
        } finally {
            this.gpu.selectRowExact(false);
        }
    }

    @Override
    protected final void retired(boolean committed) {
        Quantum context = context();
        try {
            if (committed) commit();
        } finally {
            releaseTemporary(context);
        }
    }

    public final ExecutionPlan.Instruction instruction() {
        return this.instruction;
    }

    protected final ExecutionGpu gpu() {
        return this.gpu;
    }

    protected final Quantum context() {
        return (Quantum) graph().quantum();
    }
}
