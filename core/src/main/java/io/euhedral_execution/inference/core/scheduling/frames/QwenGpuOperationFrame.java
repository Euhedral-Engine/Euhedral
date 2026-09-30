package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.scheduling.AttentionKvState;
import io.euhedral_execution.inference.core.scheduling.AttentionSequenceStates;
import io.euhedral_execution.inference.core.scheduling.GdnSequenceStates;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenGdnSequenceState;
import io.euhedral_execution.inference.core.scheduling.QwenWorkGenerator;

/// Executes one stateful or elementwise GPU instruction using its immutable buffer operands.
public final class QwenGpuOperationFrame extends QwenInstructionFrame {

    private AttentionKvState pendingAppendState;
    private int pendingAppendTokens;

    public QwenGpuOperationFrame(
            long idHash,
            FrameManager<QwenExecutionContext, QwenGpuOperationFrame> recycler,
            QwenExecutionContext context,
            QwenExecutionPlan.Instruction instruction,
            ExecutionGpu gpu,
            QwenWorkGenerator generator) {
        super(idHash, recycler, context, instruction, gpu, generator);
    }

    @Override
    protected void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        switch (instruction.kind()) {
            case FFN_STREAMED ->
                gpu().q3FfnStreamedBf16(
                                input(context, instruction, 0),
                                instruction.weightAddress(0),
                                instruction.weightAddress(1),
                                output(context, instruction, 0),
                                output(context, instruction, 1),
                                output(context, instruction, 2),
                                context.inputTokenCount(),
                                instruction.inputWidth(),
                                context.plan().weights().config().intermediateSize(),
                                instruction.weightByteSize(0),
                                instruction.weightByteSize(1));
            case Q3_GATE_UP_SWIGLU ->
                gpu().q3GateUpSwiGluBf16(
                                input(context, instruction, 0),
                                instruction.weightAddress(),
                                output(context, instruction, 0),
                                context.inputTokenCount(),
                                instruction.inputWidth(),
                                instruction.outputWidth(),
                                instruction.weightByteSize());
            case RESIDUAL_RMS_NORM ->
                gpu().residualRmsNormBf16(
                                input(context, instruction, 0),
                                input(context, instruction, 1),
                                instruction.weightAddress(),
                                output(context, instruction, 0),
                                output(context, instruction, 1),
                                context.inputTokenCount(),
                                instruction.outputWidth(),
                                (float) context.plan().weights().config().rmsNormEpsilon());
            case GDN_PROJECT_CONTROL ->
                gpu().gdnProjectControlFp32(
                                input(context, instruction, 0),
                                instruction.weightAddress(0),
                                instruction.weightAddress(1),
                                instruction.weightAddress(2),
                                instruction.weightAddress(3),
                                output(context, instruction, 0),
                                output(context, instruction, 1),
                                context.inputTokenCount(),
                                instruction.inputWidth(),
                                instruction.outputWidth());
            case GDN_PROJECTIONS -> runGdnProjections(context, instruction);
            case GDN_CONTROL -> runControl(context, instruction);
            case GDN_CONVOLUTION -> runConvolution(context, instruction);
            case GDN_RECURRENCE -> runRecurrence(context, instruction);
            case GDN_GATED_RMS_NORM -> runGatedRmsNorm(context, instruction);
            case ATTENTION_PRODUCERS -> runAttentionProducers(context, instruction);
            case ATTENTION_QK_NORM_ROPE -> runAttentionQkNormRope(context, instruction);
            case ATTENTION_KV_APPEND -> runAttentionKvAppend(context, instruction);
            case ATTENTION_CAUSAL -> runAttentionCausal(context, instruction);
            case RESIDUAL_ADD -> runResidualAdd(context, instruction);
            case SWIGLU -> runSwiGlu(context, instruction);
            default ->
                throw new IllegalArgumentException(
                        "GPU operation frame received unsupported instruction: " + instruction.kind());
        }
    }

    private void runGdnProjections(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        QwenConfig config = context.plan().weights().config();
        int valueZWidth = 2 * config.linearNumValueHeads() * config.linearValueHeadDim();
        gpu().gdnProjectionsBf16(
                        input(context, instruction, 0),
                        instruction.weightAddress(0),
                        instruction.weightAddress(1),
                        output(context, instruction, 0),
                        output(context, instruction, 1),
                        context.inputTokenCount(),
                        instruction.inputWidth(),
                        instruction.outputWidth(),
                        valueZWidth,
                        instruction.weightByteSize(0),
                        instruction.weightByteSize(1));
    }

    private void runControl(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        gpu().gdnControlFp32(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        instruction.weightAddress(0),
                        instruction.weightAddress(1),
                        output(context, instruction, 0),
                        output(context, instruction, 1),
                        context.inputTokenCount(),
                        instruction.outputWidth());
    }

    private void runConvolution(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        QwenConfig config = context.plan().weights().config();
        int queryKeyWidth = 2 * config.linearNumKeyHeads() * config.linearKeyHeadDim();
        int valueWidth = config.linearNumValueHeads() * config.linearValueHeadDim();
        gpu().gdnConvolutionBf16(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        instruction.weightAddress(0),
                        sequenceState(context, instruction).convolutionStateAddress(),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        queryKeyWidth,
                        valueWidth,
                        instruction.outputWidth(),
                        config.linearConvKernelDim());
    }

    private void runRecurrence(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        QwenConfig config = context.plan().weights().config();
        float outputScale = (float) (1.0 / Math.sqrt(config.linearKeyHeadDim()));
        gpu().gdnRecurrenceBf16(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        input(context, instruction, 2),
                        sequenceState(context, instruction).recurrentStateAddress(),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        config.linearNumKeyHeads(),
                        config.linearNumValueHeads(),
                        config.linearKeyHeadDim(),
                        config.linearValueHeadDim(),
                        outputScale);
    }

    private void runGatedRmsNorm(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        QwenConfig config = context.plan().weights().config();
        gpu().gdnGatedRmsNormBf16(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        instruction.weightAddress(0),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        config.linearNumValueHeads(),
                        config.linearValueHeadDim(),
                        (float) config.rmsNormEpsilon());
    }

    private void runAttentionQkNormRope(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        QwenConfig config = context.plan().weights().config();
        int rotaryDim = (int) Math.round(config.attentionHeadDim() * config.partialRotaryFactor());
        gpu().attentionQkNormRopeBf16(
                        input(context, instruction, 0),
                        instruction.weightAddress(0),
                        instruction.weightAddress(1),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        config.numAttentionHeads(),
                        config.numKeyValueHeads(),
                        config.attentionHeadDim(),
                        rotaryDim,
                        context.startPosition(),
                        (float) config.rmsNormEpsilon(),
                        config.ropeTheta());
    }

    private void runAttentionProducers(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        QwenConfig config = context.plan().weights().config();
        AttentionKvState state = attentionState(context, instruction);
        state.prepareAppend(context.startPosition(), context.inputTokenCount());
        gpu().attentionProducersNvfp4(
                        input(context, instruction, 0),
                        instruction.weightAddress(0),
                        instruction.weightAddress(1),
                        instruction.weightAddress(2),
                        instruction.weightAddress(3),
                        output(context, instruction, 0),
                        output(context, instruction, 1),
                        state.keyCacheAddress(),
                        state.valueCacheAddress(),
                        context.inputTokenCount(),
                        instruction.inputWidth(),
                        config.numAttentionHeads(),
                        config.numKeyValueHeads(),
                        config.attentionHeadDim(),
                        (int) Math.round(config.attentionHeadDim() * config.partialRotaryFactor()),
                        context.startPosition(),
                        (float) config.rmsNormEpsilon(),
                        config.ropeTheta(),
                        instruction.weightByteSize(0),
                        instruction.weightByteSize(1));
        this.pendingAppendState = state;
        this.pendingAppendTokens = context.inputTokenCount();
    }

    private void runAttentionKvAppend(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        QwenConfig config = context.plan().weights().config();
        int queryWidth = config.numAttentionHeads() * config.attentionHeadDim();
        int keyValueWidth = config.numKeyValueHeads() * config.attentionHeadDim();
        AttentionKvState state = attentionState(context, instruction);
        state.prepareAppend(context.startPosition(), context.inputTokenCount());
        gpu().attentionKvAppendNvfp4(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        state.keyCacheAddress(),
                        state.valueCacheAddress(),
                        context.inputTokenCount(),
                        queryWidth,
                        keyValueWidth,
                        context.startPosition());
        this.pendingAppendState = state;
        this.pendingAppendTokens = context.inputTokenCount();
    }

    @Override
    protected void gpuCompleted() {
        if (this.pendingAppendState != null) {
            try {
                this.pendingAppendState.commitAppend(this.pendingAppendTokens);
            } finally {
                this.pendingAppendState = null;
                this.pendingAppendTokens = 0;
            }
        }
    }

    @Override
    protected void releaseTemporary(QwenExecutionContext context) {
        this.pendingAppendState = null;
        this.pendingAppendTokens = 0;
    }

    private void runAttentionCausal(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        QwenConfig config = context.plan().weights().config();
        AttentionKvState state = attentionState(context, instruction);
        gpu().attentionCausalNvfp4(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        state.keyCacheAddress(),
                        state.valueCacheAddress(),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        config.numAttentionHeads(),
                        config.numKeyValueHeads(),
                        config.attentionHeadDim(),
                        state.length(),
                        context.startPosition(),
                        context.inputTokenCount() == 1 ? state.decodeScratchAddress(config.numAttentionHeads()) : 0);
    }

    private void runResidualAdd(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        gpu().residualAddBf16(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        instruction.outputWidth());
    }

    private void runSwiGlu(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        gpu().swiGluBf16(
                        input(context, instruction, 0),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        context.plan().weights().config().intermediateSize());
    }

    private static QwenGdnSequenceState sequenceState(
            QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        Object state = context.sequenceState().recurrentState();
        if (state instanceof GdnSequenceStates states) {
            return states.forLayer(instruction.layerIndex());
        }
        if (!(state instanceof QwenGdnSequenceState gdnState)) {
            throw new IllegalStateException("sequence is missing its persistent GDN state");
        }
        return gdnState;
    }

    private static AttentionKvState attentionState(
            QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        Object state = context.sequenceState().kvCacheState();
        if (!(state instanceof AttentionSequenceStates states)) {
            throw new IllegalStateException("sequence is missing its persistent attention KV state");
        }
        return states.forLayer(instruction.layerIndex());
    }

    private static long input(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction, int index) {
        return context.workspace().address(instruction.inputBuffers().get(index));
    }

    private static long output(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction, int index) {
        return context.workspace().address(instruction.outputBuffers().get(index));
    }
}
