package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;
import io.euhedral_execution.inference.core.state.AttentionKvState;

/// Executes one stateful or elementwise GPU instruction using its immutable buffer operands.
public final class OperationFrame extends InstructionFrame {

    private AttentionKvState pendingAppendState;
    /// The GDN state a verification checkpointed in this frame; its commit sets the replay.
    private GdnState pendingSpeculativeState;

    OperationFrame(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        super(graph, instruction, gpu);
    }

    @Override
    protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
        switch (instruction.kind()) {
            case Q3_GATE_UP_SWIGLU -> {
                if (instruction.weightFormat() == WeightFormat.NVFP4)
                    gpu().nvfp4GateUpSwiGluBf16(
                                    input(context, instruction, 0),
                                    instruction.weightAddress(),
                                    output(context, instruction, 0),
                                    context.inputTokenCount(),
                                    instruction.inputWidth(),
                                    instruction.outputWidth(),
                                    instruction.weightByteSize());
                else
                    gpu().q3GateUpSwiGluBf16(
                                    input(context, instruction, 0),
                                    instruction.weightAddress(),
                                    output(context, instruction, 0),
                                    context.inputTokenCount(),
                                    instruction.inputWidth(),
                                    instruction.outputWidth(),
                                    instruction.weightByteSize(),
                                    instruction.weightLayout());
            }
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
            case MTP_STEM -> runMtpStem(context, instruction);
            case GDN_CONTROL -> runControl(context, instruction);
            case GDN_CONVOLUTION -> runConvolution(context, instruction);
            case GDN_RECURRENCE -> runRecurrence(context, instruction);
            case GDN_GATED_RMS_NORM -> runGatedRmsNorm(context, instruction);
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

    private void runControl(Quantum context, ExecutionPlan.Instruction instruction) {
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

    private void runConvolution(Quantum context, ExecutionPlan.Instruction instruction) {
        Qwen38Config config = context.plan().weights().config();
        int queryKeyWidth = 2 * config.linearNumKeyHeads() * config.linearKeyHeadDim();
        int valueWidth = config.linearNumValueHeads() * config.linearValueHeadDim();
        GdnState state = sequenceState(context, instruction);
        // A previous verification committed only part of its rows: rebuild this layer's state first.
        if (state.pendingReplayRows() > 0) replay(context, instruction, state);
        if (context.kind() == Quantum.ExecutionKind.VERIFY) {
            int rows = context.inputTokenCount();
            var speculative = state.speculative(
                    rows,
                    queryKeyWidth,
                    2 * valueWidth,
                    config.linearNumValueHeads(),
                    instruction.outputWidth(),
                    valueWidth);
            gpu().copyDeviceToDevice(
                            speculative.convolutionCheckpoint(),
                            state.convolutionStateAddress(),
                            state.convolutionBytes());
            gpu().copyDeviceToDevice(
                            speculative.queryKeyRows(),
                            input(context, instruction, 0),
                            (long) rows * queryKeyWidth * Short.BYTES);
            gpu().copyDeviceToDevice(
                            speculative.valueZRows(),
                            input(context, instruction, 1),
                            (long) rows * 2 * valueWidth * Short.BYTES);
            this.pendingSpeculativeState = state;
        }
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

    private void runRecurrence(Quantum context, ExecutionPlan.Instruction instruction) {
        Qwen38Config config = context.plan().weights().config();
        float outputScale = (float) (1.0 / Math.sqrt(config.linearKeyHeadDim()));
        if (context.kind() == Quantum.ExecutionKind.VERIFY) {
            GdnState state = sequenceState(context, instruction);
            var speculative = state.speculative();
            long rowBytes = (long) context.inputTokenCount() * config.linearNumValueHeads() * Float.BYTES;
            gpu().copyDeviceToDevice(
                            speculative.recurrentCheckpoint(), state.recurrentStateAddress(), state.recurrentBytes());
            gpu().copyDeviceToDevice(speculative.alphaRows(), input(context, instruction, 1), rowBytes);
            gpu().copyDeviceToDevice(speculative.betaRows(), input(context, instruction, 2), rowBytes);
        }
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

    /// ReplaySSM: restores this layer's state from the checkpoint taken before the last verification and
    /// re-runs its committed rows through the same convolution and recurrence kernels, row-exact, so the
    /// state is bit for bit what one-row decode of those rows leaves.
    private void replay(Quantum context, ExecutionPlan.Instruction instruction, GdnState state) {
        Qwen38Config config = context.plan().weights().config();
        int rows = state.pendingReplayRows();
        var speculative = state.speculative();
        int queryKeyWidth = 2 * config.linearNumKeyHeads() * config.linearKeyHeadDim();
        int valueWidth = config.linearNumValueHeads() * config.linearValueHeadDim();
        gpu().selectRowExact(true);
        try {
            gpu().copyDeviceToDevice(
                            state.convolutionStateAddress(),
                            speculative.convolutionCheckpoint(),
                            state.convolutionBytes());
            gpu().gdnConvolutionBf16(
                            speculative.queryKeyRows(),
                            speculative.valueZRows(),
                            instruction.weightAddress(0),
                            state.convolutionStateAddress(),
                            speculative.replayConvolved(),
                            rows,
                            queryKeyWidth,
                            valueWidth,
                            instruction.outputWidth(),
                            config.linearConvKernelDim());
            gpu().copyDeviceToDevice(
                            state.recurrentStateAddress(), speculative.recurrentCheckpoint(), state.recurrentBytes());
            gpu().gdnRecurrenceBf16(
                            speculative.replayConvolved(),
                            speculative.alphaRows(),
                            speculative.betaRows(),
                            state.recurrentStateAddress(),
                            speculative.replayOutput(),
                            rows,
                            config.linearNumKeyHeads(),
                            config.linearNumValueHeads(),
                            config.linearKeyHeadDim(),
                            config.linearValueHeadDim(),
                            (float) (1.0 / Math.sqrt(config.linearKeyHeadDim())));
        } finally {
            gpu().selectRowExact(context.kind() == Quantum.ExecutionKind.VERIFY);
        }
        state.setPendingReplayRows(0);
    }

    /// MTP stem (docs/MTP_CONTRACT.md §2): row r of the packed output is
    /// [RMSNorm₁₊w(embedding r, embedding_norm); RMSNorm₁₊w(seed r, hidden_norm)], the seed rows coming from
    /// the context (base hidden rows for catch-up, the previous MTP hidden for a recursive row).
    private void runMtpStem(Quantum context, ExecutionPlan.Instruction instruction) {
        int rows = context.inputTokenCount();
        int hidden = instruction.inputWidth();
        long rowBytes = (long) hidden * Short.BYTES;
        long packed = output(context, instruction, 0);
        long normed = context.workspace().address(ExecutionPlan.Buffer.MTP_NORMED);
        float epsilon = (float) context.plan().weights().config().rmsNormEpsilon();
        gpu().rmsNormUnitOffsetBf16(
                        input(context, instruction, 0), instruction.weightAddress(0), normed, rows, hidden, epsilon);
        gpu().copyRowsDeviceToDevice(packed, 2 * rowBytes, normed, rowBytes, rows);
        gpu().rmsNormUnitOffsetBf16(
                        context.draftSeedAddress(), instruction.weightAddress(1), normed, rows, hidden, epsilon);
        gpu().copyRowsDeviceToDevice(packed + rowBytes, 2 * rowBytes, normed, rowBytes, rows);
    }

    private void runGatedRmsNorm(Quantum context, ExecutionPlan.Instruction instruction) {
        Qwen38Config config = context.plan().weights().config();
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

    private void runAttentionQkNormRope(Quantum context, ExecutionPlan.Instruction instruction) {
        Qwen38Config config = context.plan().weights().config();
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
                        context.workspace().positionAddress(),
                        (float) config.rmsNormEpsilon(),
                        config.ropeTheta());
    }

    private void runAttentionKvAppend(Quantum context, ExecutionPlan.Instruction instruction) {
        Qwen38Config config = context.plan().weights().config();
        int queryWidth = config.numAttentionHeads() * config.attentionHeadDim();
        int keyValueWidth = config.numKeyValueHeads() * config.attentionHeadDim();
        AttentionKvState state = attentionState(context, instruction);
        // Retirement settles the reservation, including staging a failed launch leaves queued.
        this.pendingAppendState = state;
        state.prepareAppend(context.startPosition(), context.inputTokenCount());
        gpu().attentionKvAppendNvfp4(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        state.keyCacheAddress(),
                        state.valueCacheAddress(),
                        context.inputTokenCount(),
                        queryWidth,
                        keyValueWidth,
                        context.startPosition(),
                        context.workspace().positionAddress());
        state.appendSubmitted(context.inputTokenCount());
    }

    /// Publishes the appended rows once the quantum's device work has retired; a verification whose
    /// GDN state ran past its committed rows leaves a replay for the next quantum.
    @Override
    protected void commit() {
        if (this.pendingAppendState != null)
            this.pendingAppendState.commitSubmitted(context().committedRowCount());
        GdnState speculative = this.pendingSpeculativeState;
        if (speculative != null) {
            int committed = context().committedRowCount();
            speculative.setPendingReplayRows(committed < context().inputTokenCount() ? committed : 0);
        }
    }

    @Override
    protected void releaseTemporary(Quantum context) {
        this.pendingSpeculativeState = null;
        AttentionKvState state = this.pendingAppendState;
        this.pendingAppendState = null;
        // A committed frontier is unaffected; an uncommitted one never becomes visible.
        if (state != null) state.discardSubmitted();
    }

    private void runAttentionCausal(Quantum context, ExecutionPlan.Instruction instruction) {
        Qwen38Config config = context.plan().weights().config();
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
                        // The append stage submitted these rows earlier on this quantum's stream; they are
                        // readable here but not committed until the quantum retires.
                        state.submittedLength(),
                        context.startPosition(),
                        context.workspace().positionAddress(),
                        context.inputTokenCount() == 1
                                        || context.kind() == Quantum.ExecutionKind.VERIFY
                                        || draftRowTwins(context)
                                ? ((AttentionStates) context.sequenceState().kvCacheState())
                                        .decodeScratch(config.numAttentionHeads(), context.inputTokenCount())
                                : 0);
    }

    /// Small draft quanta (MTP catch-up, at most 8 rows) bring decode scratch, so their attention runs on
    /// the decode row twins: far more parallel over a long cache than a 32-row prefill tile (at 32K keys
    /// about 0.45-0.75 ms against 9.5 ms for 2-4 rows), and drafts only propose tokens.
    private static boolean draftRowTwins(Quantum context) {
        return context.kind() == Quantum.ExecutionKind.DRAFT && context.inputTokenCount() <= 8;
    }

    private void runResidualAdd(Quantum context, ExecutionPlan.Instruction instruction) {
        gpu().residualAddBf16(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        instruction.outputWidth());
    }

    private void runSwiGlu(Quantum context, ExecutionPlan.Instruction instruction) {
        gpu().swiGluBf16(
                        input(context, instruction, 0),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        context.plan().weights().config().intermediateSize());
    }

    private static GdnState sequenceState(Quantum context, ExecutionPlan.Instruction instruction) {
        Object state = context.sequenceState().recurrentState();
        if (state instanceof GdnStates states) {
            return states.forLayer(instruction.layerIndex());
        }
        if (!(state instanceof GdnState gdnState)) {
            throw new IllegalStateException("sequence is missing its persistent GDN state");
        }
        return gdnState;
    }

    private static AttentionKvState attentionState(Quantum context, ExecutionPlan.Instruction instruction) {
        Object state = context.sequenceState().kvCacheState();
        if (!(state instanceof AttentionStates states)) {
            throw new IllegalStateException("sequence is missing its persistent attention KV state");
        }
        return states.forLayer(instruction.layerIndex());
    }

    private static long input(Quantum context, ExecutionPlan.Instruction instruction, int index) {
        return context.workspace().address(instruction.inputBuffers().get(index));
    }

    private static long output(Quantum context, ExecutionPlan.Instruction instruction, int index) {
        return context.workspace().address(instruction.outputBuffers().get(index));
    }
}
