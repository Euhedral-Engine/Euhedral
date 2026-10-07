package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Config;
import io.euhedral_execution.inference.core.model.qwen38.speculative.DFlash2State;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;

/// Runs one DFlash2 instruction (docs/DFLASH2.md): the target's taps, and the drafter's context and block stages.
public final class DFlash2Frame extends InstructionFrame {

    /// Sees each DFlash2 stage right after it was submitted (tests compare the drafter's intermediates with the
    /// reference this way); null in production.
    public interface Observer {
        void submitted(Quantum context, ExecutionPlan.Instruction instruction);
    }

    private static volatile Observer observer;

    public static void observe(Observer stages) {
        observer = stages;
    }

    DFlash2Frame(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        super(graph, instruction, gpu);
    }

    private static DFlash2State state(Quantum context) {
        DFlash2Config config = context.plan().weights().dflash2().config();
        return ((AttentionStates) context.sequenceState().kvCacheState()).dflash2(config);
    }

    private static long input(Quantum context, ExecutionPlan.Instruction instruction, int index) {
        return context.workspace().address(instruction.inputBuffers().get(index));
    }

    private static long output(Quantum context, ExecutionPlan.Instruction instruction, int index) {
        return context.workspace().address(instruction.outputBuffers().get(index));
    }

    @Override
    protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
        submit(context, instruction);
        Observer stages = observer;
        if (stages != null) stages.submitted(context, instruction);
    }

    private void submit(Quantum context, ExecutionPlan.Instruction instruction) {
        int rows = context.inputTokenCount();
        switch (instruction.kind()) {
            case DFLASH_TAP -> {
                if (!context.seedsDraft()) return;
                DFlash2Config config = context.plan().weights().dflash2().config();
                long hiddenBytes = (long) config.hiddenSize() * Short.BYTES;
                gpu().copyRowsDeviceToDevice(
                                state(context).taps(rows) + instruction.outputBufferIndex() * hiddenBytes,
                                (long) config.tapWidth() * Short.BYTES,
                                input(context, instruction, 0),
                                hiddenBytes,
                                rows);
            }
            case DFLASH_LINEAR -> {
                long in = instruction.inputBuffers().isEmpty()
                        ? state(context).taps(rows)
                        : input(context, instruction, 0);
                linear(instruction, in, output(context, instruction, 0), rows);
            }
            case DFLASH_RMS_NORM ->
                gpu().dflashRmsNormBf16(
                                input(context, instruction, 0),
                                instruction.weightAddress(),
                                output(context, instruction, 0),
                                rows,
                                instruction.outputWidth(),
                                config(context).rmsNormEpsilon());
            case DFLASH_CONV -> {
                DFlash2Config config = config(context);
                gpu().dflashConvBf16(
                                input(context, instruction, 0),
                                input(context, instruction, 1),
                                instruction.weightAddress(),
                                output(context, instruction, 0),
                                rows,
                                config.hiddenSize(),
                                config.convGroupSize(),
                                config.convKernel(),
                                instruction.outputBufferIndex());
            }
            case DFLASH_CONTEXT_KV -> {
                DFlash2Config config = config(context);
                DFlash2State state = state(context);
                gpu().dflashContextKvBf16(
                                input(context, instruction, 0),
                                instruction.weightAddress(),
                                state.ringKeys(instruction.layerIndex()),
                                state.ringValues(instruction.layerIndex()),
                                rows,
                                context.workspace().positionAddress(),
                                config.slidingWindow(),
                                config.keyValueHeads(),
                                config.headDim(),
                                config.rmsNormEpsilon(),
                                config.ropeTheta());
            }
            case DFLASH_BLOCK_QK -> {
                DFlash2Config config = config(context);
                gpu().dflashBlockQkBf16(
                                input(context, instruction, 0),
                                input(context, instruction, 1),
                                instruction.weightAddress(0),
                                instruction.weightAddress(1),
                                output(context, instruction, 0),
                                output(context, instruction, 1),
                                rows,
                                context.workspace().positionAddress(),
                                config.attentionHeads(),
                                config.keyValueHeads(),
                                config.headDim(),
                                config.rmsNormEpsilon(),
                                config.ropeTheta());
            }
            case DFLASH_ATTENTION -> {
                DFlash2Config config = config(context);
                DFlash2State state = state(context);
                gpu().dflashAttentionBf16(
                                input(context, instruction, 0),
                                input(context, instruction, 1),
                                input(context, instruction, 2),
                                state.ringKeys(instruction.layerIndex()),
                                state.ringValues(instruction.layerIndex()),
                                output(context, instruction, 0),
                                output(context, instruction, 1),
                                rows,
                                context.workspace().positionAddress(),
                                config.slidingWindow(),
                                config.attentionHeads(),
                                config.keyValueHeads(),
                                config.headDim());
            }
            case DFLASH_SWIGLU ->
                gpu().dflashSwiGluBf16(
                                input(context, instruction, 0),
                                output(context, instruction, 0),
                                rows,
                                instruction.outputWidth());
            case DFLASH_LM_HEAD -> {
                // Every row but the anchor proposes a token.
                long in = input(context, instruction, 0) + (long) instruction.inputWidth() * Short.BYTES;
                long out = output(context, instruction, 0);
                if (instruction.weightFormat() == WeightFormat.NVFP4)
                    gpu().linearNvfp4Bf16(
                                    in,
                                    instruction.weightAddress(),
                                    out,
                                    rows - 1,
                                    instruction.inputWidth(),
                                    instruction.outputWidth(),
                                    instruction.weightByteSize());
                else
                    gpu().linearQ3Bf16(
                                    in,
                                    instruction.weightAddress(),
                                    out,
                                    rows - 1,
                                    instruction.inputWidth(),
                                    instruction.outputWidth(),
                                    instruction.weightByteSize(),
                                    instruction.weightLayout());
            }
            case DFLASH_TOPK ->
                gpu().dflashTopKBf16(
                                input(context, instruction, 0),
                                rows - 1,
                                instruction.inputWidth(),
                                output(context, instruction, 0),
                                output(context, instruction, 1),
                                output(context, instruction, 2));
            case DFLASH_SELECT -> {
                long hidden = input(context, instruction, 0) + (long) instruction.inputWidth() * Short.BYTES;
                long tokens = output(context, instruction, 0);
                long scores = output(context, instruction, 1);
                gpu().dflashSelectBf16(
                                hidden,
                                input(context, instruction, 1),
                                input(context, instruction, 2),
                                instruction.weightAddress(0),
                                instruction.weightAddress(1),
                                context.workspace().tokenIdsAddress(),
                                rows - 1,
                                instruction.inputWidth(),
                                tokens,
                                scores);
                if (context.proposal() != null)
                    context.proposal().queue(tokens, scores, input(context, instruction, 2));
            }
            default ->
                throw new IllegalArgumentException(
                        "DFlash2 frame received unsupported instruction: " + instruction.kind());
        }
    }

    private void linear(ExecutionPlan.Instruction instruction, long input, long output, int rows) {
        if (instruction.weightFormat() == WeightFormat.NVFP4)
            gpu().linearNvfp4Bf16(
                            input,
                            instruction.weightAddress(),
                            output,
                            rows,
                            instruction.inputWidth(),
                            instruction.outputWidth(),
                            instruction.weightByteSize());
        else
            gpu().dflashLinearBf16(
                            input,
                            instruction.weightAddress(),
                            output,
                            rows,
                            instruction.inputWidth(),
                            instruction.outputWidth());
    }

    private static DFlash2Config config(Quantum context) {
        return context.plan().weights().dflash2().config();
    }

    /// A context quantum's rows are the drafter's context once its last layer's keys and values are written.
    @Override
    protected void commit() {
        ExecutionPlan.Instruction instruction = instruction();
        Quantum context = context();
        if (instruction.kind() != ExecutionPlan.Kind.DFLASH_CONTEXT_KV
                || instruction.layerIndex() != config(context).layers() - 1) return;
        state(context).commitContext(Math.toIntExact(context.startPosition() + context.inputTokenCount()));
    }
}
