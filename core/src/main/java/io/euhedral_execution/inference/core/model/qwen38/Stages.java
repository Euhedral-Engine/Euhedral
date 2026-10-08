package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.ScratchUse;
import io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Config;
import io.euhedral_execution.inference.core.model.qwen38.speculative.DFlash2State;
import io.euhedral_execution.inference.core.runtime.graph.StageFrame;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import java.util.Objects;

/// The stage frames of 3.8 views: one class per instruction kind, each running its instruction (the stage's spec) on
/// the quantum bound to its graph. A frame is built once per reusable graph with its instruction and weights; each
/// quantum rebinds only the graph's quantum, and the frame launches its operation on that quantum's stream.
public final class Stages {

    private Stages() {}

    /// The stage frame of one instruction.
    public static Stage create(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        return switch (instruction.kind()) {
            case EMBEDDING -> new Embed(graph, instruction, gpu);
            case WEIGHT_TRANSFER -> new WeightTransfer(graph, instruction, gpu);
            case RMS_NORM, RMS_NORM_UNIT_OFFSET -> new RmsNorm(graph, instruction, gpu);
            case Q3_LINEAR, Q4_LINEAR, Q5_LINEAR, BF16_LINEAR -> new Linear(graph, instruction, gpu);
            case Q3_GATE_UP_SWIGLU -> new GateUpSwiGlu(graph, instruction, gpu);
            case RESIDUAL_RMS_NORM -> new ResidualNorm(graph, instruction, gpu);
            case GDN_PROJECT_CONTROL -> new GdnProjectControl(graph, instruction, gpu);
            case MTP_STEM -> new MtpStem(graph, instruction, gpu);
            case GDN_CONTROL -> new GdnControl(graph, instruction, gpu);
            case GDN_CONVOLUTION -> new GdnConvolution(graph, instruction, gpu);
            case GDN_RECURRENCE -> new GdnRecurrence(graph, instruction, gpu);
            case GDN_GATED_RMS_NORM -> new GdnGatedNorm(graph, instruction, gpu);
            case ATTENTION_QK_NORM_ROPE -> new QkNormRope(graph, instruction, gpu);
            case ATTENTION_KV_APPEND -> new KvAppend(graph, instruction, gpu);
            case ATTENTION_CAUSAL -> new CausalAttention(graph, instruction, gpu);
            case RESIDUAL_ADD -> new ResidualAdd(graph, instruction, gpu);
            case SWIGLU -> new SwiGlu(graph, instruction, gpu);
            case DFLASH_TAP -> new DFlash2.Tap(graph, instruction, gpu);
            case DFLASH_LINEAR -> new DFlash2.Linear(graph, instruction, gpu);
            case DFLASH_RMS_NORM -> new DFlash2.RmsNorm(graph, instruction, gpu);
            case DFLASH_CONV -> new DFlash2.Conv(graph, instruction, gpu);
            case DFLASH_CONTEXT_KV -> new DFlash2.ContextKv(graph, instruction, gpu);
            case DFLASH_BLOCK_QK -> new DFlash2.BlockQk(graph, instruction, gpu);
            case DFLASH_ATTENTION -> new DFlash2.Attention(graph, instruction, gpu);
            case DFLASH_SWIGLU -> new DFlash2.SwiGlu(graph, instruction, gpu);
            case DFLASH_LM_HEAD -> new DFlash2.LmHead(graph, instruction, gpu);
            case DFLASH_TOPK -> new DFlash2.TopK(graph, instruction, gpu);
            case DFLASH_SELECT -> new DFlash2.Select(graph, instruction, gpu);
        };
    }

    /// The base of every 3.8 stage: its instruction, the GPU it launches on, and the quantum it reads.
    public abstract static class Stage extends StageFrame {

        private final ExecutionPlan.Instruction instruction;
        private final ExecutionGpu gpu;
        /// The scratch route its shape declared for it, or null ([Shape#scratchUse]).
        ScratchUse scratchUse;

        protected Stage(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, Objects.requireNonNull(instruction, "instruction").id());
            this.instruction = instruction;
            this.gpu = Objects.requireNonNull(gpu, "gpu");
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
            // A stage the shape declared as a scratch user takes the workspace's one scratch buffer, which the
            // shape's edges (and the workspace's, across graphs) give it alone.
            long scratch = this.scratchUse != null ? context.workspace().scratchAddress() : 0;
            if (scratch != 0) this.gpu.withScratch(scratch, context.workspace().scratchBytes(), () -> run(context));
            else run(context);
        }

        private void run(Quantum context) {
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

        protected static long input(Quantum context, ExecutionPlan.Instruction instruction, int index) {
            return context.workspace().address(instruction.inputBuffers().get(index));
        }

        protected static long output(Quantum context, ExecutionPlan.Instruction instruction, int index) {
            return context.workspace().address(instruction.outputBuffers().get(index));
        }

        protected static GdnState sequenceState(Quantum context, ExecutionPlan.Instruction instruction) {
            Object state = context.sequenceState().recurrentState();
            if (state instanceof GdnStates states) {
                return states.forLayer(instruction.layerIndex());
            }
            if (!(state instanceof GdnState gdnState)) {
                throw new IllegalStateException("sequence is missing its persistent GDN state");
            }
            return gdnState;
        }

        protected static AttentionKvState attentionState(Quantum context, ExecutionPlan.Instruction instruction) {
            Object state = context.sequenceState().kvCacheState();
            if (!(state instanceof AttentionStates states)) {
                throw new IllegalStateException("sequence is missing its persistent attention KV state");
            }
            return states.forLayer(instruction.layerIndex());
        }
    }

    /// Runs the embedding lookup over the token IDs the quantum uploaded at admission.
    public static final class Embed extends Stage {

        Embed(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
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

    /// Copies a host-backed weight into its staging slot on the pool's transfer lane, so the copy engine
    /// overlaps the compute lanes. Its consumer awaits it like any cross-lane predecessor.
    public static final class WeightTransfer extends Stage {
        private final long source;
        private final long destination;
        private final long byteSize;

        WeightTransfer(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
            TensorHandle weight = instruction.weight();
            this.source = weight.hostAddress();
            this.destination = weight.deviceAddress();
            this.byteSize = weight.byteSize();
        }

        @Override
        protected boolean transfers() {
            return true;
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
            gpu().copyHostWeightsToDevice(this.destination, this.source, this.byteSize);
        }
    }

    /// Runs the standalone BF16 RMSNorm instruction.
    public static final class RmsNorm extends Stage {

        RmsNorm(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
            boolean finalNorm = instruction.outputBuffers().contains(ExecutionPlan.Buffer.FINAL_NORMALIZED);
            int rows = finalNorm ? context.logitsRowCount() : context.inputTokenCount();
            if (finalNorm && context.seedsDraft() && context.plan().drafts()) {
                // Speculative drafting reads every row's post-final-norm hidden (docs/MTP_CONTRACT.md §2).
                var states = (AttentionStates) context.sequenceState().kvCacheState();
                int all = context.inputTokenCount();
                normalize(
                        instruction,
                        context.workspace().address(instruction.inputBuffers().getFirst()),
                        states.draftSeedRows(all, instruction.outputWidth()),
                        all,
                        (float) context.plan().weights().config().rmsNormEpsilon());
            }
            if (rows == 0) return;
            long input = context.shape().hasFirstLayer()
                    ? context.workspace().address(instruction.inputBuffers().getFirst())
                    : context.workspace().hiddenStateAddress();
            if (finalNorm && rows != context.inputTokenCount()) {
                input += (long) (context.inputTokenCount() - 1) * instruction.inputWidth() * Short.BYTES;
            }
            long output = context.shape().hasFirstLayer()
                    ? context.workspace().address(instruction.outputBuffers().getFirst())
                    : context.workspace().normalizedStateAddress();
            float epsilon = (float) context.plan().weights().config().rmsNormEpsilon();
            normalize(instruction, input, output, rows, epsilon);
            if (finalNorm && context.kind() == Quantum.ExecutionKind.DRAFT) {
                // The last MTP row's post-mtp.norm hidden seeds the next recursive draft row.
                var states = (AttentionStates) context.sequenceState().kvCacheState();
                gpu().copyDeviceToDevice(
                                states.draftRecursionHidden(instruction.outputWidth()),
                                output + (long) (rows - 1) * instruction.outputWidth() * Short.BYTES,
                                (long) instruction.outputWidth() * Short.BYTES);
            }
        }

        private void normalize(
                ExecutionPlan.Instruction instruction, long input, long output, int rows, float epsilon) {
            if (instruction.kind() == ExecutionPlan.Kind.RMS_NORM_UNIT_OFFSET) {
                gpu().rmsNormUnitOffsetBf16(
                                input, instruction.weightAddress(), output, rows, instruction.outputWidth(), epsilon);
            } else {
                gpu().rmsNormBf16(input, instruction.weightAddress(), output, rows, instruction.outputWidth(), epsilon);
            }
        }
    }

    /// Runs one independent quantized or BF16 projection instruction.
    public static final class Linear extends Stage {

        Linear(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
            boolean logits = instruction.outputBuffers().contains(ExecutionPlan.Buffer.LOGITS);
            int rows = logits ? context.logitsRowCount() : context.inputTokenCount();
            if (rows == 0) return;
            long input = context.shape().hasFirstLayer()
                    ? context.workspace().address(instruction.inputBuffers().getFirst())
                    : context.workspace().normalizedStateAddress();
            long output = context.shape().hasFirstLayer()
                    ? context.workspace().address(instruction.outputBuffers().getFirst())
                    : context.workspace().projectionAddress(instruction.outputBufferIndex());
            if (instruction.weights().size() == 1
                    && instruction.weightFormat() == WeightFormat.NVFP4
                    && (instruction.kind() == ExecutionPlan.Kind.Q3_LINEAR
                            || instruction.kind() == ExecutionPlan.Kind.Q4_LINEAR
                            || instruction.kind() == ExecutionPlan.Kind.Q5_LINEAR)) {
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
            boolean quantized = instruction.kind() == ExecutionPlan.Kind.Q3_LINEAR
                    || instruction.kind() == ExecutionPlan.Kind.Q4_LINEAR
                    || instruction.kind() == ExecutionPlan.Kind.Q5_LINEAR;
            if (context.kind() == Quantum.ExecutionKind.VERIFY
                    && rows > 1
                    && !(quantized && gpu().rowExactQuantizedLinears())) {
                // Row-exact verification: each row takes exactly the route and kernel one-row decode does.
                int outputBytes = instruction.kind() == ExecutionPlan.Kind.BF16_LINEAR ? Float.BYTES : Short.BYTES;
                for (int row = 0; row < rows; row++) {
                    launch(
                            instruction,
                            input + (long) row * instruction.inputWidth() * Short.BYTES,
                            output + (long) row * instruction.outputWidth() * outputBytes,
                            1);
                }
            } else launch(instruction, input, output, rows);
            if (logits) context.logitsProduced(output);
        }

        private void launch(ExecutionPlan.Instruction instruction, long input, long output, int rows) {
            switch (instruction.kind()) {
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
                            "linear stage received non-linear instruction: " + instruction.kind());
            }
        }
    }

    /// The fused gate/up projection and SwiGLU of a prefill FFN region.
    public static final class GateUpSwiGlu extends Stage {

        GateUpSwiGlu(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
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
    }

    /// The rounded residual add and RMSNorm of a region view, in one launch.
    public static final class ResidualNorm extends Stage {

        ResidualNorm(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
            gpu().residualRmsNormBf16(
                            input(context, instruction, 0),
                            input(context, instruction, 1),
                            instruction.weightAddress(),
                            output(context, instruction, 0),
                            output(context, instruction, 1),
                            context.inputTokenCount(),
                            instruction.outputWidth(),
                            (float) context.plan().weights().config().rmsNormEpsilon());
        }
    }

    /// The joint GDN A/B projection and control of a region view.
    public static final class GdnProjectControl extends Stage {

        GdnProjectControl(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
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
        }
    }

    /// MTP stem (docs/MTP_CONTRACT.md §2): row r of the packed output is
    /// [RMSNorm₁₊w(embedding r, embedding_norm); RMSNorm₁₊w(seed r, hidden_norm)], the seed rows coming from
    /// the context (base hidden rows for catch-up, the previous MTP hidden for a recursive row).
    public static final class MtpStem extends Stage {

        MtpStem(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
            int rows = context.inputTokenCount();
            int hidden = instruction.inputWidth();
            long rowBytes = (long) hidden * Short.BYTES;
            long packed = output(context, instruction, 0);
            long normed = context.workspace().address(ExecutionPlan.Buffer.MTP_NORMED);
            float epsilon = (float) context.plan().weights().config().rmsNormEpsilon();
            gpu().rmsNormUnitOffsetBf16(
                            input(context, instruction, 0),
                            instruction.weightAddress(0),
                            normed,
                            rows,
                            hidden,
                            epsilon);
            gpu().copyRowsDeviceToDevice(packed, 2 * rowBytes, normed, rowBytes, rows);
            gpu().rmsNormUnitOffsetBf16(
                            context.draftSeedAddress(), instruction.weightAddress(1), normed, rows, hidden, epsilon);
            gpu().copyRowsDeviceToDevice(packed + rowBytes, 2 * rowBytes, normed, rowBytes, rows);
        }
    }

    /// The GDN control (alpha and beta) of an unfused view.
    public static final class GdnControl extends Stage {

        GdnControl(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
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
    }

    /// The GDN convolution over the sequence's persistent convolution state. A verification checkpoints the state
    /// and its input rows first; a previous verification that committed only part of its rows leaves a replay this
    /// stage runs before its own rows.
    public static final class GdnConvolution extends Stage {

        /// The GDN state a verification checkpointed in this stage; its commit sets the replay.
        private GdnState pendingSpeculativeState;

        GdnConvolution(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
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
                                state.recurrentStateAddress(),
                                speculative.recurrentCheckpoint(),
                                state.recurrentBytes());
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

        /// A verification whose GDN state ran past its committed rows leaves a replay for the next quantum.
        @Override
        protected void commit() {
            GdnState speculative = this.pendingSpeculativeState;
            if (speculative != null) {
                int committed = context().committedRowCount();
                speculative.setPendingReplayRows(committed < context().inputTokenCount() ? committed : 0);
            }
        }

        @Override
        protected void releaseTemporary(Quantum context) {
            this.pendingSpeculativeState = null;
        }
    }

    /// The GDN recurrence over the sequence's persistent recurrent state; a verification checkpoints it first.
    public static final class GdnRecurrence extends Stage {

        GdnRecurrence(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
            Qwen38Config config = context.plan().weights().config();
            float outputScale = (float) (1.0 / Math.sqrt(config.linearKeyHeadDim()));
            if (context.kind() == Quantum.ExecutionKind.VERIFY) {
                GdnState state = sequenceState(context, instruction);
                var speculative = state.speculative();
                long rowBytes = (long) context.inputTokenCount() * config.linearNumValueHeads() * Float.BYTES;
                gpu().copyDeviceToDevice(
                                speculative.recurrentCheckpoint(),
                                state.recurrentStateAddress(),
                                state.recurrentBytes());
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
    }

    /// The GDN gated RMSNorm.
    public static final class GdnGatedNorm extends Stage {

        GdnGatedNorm(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
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
    }

    /// The attention Q/K RMSNorm and RoPE.
    public static final class QkNormRope extends Stage {

        QkNormRope(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
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
    }

    /// Appends the quantum's keys and values to the sequence's KV cache. The rows are submitted at once and
    /// published (committed) only when the quantum's device work retired; an uncommitted append never
    /// becomes visible.
    public static final class KvAppend extends Stage {

        private AttentionKvState pendingAppendState;

        KvAppend(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
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

        /// Publishes the appended rows once the quantum's device work has retired.
        @Override
        protected void commit() {
            if (this.pendingAppendState != null)
                this.pendingAppendState.commitSubmitted(context().committedRowCount());
        }

        @Override
        protected void releaseTemporary(Quantum context) {
            AttentionKvState state = this.pendingAppendState;
            this.pendingAppendState = null;
            // A committed frontier is unaffected; an uncommitted one never becomes visible.
            if (state != null) state.discardSubmitted();
        }
    }

    /// Causal attention over the sequence's KV cache, including the rows this quantum appended.
    public static final class CausalAttention extends Stage {

        CausalAttention(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
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
    }

    /// Adds a residual into the hidden state.
    public static final class ResidualAdd extends Stage {

        ResidualAdd(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
            gpu().residualAddBf16(
                            input(context, instruction, 0),
                            input(context, instruction, 1),
                            output(context, instruction, 0),
                            context.inputTokenCount(),
                            instruction.outputWidth());
        }
    }

    /// The FFN's SwiGLU of an unfused view.
    public static final class SwiGlu extends Stage {

        SwiGlu(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        @Override
        protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
            gpu().swiGluBf16(
                            input(context, instruction, 0),
                            output(context, instruction, 0),
                            context.inputTokenCount(),
                            context.plan().weights().config().intermediateSize());
        }
    }

    /// The DFlash2 stages (docs/DFLASH2.md): the target's taps, and the drafter's context and block stages. Each is
    /// shown to the observer right after it was submitted.
    public abstract static class DFlash2 extends Stage {

        /// Sees each DFlash2 stage right after it was submitted (tests compare the drafter's intermediates with the
        /// reference this way); null in production.
        public interface Observer {
            void submitted(Quantum context, ExecutionPlan.Instruction instruction);
        }

        private static volatile Observer observer;

        public static void observe(Observer stages) {
            observer = stages;
        }

        DFlash2(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
            super(graph, instruction, gpu);
        }

        /// Launches the stage's DFlash2 operation.
        protected abstract void submit(Quantum context, ExecutionPlan.Instruction instruction, int rows);

        @Override
        protected final void perform(Quantum context, ExecutionPlan.Instruction instruction) {
            submit(context, instruction, context.inputTokenCount());
            Observer stages = observer;
            if (stages != null) stages.submitted(context, instruction);
        }

        static DFlash2State state(Quantum context) {
            DFlash2Config config = context.plan().weights().dflash2().config();
            return ((AttentionStates) context.sequenceState().kvCacheState()).dflash2(config);
        }

        static DFlash2Config config(Quantum context) {
            return context.plan().weights().dflash2().config();
        }

        /// Copies the target's hidden rows after one tapped layer into the sequence's DFlash2 tap rows, in quanta
        /// that seed drafting.
        public static final class Tap extends DFlash2 {
            Tap(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
                super(graph, instruction, gpu);
            }

            @Override
            protected void submit(Quantum context, ExecutionPlan.Instruction instruction, int rows) {
                if (!context.seedsDraft()) return;
                DFlash2Config config = config(context);
                long hiddenBytes = (long) config.hiddenSize() * Short.BYTES;
                gpu().copyRowsDeviceToDevice(
                                state(context).taps(rows) + instruction.outputBufferIndex() * hiddenBytes,
                                (long) config.tapWidth() * Short.BYTES,
                                input(context, instruction, 0),
                                hiddenBytes,
                                rows);
            }
        }

        /// A drafter projection; with no input buffer it reads the sequence's tap rows.
        public static final class Linear extends DFlash2 {
            Linear(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
                super(graph, instruction, gpu);
            }

            @Override
            protected void submit(Quantum context, ExecutionPlan.Instruction instruction, int rows) {
                long in = instruction.inputBuffers().isEmpty()
                        ? state(context).taps(rows)
                        : input(context, instruction, 0);
                long out = output(context, instruction, 0);
                if (instruction.weightFormat() == WeightFormat.NVFP4)
                    gpu().linearNvfp4Bf16(
                                    in,
                                    instruction.weightAddress(),
                                    out,
                                    rows,
                                    instruction.inputWidth(),
                                    instruction.outputWidth(),
                                    instruction.weightByteSize());
                else
                    gpu().dflashLinearBf16(
                                    in,
                                    instruction.weightAddress(),
                                    out,
                                    rows,
                                    instruction.inputWidth(),
                                    instruction.outputWidth());
            }
        }

        public static final class RmsNorm extends DFlash2 {
            RmsNorm(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
                super(graph, instruction, gpu);
            }

            @Override
            protected void submit(Quantum context, ExecutionPlan.Instruction instruction, int rows) {
                gpu().dflashRmsNormBf16(
                                input(context, instruction, 0),
                                instruction.weightAddress(),
                                output(context, instruction, 0),
                                rows,
                                instruction.outputWidth(),
                                config(context).rmsNormEpsilon());
            }
        }

        /// The dynamic convolution's prepare (`outputBufferIndex` 0) or finish (1) kernel.
        public static final class Conv extends DFlash2 {
            Conv(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
                super(graph, instruction, gpu);
            }

            @Override
            protected void submit(Quantum context, ExecutionPlan.Instruction instruction, int rows) {
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
        }

        /// The drafter's keys and values of committed rows into its own cache. A context quantum's rows are the
        /// drafter's context once its last layer's keys and values are written.
        public static final class ContextKv extends DFlash2 {
            ContextKv(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
                super(graph, instruction, gpu);
            }

            @Override
            protected void submit(Quantum context, ExecutionPlan.Instruction instruction, int rows) {
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

            @Override
            protected void commit() {
                ExecutionPlan.Instruction instruction = instruction();
                Quantum context = context();
                if (instruction.layerIndex() != config(context).layers() - 1) return;
                state(context).commitContext(Math.toIntExact(context.startPosition() + context.inputTokenCount()));
            }
        }

        public static final class BlockQk extends DFlash2 {
            BlockQk(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
                super(graph, instruction, gpu);
            }

            @Override
            protected void submit(Quantum context, ExecutionPlan.Instruction instruction, int rows) {
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
        }

        public static final class Attention extends DFlash2 {
            Attention(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
                super(graph, instruction, gpu);
            }

            @Override
            protected void submit(Quantum context, ExecutionPlan.Instruction instruction, int rows) {
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
        }

        public static final class SwiGlu extends DFlash2 {
            SwiGlu(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
                super(graph, instruction, gpu);
            }

            @Override
            protected void submit(Quantum context, ExecutionPlan.Instruction instruction, int rows) {
                gpu().dflashSwiGluBf16(
                                input(context, instruction, 0),
                                output(context, instruction, 0),
                                rows,
                                instruction.outputWidth());
            }
        }

        /// The target's output head over the block's proposal rows: every row but the anchor proposes a token.
        public static final class LmHead extends DFlash2 {
            LmHead(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
                super(graph, instruction, gpu);
            }

            @Override
            protected void submit(Quantum context, ExecutionPlan.Instruction instruction, int rows) {
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
        }

        public static final class TopK extends DFlash2 {
            TopK(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
                super(graph, instruction, gpu);
            }

            @Override
            protected void submit(Quantum context, ExecutionPlan.Instruction instruction, int rows) {
                gpu().dflashTopKBf16(
                                input(context, instruction, 0),
                                rows - 1,
                                instruction.inputWidth(),
                                output(context, instruction, 0),
                                output(context, instruction, 1),
                                output(context, instruction, 2));
            }
        }

        /// The selector's path over the candidates: the block's proposal, queued for the host.
        public static final class Select extends DFlash2 {
            Select(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
                super(graph, instruction, gpu);
            }

            @Override
            protected void submit(Quantum context, ExecutionPlan.Instruction instruction, int rows) {
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
        }
    }
}
