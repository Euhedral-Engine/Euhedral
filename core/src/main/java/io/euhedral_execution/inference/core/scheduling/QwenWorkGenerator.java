package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.impl.FrameFactory;
import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.frames.EmbeddingFrame;
import io.euhedral_execution.inference.core.scheduling.frames.LinearFrame;
import io.euhedral_execution.inference.core.scheduling.frames.QwenGpuOperationFrame;
import io.euhedral_execution.inference.core.scheduling.frames.QwenInstructionFrame;
import io.euhedral_execution.inference.core.scheduling.frames.RmsNormFrame;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;

/// Tracks dependency readiness and materializes pooled instruction frames on the serialized source path.
public final class QwenWorkGenerator {

    private static final int FRAME_POOL_CAPACITY = 256;
    private static final long FRAME_POOL_PASSWORD = 0x5147454eL;

    private final QwenExecutionPlan plan;
    private final ExecutionGpu gpu;
    private final QwenExecutionRunner workQueue;
    private final Consumer<? super QwenExecutionContext> terminalConsumer;
    private final FrameManager<QwenExecutionContext, EmbeddingFrame> embeddingFrames;
    private final Map<Integer, FrameManager<QwenExecutionContext, RmsNormFrame>> rmsNormFrames;
    private final Map<Integer, FrameManager<QwenExecutionContext, LinearFrame>> linearFrames;
    private final Map<Integer, FrameManager<QwenExecutionContext, QwenGpuOperationFrame>> operationFrames;
    // These fields are owned exclusively by the serialized source pull/request path.
    private int nextInstructionId;
    private int pendingInstructionId = -1;
    private QwenExecutionContext pendingContext;
    private QwenInstructionFrame pendingFrame;

    QwenWorkGenerator(
            QwenExecutionPlan plan,
            ExecutionGpu gpu,
            QwenExecutionRunner workQueue,
            Consumer<? super QwenExecutionContext> terminalConsumer) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.workQueue = Objects.requireNonNull(workQueue, "workQueue");
        this.terminalConsumer = Objects.requireNonNull(terminalConsumer, "terminalConsumer");
        this.embeddingFrames = embeddingManager(plan.instructions().getFirst());
        Map<Integer, FrameManager<QwenExecutionContext, RmsNormFrame>> rmsManagers = new HashMap<>();
        Map<Integer, FrameManager<QwenExecutionContext, LinearFrame>> linearManagers = new HashMap<>();
        Map<Integer, FrameManager<QwenExecutionContext, QwenGpuOperationFrame>> operationManagers = new HashMap<>();
        for (QwenExecutionPlan.Instruction instruction : plan.instructions()) {
            if (instruction.kind() == QwenExecutionPlan.Kind.RMS_NORM
                    || instruction.kind() == QwenExecutionPlan.Kind.RMS_NORM_UNIT_OFFSET) {
                rmsManagers.put(instruction.id(), rmsNormManager(instruction));
            } else if (instruction.kind() == QwenExecutionPlan.Kind.Q3_LINEAR
                    || instruction.kind() == QwenExecutionPlan.Kind.Q4_LINEAR
                    || instruction.kind() == QwenExecutionPlan.Kind.Q5_LINEAR
                    || instruction.kind() == QwenExecutionPlan.Kind.BF16_LINEAR) {
                linearManagers.put(instruction.id(), linearManager(instruction));
            } else if (instruction.kind() == QwenExecutionPlan.Kind.GDN_CONTROL
                    || instruction.kind() == QwenExecutionPlan.Kind.GDN_CONVOLUTION
                    || instruction.kind() == QwenExecutionPlan.Kind.GDN_RECURRENCE
                    || instruction.kind() == QwenExecutionPlan.Kind.GDN_GATED_RMS_NORM
                    || instruction.kind() == QwenExecutionPlan.Kind.FFN_STREAMED
                    || instruction.kind() == QwenExecutionPlan.Kind.Q3_GATE_UP_SWIGLU
                    || instruction.kind() == QwenExecutionPlan.Kind.RESIDUAL_RMS_NORM
                    || instruction.kind() == QwenExecutionPlan.Kind.GDN_PROJECT_CONTROL
                    || instruction.kind() == QwenExecutionPlan.Kind.RESIDUAL_ADD
                    || instruction.kind() == QwenExecutionPlan.Kind.SWIGLU
                    || instruction.kind() == QwenExecutionPlan.Kind.ATTENTION_QK_NORM_ROPE
                    || instruction.kind() == QwenExecutionPlan.Kind.ATTENTION_PRODUCERS
                    || instruction.kind() == QwenExecutionPlan.Kind.ATTENTION_KV_APPEND
                    || instruction.kind() == QwenExecutionPlan.Kind.ATTENTION_CAUSAL) {
                operationManagers.put(instruction.id(), operationManager(instruction));
            }
        }
        this.rmsNormFrames = Map.copyOf(rmsManagers);
        this.linearFrames = Map.copyOf(linearManagers);
        this.operationFrames = Map.copyOf(operationManagers);
    }

    private FrameManager<QwenExecutionContext, EmbeddingFrame> embeddingManager(
            QwenExecutionPlan.Instruction instruction) {
        var manager = new FrameManager<QwenExecutionContext, EmbeddingFrame>(FRAME_POOL_CAPACITY, FRAME_POOL_PASSWORD);
        manager.setFactory(new FrameFactory<>(
                (idHash, context) -> new EmbeddingFrame(idHash, manager, context, instruction, this.gpu, this),
                (context, frame) -> frame.replace(context)));
        return manager;
    }

    private FrameManager<QwenExecutionContext, RmsNormFrame> rmsNormManager(QwenExecutionPlan.Instruction instruction) {
        var manager = new FrameManager<QwenExecutionContext, RmsNormFrame>(FRAME_POOL_CAPACITY, FRAME_POOL_PASSWORD);
        manager.setFactory(new FrameFactory<>(
                (idHash, context) -> new RmsNormFrame(idHash, manager, context, instruction, this.gpu, this),
                (context, frame) -> frame.replace(context)));
        return manager;
    }

    private FrameManager<QwenExecutionContext, LinearFrame> linearManager(QwenExecutionPlan.Instruction instruction) {
        var manager = new FrameManager<QwenExecutionContext, LinearFrame>(FRAME_POOL_CAPACITY, FRAME_POOL_PASSWORD);
        manager.setFactory(new FrameFactory<>(
                (idHash, context) -> new LinearFrame(idHash, manager, context, instruction, this.gpu, this),
                (context, frame) -> frame.replace(context)));
        return manager;
    }

    private FrameManager<QwenExecutionContext, QwenGpuOperationFrame> operationManager(
            QwenExecutionPlan.Instruction instruction) {
        var manager =
                new FrameManager<QwenExecutionContext, QwenGpuOperationFrame>(FRAME_POOL_CAPACITY, FRAME_POOL_PASSWORD);
        manager.setFactory(new FrameFactory<>(
                (idHash, context) -> new QwenGpuOperationFrame(idHash, manager, context, instruction, this.gpu, this),
                (context, frame) -> frame.replace(context)));
        return manager;
    }

    void start(QwenExecutionContext context) {
        enqueueReady(context, this.plan.instructions().getFirst().id());
    }

    public void completed(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction, Throwable error) {
        if (error != null) context.fail(error);
        if (!context.hasFailureOrCancellation()) {
            try {
                for (int successorId : this.plan.successors(instruction.id())) {
                    if (context.dependencyCompleted(successorId)) {
                        enqueueReady(context, successorId);
                        if (context.hasFailureOrCancellation()) break;
                    }
                }
            } catch (RuntimeException | Error failure) {
                context.fail(failure);
            }
        }
        if (context.releaseWork()) context.finish(this.terminalConsumer, this.gpu);
    }

    private void enqueueReady(QwenExecutionContext context, int instructionId) {
        context.reserveWork();
        try {
            if (this.workQueue.offerReady(instructionId, context)) return;
            context.fail(new IllegalStateException("Euhedral work queue rejected a ready instruction"));
        } catch (RuntimeException | Error failure) {
            context.fail(failure);
        }
        if (context.releaseWork()) context.finish(this.terminalConsumer, this.gpu);
    }

    long drain(
            Consumer<? super AbstractFrame> consumer, Function<AbstractFrame, Boolean> stopCondition, long requested) {
        Objects.requireNonNull(consumer, "consumer");
        Objects.requireNonNull(stopCondition, "stopCondition");
        if (requested <= 0) return 0;

        long delivered = 0;
        while (delivered < requested) {
            int instructionId = this.pendingFrame == null ? findReadyInstruction() : this.pendingInstructionId;
            if (instructionId < 0) break;

            QwenExecutionContext context =
                    this.pendingFrame == null ? this.workQueue.peekReady(instructionId) : this.pendingContext;
            if (context == null) {
                advanceInstruction(instructionId);
                continue;
            }

            QwenInstructionFrame frame = this.pendingFrame;
            if (context.hasFailureOrCancellation()) {
                discardReady(instructionId, context, frame);
                continue;
            }

            if (frame == null) {
                try {
                    frame = create(context, this.plan.instructions().get(instructionId));
                } catch (RuntimeException | Error failure) {
                    context.fail(failure);
                    discardReady(instructionId, context, null);
                    continue;
                }
                if (context.hasFailureOrCancellation()) {
                    discardReady(instructionId, context, frame);
                    continue;
                }
            }

            boolean stop;
            try {
                stop = stopCondition.apply(frame);
            } catch (RuntimeException | Error failure) {
                if (this.pendingFrame != frame) frame.abandonBeforePublication();
                throw failure;
            }
            if (stop) {
                this.pendingInstructionId = instructionId;
                this.pendingContext = context;
                this.pendingFrame = frame;
                return delivered;
            }

            removeReady(instructionId, context);
            clearPending(frame);
            advanceInstruction(instructionId);
            consumer.accept(frame);
            delivered++;
        }
        return delivered;
    }

    private int findReadyInstruction() {
        int instructionCount = this.plan.instructions().size();
        for (int offset = 0; offset < instructionCount; offset++) {
            int candidate = (this.nextInstructionId + offset) % instructionCount;
            if (this.workQueue.peekReady(candidate) != null) return candidate;
        }
        return -1;
    }

    private QwenInstructionFrame create(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        return switch (instruction.kind()) {
            case EMBEDDING -> this.embeddingFrames.getOrCreate(context, FRAME_POOL_PASSWORD);
            case RMS_NORM, RMS_NORM_UNIT_OFFSET ->
                Objects.requireNonNull(this.rmsNormFrames.get(instruction.id()))
                        .getOrCreate(context, FRAME_POOL_PASSWORD);
            case Q3_LINEAR, Q4_LINEAR, Q5_LINEAR, BF16_LINEAR ->
                Objects.requireNonNull(this.linearFrames.get(instruction.id()))
                        .getOrCreate(context, FRAME_POOL_PASSWORD);
            case GDN_CONTROL,
                    GDN_CONVOLUTION,
                    GDN_RECURRENCE,
                    GDN_GATED_RMS_NORM,
                    FFN_STREAMED,
                    Q3_GATE_UP_SWIGLU,
                    RESIDUAL_RMS_NORM,
                    GDN_PROJECT_CONTROL,
                    RESIDUAL_ADD,
                    SWIGLU,
                    ATTENTION_QK_NORM_ROPE,
                    ATTENTION_PRODUCERS,
                    ATTENTION_KV_APPEND,
                    ATTENTION_CAUSAL ->
                Objects.requireNonNull(this.operationFrames.get(instruction.id()))
                        .getOrCreate(context, FRAME_POOL_PASSWORD);
        };
    }

    private void discardReady(int instructionId, QwenExecutionContext context, QwenInstructionFrame frame) {
        removeReady(instructionId, context);
        clearPending(frame);
        if (frame != null) frame.abandonBeforePublication();
        advanceInstruction(instructionId);
        if (context.releaseWork()) context.finish(this.terminalConsumer, this.gpu);
    }

    private void removeReady(int instructionId, QwenExecutionContext expected) {
        if (this.workQueue.pollReady(instructionId) != expected) {
            throw new IllegalStateException("ready instruction queue changed outside its source owner");
        }
    }

    private void clearPending(QwenInstructionFrame frame) {
        if (this.pendingFrame == frame) {
            this.pendingFrame = null;
            this.pendingContext = null;
            this.pendingInstructionId = -1;
        }
    }

    private void advanceInstruction(int instructionId) {
        this.nextInstructionId = (instructionId + 1) % this.plan.instructions().size();
    }
}
