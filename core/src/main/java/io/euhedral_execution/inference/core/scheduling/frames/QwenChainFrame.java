package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenWorkGenerator;

/// Launches one decode quantum's whole instruction list, in plan order, on a single CUDA stream.
///
/// Stream order already enforces every kernel-to-kernel dependency, so the per-instruction host
/// completion round trip is unnecessary: the chain records one completion event after the last
/// launch and finalizes every instruction's temporary resources when it retires.
public final class QwenChainFrame extends QwenInstructionFrame {
    /// Programmatic dependent launch won 12 of 12 paired forks (about +1% decode) at a 64-token
    /// context but only 10 of 12 at 1024, so it is enabled only below this starting position.
    static final long PDL_MAX_START_POSITION = 1024;

    private final QwenWorkGenerator generator;
    private final QwenInstructionFrame[] performers;
    private int performed;

    public QwenChainFrame(
            long idHash,
            FrameManager<QwenExecutionContext, QwenChainFrame> recycler,
            QwenExecutionContext context,
            QwenExecutionPlan.Instruction instruction,
            ExecutionGpu gpu,
            QwenWorkGenerator generator) {
        super(idHash, recycler, context, instruction, gpu, generator);
        this.generator = generator;
        this.performers = new QwenInstructionFrame[context.plan().instructions().size()];
    }

    @Override
    protected void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction ignored) {
        var instructions = context.plan().instructions();
        // Decode kernels begin with griddepcontrol.wait, so short-context chains may overlap launch tails.
        boolean overlap = context.startPosition() < PDL_MAX_START_POSITION;
        if (overlap) gpu().programmaticDependentLaunch(true);
        try {
            launch(context, instructions);
        } finally {
            if (overlap) gpu().programmaticDependentLaunch(false);
        }
    }

    private void launch(QwenExecutionContext context, java.util.List<QwenExecutionPlan.Instruction> instructions) {
        for (int index = 0; index < instructions.size(); index++) {
            QwenInstructionFrame performer = this.performers[index];
            if (performer == null) {
                performer = this.generator.newPerformer(context, instructions.get(index));
                this.performers[index] = performer;
            }
            performer.bindChained(context);
            // Counted before the launch: a failed launch may already own temporary resources.
            this.performed = index + 1;
            performer.performChained();
        }
    }

    @Override
    protected void gpuCompleted() {
        for (int index = 0; index < this.performed; index++) this.performers[index].gpuCompletedChained();
    }

    @Override
    protected void releaseTemporary(QwenExecutionContext context) {
        RuntimeException failure = null;
        int count = this.performed;
        this.performed = 0;
        for (int index = 0; index < count; index++) {
            try {
                this.performers[index].releaseChained(context);
            } catch (RuntimeException cleanupFailure) {
                if (failure == null) failure = cleanupFailure;
                else failure.addSuppressed(cleanupFailure);
            }
        }
        if (failure != null) throw failure;
    }
}
