package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenWorkGenerator;
import java.util.Objects;

/// Base lifecycle for one reusable Qwen instruction frame.
public abstract class QwenInstructionFrame extends AbstractFrame {

    private final ExecutionGpu gpu;
    private final QwenWorkGenerator generator;
    private QwenExecutionContext context;
    private final QwenExecutionPlan.Instruction instruction;
    private boolean completed;

    @SuppressWarnings("rawtypes")
    protected QwenInstructionFrame(
            long idHash,
            FrameManager recycler,
            QwenExecutionContext context,
            QwenExecutionPlan.Instruction instruction,
            ExecutionGpu gpu,
            QwenWorkGenerator generator) {
        super(idHash, recycler, null);
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.generator = Objects.requireNonNull(generator, "generator");
        this.context = Objects.requireNonNull(context, "context");
        this.instruction = Objects.requireNonNull(instruction, "instruction");
        randomizeHash(instruction.id());
    }

    /// Rebind only the per-quantum state; this pool's instruction and weights never change.
    public final void replace(QwenExecutionContext context) {
        this.context = Objects.requireNonNull(context, "context");
        this.completed = false;
        randomizeHash(this.instruction.id());
    }

    protected abstract void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction);

    protected void releaseTemporary(QwenExecutionContext context) {}

    protected void gpuCompleted() {}

    /// Rebinds a frame that a decode chain owns; chained frames are never pooled or published.
    public final void bindChained(QwenExecutionContext context) {
        this.context = Objects.requireNonNull(context, "context");
        this.completed = false;
    }

    /// Launches this instruction on the chain frame's current submission stream.
    public final void performChained() {
        perform(this.context, this.instruction);
    }

    /// Runs this instruction's success-only finalization once the whole chain's device work retired.
    public final void gpuCompletedChained() {
        gpuCompleted();
    }

    /// Releases this instruction's temporary resources once the whole chain's device work retired.
    public final void releaseChained(QwenExecutionContext context) {
        try {
            releaseTemporary(context);
        } finally {
            this.context = null;
            this.completed = false;
        }
    }

    /// Clears an instruction frame that could not be published to Euhedral.
    public final void abandonBeforePublication() {
        this.context = null;
        this.completed = false;
        resetHash();
        recycle();
    }

    protected final ExecutionGpu gpu() {
        return this.gpu;
    }

    @Override
    public final void execute() {
        QwenExecutionContext current = this.context;
        if (current.hasFailureOrCancellation()) {
            return;
        }
        try {
            this.gpu.submit(() -> perform(current, this.instruction));
            if (!this.gpu.asynchronous()) this.gpu.synchronize();
        } catch (RuntimeException | Error failure) {
            // A launch or post-launch host operation may fail after earlier work was submitted.
            // A streamed FFN owns internal streams even when its outer execution mode is SYNC.
            // Neither a failed submission nor a failed completion barrier proves they retired.
            if (this.gpu.asynchronous() || this.instruction.kind() == QwenExecutionPlan.Kind.FFN_STREAMED) {
                try {
                    this.gpu.synchronize();
                } catch (RuntimeException | Error synchronizationFailure) {
                    failure.addSuppressed(synchronizationFailure);
                    this.gpu.poison(failure);
                }
            }
            if (failure instanceof Error fatal) {
                // Euhedral's executor catches Exception, not Error. Finalize while this
                // worker still owns the frame, then allow the fatal error to escape.
                try {
                    finish(fatal);
                } catch (Throwable cleanupFailure) {
                    if (cleanupFailure != fatal) fatal.addSuppressed(cleanupFailure);
                }
            }
            throw failure;
        }
        this.completed = true;
    }

    @Override
    public final void doFinally() {
        if (this.completed && this.gpu.asynchronous()) {
            try {
                this.gpu.deferCompletion(() -> finish(null), failure -> finish(failure));
            } catch (RuntimeException | Error registrationFailure) {
                // Registration may fail after a launch. Prove device completion before
                // releasing the frame, workspace, and sequence ownership.
                try {
                    this.gpu.synchronize();
                } catch (RuntimeException | Error synchronizationFailure) {
                    registrationFailure.addSuppressed(synchronizationFailure);
                    this.gpu.poison(registrationFailure);
                }
                finish(registrationFailure);
            }
        } else {
            finish(this.completed ? null : this.context.failure());
        }
    }

    @Override
    public final void doFinallyWithError(Throwable error) {
        // This path also handles errors thrown before the frame body ran. A failed native
        // submission is drained by execute() before this finalizer may recycle the frame.
        finish(Objects.requireNonNull(error, "error"));
    }

    private void finish(Throwable error) {
        QwenExecutionContext current = this.context;
        QwenExecutionPlan.Instruction completedInstruction = this.instruction;
        try {
            if (error == null) gpuCompleted();
            releaseTemporary(current);
        } catch (RuntimeException | Error cleanupFailure) {
            if (error == null) {
                error = cleanupFailure;
            } else {
                error.addSuppressed(cleanupFailure);
            }
        }
        try {
            this.generator.completed(current, completedInstruction, error);
        } finally {
            this.context = null;
            this.completed = false;
            resetHash();
            recycle();
        }
    }
}
