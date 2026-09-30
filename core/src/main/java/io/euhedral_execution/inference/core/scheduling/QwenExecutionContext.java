package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.scheduling.graph.StageQuantum;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/// Mutable state for one inference quantum: its token range, sequence lease, workspace, and outcome.
///
/// While it runs, the quantum is bound to a reusable stage graph whose frames perform the operations.
/// Its sequence reference retains sequence-lifetime state, which is published only at retirement.
public final class QwenExecutionContext implements StageQuantum {

    /// Decode quanta that start below this position overlap registered kernels with their predecessor
    /// (programmatic dependent launch). It won 12 of 12 paired forks, about +1% decode, at a 64-token
    /// context and only 10 of 12 at 1024, so longer contexts keep ordinary launches.
    static final long OVERLAP_MAX_START_POSITION = 1024;

    private static final Throwable TERMINAL_SUCCESS = new IllegalStateException("quantum already finalized");
    private static final Runnable NO_OP = () -> {};

    public enum ExecutionKind {
        PREFILL,
        DECODE
    }

    public enum Status {
        SUCCESS,
        CANCELLED,
        FAILED
    }

    public record Outcome(Status status, Throwable failure) {}

    static final class DuplicateAdmissionException extends IllegalStateException {
        DuplicateAdmissionException() {
            super("quantum was already submitted");
        }
    }

    private final QwenExecutionPlan plan;
    private final QwenSequenceState sequence;
    private final ExecutionKind kind;
    private final QwenLogitsRequirement logitsRequirement;
    private final long startPosition;
    private final int[] tokenIds;
    private final AtomicBoolean submitted = new AtomicBoolean();
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private final CompletableFuture<Outcome> outcome = new CompletableFuture<>();
    private QwenSequenceState.ExecutionLease lease;
    private QwenExecutionWorkspace workspace;
    private QwenDeviceLogits logitsOutput;
    private long temporaryTokenIdsAddress;
    private ExecutionGpu gpu;
    private Consumer<? super QwenExecutionContext> terminalConsumer;
    private Outcome pendingOutcome;

    public QwenExecutionContext(
            QwenExecutionPlan plan,
            QwenSequenceState sequence,
            ExecutionKind kind,
            long startPosition,
            int[] tokenIds) {
        this(plan, sequence, kind, startPosition, tokenIds, QwenLogitsRequirement.ALL_TOKENS);
    }

    public QwenExecutionContext(
            QwenExecutionPlan plan,
            QwenSequenceState sequence,
            ExecutionKind kind,
            long startPosition,
            int[] tokenIds,
            QwenLogitsRequirement logitsRequirement) {
        this.plan = Objects.requireNonNull(plan, "plan")
                .forExecution(kind, Objects.requireNonNull(tokenIds, "tokenIds").length);
        this.logitsRequirement = Objects.requireNonNull(logitsRequirement, "logitsRequirement");
        this.sequence = Objects.requireNonNull(sequence, "sequence");
        this.kind = Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(tokenIds, "tokenIds");
        if (startPosition < 0 || tokenIds.length == 0) {
            throw new IllegalArgumentException("invalid token range");
        }
        this.startPosition = startPosition;
        this.tokenIds = tokenIds.clone();
    }

    public QwenExecutionPlan plan() {
        return this.plan;
    }

    public QwenSequenceState sequenceState() {
        return this.sequence;
    }

    public int inputTokenCount() {
        return this.tokenIds.length;
    }

    public QwenLogitsRequirement logitsRequirement() {
        return this.logitsRequirement;
    }

    public int logitsRowCount() {
        return this.logitsRequirement.outputRows(this.tokenIds.length);
    }

    public long startPosition() {
        return this.startPosition;
    }

    public int[] inputTokenIds() {
        return this.tokenIds.clone();
    }

    public QwenExecutionWorkspace workspace() {
        QwenExecutionWorkspace current = this.workspace;
        if (current == null) {
            throw new IllegalStateException("workspace has not been allocated");
        }
        return current;
    }

    public CompletableFuture<Outcome> outcome() {
        return this.outcome.copy();
    }

    /// Returns GPU-resident logits after successful completion; the caller owns and must close them.
    public Optional<QwenDeviceLogits> logitsOutput() {
        return Optional.ofNullable(this.logitsOutput);
    }

    CompletableFuture<Outcome> completion() {
        return this.outcome;
    }

    public void cancel() {
        this.sequence.cancel();
    }

    public void fail(Throwable cause) {
        Objects.requireNonNull(cause, "cause");
        if (this.failure.compareAndSet(null, cause)) return;
        Throwable first = this.failure.get();
        if (first != TERMINAL_SUCCESS && first != cause) first.addSuppressed(cause);
    }

    /// Reports whether this quantum must stop admitting dependent instructions.
    public boolean hasFailureOrCancellation() {
        return this.failure.get() != null || this.sequence.cancellationRequested();
    }

    @Override
    public boolean stopRequested() {
        return hasFailureOrCancellation();
    }

    @Override
    public boolean overlapLaunches() {
        return this.kind == ExecutionKind.DECODE && this.startPosition < OVERLAP_MAX_START_POSITION;
    }

    /// Returns the first operation failure, if one has been recorded.
    public Throwable failure() {
        return this.failure.get();
    }

    public ExecutionKind kind() {
        return this.kind;
    }

    /// Allocates this quantum's temporary token-ID upload buffer.
    public long allocateTemporaryTokenIds(ExecutionGpu gpu, long bytes) {
        this.temporaryTokenIdsAddress = gpu.allocate(bytes);
        if (this.temporaryTokenIdsAddress == 0) {
            throw new IllegalStateException("GPU returned a null token-ID address");
        }
        return this.temporaryTokenIdsAddress;
    }

    /// Releases the temporary token-ID upload buffer when its embedding instruction is finalized.
    public void releaseTemporaryTokenIds(ExecutionGpu gpu) {
        if (this.temporaryTokenIdsAddress != 0) {
            gpu.free(this.temporaryTokenIdsAddress);
            this.temporaryTokenIdsAddress = 0;
        }
    }

    void begin(ExecutionGpu gpu) {
        begin(gpu, null, null, NO_OP);
    }

    /// Package-private hook to deterministically exercise cancellation at the lease-claim boundary.
    void begin(ExecutionGpu gpu, Runnable beforeClaim) {
        begin(gpu, null, null, beforeClaim);
    }

    /// Claims the sequence and prepares quantum-owned storage with `stream` selected, so any
    /// initialization it queues precedes every stage of the quantum. The quantum proceeds only if its
    /// outcome is still open afterwards; a failed preparation has already reached its terminal outcome.
    void begin(ExecutionGpu gpu, GpuStream stream, Consumer<? super QwenExecutionContext> terminalConsumer) {
        begin(gpu, stream, terminalConsumer, NO_OP);
    }

    private void begin(
            ExecutionGpu gpu,
            GpuStream stream,
            Consumer<? super QwenExecutionContext> terminalConsumer,
            Runnable beforeClaim) {
        if (!this.submitted.compareAndSet(false, true)) {
            throw new DuplicateAdmissionException();
        }
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.terminalConsumer = terminalConsumer;
        if (this.sequence.cancellationRequested()) {
            this.outcome.complete(new Outcome(Status.CANCELLED, null));
            return;
        }
        try {
            long end = Math.addExact(this.startPosition, this.tokenIds.length);
            if (end < 0) {
                throw new IllegalArgumentException("token range overflows");
            }
            int vocabulary = this.plan.weights().config().vocabSize();
            for (int index = 0; index < this.tokenIds.length; index++) {
                if (this.tokenIds[index] < 0 || this.tokenIds[index] >= vocabulary) {
                    throw new IllegalArgumentException("token ID outside vocabulary at index " + index);
                }
            }
            beforeClaim.run();
            try {
                this.lease = this.sequence.claimExecution(this.startPosition);
            } catch (IllegalStateException claimFailure) {
                if (this.sequence.terminalState() == QwenSequenceState.TerminalState.CANCELLED) {
                    this.outcome.complete(new Outcome(Status.CANCELLED, null));
                    return;
                }
                throw claimFailure;
            }
            initializeSequenceState(gpu);
            this.workspace = this.plan.hasFirstLayer()
                    ? new QwenExecutionWorkspace(gpu, this.tokenIds.length, this.plan, this.logitsRequirement)
                    : new QwenExecutionWorkspace(
                            gpu,
                            this.tokenIds.length,
                            this.plan.weights().config().hiddenSize(),
                            this.plan.projectionWidths());
            this.workspace.allocateBuffers();
        } catch (RuntimeException | Error error) {
            // Initialization can queue zeroes before a later allocation fails. Keep every allocation
            // when the stream cannot prove that those writes stopped.
            if (stream != null) stream.recover(error);
            fail(error);
            finish();
        }
    }

    private void initializeSequenceState(ExecutionGpu gpu) {
        if (!this.plan.hasFirstLayer()) return;
        Object current = this.sequence.recurrentState();
        Object currentKv = this.sequence.kvCacheState();
        QwenConfig config = this.plan.weights().config();
        if (current == null) {
            AutoCloseable createdRecurrent = null;
            AutoCloseable createdKv = null;
            try {
                if (this.plan.weights().layers().length > 1) {
                    createdRecurrent = GdnSequenceStates.allocate(
                            gpu,
                            config.layerTypes(),
                            config.linearNumKeyHeads(),
                            config.linearNumValueHeads(),
                            config.linearKeyHeadDim(),
                            config.linearValueHeadDim(),
                            config.linearConvKernelDim());
                    createdKv = AttentionSequenceStates.allocate(
                            gpu, config.layerTypes(), config.numKeyValueHeads() * config.attentionHeadDim());
                } else {
                    createdRecurrent = QwenGdnSequenceState.allocate(
                            gpu,
                            config.linearNumKeyHeads(),
                            config.linearNumValueHeads(),
                            config.linearKeyHeadDim(),
                            config.linearValueHeadDim(),
                            config.linearConvKernelDim());
                }
                this.sequence.setRecurrentState(this.lease, createdRecurrent);
                if (createdKv != null) this.sequence.setKvCacheState(this.lease, createdKv);
            } catch (RuntimeException | Error attachmentFailure) {
                closeCreatedState(createdKv, attachmentFailure);
                closeCreatedState(createdRecurrent, attachmentFailure);
                throw attachmentFailure;
            }
        } else if (this.plan.weights().layers().length > 1) {
            if (!(current instanceof GdnSequenceStates)) {
                throw new IllegalStateException("sequence already owns incompatible full-model GDN state");
            }
            if (!(currentKv instanceof AttentionSequenceStates)) {
                throw new IllegalStateException("sequence already owns incompatible full-attention KV state");
            }
        } else if (!(current instanceof QwenGdnSequenceState)) {
            throw new IllegalStateException("sequence already owns incompatible recurrent state");
        }
    }

    private static void closeCreatedState(AutoCloseable state, Throwable failure) {
        if (state == null) return;
        try {
            state.close();
        } catch (Exception cleanupFailure) {
            failure.addSuppressed(cleanupFailure);
        }
    }

    /// Terminal work for a quantum whose device work has retired. Releases quantum storage, publishes
    /// sequence state, and prepares the outcome that [#publishOutcome] reports.
    @Override
    public void retire(Throwable deviceFailure) {
        if (this.outcome.isDone() || this.pendingOutcome != null) {
            return;
        }
        if (deviceFailure != null) fail(deviceFailure);
        ExecutionGpu gpu = this.gpu;
        try {
            releaseTemporaryTokenIds(gpu);
        } catch (Throwable cleanupFailure) {
            fail(cleanupFailure);
        }
        if (this.failure.get() == null && !this.sequence.cancellationRequested()) {
            try {
                retainLogits(gpu);
            } catch (Throwable retentionFailure) {
                fail(retentionFailure);
            }
        }
        if (this.failure.get() == null && !this.sequence.cancellationRequested() && this.terminalConsumer != null) {
            try {
                this.terminalConsumer.accept(this);
            } catch (Throwable consumerFailure) {
                fail(consumerFailure);
            }
        }
        if (this.workspace != null) {
            try {
                this.workspace.close();
            } catch (Throwable cleanupFailure) {
                fail(cleanupFailure);
                try {
                    this.workspace.close();
                } catch (Throwable retryFailure) {
                    fail(retryFailure);
                }
            }
        }
        Throwable error = this.failure.get();
        if (error == null && !this.failure.compareAndSet(null, TERMINAL_SUCCESS)) {
            error = this.failure.get();
        }
        if (error == TERMINAL_SUCCESS) error = null;
        if (error != null || this.sequence.cancellationRequested()) {
            error = releaseLogits(error);
        }
        Outcome completed;
        try {
            if (error != null) {
                if (this.lease != null) {
                    this.sequence.markFailed(this.lease, error);
                }
                completed = new Outcome(Status.FAILED, error);
            } else if (this.sequence.cancellationRequested()) {
                if (this.lease != null) {
                    this.sequence.markCancelled(this.lease);
                }
                completed = new Outcome(Status.CANCELLED, null);
            } else {
                boolean cancelled = this.sequence.releaseExecutionAndCheckCancellation(
                        this.lease, this.startPosition + this.tokenIds.length);
                completed = new Outcome(cancelled ? Status.CANCELLED : Status.SUCCESS, null);
            }
        } catch (Throwable cleanupFailure) {
            // Terminal state is published before persistent cleanup. Never strand the caller's future
            // if a device free fails; the sequence owner can retry cleanup after observing this failure.
            if (error == null) error = cleanupFailure;
            else if (error != cleanupFailure) error.addSuppressed(cleanupFailure);
            completed = new Outcome(Status.FAILED, releaseLogits(error));
        }
        this.lease = null;
        this.pendingOutcome = completed;
    }

    /// Completes the caller-visible outcome prepared by [#retire].
    @Override
    public void publishOutcome() {
        Outcome completed = this.pendingOutcome;
        if (completed == null) {
            if (this.outcome.isDone()) return;
            throw new IllegalStateException("quantum has not retired");
        }
        this.pendingOutcome = null;
        this.outcome.complete(completed);
    }

    /// Retires a quantum that never started stages, then publishes its outcome.
    void finish() {
        retire(null);
        publishOutcome();
    }

    /// Terminal cleanup for a quantum driven outside a stage graph.
    void finish(Consumer<? super QwenExecutionContext> terminalConsumer, ExecutionGpu gpu) {
        if (this.gpu == null) this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (terminalConsumer != null) this.terminalConsumer = terminalConsumer;
        finish();
    }

    private void retainLogits(ExecutionGpu gpu) {
        if (this.workspace == null || !this.workspace.hasBuffer(QwenExecutionPlan.Buffer.LOGITS)) return;
        long address = this.workspace.detachAddress(QwenExecutionPlan.Buffer.LOGITS);
        try {
            this.logitsOutput = new QwenDeviceLogits(
                    gpu, address, logitsRowCount(), this.plan.weights().config().vocabSize());
        } catch (RuntimeException | Error constructionFailure) {
            try {
                gpu.free(address);
            } catch (Throwable cleanupFailure) {
                constructionFailure.addSuppressed(cleanupFailure);
            }
            throw constructionFailure;
        }
    }

    private Throwable releaseLogits(Throwable priorFailure) {
        if (this.logitsOutput == null) return priorFailure;
        try {
            this.logitsOutput.close();
            this.logitsOutput = null;
        } catch (Throwable cleanupFailure) {
            if (priorFailure != null) priorFailure.addSuppressed(cleanupFailure);
            else priorFailure = cleanupFailure;
        }
        return priorFailure;
    }
}
