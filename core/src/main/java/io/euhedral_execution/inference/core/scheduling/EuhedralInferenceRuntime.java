package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.frames.RunnableFrame;
import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.core.ingest.QueueIngestSink;
import io.euhedral_execution.data_structures.queues.PartitionedMpscQueue;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/// Owns temporary execution-source attachment while borrowing model and lattice lifetimes.
public final class EuhedralInferenceRuntime {

    private final LatticeTerminal lattice;
    private final QwenExecutionPlan plan;
    private final ExecutionGpu gpu;
    private final QueueIngestSink completionSink;
    private final AtomicReference<QwenExecutionRunner> attachedRunner = new AtomicReference<>();
    private final Set<QwenExecutionRunner> executingRunners = ConcurrentHashMap.newKeySet();
    private final Object admissionLock = new Object();
    private final CompletableFuture<Void> admissionsDrained = new CompletableFuture<>();
    private boolean closing;
    private int admissions;

    public EuhedralInferenceRuntime(LatticeTerminal lattice, QwenExecutionPlan plan, ExecutionGpu gpu) {
        this.lattice = Objects.requireNonNull(lattice, "lattice");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (gpu.asynchronous()) {
            var sink = new QueueIngestSink(new PartitionedMpscQueue<>(64));
            this.lattice.addUpstream(sink.getDelegate());
            try {
                gpu.bindCompletionSink(completion -> {
                    if (!sink.offer(new RunnableFrame(0L, completion)))
                        throw new IllegalStateException("CUDA completion sink rejected a ready frame");
                });
            } catch (RuntimeException | Error failure) {
                sink.complete();
                throw failure;
            }
            this.completionSink = sink;
        } else {
            this.completionSink = null;
        }
    }

    /// Executes admitted quanta through one temporary Euhedral source and then detaches it.
    public List<QwenExecutionContext.Outcome> execute(List<QwenExecutionContext> contexts)
            throws InterruptedException, ExecutionException {
        return execute(contexts, ignored -> {});
    }

    /// Executes quanta while their terminal callback can still inspect the live workspace.
    public List<QwenExecutionContext.Outcome> execute(
            List<QwenExecutionContext> contexts, Consumer<? super QwenExecutionContext> terminalConsumer)
            throws InterruptedException, ExecutionException {
        List<QwenExecutionContext> acceptedContexts = List.copyOf(Objects.requireNonNull(contexts, "contexts"));
        if (acceptedContexts.isEmpty()) return List.of();
        QwenExecutionPlan prefillPlan = this.plan.forExecution(
                QwenExecutionContext.ExecutionKind.PREFILL,
                acceptedContexts.getFirst().inputTokenCount());
        QwenExecutionPlan executionPlan = acceptedContexts.getFirst().plan() == prefillPlan ? prefillPlan : this.plan;
        // Each call owns its source and completion futures; the fabric schedules independent inputs.
        Objects.requireNonNull(terminalConsumer, "terminalConsumer");
        boolean homogeneous = acceptedContexts.stream().allMatch(context -> context.plan() == executionPlan);
        var runner = homogeneous
                ? QwenExecutionRunner.concrete(executionPlan, this.gpu, terminalConsumer)
                : new QwenExecutionRunner(this.plan, this.gpu, terminalConsumer);
        admit();
        Throwable executionFailure = null;
        try {
            this.executingRunners.add(runner);
            this.lattice.addUpstream(runner);
            List<CompletableFuture<QwenExecutionContext.Outcome>> completions =
                    new ArrayList<>(acceptedContexts.size());
            for (QwenExecutionContext context : acceptedContexts) completions.add(runner.submit(context));
            CompletableFuture.allOf(completions.toArray(CompletableFuture[]::new))
                    .get();
            return completions.stream().map(CompletableFuture::join).toList();
        } catch (InterruptedException | ExecutionException | RuntimeException | Error failure) {
            executionFailure = failure;
            throw failure;
        } finally {
            try {
                runner.completeGracefully();
                runner.awaitTermination();
            } catch (RuntimeException | Error cleanupFailure) {
                if (executionFailure == null) throw cleanupFailure;
                if (cleanupFailure != executionFailure) executionFailure.addSuppressed(cleanupFailure);
            } finally {
                if (runner.isComplete() && !runner.isAttached()) this.executingRunners.remove(runner);
                releaseAdmission();
            }
        }
    }

    /// Creates the manually managed source used by submit; execute calls own separate sources.
    public QwenExecutionRunner attachRunner() {
        return attachRunner(ignored -> {});
    }

    public QwenExecutionRunner attachRunner(Consumer<? super QwenExecutionContext> terminalConsumer) {
        admit();
        try {
            return attachAdmittedRunner(terminalConsumer);
        } finally {
            releaseAdmission();
        }
    }

    private QwenExecutionRunner attachAdmittedRunner(Consumer<? super QwenExecutionContext> terminalConsumer) {
        QwenExecutionRunner runner = new QwenExecutionRunner(
                this.plan, this.gpu, Objects.requireNonNull(terminalConsumer, "terminalConsumer"));
        if (!this.attachedRunner.compareAndSet(null, runner)) {
            throw new IllegalStateException("an execution runner is already attached to this runtime");
        }
        try {
            this.lattice.addUpstream(runner);
            return runner;
        } catch (RuntimeException | Error attachmentFailure) {
            try {
                runner.completeGracefully();
                runner.awaitTermination();
            } catch (RuntimeException | Error cleanupFailure) {
                attachmentFailure.addSuppressed(cleanupFailure);
            }
            this.attachedRunner.compareAndSet(runner, null);
            throw attachmentFailure;
        }
    }

    /// Submits one quantum to the currently attached source.
    public CompletableFuture<QwenExecutionContext.Outcome> submit(QwenExecutionContext context) {
        QwenExecutionRunner runner = this.attachedRunner.get();
        if (runner == null) throw new IllegalStateException("no execution runner is attached");
        return runner.submit(context);
    }

    /// Waits on the quantum's existing completion future.
    public QwenExecutionContext.Outcome await(CompletableFuture<QwenExecutionContext.Outcome> completion)
            throws InterruptedException, ExecutionException {
        return Objects.requireNonNull(completion, "completion").get();
    }

    /// Closes runner admission, drains accepted quanta, and waits for Euhedral to detach the source.
    public void disconnectRunner() {
        for (var executing : this.executingRunners) {
            executing.completeGracefully();
            executing.awaitTermination();
            if (executing.isComplete() && !executing.isAttached()) this.executingRunners.remove(executing);
        }
        QwenExecutionRunner runner = this.attachedRunner.get();
        if (runner == null) return;
        try {
            runner.completeGracefully();
            runner.awaitTermination();
        } finally {
            if (runner.isComplete() && !runner.isAttached()) this.attachedRunner.compareAndSet(runner, null);
        }
    }

    /// Detaches the process-wide completion source only after the engine stops all admissions.
    public void closeCompletionSink() {
        synchronized (this.admissionLock) {
            this.closing = true;
            if (this.admissions == 0) this.admissionsDrained.complete(null);
        }
        this.admissionsDrained.join();
        disconnectRunner();
        if (this.completionSink != null) this.completionSink.complete();
    }

    private void admit() {
        synchronized (this.admissionLock) {
            if (this.closing) throw new IllegalStateException("inference runtime is closing");
            this.admissions++;
        }
    }

    private void releaseAdmission() {
        synchronized (this.admissionLock) {
            if (--this.admissions == 0 && this.closing) this.admissionsDrained.complete(null);
        }
    }

    public boolean hasAttachedRunner() {
        return this.attachedRunner.get() != null || !this.executingRunners.isEmpty();
    }
}
