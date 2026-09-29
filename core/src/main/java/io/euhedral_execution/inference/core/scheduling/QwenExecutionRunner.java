package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.data_structures.queues.PartitionedMpscQueue;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Function;

/// A directly pluggable Euhedral source. Producers publish ready operation state to an MPSC queue;
/// Euhedral owns `pull`, `request`, frame checkout, and synchronous pushes.
public final class QwenExecutionRunner implements LatticeSource {

    private static final Function<AbstractFrame, Boolean> NEVER_STOP = ignored -> false;

    private final QwenExecutionPlan plan;
    private final ExecutionGpu gpu;
    private final PartitionedMpscQueue<QwenExecutionContext> ready;
    private final QwenWorkGenerator generator;
    private final java.util.Map<QwenExecutionPlan, QwenExecutionRunner> variantRunners;
    private final AtomicInteger active = new AtomicInteger();
    private final AtomicBoolean attached = new AtomicBoolean();

    private final AtomicReference<LatticeReceiver> downstream = new AtomicReference<>();
    private final AtomicBoolean finished = new AtomicBoolean();
    private final CompletableFuture<Void> termination = new CompletableFuture<>();

    public QwenExecutionRunner(QwenExecutionPlan plan, ExecutionGpu gpu) {
        this(plan, gpu, ignored -> {});
    }

    public QwenExecutionRunner(
            QwenExecutionPlan plan, ExecutionGpu gpu, Consumer<? super QwenExecutionContext> terminalConsumer) {
        this(Objects.requireNonNull(plan, "plan").executionOwner(), gpu, terminalConsumer, true);
    }

    static QwenExecutionRunner concrete(
            QwenExecutionPlan plan, ExecutionGpu gpu, Consumer<? super QwenExecutionContext> terminalConsumer) {
        return new QwenExecutionRunner(plan, gpu, terminalConsumer, false);
    }

    private QwenExecutionRunner(
            QwenExecutionPlan plan,
            ExecutionGpu gpu,
            Consumer<? super QwenExecutionContext> terminalConsumer,
            boolean includeVariants) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.ready = new PartitionedMpscQueue<>(plan.instructions().size(), 64);
        this.generator = new QwenWorkGenerator(plan, gpu, this, terminalConsumer);
        var variants = new java.util.LinkedHashMap<QwenExecutionPlan, QwenExecutionRunner>();
        if (includeVariants)
            for (var variant : plan.executionVariants())
                variants.put(variant, new QwenExecutionRunner(variant, gpu, terminalConsumer, false));
        this.variantRunners = java.util.Collections.unmodifiableMap(variants);
    }

    public CompletableFuture<QwenExecutionContext.Outcome> submit(QwenExecutionContext context) {
        Objects.requireNonNull(context, "context");
        if (context.plan() != plan) {
            QwenExecutionRunner variant = this.variantRunners.get(context.plan());
            if (variant == null) throw new IllegalArgumentException("quantum belongs to another execution plan");
            return submitVariant(variant, context);
        }
        gpu.ensureHealthy();
        while (true) {
            int count = active.get();
            if (count < 0) throw new IllegalStateException("Qwen runner admission is closed");
            if (count == Integer.MAX_VALUE) throw new IllegalStateException("too many active quanta");
            if (active.compareAndSet(count, count + 1)) break;
        }
        try {
            gpu.prepare(() -> context.begin(gpu));
        } catch (RuntimeException | Error failure) {
            if (!(failure instanceof QwenExecutionContext.DuplicateAdmissionException)
                    && !context.completion().isDone()) {
                if (gpu.asynchronous()) {
                    try {
                        gpu.synchronize();
                    } catch (RuntimeException | Error synchronizationFailure) {
                        failure.addSuppressed(synchronizationFailure);
                        gpu.poison(failure);
                    }
                }
                context.fail(failure);
                context.finish(null, gpu);
            }
            if (active.decrementAndGet() == Integer.MIN_VALUE) signalComplete();
            throw failure;
        }
        var outcome = context.completion();
        outcome.whenComplete((ignored, failure) -> {
            int remaining = active.decrementAndGet();
            if (remaining == Integer.MIN_VALUE) signalComplete();
        });
        if (!outcome.isDone()) generator.start(context);
        return outcome.copy();
    }

    private CompletableFuture<QwenExecutionContext.Outcome> submitVariant(
            QwenExecutionRunner variant, QwenExecutionContext context) {
        gpu.ensureHealthy();
        while (true) {
            int count = active.get();
            if (count < 0) throw new IllegalStateException("Qwen runner admission is closed");
            if (count == Integer.MAX_VALUE) throw new IllegalStateException("too many active quanta");
            if (active.compareAndSet(count, count + 1)) break;
        }
        CompletableFuture<QwenExecutionContext.Outcome> completion;
        try {
            completion = variant.submit(context);
        } catch (RuntimeException | Error failure) {
            if (active.decrementAndGet() == Integer.MIN_VALUE) signalComplete();
            throw failure;
        }
        // Caller-visible futures can be cancelled independently of admitted GPU work.
        context.completion().whenComplete((ignored, failure) -> {
            if (active.decrementAndGet() == Integer.MIN_VALUE) signalComplete();
        });
        return completion;
    }

    /// Called by producers; each partition is bound to one immutable plan instruction.
    boolean offerReady(int instructionId, QwenExecutionContext context) {
        return this.ready.offer(instructionId, context);
    }

    QwenExecutionContext peekReady(int instructionId) {
        return this.ready.peek(instructionId);
    }

    QwenExecutionContext pollReady(int instructionId) {
        return this.ready.poll(instructionId);
    }

    @Override
    public void addDownstream(LatticeReceiver receiver) {
        Objects.requireNonNull(receiver, "receiver");
        if (!attached.compareAndSet(false, true)) {
            receiver.onError(new IllegalStateException("Qwen source already has a downstream"));
        } else if (finished.get()) {
            notifyComplete(receiver);
        } else {
            downstream.set(receiver);
            if (finished.get() && downstream.compareAndSet(receiver, null)) notifyComplete(receiver);
        }
    }

    @Override
    public long pull(Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stopCondition, long requested) {
        Objects.requireNonNull(consumer, "consumer");
        Objects.requireNonNull(stopCondition, "stopCondition");
        if (requested <= 0 || finished.get()) return 0;
        return drain(consumer, stopCondition, requested);
    }

    private long drain(Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stop, long requested) {
        if (this.variantRunners.isEmpty()) return this.generator.drain(consumer, stop, requested);
        // Only mixed-topology/manual sources need this per-pull stop witness. The ordinary
        // runtime uses a concrete runner, retaining its existing allocation-free drain path.
        boolean[] stopped = {false};
        Function<AbstractFrame, Boolean> guardedStop = frame -> {
            stopped[0] = stop.apply(frame);
            return stopped[0];
        };
        long count = this.generator.drain(consumer, guardedStop, requested);
        for (var variant : this.variantRunners.values()) {
            if (stopped[0] || count >= requested) break;
            count += variant.generator.drain(consumer, guardedStop, requested - count);
        }
        return count;
    }

    @Override
    public void request(long requested) {
        if (requested <= 0 || finished.get()) return;
        LatticeReceiver receiver = downstream.get();
        if (receiver != null) drain(receiver::push, NEVER_STOP, requested);
    }

    @Override
    public void complete() {
        completeGracefully();
    }

    public void completeGracefully() {
        int old = active.getAndUpdate(value -> value | Integer.MIN_VALUE);
        for (var variant : this.variantRunners.values()) variant.completeGracefully();
        if ((old & Integer.MAX_VALUE) == 0) signalComplete();
    }

    private void signalComplete() {
        if (finished.compareAndSet(false, true)) {
            LatticeReceiver receiver = downstream.getAndSet(null);
            notifyComplete(receiver);
        }
    }

    private void notifyComplete(LatticeReceiver receiver) {
        try {
            if (receiver != null) receiver.onComplete();
        } catch (RuntimeException | Error failure) {
            this.termination.completeExceptionally(failure);
            throw failure;
        }
        this.termination.complete(null);
    }

    /// Waits until the downstream completion callback has returned to its source owner.
    void awaitTermination() {
        this.termination.join();
        for (var variant : this.variantRunners.values()) variant.awaitTermination();
    }

    @Override
    public boolean isComplete() {
        return finished.get();
    }

    /// Whether this source currently retains its lattice receiver.
    public boolean isAttached() {
        return this.downstream.get() != null;
    }
}
