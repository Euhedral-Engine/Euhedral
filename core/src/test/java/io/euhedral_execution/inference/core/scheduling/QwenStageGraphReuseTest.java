package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.gpu.InlineGpuStream;
import io.euhedral_execution.inference.core.scheduling.frames.QwenStageFrame;
import io.euhedral_execution.inference.core.scheduling.graph.LanePool;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

/// Stage frames are built once per reusable graph, rebound to later quanta, and never recycled while
/// their quantum's device work may still use its storage.
class QwenStageGraphReuseTest {

    private final QwenExecutionFixtures.ManualLattice lattice = new QwenExecutionFixtures.ManualLattice();

    @Test
    void stageErrorStillRetiresTheQuantumAtTheRealExecutorBoundary() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu() {
            @Override
            public GpuStream openStream() {
                return new InlineGpuStream() {
                    @Override
                    public void submit(Runnable launches, boolean overlapPredecessor) {
                        launches.run();
                        // Admission's own preparation succeeds; the stage's launch hits the host failure.
                        if (launches instanceof QwenStageFrame)
                            throw new OutOfMemoryError("injected post-launch host exhaustion");
                    }
                };
            }
        };
        var runtime = new EuhedralInferenceRuntime(this.lattice, plan, gpu);
        var context = context(plan, 507);
        runtime.submit(context);
        AbstractFrame frame = pullOne();
        var receiver = new AtomicReference<LatticeReceiver>();
        new DefaultExecutor().input(new LatticeSource() {
            @Override
            public void addDownstream(LatticeReceiver downstream) {
                receiver.set(downstream);
            }

            @Override
            public long pull(Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stop, long requested) {
                return 0;
            }

            @Override
            public void request(long requested) {}

            @Override
            public void complete() {}

            @Override
            public boolean isComplete() {
                return false;
            }
        });

        // Escaping into Euhedral, the Error would complete the graph's source or end the worker.
        assertDoesNotThrow(() -> receiver.get().push(frame));
        this.lattice.drive();
        assertTrue(context.outcome().isDone(), "an Error must not strand the quantum");
        var outcome = context.outcome().join();
        assertEquals(QwenExecutionContext.Status.FAILED, outcome.status());
        assertInstanceOf(OutOfMemoryError.class, outcome.failure());
        runtime.close();
    }

    @Test
    void stageFramesHaveDedicatedTypesAndAreReusedAcrossQuanta() {
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.weights(),
                QwenExecutionFixtures.norm(),
                List.of(
                        QwenExecutionFixtures.q3("projection-a", 64, 201),
                        QwenExecutionFixtures.q3("projection-b", 64, 202)));
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var runtime = new EuhedralInferenceRuntime(this.lattice, plan, gpu);

        List<AbstractFrame> firstRun = runQuantum(runtime, plan, 301);
        List<AbstractFrame> secondRun = runQuantum(runtime, plan, 302);

        assertEquals(
                List.of(
                        "io.euhedral_execution.inference.core.scheduling.frames.EmbeddingFrame",
                        "io.euhedral_execution.inference.core.scheduling.frames.RmsNormFrame",
                        "io.euhedral_execution.inference.core.scheduling.frames.LinearFrame",
                        "io.euhedral_execution.inference.core.scheduling.frames.LinearFrame"),
                firstRun.stream().map(frame -> frame.getClass().getName()).toList());
        assertNotSame(firstRun.get(2), firstRun.get(3));
        for (int index = 0; index < firstRun.size(); index++) {
            assertSame(firstRun.get(index), secondRun.get(index), "stage " + index + " was not reused");
        }
        assertEquals(
                List.of("embed", "norm", "linear:201", "linear:202", "embed", "norm", "linear:201", "linear:202"),
                gpu.operations);
        runtime.close();
    }

    @Test
    void failedStageIsReboundToTheNextQuantum() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var runtime = new EuhedralInferenceRuntime(this.lattice, plan, gpu);
        RuntimeException failure = new IllegalStateException("injected embedding failure");
        gpu.afterEmbedding = () -> {
            throw failure;
        };
        var first = context(plan, 401);
        runtime.submit(first);
        AbstractFrame failedStage = pullOne();
        // The stage records its failure instead of throwing into Euhedral, which then runs doFinally.
        failedStage.execute();
        failedStage.doFinally();
        this.lattice.drive();
        var failed = first.outcome().join();
        assertEquals(QwenExecutionContext.Status.FAILED, failed.status());
        assertSame(failure, failed.failure());

        gpu.afterEmbedding = () -> {};
        var second = context(plan, 402);
        runtime.submit(second);
        AbstractFrame reused = pullOne();

        assertSame(failedStage, reused);
        reused.execute();
        reused.doFinally();
        this.lattice.drive();
        assertEquals(
                QwenExecutionContext.Status.SUCCESS, second.outcome().join().status());
        runtime.close();
    }

    @Test
    void boundaryRegistrationFailureRecoversTheStreamAndRetiresOnce() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var recoveries = new int[1];
        var gpu = new QwenExecutionFixtures.RecordingGpu() {
            @Override
            public GpuStream openStream() {
                return new InlineGpuStream() {
                    @Override
                    public long notifyRetired(RetirementListener listener) {
                        throw new IllegalStateException("injected event registration failure");
                    }

                    @Override
                    public void recover(Throwable failure) {
                        recoveries[0]++;
                    }
                };
            }
        };
        // One lane: the quantum's single stream is the one that must prove idleness.
        var runtime = new EuhedralInferenceRuntime(this.lattice, plan, gpu, 1, LanePool.Placement.CHAIN);
        var first = context(plan, 501);
        runtime.submit(first);
        AbstractFrame stage = pullOne();
        stage.execute();
        stage.doFinally();
        this.lattice.drive();

        assertEquals(1, recoveries[0], "the stream proves idleness before storage is released");
        assertEquals(QwenExecutionContext.Status.FAILED, first.outcome().join().status());
        var second = context(plan, 502);
        runtime.submit(second);
        AbstractFrame reused = pullOne();
        assertSame(stage, reused, "the failed quantum recycled its graph exactly once");
        reused.execute();
        reused.doFinally();
        this.lattice.drive();
        assertEquals(QwenExecutionContext.Status.FAILED, second.outcome().join().status());
        runtime.close();
        QwenExecutionFixtures.assertEachAllocationFreedOnce(gpu);
    }

    @Test
    void embeddingUploadLivesUntilTheQuantumRetires() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var stream = new QwenExecutionFixtures.HoldingStream();
        var uploads = new int[2];
        var gpu = new QwenExecutionFixtures.RecordingGpu() {
            @Override
            public GpuStream openStream() {
                return stream;
            }

            @Override
            public UploadBuffer allocateUploadBuffer(long bytes) {
                uploads[0]++;
                UploadBuffer allocated = super.allocateUploadBuffer(bytes);
                return new UploadBuffer(allocated.segment(), () -> {
                    uploads[1]++;
                    allocated.close();
                });
            }
        };
        var runtime = new EuhedralInferenceRuntime(this.lattice, plan, gpu);
        var context = context(plan, 506);
        runtime.submit(context);
        this.lattice.drive();

        assertEquals(1, uploads[0]);
        assertEquals(0, uploads[1], "the device may still read the staged token IDs");
        assertFalse(context.outcome().isDone());
        stream.release(null);
        assertEquals(0, uploads[1], "the driver callback only enqueues the retirement");
        this.lattice.drive();
        assertEquals(1, uploads[1]);
        assertEquals(
                QwenExecutionContext.Status.SUCCESS, context.outcome().join().status());
        runtime.close();
    }

    @Test
    void unprovableRetirementPoisonsGpuAndNeverReleasesUnprovenBuffers() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new UnrecoverableGpu(false);
        var runtime = new EuhedralInferenceRuntime(this.lattice, plan, gpu);
        var context = context(plan, 503);
        runtime.submit(context);
        AbstractFrame stage = pullOne();
        stage.execute();
        stage.doFinally();
        this.lattice.drive();

        assertEquals(
                QwenExecutionContext.Status.FAILED, context.outcome().join().status());
        assertTrue(gpu.poisoned);
        assertEquals(0, gpu.nativeFrees, "uncertain GPU work may still use every submitted buffer");
        assertEquals(0, gpu.hostReleases, "uncertain DMA may still read the pinned upload");
        assertThrows(IllegalStateException.class, () -> runtime.submit(context(plan, 504)));
        runtime.close();
        assertEquals(0, gpu.nativeFrees, "closing the runtime keeps the poisoned graph's workspace storage");
        assertTrue(runtime.retainedWorkspaceBytes() > 0);
    }

    @Test
    void failedRegistrationAndRecoveryFailsTheQuantumWithoutReleasingBuffers() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new UnrecoverableGpu(true);
        var runtime = new EuhedralInferenceRuntime(this.lattice, plan, gpu);
        var context = context(plan, 505);
        runtime.submit(context);
        AbstractFrame stage = pullOne();
        stage.execute();
        stage.doFinally();
        this.lattice.drive();

        assertTrue(gpu.poisoned);
        assertEquals(
                QwenExecutionContext.Status.FAILED, context.outcome().join().status());
        assertEquals(0, gpu.nativeFrees);
        runtime.close();
    }

    /// Launches fail after reaching the device; the stream can prove neither registration nor retirement.
    private static final class UnrecoverableGpu extends QwenExecutionFixtures.RecordingGpu {
        private final boolean failRegistration;
        boolean poisoned;
        int nativeFrees;
        int hostReleases;

        UnrecoverableGpu(boolean failRegistration) {
            this.failRegistration = failRegistration;
        }

        @Override
        public GpuStream openStream() {
            return new InlineGpuStream() {
                @Override
                public void submit(Runnable launches, boolean overlapPredecessor) {
                    launches.run();
                    if (!failRegistration && launches instanceof QwenStageFrame)
                        throw new IllegalStateException("post-launch host error");
                }

                @Override
                public long notifyRetired(RetirementListener listener) {
                    if (failRegistration) throw new IllegalStateException("injected event registration failure");
                    return super.notifyRetired(listener);
                }

                @Override
                public Throwable confirmRetired(long ticket) {
                    IllegalStateException failure = new IllegalStateException("device retirement unproven");
                    poison(failure);
                    return failure;
                }

                @Override
                public void recover(Throwable failure) {
                    poison(failure);
                }
            };
        }

        @Override
        public void free(long address) {
            if (poisoned) throw new IllegalStateException("poisoned GPU retains the allocation");
            nativeFrees++;
            super.free(address);
        }

        @Override
        public UploadBuffer allocateUploadBuffer(long bytes) {
            UploadBuffer allocated = super.allocateUploadBuffer(bytes);
            return new UploadBuffer(allocated.segment(), () -> {
                hostReleases++;
                allocated.close();
            });
        }

        @Override
        public boolean completionProven() {
            return !poisoned;
        }

        @Override
        public void poison(Throwable failure) {
            poisoned = true;
        }

        @Override
        public void ensureHealthy() {
            if (poisoned) throw new IllegalStateException("GPU engine is poisoned");
        }
    }

    @Test
    void cancelledQuantumsQueuedRootRetiresWithoutSubmittingWork() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var runtime = new EuhedralInferenceRuntime(this.lattice, plan, gpu);
        var cancelled = context(plan, 403);
        var cancelledOutcome = runtime.submit(cancelled);
        assertEquals(0, this.lattice.pull(ignored -> {}, frame -> true, 1), "the stop predicate holds the root");
        cancelled.cancel();
        this.lattice.drive();
        assertEquals(
                QwenExecutionContext.Status.CANCELLED, cancelledOutcome.join().status());
        assertTrue(gpu.operations.isEmpty(), "a cancelled root submits nothing");
        assertTrue(gpu.frees.isEmpty(), "the graph keeps its workspace storage");
        var next = context(plan, 404);
        var nextOutcome = runtime.submit(next);
        this.lattice.drive();
        assertEquals(QwenExecutionContext.Status.SUCCESS, nextOutcome.join().status());
        runtime.close();
        QwenExecutionFixtures.assertEachAllocationFreedOnce(gpu);
    }

    @Test
    void concurrentFinishersRecycleGraphsForLaterQuanta() throws Exception {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var runtime = new EuhedralInferenceRuntime(this.lattice, plan, gpu);
        var firstContexts = List.of(context(plan, 409), context(plan, 410));
        List<AbstractFrame> firstRoots = admitAndPull(runtime, firstContexts);
        assertNotSame(firstRoots.getFirst(), firstRoots.getLast(), "concurrent quanta use separate graphs");
        firstRoots.forEach(AbstractFrame::execute);
        runConcurrently(firstRoots, AbstractFrame::doFinally);
        this.lattice.drive();
        for (QwenExecutionContext context : firstContexts) {
            assertEquals(
                    QwenExecutionContext.Status.SUCCESS,
                    context.outcome().join().status());
        }

        var secondContexts = List.of(context(plan, 411), context(plan, 412));
        List<AbstractFrame> secondRoots = admitAndPull(runtime, secondContexts);
        assertEquals(new HashSet<>(firstRoots), new HashSet<>(secondRoots), "no graph was built on the hot path");
        secondRoots.forEach(frame -> {
            frame.execute();
            frame.doFinally();
        });
        this.lattice.drive();
        for (QwenExecutionContext context : secondContexts) {
            assertEquals(
                    QwenExecutionContext.Status.SUCCESS,
                    context.outcome().join().status());
        }
        runtime.close();
    }

    private List<AbstractFrame> admitAndPull(EuhedralInferenceRuntime runtime, List<QwenExecutionContext> contexts) {
        for (QwenExecutionContext context : contexts) runtime.submit(context);
        List<AbstractFrame> frames = new ArrayList<>(contexts.size());
        assertEquals(contexts.size(), this.lattice.pull(frames::add, frame -> false, contexts.size()));
        return frames;
    }

    private static void runConcurrently(List<AbstractFrame> frames, Consumer<? super AbstractFrame> action)
            throws Exception {
        try (ExecutorService workers = Executors.newFixedThreadPool(frames.size())) {
            CountDownLatch ready = new CountDownLatch(frames.size());
            CountDownLatch start = new CountDownLatch(1);
            List<Future<?>> returned = new ArrayList<>(frames.size());
            for (AbstractFrame frame : frames) {
                returned.add(workers.submit(() -> {
                    ready.countDown();
                    start.await();
                    action.accept(frame);
                    return null;
                }));
            }
            ready.await();
            start.countDown();
            for (Future<?> future : returned) future.get();
        }
    }

    private static QwenExecutionContext context(QwenExecutionPlan plan, long sequenceId) {
        return new QwenExecutionContext(
                plan, new QwenSequenceState(sequenceId), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
    }

    /// Pulls one frame at a time, as a worker would, and returns the quantum's stage frames in order.
    private List<AbstractFrame> runQuantum(EuhedralInferenceRuntime runtime, QwenExecutionPlan plan, long sequenceId) {
        var context = context(plan, sequenceId);
        var outcome = runtime.submit(context);
        List<AbstractFrame> stages = new ArrayList<>();
        while (!outcome.isDone()) {
            AbstractFrame frame = pullOne();
            if (frame instanceof QwenStageFrame) stages.add(frame);
            frame.execute();
            frame.doFinally();
        }
        assertEquals(QwenExecutionContext.Status.SUCCESS, outcome.join().status());
        return stages;
    }

    private AbstractFrame pullOne() {
        List<AbstractFrame> next = new ArrayList<>(1);
        assertEquals(1, this.lattice.pull(next::add, frame -> false, 1));
        return next.getFirst();
    }
}
