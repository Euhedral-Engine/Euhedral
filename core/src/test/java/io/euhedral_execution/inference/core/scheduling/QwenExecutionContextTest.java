package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.gpu.InlineGpuStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class QwenExecutionContextTest {

    /// A stream whose recovery cannot prove that submitted initialization stopped.
    private static class UnrecoverableStream extends InlineGpuStream {
        private final QwenExecutionFixtures.RecordingGpu gpu;
        private final boolean failSubmission;

        UnrecoverableStream(QwenExecutionFixtures.RecordingGpu gpu, boolean failSubmission) {
            this.gpu = gpu;
            this.failSubmission = failSubmission;
        }

        @Override
        public void submit(Runnable launches, boolean overlapPredecessor) {
            launches.run();
            if (this.failSubmission) throw new IllegalStateException("post-initialization submission failure");
        }

        @Override
        public void recover(Throwable failure) {
            this.gpu.poison(failure);
        }
    }

    private static class PoisonableGpu extends QwenExecutionFixtures.RecordingGpu {
        boolean poisoned;
        boolean failSubmission;

        @Override
        public GpuStream openStream() {
            return new UnrecoverableStream(this, this.failSubmission);
        }

        @Override
        public void poison(Throwable failure) {
            poisoned = true;
        }

        @Override
        public void free(long address) {
            if (poisoned) throw new IllegalStateException("unproven allocation retained");
            super.free(address);
        }
    }

    @Test
    void failedSubmissionAfterInitializationFailsWithoutFreeingUnprovenWork() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new PoisonableGpu();
        gpu.failSubmission = true;
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(899), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        assertThrows(IllegalStateException.class, () -> runtime.submit(context));
        assertTrue(gpu.poisoned);
        assertTrue(context.outcome().isDone());
        assertEquals(
                QwenExecutionContext.Status.FAILED, context.outcome().join().status());
        assertTrue(gpu.frees.isEmpty());
        assertEquals(0, runtime.activeQuanta());
        runtime.close();
    }

    @Test
    void failedInitializationRetainsBuffersIfRecoveryCannotComplete() throws Exception {
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.weights(),
                QwenExecutionFixtures.norm(),
                List.of(QwenExecutionFixtures.q3("projection", 64, 201)));
        var gpu = new PoisonableGpu();
        gpu.failAllocationAt = 2;
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(900), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        assertEquals(
                QwenExecutionContext.Status.FAILED,
                runtime.submit(context).get(2, TimeUnit.SECONDS).status());
        assertTrue(gpu.poisoned);
        assertEquals(0, gpu.frees.size());
        runtime.close();
    }

    @Test
    void cancellationCleanupFailureStillPublishesOutcomeAndCanBeRetried() throws Exception {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var sequence = new QwenSequenceState(901);
        var lease = sequence.claimExecution(0);
        var attempts = new AtomicInteger();
        sequence.setRecurrentState(lease, (AutoCloseable) () -> {
            if (attempts.incrementAndGet() == 1) throw new IllegalStateException("transient free failure");
        });
        sequence.releaseExecution(lease, 0);
        var context =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        gpu.afterEmbedding = context::cancel;
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context);
        assertTrue(outcome.isDone(), "terminal cleanup failure stranded the generation future");
        assertEquals(
                QwenExecutionContext.Status.FAILED,
                outcome.get(1, TimeUnit.SECONDS).status());
        assertFalse(sequence.isExecutionClaimed());
        sequence.complete();
        assertEquals(2, attempts.get());
        runtime.close();
        assertFalse(runtime.isAttached());
    }

    @Test
    void stagesRunEmbeddingNormAndIndependentProjectionsWithoutDeviceBarriers() throws Exception {
        var weights = QwenExecutionFixtures.weights();
        var plan = new QwenExecutionPlan(
                weights,
                QwenExecutionFixtures.norm(),
                List.of(QwenExecutionFixtures.q3("first", 64, 201), QwenExecutionFixtures.q3("second", 128, 202)));
        assertEquals(List.of(2, 3), plan.successors(1));
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var sequence = new QwenSequenceState(10);
        var context = new QwenExecutionContext(
                plan, sequence, QwenExecutionContext.ExecutionKind.PREFILL, 0, new int[] {1, 2});
        List<Long> outputs = new ArrayList<>();
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context, completed -> {
            outputs.add(completed.workspace().projectionAddress(0));
            outputs.add(completed.workspace().projectionAddress(1));
            assertTrue(sequence.isExecutionClaimed());
            assertFalse(completed.workspace().isClosed());
        });

        assertEquals(
                QwenExecutionContext.Status.SUCCESS,
                outcome.get(5, TimeUnit.SECONDS).status());
        assertEquals(List.of("embed", "norm", "linear:201", "linear:202"), gpu.operations);
        assertEquals(0, gpu.synchronizations, "stream order, not device barriers, sequences the stages");
        assertNotEquals(outputs.get(0), outputs.get(1));
        assertEquals(gpu.allocations.size(), gpu.frees.size());
        assertFalse(gpu.frees.contains(QwenExecutionFixtures.MODEL_ADDRESS));
        assertTrue(context.workspace().isClosed());
        assertEquals(2, sequence.currentTokenPosition());
        runtime.close();
    }

    @Test
    void pullHonorsStopWithoutConsumingOrGeneratingFrame() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var lattice = new QwenExecutionFixtures.ManualLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, new QwenExecutionFixtures.RecordingGpu());
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(11), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        runtime.submit(context);
        List<AbstractFrame> pulled = new ArrayList<>();
        assertEquals(0, lattice.source.pull(pulled::add, frame -> true, 1));
        assertTrue(pulled.isEmpty());
        assertEquals(1, lattice.source.pull(pulled::add, frame -> false, 1));
        assertEquals(1, pulled.size(), "admission exposes only the root stage");
        // The caller owns the pulled frame, including its execution and terminal notification.
        pulled.getFirst().execute();
        pulled.getFirst().doFinally();
        assertFalse(context.outcome().isDone(), "the retirement frame still waits for Euhedral");
        lattice.drive();
        assertEquals(
                QwenExecutionContext.Status.SUCCESS, context.outcome().join().status());
        runtime.close();
    }

    @Test
    void pullDrainsReadyFramesWithoutBorrowingBeyondDemand() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var lattice = new QwenExecutionFixtures.ManualLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, new QwenExecutionFixtures.RecordingGpu());
        var first = new QwenExecutionContext(
                plan, new QwenSequenceState(101), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var second = new QwenExecutionContext(
                plan, new QwenSequenceState(102), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {2});
        runtime.submit(first);
        runtime.submit(second);
        List<AbstractFrame> borrowed = new ArrayList<>();
        assertEquals(2, lattice.pull(borrowed::add, frame -> false, 3), "each quantum exposes only its root");
        assertEquals(0, lattice.pull(borrowed::add, frame -> false, 3));
        for (AbstractFrame frame : borrowed) {
            frame.execute();
            frame.doFinally();
        }
        lattice.drive();
        assertEquals(QwenExecutionContext.Status.SUCCESS, first.outcome().join().status());
        assertEquals(
                QwenExecutionContext.Status.SUCCESS, second.outcome().join().status());
        runtime.close();
    }

    @Test
    void demandLeftByAnEmptyRequestIsServedWhenAStageBecomesReady() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var pushes = new AtomicInteger();
        var source = new java.util.concurrent.atomic.AtomicReference<LatticeSource>();
        var runtime = new EuhedralInferenceRuntime(
                attached -> {
                    source.set(attached);
                    attached.addDownstream(new LatticeReceiver() {
                        @Override
                        public void addUpstream(LatticeSource upstream) {}

                        @Override
                        public void push(AbstractFrame frame) {
                            pushes.incrementAndGet();
                            frame.execute();
                            frame.doFinally();
                        }

                        @Override
                        public void onComplete() {}

                        @Override
                        public void onError(Throwable error) {
                            throw new AssertionError(error);
                        }
                    });
                },
                plan,
                new QwenExecutionFixtures.RecordingGpu());
        // The graph and its source are built by the first quantum; pull drives it without leaving demand.
        var warm = runtime.submit(new QwenExecutionContext(
                plan, new QwenSequenceState(15), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1}));
        source.get().pull(PullingLattice::run, frame -> false, Long.MAX_VALUE);
        assertEquals(QwenExecutionContext.Status.SUCCESS, warm.join().status());
        assertEquals(0, pushes.get());

        source.get().request(1);
        assertEquals(0, pushes.get(), "nothing was ready");
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(16), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var outcome = runtime.submit(context);
        assertEquals(1, pushes.get(), "the earlier demand delivered the root when admission published it");
        assertFalse(outcome.isDone(), "the demand was consumed; retirement waits for more");
        source.get().request(1);
        assertEquals(2, pushes.get());
        assertEquals(QwenExecutionContext.Status.SUCCESS, outcome.join().status());
        runtime.close();
    }

    @Test
    void cancellationAfterEmbeddingStopsSuccessorsAndReleasesWorkspace() throws Exception {
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.weights(),
                QwenExecutionFixtures.norm(),
                List.of(QwenExecutionFixtures.q3("projection", 64, 201)));
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(12), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        gpu.afterEmbedding = context::cancel;
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context);
        assertEquals(
                QwenExecutionContext.Status.CANCELLED,
                outcome.get(5, TimeUnit.SECONDS).status());
        assertEquals(List.of("embed"), gpu.operations);
        assertEquals(gpu.allocations.size(), gpu.frees.size());
        assertTrue(context.workspace().isClosed());
        runtime.close();
    }

    @Test
    void linearFailureStopsOtherWorkAndDoesNotFreeWeights() throws Exception {
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.weights(),
                QwenExecutionFixtures.norm(),
                List.of(QwenExecutionFixtures.q3("first", 64, 201), QwenExecutionFixtures.q3("second", 64, 202)));
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var failure = new IllegalStateException("injected linear failure");
        gpu.linearFailure = failure;
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(13), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context);
        assertEquals(
                QwenExecutionContext.Status.FAILED,
                outcome.get(5, TimeUnit.SECONDS).status());
        assertSame(failure, outcome.get().failure());
        assertEquals(List.of("embed", "norm", "linear:201"), gpu.operations);
        assertEquals(gpu.allocations.size(), gpu.frees.size());
        assertFalse(gpu.frees.contains(QwenExecutionFixtures.MODEL_ADDRESS));
        runtime.close();
    }

    @Test
    void failedTemporaryFreeIsRetriedAtTerminalCleanup() throws Exception {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        gpu.freeFailures = 1;
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(14), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context);
        assertEquals(
                QwenExecutionContext.Status.FAILED,
                outcome.get(5, TimeUnit.SECONDS).status());
        assertEquals(gpu.allocations.size(), gpu.frees.size());
        runtime.close();
    }

    @Test
    void failedWorkspaceFreeIsRetriedBeforePublishingOutcome() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        gpu.failAddressOnce = 1000;
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(20), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context);
        assertEquals(QwenExecutionContext.Status.FAILED, outcome.join().status());
        assertTrue(context.workspace().isClosed());
        assertEquals(gpu.allocations.size(), gpu.frees.size());
        runtime.close();
    }

    @Test
    void partiallyAllocatedWorkspaceRemainsOwnedWhenConstructionCleanupFails() {
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.weights(),
                QwenExecutionFixtures.norm(),
                List.of(QwenExecutionFixtures.q3("projection", 64, 201)));
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        gpu.failAllocationAt = 2;
        gpu.freeFailures = 1;
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(103), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);

        assertEquals(
                QwenExecutionContext.Status.FAILED,
                runtime.submit(context).join().status());
        assertEquals(gpu.allocations, gpu.frees);
        assertTrue(context.workspace().isClosed());
        runtime.close();
    }

    @Test
    void cancellationBetweenAdmissionCheckAndLeaseClaimPublishesCancelled() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var sequence = new QwenSequenceState(104);
        var context =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});

        context.begin(gpu, sequence::cancel);

        assertEquals(
                QwenExecutionContext.Status.CANCELLED, context.outcome().join().status());
        assertEquals(QwenSequenceState.TerminalState.CANCELLED, sequence.terminalState());
        assertTrue(gpu.allocations.isEmpty());
    }

    @Test
    void rejectedTokenDoesNotPoisonUnclaimedSequence() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var sequence = new QwenSequenceState(15);
        var context = new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {
            QwenExecutionFixtures.VOCABULARY
        });
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);
        assertEquals(
                QwenExecutionContext.Status.FAILED,
                runtime.submit(context).join().status());
        assertEquals(QwenSequenceState.TerminalState.ACTIVE, sequence.terminalState());
        assertTrue(gpu.allocations.isEmpty());
        assertEquals(0, runtime.activeQuanta());
        runtime.close();
    }

    @Test
    void duplicateAdmissionDoesNotRegisterAnotherTerminalOwner() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var lattice = new QwenExecutionFixtures.ManualLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, new QwenExecutionFixtures.RecordingGpu());
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(17), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var first = runtime.submit(context);
        int terminalDependents = context.completion().getNumberOfDependents();
        assertThrows(IllegalStateException.class, () -> runtime.submit(context));
        assertEquals(terminalDependents, context.completion().getNumberOfDependents());
        assertEquals(1, runtime.activeQuanta());
        lattice.drive();
        assertEquals(QwenExecutionContext.Status.SUCCESS, first.join().status());
        runtime.close();
        assertTrue(lattice.source.isComplete());
    }

    @Test
    void callersCannotCompleteTheInternalQuantumOutcome() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(18), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var lattice = new QwenExecutionFixtures.ManualLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var exposed = runtime.submit(context);
        exposed.complete(new QwenExecutionContext.Outcome(QwenExecutionContext.Status.SUCCESS, null));
        assertFalse(context.outcome().isDone());
        assertEquals(1, runtime.activeQuanta());
        lattice.drive();
        assertEquals(
                QwenExecutionContext.Status.SUCCESS, context.outcome().join().status());
        assertTrue(context.workspace().isClosed());
        runtime.close();
        assertTrue(lattice.source.isComplete());
    }

    @Test
    void failureRecordedDuringTerminalConsumerCannotCommitSuccess() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(19), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var failure = new IllegalStateException("external quantum failure");
        var runtime = QwenExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context, ignored -> context.fail(failure));
        assertEquals(QwenExecutionContext.Status.FAILED, outcome.join().status());
        assertSame(failure, outcome.join().failure());
        assertEquals(
                QwenSequenceState.TerminalState.FAILED, context.sequenceState().terminalState());
        runtime.close();
    }
}
