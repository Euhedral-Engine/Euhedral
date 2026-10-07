package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.gpu.InlineGpuStream;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class QuantumTest {

    /// A stream whose recovery cannot prove that submitted initialization stopped.
    private static class UnrecoverableStream extends InlineGpuStream {
        private final ExecutionFixtures.RecordingGpu gpu;
        private final boolean failSubmission;

        UnrecoverableStream(ExecutionFixtures.RecordingGpu gpu, boolean failSubmission) {
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

    private static class PoisonableGpu extends ExecutionFixtures.RecordingGpu {
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
        public boolean completionProven() {
            return !poisoned;
        }

        @Override
        public void free(long address) {
            if (poisoned) throw new IllegalStateException("unproven allocation retained");
            super.free(address);
        }
    }

    @Test
    void failedSubmissionAfterInitializationFailsWithoutFreeingUnprovenWork() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new PoisonableGpu();
        gpu.failSubmission = true;
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var context = new Quantum(plan, new Sequence(899), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        assertThrows(IllegalStateException.class, () -> runtime.submit(context));
        assertTrue(gpu.poisoned);
        assertTrue(context.outcome().isDone());
        assertEquals(Quantum.Status.FAILED, context.outcome().join().status());
        assertTrue(gpu.frees.isEmpty());
        assertEquals(0, runtime.activeQuanta());
        runtime.close();
    }

    @Test
    void admissionThatFailsBeforePreparationFailsTheQuantumAndRejectsARetry() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var selectionFailure = new IllegalStateException("stream selection failed");
        var failSelection = new AtomicBoolean(true);
        var gpu = new ExecutionFixtures.RecordingGpu() {
            @Override
            public GpuStream openStream() {
                return new InlineGpuStream() {
                    @Override
                    public void submit(Runnable launches, boolean overlapPredecessor) {
                        if (failSelection.get()) throw selectionFailure;
                        launches.run();
                    }
                };
            }
        };
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var sequence = new Sequence(905);
        var context = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});

        assertSame(selectionFailure, assertThrows(IllegalStateException.class, () -> runtime.submit(context)));
        assertEquals(Quantum.Status.FAILED, context.outcome().join().status());
        failSelection.set(false);
        // A retry would claim a second lease and workspace that the finished outcome never releases.
        assertThrows(Quantum.DuplicateAdmissionException.class, () -> runtime.submit(context));
        assertFalse(sequence.isExecutionClaimed());
        assertTrue(gpu.allocations.isEmpty());
        assertEquals(0, runtime.activeQuanta());

        var next = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        assertEquals(
                Quantum.Status.SUCCESS,
                runtime.submit(next).get(2, TimeUnit.SECONDS).status());
        runtime.close();
    }

    @Test
    void outcomeReachedAtAdmissionIsPublishedWithTheStreamDeselected() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var selected = new AtomicInteger();
        var gpu = new ExecutionFixtures.RecordingGpu() {
            @Override
            public GpuStream openStream() {
                return new InlineGpuStream() {
                    @Override
                    public void submit(Runnable launches, boolean overlapPredecessor) {
                        selected.incrementAndGet();
                        try {
                            launches.run();
                        } finally {
                            selected.decrementAndGet();
                        }
                    }
                };
            }
        };
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var cancelledSequence = new Sequence(906);
        cancelledSequence.cancel();
        var cancelled = new Quantum(plan, cancelledSequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        var invalid =
                new Quantum(plan, new Sequence(907), Quantum.ExecutionKind.DECODE, 0, new int[] {Integer.MAX_VALUE});
        List<Integer> selectedAtOutcome = new ArrayList<>();
        cancelled.outcome().whenComplete((outcome, failure) -> selectedAtOutcome.add(selected.get()));
        invalid.outcome().whenComplete((outcome, failure) -> selectedAtOutcome.add(selected.get()));

        assertEquals(Quantum.Status.CANCELLED, runtime.submit(cancelled).join().status());
        assertEquals(Quantum.Status.FAILED, runtime.submit(invalid).join().status());
        // An outcome callback that launched work would otherwise land on the graph's stream.
        assertEquals(List.of(0, 0), selectedAtOutcome);
        runtime.close();
    }

    @Test
    void admissionRefusedByAClosedRuntimeFailsTheQuantum() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var runtime = ExecutionFixtures.runtime(plan, new ExecutionFixtures.RecordingGpu());
        runtime.close();
        var context = new Quantum(plan, new Sequence(908), Quantum.ExecutionKind.DECODE, 0, new int[] {1});

        assertThrows(IllegalStateException.class, () -> runtime.submit(context));
        assertEquals(Quantum.Status.FAILED, context.outcome().join().status());
        assertThrows(Quantum.DuplicateAdmissionException.class, () -> runtime.submit(context));
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void closeDuringGraphBuildReleasesTheNewGraphsStream() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var building = new CountDownLatch(1);
        var closed = new CountDownLatch(1);
        var streamClosed = new AtomicBoolean();
        var gpu = new ExecutionFixtures.RecordingGpu() {
            @Override
            public GpuStream openStream() {
                building.countDown();
                try {
                    closed.await();
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(interrupted);
                }
                return new InlineGpuStream() {
                    @Override
                    public void close() {
                        streamClosed.set(true);
                    }
                };
            }
        };
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var context = new Quantum(plan, new Sequence(909), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        try (var admission = Executors.newSingleThreadExecutor()) {
            Future<?> submitted = admission.submit(() -> runtime.submit(context));
            assertTrue(building.await(10, TimeUnit.SECONDS));
            runtime.close();
            closed.countDown();
            var failure = assertThrows(ExecutionException.class, () -> submitted.get(10, TimeUnit.SECONDS));
            assertInstanceOf(IllegalStateException.class, failure.getCause());
        }
        assertTrue(streamClosed.get(), "close() never saw this graph, so its build must release the stream");
        assertEquals(Quantum.Status.FAILED, context.outcome().join().status());
    }

    @Test
    void failedInitializationRetainsBuffersIfRecoveryCannotComplete() throws Exception {
        var plan = new ExecutionPlan(
                ExecutionFixtures.weights(),
                ExecutionFixtures.norm(),
                List.of(ExecutionFixtures.q3("projection", 64, 201)));
        var gpu = new PoisonableGpu();
        gpu.failAllocationAt = 2;
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var context = new Quantum(plan, new Sequence(900), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        assertEquals(
                Quantum.Status.FAILED,
                runtime.submit(context).get(2, TimeUnit.SECONDS).status());
        assertTrue(gpu.poisoned);
        assertEquals(0, gpu.frees.size());
        runtime.close();
    }

    @Test
    void cancellationCleanupFailureStillPublishesOutcomeAndCanBeRetried() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new ExecutionFixtures.RecordingGpu();
        var sequence = new Sequence(901);
        var lease = sequence.claimExecution(0);
        var attempts = new AtomicInteger();
        sequence.setRecurrentState(lease, (AutoCloseable) () -> {
            if (attempts.incrementAndGet() == 1) throw new IllegalStateException("transient free failure");
        });
        sequence.releaseExecution(lease, 0);
        var context = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        gpu.afterEmbedding = context::cancel;
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context);
        assertEquals(
                Quantum.Status.FAILED,
                outcome.get(10, TimeUnit.SECONDS).status(),
                "terminal cleanup failure stranded the generation future");
        assertFalse(sequence.isExecutionClaimed());
        sequence.complete();
        assertEquals(2, attempts.get());
        runtime.close();
        assertFalse(runtime.isAttached());
    }

    @Test
    void stagesRunEmbeddingNormAndIndependentProjectionsWithoutDeviceBarriers() throws Exception {
        var weights = ExecutionFixtures.weights();
        var plan = new ExecutionPlan(
                weights,
                ExecutionFixtures.norm(),
                List.of(ExecutionFixtures.q3("first", 64, 201), ExecutionFixtures.q3("second", 128, 202)));
        assertEquals(List.of(2, 3), plan.successors(1));
        var gpu = new ExecutionFixtures.RecordingGpu();
        var sequence = new Sequence(10);
        var context = new Quantum(plan, sequence, Quantum.ExecutionKind.PREFILL, 0, new int[] {1, 2});
        List<Long> outputs = new ArrayList<>();
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context, completed -> {
            outputs.add(completed.workspace().projectionAddress(0));
            outputs.add(completed.workspace().projectionAddress(1));
            assertTrue(sequence.isExecutionClaimed());
            assertFalse(completed.workspace().isClosed());
        });

        assertEquals(Quantum.Status.SUCCESS, outcome.get(5, TimeUnit.SECONDS).status());
        assertEquals(List.of("embed", "norm", "linear:201", "linear:202"), gpu.operations);
        assertEquals(0, gpu.synchronizations, "stream order, not device barriers, sequences the stages");
        assertNotEquals(outputs.get(0), outputs.get(1));
        assertTrue(gpu.frees.isEmpty(), "the graph retains its workspace storage for its next quantum");
        assertTrue(context.workspace().isClosed());
        assertEquals(2, sequence.currentTokenPosition());
        runtime.close();
        ExecutionFixtures.assertEachAllocationFreedOnce(gpu);
        assertFalse(gpu.frees.contains(ExecutionFixtures.MODEL_ADDRESS));
    }

    @Test
    void pullHonorsStopWithoutConsumingOrGeneratingFrame() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, new ExecutionFixtures.RecordingGpu());
        var context = new Quantum(plan, new Sequence(11), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        runtime.submit(context);
        List<AbstractFrame> pulled = new ArrayList<>();
        assertEquals(0, lattice.pull(pulled::add, frame -> true, 1));
        assertTrue(pulled.isEmpty());
        assertEquals(1, lattice.pull(pulled::add, frame -> false, 1));
        assertEquals(1, pulled.size(), "admission exposes only the root stage");
        // The caller owns the pulled frame, including its execution and terminal notification.
        pulled.getFirst().execute();
        pulled.getFirst().doFinally();
        assertFalse(context.outcome().isDone(), "the retirement frame still waits for Euhedral");
        lattice.drive();
        assertEquals(Quantum.Status.SUCCESS, context.outcome().join().status());
        runtime.close();
    }

    @Test
    void pullDrainsReadyFramesWithoutBorrowingBeyondDemand() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, new ExecutionFixtures.RecordingGpu());
        var first = new Quantum(plan, new Sequence(101), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        var second = new Quantum(plan, new Sequence(102), Quantum.ExecutionKind.DECODE, 0, new int[] {2});
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
        assertEquals(Quantum.Status.SUCCESS, first.outcome().join().status());
        assertEquals(Quantum.Status.SUCCESS, second.outcome().join().status());
        runtime.close();
    }

    @Test
    void cancellationAfterEmbeddingStopsSuccessorsAndReleasesWorkspace() throws Exception {
        var plan = new ExecutionPlan(
                ExecutionFixtures.weights(),
                ExecutionFixtures.norm(),
                List.of(ExecutionFixtures.q3("projection", 64, 201)));
        var gpu = new ExecutionFixtures.RecordingGpu();
        var context = new Quantum(plan, new Sequence(12), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        gpu.afterEmbedding = context::cancel;
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context);
        assertEquals(Quantum.Status.CANCELLED, outcome.get(5, TimeUnit.SECONDS).status());
        assertEquals(List.of("embed"), gpu.operations);
        assertTrue(context.workspace().isClosed());
        runtime.close();
        ExecutionFixtures.assertEachAllocationFreedOnce(gpu);
    }

    @Test
    void linearFailureStopsOtherWorkAndDoesNotFreeWeights() throws Exception {
        var plan = new ExecutionPlan(
                ExecutionFixtures.weights(),
                ExecutionFixtures.norm(),
                List.of(ExecutionFixtures.q3("first", 64, 201), ExecutionFixtures.q3("second", 64, 202)));
        var gpu = new ExecutionFixtures.RecordingGpu();
        var failure = new IllegalStateException("injected linear failure");
        gpu.linearFailure = failure;
        var context = new Quantum(plan, new Sequence(13), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context);
        assertEquals(Quantum.Status.FAILED, outcome.get(5, TimeUnit.SECONDS).status());
        assertSame(failure, outcome.get().failure());
        assertEquals(List.of("embed", "norm", "linear:201"), gpu.operations);
        runtime.close();
        ExecutionFixtures.assertEachAllocationFreedOnce(gpu);
        assertFalse(gpu.frees.contains(ExecutionFixtures.MODEL_ADDRESS));
    }

    @Test
    void retirementReleasesTheWorkspaceBindingWithoutFreeingDeviceStorage() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new ExecutionFixtures.RecordingGpu();
        // Nothing is freed on the retirement path, so a free failure cannot fail the quantum.
        gpu.freeFailures = 1;
        var context = new Quantum(plan, new Sequence(14), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context);
        assertEquals(Quantum.Status.SUCCESS, outcome.get(5, TimeUnit.SECONDS).status());
        assertTrue(context.workspace().isClosed());
        assertThrows(IllegalStateException.class, () -> context.workspace().hiddenStateAddress());
        assertTrue(gpu.frees.isEmpty());
        assertEquals(2, gpu.allocations.size(), "hidden state and token IDs");
        // Release happens once the runtime proves every graph retired; a failed free is reported there.
        assertThrows(IllegalStateException.class, runtime::close);
        assertEquals(1, gpu.frees.size());
    }

    @Test
    void partiallyAllocatedWorkspaceStaysWithItsGraphForTheNextQuantum() {
        var plan = new ExecutionPlan(
                ExecutionFixtures.weights(),
                ExecutionFixtures.norm(),
                List.of(ExecutionFixtures.q3("projection", 64, 201)));
        var gpu = new ExecutionFixtures.RecordingGpu();
        gpu.failAllocationAt = 2;
        var context = new Quantum(plan, new Sequence(103), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        var runtime = ExecutionFixtures.runtime(plan, gpu);

        assertEquals(Quantum.Status.FAILED, runtime.submit(context).join().status());
        assertEquals(1, gpu.allocations.size());
        assertTrue(gpu.frees.isEmpty(), "the failed admission queued nothing; its graph keeps the allocation");
        assertTrue(context.workspace().isClosed());

        var next = new Quantum(plan, new Sequence(104), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        assertEquals(Quantum.Status.SUCCESS, runtime.submit(next).join().status());
        assertEquals(4, gpu.allocations.size(), "the next quantum reused the first slot and filled the rest");
        runtime.close();
        ExecutionFixtures.assertEachAllocationFreedOnce(gpu);
    }

    @Test
    void cancellationBetweenAdmissionCheckAndLeaseClaimPublishesCancelled() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new ExecutionFixtures.RecordingGpu();
        var sequence = new Sequence(104);
        var context = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});

        context.begin(gpu, sequence::cancel);

        assertEquals(Quantum.Status.CANCELLED, context.outcome().join().status());
        assertEquals(Sequence.TerminalState.CANCELLED, sequence.terminalState());
        assertTrue(gpu.allocations.isEmpty());
    }

    @Test
    void rejectedTokenDoesNotPoisonUnclaimedSequence() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new ExecutionFixtures.RecordingGpu();
        var sequence = new Sequence(15);
        var context =
                new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {ExecutionFixtures.VOCABULARY});
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        assertEquals(Quantum.Status.FAILED, runtime.submit(context).join().status());
        assertEquals(Sequence.TerminalState.ACTIVE, sequence.terminalState());
        assertTrue(gpu.allocations.isEmpty());
        assertEquals(0, runtime.activeQuanta());
        runtime.close();
    }

    @Test
    void duplicateAdmissionDoesNotRegisterAnotherTerminalOwner() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, new ExecutionFixtures.RecordingGpu());
        var context = new Quantum(plan, new Sequence(17), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        var first = runtime.submit(context);
        int terminalDependents = context.completion().getNumberOfDependents();
        assertThrows(IllegalStateException.class, () -> runtime.submit(context));
        assertEquals(terminalDependents, context.completion().getNumberOfDependents());
        assertEquals(1, runtime.activeQuanta());
        lattice.drive();
        assertEquals(Quantum.Status.SUCCESS, first.join().status());
        runtime.close();
        assertTrue(lattice.source.isComplete());
    }

    @Test
    void callersCannotCompleteTheInternalQuantumOutcome() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new ExecutionFixtures.RecordingGpu();
        var context = new Quantum(plan, new Sequence(18), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        var exposed = runtime.submit(context);
        exposed.complete(new Quantum.Outcome(Quantum.Status.SUCCESS, null));
        assertFalse(context.outcome().isDone());
        assertEquals(1, runtime.activeQuanta());
        lattice.drive();
        assertEquals(Quantum.Status.SUCCESS, context.outcome().join().status());
        assertTrue(context.workspace().isClosed());
        runtime.close();
        assertTrue(lattice.source.isComplete());
    }

    @Test
    void failureRecordedDuringTerminalConsumerCannotCommitSuccess() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new ExecutionFixtures.RecordingGpu();
        var context = new Quantum(plan, new Sequence(19), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        var failure = new IllegalStateException("external quantum failure");
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context, ignored -> context.fail(failure));
        assertEquals(Quantum.Status.FAILED, outcome.join().status());
        assertSame(failure, outcome.join().failure());
        assertEquals(Sequence.TerminalState.FAILED, context.sequenceState().terminalState());
        runtime.close();
    }
}
