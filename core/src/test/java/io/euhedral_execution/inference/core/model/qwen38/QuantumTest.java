package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu.UploadBuffer;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.gpu.InlineGpuStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
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
        // The refused admission is the quantum's outcome; nothing is thrown to the submitter.
        assertInstanceOf(
                IllegalStateException.class, runtime.submit(context).join().failure());
        assertTrue(gpu.poisoned);
        assertTrue((context.conclusion() != null));
        assertEquals(Quantum.Status.FAILED, context.conclusion().status());
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
        int atLoad = gpu.allocations.size();
        var sequence = new Sequence(905);
        var context = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});

        var refused = runtime.submit(context).get(2, TimeUnit.SECONDS);
        assertEquals(Quantum.Status.FAILED, refused.status());
        assertSame(selectionFailure, refused.failure());
        failSelection.set(false);
        // A retry would take a second sequence admission and workspace that the finished outcome never releases.
        assertThrows(Quantum.DuplicateAdmissionException.class, () -> runtime.submit(context));
        assertFalse(sequence.inFlight());
        assertEquals(atLoad, gpu.allocations.size(), "the refused quantum allocated nothing");
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
        // The callbacks may run on two threads at once: the test thread and the worker that completed the outcome.
        List<Integer> selectedAtOutcome = new java.util.concurrent.CopyOnWriteArrayList<>();
        var cancelledOutcome = runtime.submit(cancelled);
        var cancelledSeen = cancelledOutcome.whenComplete((outcome, failure) -> selectedAtOutcome.add(selected.get()));
        var invalidOutcome = runtime.submit(invalid);
        var invalidSeen = invalidOutcome.whenComplete((outcome, failure) -> selectedAtOutcome.add(selected.get()));

        assertEquals(Quantum.Status.CANCELLED, cancelledOutcome.join().status());
        assertEquals(Quantum.Status.FAILED, invalidOutcome.join().status());
        cancelledSeen.join();
        invalidSeen.join();
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

        assertEquals(Quantum.Status.FAILED, runtime.submit(context).join().status());
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
            Future<java.util.concurrent.CompletableFuture<Quantum.Outcome>> submitted =
                    admission.submit(() -> runtime.submit(context));
            assertTrue(building.await(10, TimeUnit.SECONDS));
            var closing = admission.submit(runtime::close);
            closed.countDown();
            var outcome = submitted.get(10, TimeUnit.SECONDS).get(10, TimeUnit.SECONDS);
            assertInstanceOf(IllegalStateException.class, outcome.failure());
            closing.get(10, TimeUnit.SECONDS);
        }
        assertTrue(streamClosed.get(), "close() never saw this graph, so its build must release the stream");
        assertEquals(Quantum.Status.FAILED, context.conclusion().status());
    }

    @Test
    void failedInitializationRetainsBuffersIfRecoveryCannotComplete() throws Exception {
        var plan = new ExecutionPlan(
                ExecutionFixtures.weights(),
                ExecutionFixtures.norm(),
                List.of(ExecutionFixtures.q3("projection", 64, 201)));
        var gpu = new PoisonableGpu();
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        // The workspace allocated at load; the quantum's own input record is the next allocation, and it fails.
        gpu.failAllocationAt = gpu.allocationAttempts + 1;
        var context = new Quantum(plan, new Sequence(900), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        assertEquals(
                Quantum.Status.FAILED,
                runtime.submit(context).get(2, TimeUnit.SECONDS).status());
        assertTrue(gpu.poisoned);
        assertEquals(0, gpu.frees.size());
        runtime.close();
    }

    @Test
    void aCancelledQuantumLeavesCleanupToCompleteWhichReportsAndRetriesAFailedFree() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new ExecutionFixtures.RecordingGpu();
        var sequence = new Sequence(901);
        var held = HeldWork.admit(sequence);
        var attempts = new AtomicInteger();
        sequence.setRecurrentState((AutoCloseable) () -> {
            if (attempts.incrementAndGet() == 1) throw new IllegalStateException("transient free failure");
        });
        held.commit(0);
        var context = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        gpu.afterEmbedding = context::cancel;
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context);
        assertEquals(Quantum.Status.CANCELLED, outcome.get(10, TimeUnit.SECONDS).status());
        assertFalse(sequence.inFlight());
        assertEquals(0, attempts.get(), "the sequence's state closes only when its owner completes it");
        assertThrows(IllegalStateException.class, sequence::complete);
        sequence.complete();
        assertEquals(2, attempts.get());
        runtime.close();
        assertFalse(runtime.isAttached());
    }

    @Test
    @Timeout(10)
    void aReleaseFailureAfterACancellationFailsTheQuantumAndTheSequence() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var releaseFailure = new IllegalStateException("injected upload release failure");
        var gpu = new ExecutionFixtures.RecordingGpu() {
            @Override
            public UploadBuffer allocateUploadBuffer(long bytes) {
                var arena = java.lang.foreign.Arena.ofShared();
                return new UploadBuffer(arena.allocate(bytes, Integer.BYTES), () -> {
                    arena.close();
                    throw releaseFailure;
                });
            }
        };
        var sequence = new Sequence(906);
        var context = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        gpu.afterEmbedding = context::cancel;
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var outcome = runtime.submit(context).get(5, TimeUnit.SECONDS);
        assertEquals(Quantum.Status.FAILED, outcome.status(), "the failed release is not lost in the cancellation");
        assertSame(releaseFailure, outcome.failure());
        assertEquals(Sequence.TerminalState.FAILED, sequence.terminalState());
        assertFalse(sequence.inFlight());
        runtime.close();
    }

    @Test
    @Timeout(10)
    void quantaOfOneSequenceInFlightConcludeInAdmissionOrderWhateverOrderTheyRetire() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var stream = new ExecutionFixtures.HoldingStream();
        var gpu = new ExecutionFixtures.RecordingGpu() {
            @Override
            public GpuStream openStream() {
                return stream;
            }
        };
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, gpu);
        var sequence = new Sequence(907);
        List<String> concluded = new ArrayList<>();
        var first = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        var second = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 1, new int[] {2});
        try {
            var firstOutcome = runtime.submit(first, quantum -> concluded.add("first"));
            lattice.drive();
            assertTrue(sequence.inFlight());
            assertEquals(1, sequence.submittedFrontier(), "the next quantum starts where the first will leave it");
            var secondOutcome = runtime.submit(second, quantum -> concluded.add("second"));
            lattice.drive();
            assertEquals(2, stream.held());

            stream.releaseNewest(null);
            lattice.drive();
            assertEquals(List.of(), concluded, "the second retired first but waits for the first");
            assertFalse(secondOutcome.isDone());
            assertEquals(0, sequence.committedFrontier());

            stream.release(null);
            lattice.drive();
            assertEquals(List.of("first", "second"), concluded);
            assertEquals(
                    Quantum.Status.SUCCESS,
                    firstOutcome.get(2, TimeUnit.SECONDS).status());
            assertEquals(
                    Quantum.Status.SUCCESS,
                    secondOutcome.get(2, TimeUnit.SECONDS).status());
            assertEquals(2, sequence.committedFrontier());
            assertFalse(sequence.inFlight());
        } finally {
            releaseAndClose(stream, lattice, runtime);
        }
    }

    @Test
    @Timeout(10)
    void aQuantumThatSharesSequenceStateIsRefusedWhileAnotherIsInFlight() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var plan = new ExecutionPlan(weights);
        var stream = new ExecutionFixtures.HoldingStream();
        var gpu = new EngineExecutionFixture.SamplingGpu(weights.config().vocabSize()) {
            @Override
            public GpuStream openStream() {
                return stream;
            }
        };
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, gpu);
        var sequence = new Sequence(909);
        var first = new Quantum(plan, sequence, Quantum.ExecutionKind.PREFILL, 0, new int[64], LogitsRequirement.NONE);
        var second = new Quantum(plan, sequence, Quantum.ExecutionKind.PREFILL, 64, new int[1], LogitsRequirement.NONE);
        try {
            var firstOutcome = runtime.submit(first);
            lattice.drive();
            var secondOutcome = runtime.submit(second);
            lattice.drive();
            var refused = secondOutcome.get(2, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.FAILED, refused.status());
            assertInstanceOf(IllegalStateException.class, refused.failure());
            assertEquals(Sequence.TerminalState.ACTIVE, sequence.terminalState(), "the refusal leaves the sequence");

            stream.release(null);
            lattice.drive();
            assertEquals(
                    Quantum.Status.SUCCESS,
                    firstOutcome.get(2, TimeUnit.SECONDS).status());
            assertEquals(64, sequence.committedFrontier());
        } finally {
            releaseAndClose(stream, lattice, runtime);
            sequence.complete();
        }
    }

    @Test
    @Timeout(10)
    void aQuantumAdmittedBehindAFailedOneFailsWithoutCommitting() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var stream = new ExecutionFixtures.HoldingStream();
        var gpu = new ExecutionFixtures.RecordingGpu() {
            @Override
            public GpuStream openStream() {
                return stream;
            }
        };
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, gpu);
        var sequence = new Sequence(908);
        var first = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        var second = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 1, new int[] {2});
        try {
            var firstOutcome = runtime.submit(first);
            lattice.drive();
            var secondOutcome = runtime.submit(second);
            lattice.drive();
            var deviceFailure = new IllegalStateException("injected device failure");
            stream.release(deviceFailure);
            stream.release(null);
            lattice.drive();
            assertEquals(
                    Quantum.Status.FAILED, firstOutcome.get(2, TimeUnit.SECONDS).status());
            var blocked = secondOutcome.get(2, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.FAILED, blocked.status());
            assertSame(deviceFailure, blocked.failure().getCause(), "it names the failure before it");
            assertEquals(0, sequence.committedFrontier());
            assertEquals(Sequence.TerminalState.FAILED, sequence.terminalState());
            assertFalse(sequence.inFlight());
        } finally {
            releaseAndClose(stream, lattice, runtime);
        }
    }

    /// Lets every boundary the stream still holds retire before closing: a test that failed mid-way must not leave
    /// close waiting on a quantum whose device work never retires.
    private static void releaseAndClose(
            ExecutionFixtures.HoldingStream stream, ExecutionFixtures.ManualLattice lattice, Execution runtime) {
        while (stream.held() > 0) stream.release(null);
        lattice.drive();
        runtime.close();
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
            assertTrue(sequence.inFlight());
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
        var runtime = new Execution(lattice, plan, new ExecutionFixtures.RecordingGpu());
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
        assertFalse((context.conclusion() != null), "the retirement frame still waits for Euhedral");
        lattice.drive();
        assertEquals(Quantum.Status.SUCCESS, context.conclusion().status());
        runtime.close();
    }

    @Test
    void pullDrainsReadyFramesWithoutBorrowingBeyondDemand() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, new ExecutionFixtures.RecordingGpu());
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
        assertEquals(Quantum.Status.SUCCESS, first.conclusion().status());
        assertEquals(Quantum.Status.SUCCESS, second.conclusion().status());
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
        var context = new Quantum(plan, new Sequence(103), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        int atLoad = gpu.allocations.size();
        // The workspace allocated at load; the graph's input record is the quantum's one allocation, and it fails.
        gpu.failAllocationAt = gpu.allocationAttempts + 1;

        assertEquals(Quantum.Status.FAILED, runtime.submit(context).join().status());
        assertEquals(atLoad, gpu.allocations.size());
        assertTrue(gpu.frees.isEmpty(), "the failed admission queued nothing");
        assertTrue(context.workspace() == null || context.workspace().isClosed());

        var next = new Quantum(plan, new Sequence(104), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        assertEquals(Quantum.Status.SUCCESS, runtime.submit(next).join().status());
        assertEquals(atLoad + 1, gpu.allocations.size(), "the next quantum allocated only its graph's input record");
        runtime.close();
        ExecutionFixtures.assertEachAllocationFreedOnce(gpu);
    }

    @Test
    void cancellationBetweenAdmissionCheckAndSequenceAdmissionPublishesCancelled() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new ExecutionFixtures.RecordingGpu();
        var sequence = new Sequence(104);
        var context = new Quantum(plan, sequence, Quantum.ExecutionKind.DECODE, 0, new int[] {1});

        context.begin(gpu, sequence::cancel);

        assertEquals(Quantum.Status.CANCELLED, context.conclusion().status());
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
        int atLoad = gpu.allocations.size();
        assertEquals(Quantum.Status.FAILED, runtime.submit(context).join().status());
        assertEquals(Sequence.TerminalState.ACTIVE, sequence.terminalState());
        assertEquals(atLoad, gpu.allocations.size(), "the rejected token allocated nothing");
        assertEquals(0, runtime.activeQuanta());
        runtime.close();
    }

    @Test
    void duplicateAdmissionDoesNotRegisterAnotherTerminalOwner() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, new ExecutionFixtures.RecordingGpu());
        var context = new Quantum(plan, new Sequence(17), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        var first = runtime.submit(context);
        // The quantum has one admission; a second is refused before it could bind another continuation.
        assertThrows(Quantum.DuplicateAdmissionException.class, () -> runtime.submit(context));
        lattice.drive();
        assertEquals(Quantum.Status.SUCCESS, first.join().status());
        assertEquals(0, runtime.activeQuanta());
        runtime.close();
        assertTrue(lattice.source.isComplete());
    }

    @Test
    void callersCannotCompleteTheInternalQuantumOutcome() {
        var plan = new ExecutionPlan(ExecutionFixtures.weights());
        var gpu = new ExecutionFixtures.RecordingGpu();
        var context = new Quantum(plan, new Sequence(18), Quantum.ExecutionKind.DECODE, 0, new int[] {1});
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, gpu);
        var exposed = runtime.submit(context);
        exposed.complete(new Quantum.Outcome(Quantum.Status.SUCCESS, null));
        assertFalse((context.conclusion() != null));
        lattice.drive();
        assertEquals(Quantum.Status.SUCCESS, context.conclusion().status());
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
