package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

@Timeout(30)
class SequenceTest {

    /// A piece of work on the sequence; `conclusion` runs in admission order with whether it is blocked.
    private static final class Work implements Sequence.Work {
        final long start;
        final long end;
        final BiConsumer<Work, Throwable> conclusion;
        volatile boolean ready;
        Throwable blocked;
        boolean concluded;

        Work(long start, long end, BiConsumer<Work, Throwable> conclusion) {
            this.start = start;
            this.end = end;
            this.conclusion = conclusion;
        }

        @Override
        public boolean ready() {
            return this.ready;
        }

        @Override
        public long admittedAt() {
            return this.start;
        }

        @Override
        public void concluded(Throwable blocked) {
            this.blocked = blocked;
            this.concluded = true;
            this.conclusion.accept(this, blocked);
        }

        void complete(Sequence sequence) {
            this.ready = true;
            sequence.drain();
        }
    }

    /// Commits to its end unless blocked or cancelled, as a quantum does.
    private static Work committing(Sequence sequence, long start, long end) {
        return new Work(start, end, (work, blocked) -> {
            if (blocked != null) {
                sequence.fail(blocked);
                sequence.abandon();
            } else if (sequence.cancellationRequested()) {
                sequence.abandon();
            } else {
                sequence.commit(work.end);
            }
        });
    }

    private static Work admitted(Sequence sequence, long start, long end) {
        Work work = committing(sequence, start, end);
        sequence.admit(work, start, end);
        return work;
    }

    @Test
    void stateIsLockFreeAndOnlyWorkInFlightSetsIt() {
        for (var method : Sequence.class.getDeclaredMethods()) {
            assertFalse(Modifier.isSynchronized(method.getModifiers()));
        }
        var sequence = new Sequence(21);
        assertThrows(IllegalStateException.class, () -> sequence.setKvCacheState("kv"));
        var work = admitted(sequence, 0, 1);
        sequence.setKvCacheState("kv");
        sequence.setRecurrentState("recurrent");
        assertEquals("kv", sequence.kvCacheState());
        assertEquals("recurrent", sequence.recurrentState());
        work.complete(sequence);
        assertEquals(1, sequence.currentTokenPosition());
        assertFalse(sequence.inFlight());
        assertThrows(IllegalStateException.class, () -> sequence.commit(2), "nothing is in flight");
    }

    @Test
    void aWorkMustStartAtTheSubmittedFrontier() {
        var sequence = new Sequence(22, 4);
        assertThrows(IllegalArgumentException.class, () -> admitted(sequence, 3, 5));
        assertThrows(IllegalArgumentException.class, () -> admitted(sequence, 5, 6));
        assertFalse(sequence.inFlight());
        assertEquals(4, sequence.submittedFrontier());

        admitted(sequence, 4, 6);
        assertEquals(6, sequence.submittedFrontier());
        assertEquals(4, sequence.committedFrontier());
        assertThrows(IllegalArgumentException.class, () -> admitted(sequence, 4, 5), "4 is already submitted");
        admitted(sequence, 6, 7);
        assertEquals(7, sequence.submittedFrontier());
    }

    @Test
    void severalWorksInFlightConcludeInAdmissionOrderWhateverOrderTheyComplete() {
        var sequence = new Sequence(23);
        List<Long> order = new ArrayList<>();
        var works = new ArrayList<Work>();
        for (long start = 0; start < 6; start += 2) {
            long from = start;
            var work = new Work(from, from + 2, (w, blocked) -> {
                assertNull(blocked);
                assertEquals(w.start, sequence.committedFrontier(), "everything before it committed");
                order.add(w.start);
                sequence.commit(w.end);
            });
            sequence.admit(work, from, from + 2);
            works.add(work);
        }
        works.get(2).complete(sequence);
        works.get(1).complete(sequence);
        assertEquals(List.of(), order, "the first work has not completed");
        assertEquals(0, sequence.committedFrontier());
        assertTrue(sequence.inFlight());

        works.get(0).complete(sequence);
        assertEquals(List.of(0L, 2L, 4L), order);
        assertEquals(6, sequence.committedFrontier());
        assertFalse(sequence.inFlight());
    }

    @Test
    void aFailureBlocksEveryWorkAdmittedAfterIt() {
        var sequence = new Sequence(24);
        var failure = new IllegalStateException("the quantum failed");
        var first = new Work(0, 2, (w, blocked) -> {
            sequence.fail(failure);
            sequence.abandon();
        });
        sequence.admit(first, 0, 2);
        var second = admitted(sequence, 2, 3);
        second.complete(sequence);
        first.complete(sequence);

        assertNotNull(second.blocked);
        assertSame(failure, second.blocked.getCause());
        assertEquals(0, sequence.committedFrontier());
        assertEquals(Sequence.TerminalState.FAILED, sequence.terminalState());
        assertSame(failure, sequence.terminalFailure());
        assertFalse(sequence.inFlight());
        assertThrows(IllegalStateException.class, () -> admitted(sequence, 0, 1));
    }

    @Test
    void aShortCommitBlocksAWorkAdmittedPastIt() {
        var sequence = new Sequence(25);
        var verify = new Work(0, 4, (w, blocked) -> sequence.commit(1));
        sequence.admit(verify, 0, 4);
        var next = admitted(sequence, 4, 5);
        verify.complete(sequence);
        next.complete(sequence);

        assertNotNull(next.blocked, "it started at 4 but the sequence committed only to 1");
        assertEquals(1, sequence.committedFrontier());
        assertEquals(Sequence.TerminalState.FAILED, sequence.terminalState());
    }

    @Test
    void aShortCommitWithNothingBehindItRewindsTheSubmittedFrontier() {
        var sequence = new Sequence(26);
        var verify = new Work(0, 4, (w, blocked) -> sequence.commit(1));
        sequence.admit(verify, 0, 4);
        verify.complete(sequence);
        assertEquals(1, sequence.submittedFrontier());
        admitted(sequence, 1, 2).complete(sequence);
        assertEquals(2, sequence.committedFrontier());
        assertEquals(Sequence.TerminalState.ACTIVE, sequence.terminalState());
    }

    @Test
    void cancellingWithWorkInFlightDefersCleanupToComplete() {
        var sequence = new Sequence(27);
        var recurrent = new CloseableState();
        var kv = new CloseableState();
        var work = admitted(sequence, 0, 1);
        sequence.setRecurrentState(recurrent);
        sequence.setKvCacheState(kv);

        sequence.cancel();
        assertTrue(sequence.cancellationRequested());
        assertEquals(Sequence.TerminalState.ACTIVE, sequence.terminalState(), "the work in flight concludes it");
        assertThrows(IllegalStateException.class, sequence::complete, "work is in flight");
        assertThrows(IllegalStateException.class, () -> admitted(sequence, 1, 2));
        work.complete(sequence);
        assertEquals(Sequence.TerminalState.CANCELLED, sequence.terminalState());
        assertEquals(0, sequence.committedFrontier());
        assertEquals(0, recurrent.closeCount);
        assertEquals(0, kv.closeCount);

        sequence.complete();
        assertEquals(1, recurrent.closeCount);
        assertEquals(1, kv.closeCount);
        assertEquals(Sequence.TerminalState.CANCELLED, sequence.terminalState());
    }

    @Test
    void aCancellationAfterTheCommitDecisionStillEndsTheSequenceCancelled() {
        var sequence = new Sequence(28);
        var work = new Work(0, 3, (w, blocked) -> {
            sequence.cancel();
            sequence.commit(3);
        });
        sequence.admit(work, 0, 3);
        work.complete(sequence);
        assertEquals(3, sequence.committedFrontier(), "the work had committed");
        assertEquals(Sequence.TerminalState.CANCELLED, sequence.terminalState());
        assertFalse(sequence.inFlight());
    }

    @Test
    void cancellationRacingWithSettlementAlwaysEndsCancelled() throws Exception {
        for (int round = 0; round < 500; round++) {
            var sequence = new Sequence(29);
            var work = admitted(sequence, 0, 1);
            var start = new CountDownLatch(1);
            try (var executor = Executors.newFixedThreadPool(2)) {
                var settle = executor.submit(() -> {
                    start.await();
                    work.complete(sequence);
                    return null;
                });
                var cancel = executor.submit(() -> {
                    start.await();
                    sequence.cancel();
                    return null;
                });
                start.countDown();
                settle.get(5, TimeUnit.SECONDS);
                cancel.get(5, TimeUnit.SECONDS);
            }
            assertEquals(Sequence.TerminalState.CANCELLED, sequence.terminalState(), "round " + round);
            assertFalse(sequence.inFlight());
        }
    }

    @Test
    void aDraftWorkLeavesTheFrontiersAlone() {
        var sequence = new Sequence(30, 6);
        var draft = admitted(sequence, 6, 6);
        assertTrue(sequence.inFlight());
        assertEquals(6, sequence.submittedFrontier());
        draft.complete(sequence);
        assertEquals(6, sequence.committedFrontier());
        assertFalse(sequence.inFlight());
    }

    @Test
    void aFailureIsTerminalAndACancellationCannotOverwriteIt() {
        var sequence = new Sequence(31);
        var failure = new IllegalStateException("the quantum failed");
        var work = new Work(0, 1, (w, blocked) -> {
            sequence.fail(failure);
            sequence.abandon();
        });
        sequence.admit(work, 0, 1);
        sequence.cancel();
        work.complete(sequence);
        sequence.cancel();
        assertEquals(Sequence.TerminalState.FAILED, sequence.terminalState());
        assertEquals(failure, sequence.terminalFailure());
    }

    @Test
    void successfulSequenceClosesItsPersistentStateOnComplete() {
        var sequence = new Sequence(32);
        var recurrent = new CloseableState();
        var kv = new CloseableState();
        var work = admitted(sequence, 0, 1);
        sequence.setRecurrentState(recurrent);
        sequence.setKvCacheState(kv);
        work.complete(sequence);
        assertEquals(0, recurrent.closeCount);

        sequence.complete();

        assertEquals(Sequence.TerminalState.COMPLETED, sequence.terminalState());
        assertEquals(1, recurrent.closeCount);
        assertEquals(1, kv.closeCount);
        assertThrows(IllegalStateException.class, () -> admitted(sequence, 1, 2));
    }

    @Test
    void terminalCleanupRetriesFailuresWithoutReclosingReleasedState() {
        for (boolean failSequence : List.of(false, true)) {
            var sequence = new Sequence(33);
            var work = admitted(sequence, 0, 1);
            var attempts = new java.util.concurrent.atomic.AtomicInteger();
            var kv = new CloseableState();
            sequence.setKvCacheState(kv);
            sequence.setRecurrentState((AutoCloseable) () -> {
                if (attempts.incrementAndGet() <= 2) throw new IllegalStateException("transient free failure");
            });
            work.complete(sequence);
            if (failSequence) sequence.fail(new IllegalStateException("execution failed"));
            else sequence.cancel();
            assertEquals(0, kv.closeCount);
            assertThrows(IllegalStateException.class, sequence::complete);
            assertEquals(1, kv.closeCount);
            assertThrows(IllegalStateException.class, sequence::complete);
            sequence.complete();
            sequence.complete();
            assertEquals(3, attempts.get());
            assertEquals(1, kv.closeCount);
        }
    }

    private static final class CloseableState implements AutoCloseable {
        private int closeCount;

        @Override
        public void close() {
            this.closeCount++;
        }
    }
}
