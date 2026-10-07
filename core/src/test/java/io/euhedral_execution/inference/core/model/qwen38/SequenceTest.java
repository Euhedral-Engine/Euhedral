package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Modifier;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class SequenceTest {
    @Test
    void stateIsLockFreeAndOnlyTheQuantumInFlightSetsIt() {
        for (var method : Sequence.class.getDeclaredMethods()) {
            assertFalse(Modifier.isSynchronized(method.getModifiers()));
        }
        var state = new Sequence(21);
        assertThrows(IllegalStateException.class, () -> state.setKvCacheState("kv"));
        state.admit(0, 1);
        state.setKvCacheState("kv");
        state.setRecurrentState("recurrent");
        assertEquals("kv", state.kvCacheState());
        assertEquals("recurrent", state.recurrentState());
        assertFalse(state.commit(1));
        assertEquals(1, state.currentTokenPosition());
        assertThrows(IllegalStateException.class, () -> state.commit(2), "nothing is in flight");
    }

    @Test
    void aQuantumMustStartAtTheSubmittedFrontier() {
        var state = new Sequence(22, 4);
        assertThrows(IllegalArgumentException.class, () -> state.admit(3, 5));
        assertThrows(IllegalArgumentException.class, () -> state.admit(5, 6));
        assertFalse(state.inFlight());
        assertEquals(4, state.submittedFrontier());
        assertEquals(4, state.committedFrontier());
    }

    @Test
    void onlyOneQuantumAtATime() {
        var state = new Sequence(23);
        state.admit(0, 2);
        assertThrows(IllegalStateException.class, () -> state.admit(2, 3));
        assertThrows(IllegalStateException.class, () -> state.admit(0, 2));
        state.commit(2);
        state.admit(2, 3);
        state.abandon();
        state.admit(2, 3);
        assertTrue(state.inFlight());
    }

    @Test
    void commitMovesBothFrontiersAndAbandonRestoresSubmitted() {
        var state = new Sequence(24);
        state.admit(0, 5);
        assertEquals(5, state.submittedFrontier());
        assertEquals(0, state.committedFrontier());
        state.commit(5);
        assertEquals(5, state.submittedFrontier());
        assertEquals(5, state.committedFrontier());

        state.admit(5, 9);
        assertEquals(9, state.submittedFrontier());
        state.abandon();
        assertEquals(5, state.submittedFrontier());
        assertEquals(5, state.committedFrontier());
        assertFalse(state.inFlight());
        assertEquals(Sequence.TerminalState.ACTIVE, state.terminalState());
    }

    @Test
    void cancellingWithAQuantumInFlightDefersCleanupToClose() {
        var state = new Sequence(25);
        var recurrent = new CloseableState();
        var kv = new CloseableState();
        state.admit(0, 1);
        state.setRecurrentState(recurrent);
        state.setKvCacheState(kv);

        state.cancel();
        assertTrue(state.cancellationRequested());
        assertEquals(Sequence.TerminalState.ACTIVE, state.terminalState(), "the quantum in flight decides");
        assertThrows(IllegalStateException.class, state::complete, "a quantum is in flight");
        state.abandon();
        assertEquals(Sequence.TerminalState.CANCELLED, state.terminalState());
        assertEquals(0, recurrent.closeCount);
        assertEquals(0, kv.closeCount);

        state.complete();
        assertEquals(1, recurrent.closeCount);
        assertEquals(1, kv.closeCount);
        assertEquals(Sequence.TerminalState.CANCELLED, state.terminalState());
    }

    @Test
    void aCommitAfterCancellationLeavesTheFrontiersAndReportsTheCancellation() {
        var state = new Sequence(26);
        state.admit(0, 3);
        state.cancel();
        assertTrue(state.commit(3));
        assertEquals(0, state.committedFrontier());
        assertEquals(0, state.submittedFrontier());
        assertEquals(Sequence.TerminalState.CANCELLED, state.terminalState());
        assertThrows(IllegalStateException.class, () -> state.admit(0, 1));
    }

    @Test
    void aDraftQuantumLeavesTheFrontiersAlone() {
        var state = new Sequence(27, 6);
        state.admit(6, 6);
        assertTrue(state.inFlight());
        assertThrows(IllegalStateException.class, () -> state.admit(6, 7), "a draft is still one quantum");
        assertEquals(6, state.submittedFrontier());
        state.commit(6);
        assertEquals(6, state.committedFrontier());
        assertEquals(6, state.submittedFrontier());
    }

    @Test
    void successfulSequenceClosesItsPersistentStateOnComplete() {
        var state = new Sequence(28);
        var recurrent = new CloseableState();
        var kv = new CloseableState();
        state.admit(0, 1);
        state.setRecurrentState(recurrent);
        state.setKvCacheState(kv);
        state.commit(1);
        assertEquals(0, recurrent.closeCount);

        state.complete();

        assertEquals(Sequence.TerminalState.COMPLETED, state.terminalState());
        assertEquals(1, recurrent.closeCount);
        assertEquals(1, kv.closeCount);
    }

    @Test
    void aFailureIsTerminalAndACancellationCannotOverwriteIt() {
        var state = new Sequence(29);
        state.admit(0, 1);
        state.cancel();
        var failure = new IllegalStateException("the quantum failed");
        state.fail(failure);
        state.abandon();
        state.cancel();
        assertEquals(Sequence.TerminalState.FAILED, state.terminalState());
        assertEquals(failure, state.terminalFailure());
        assertThrows(IllegalStateException.class, () -> state.admit(0, 1));
    }

    @Test
    void cancellationRacingWithAdmissionCannotCommitPosition() throws Exception {
        for (int round = 0; round < 200; round++) {
            var state = new Sequence(30);
            var start = new CountDownLatch(1);
            try (var executor = Executors.newFixedThreadPool(2)) {
                var admit = executor.submit(() -> {
                    start.await();
                    try {
                        state.admit(0, 1);
                        return true;
                    } catch (IllegalStateException cancelled) {
                        return false;
                    }
                });
                var cancel = executor.submit(() -> {
                    start.await();
                    state.cancel();
                    return null;
                });
                start.countDown();
                boolean admitted = admit.get(5, TimeUnit.SECONDS);
                cancel.get(5, TimeUnit.SECONDS);
                if (admitted) assertTrue(state.commit(1));
                assertEquals(Sequence.TerminalState.CANCELLED, state.terminalState());
                assertEquals(0, state.currentTokenPosition());
                assertFalse(state.inFlight());
            }
        }
    }

    @Test
    void terminalCleanupRetriesFailuresWithoutReclosingReleasedState() {
        for (boolean failSequence : List.of(false, true)) {
            var state = new Sequence(31);
            state.admit(0, 1);
            var attempts = new java.util.concurrent.atomic.AtomicInteger();
            var kv = new CloseableState();
            state.setKvCacheState(kv);
            state.setRecurrentState((AutoCloseable) () -> {
                if (attempts.incrementAndGet() <= 2) throw new IllegalStateException("transient free failure");
            });
            state.commit(1);
            if (failSequence) state.fail(new IllegalStateException("execution failed"));
            else state.cancel();
            assertEquals(0, kv.closeCount);
            assertThrows(IllegalStateException.class, state::complete);
            assertEquals(1, kv.closeCount);
            assertThrows(IllegalStateException.class, state::complete);
            state.complete();
            state.complete();
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
