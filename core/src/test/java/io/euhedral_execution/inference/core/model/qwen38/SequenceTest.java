package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class SequenceTest {
    @Test
    void stateIsLockFreeAndLeaseGuardsMutation() {
        for (var method : Sequence.class.getDeclaredMethods()) {
            assertFalse(Modifier.isSynchronized(method.getModifiers()));
        }
        var state = new Sequence(21);
        var lease = state.claimExecution(0);
        assertThrows(IllegalStateException.class, () -> state.claimExecution(0));
        state.setKvCacheState(lease, "kv");
        state.setRecurrentState(lease, "recurrent");
        assertEquals("kv", state.kvCacheState());
        assertEquals("recurrent", state.recurrentState());
        state.releaseExecution(lease, 1);
        assertEquals(1, state.currentTokenPosition());
        assertThrows(IllegalStateException.class, () -> state.releaseExecution(lease, 2));
    }

    @Test
    void successfulSequenceClosesItsPersistentResource() {
        var state = new Sequence(25);
        var lease = state.claimExecution(0);
        var recurrentState = new CloseableState();
        var kvCacheState = new CloseableState();
        state.setRecurrentState(lease, recurrentState);
        state.setKvCacheState(lease, kvCacheState);
        state.releaseExecution(lease, 1);

        state.complete();

        assertEquals(1, recurrentState.closeCount);
        assertEquals(1, kvCacheState.closeCount);
    }

    @Test
    void cancelledSequenceClosesItsPersistentResourceAfterTheActiveLease() {
        var state = new Sequence(26);
        var lease = state.claimExecution(0);
        var recurrentState = new CloseableState();
        state.setRecurrentState(lease, recurrentState);

        state.cancel();

        assertTrue(state.releaseExecutionAndCheckCancellation(lease, 1));
        assertEquals(1, recurrentState.closeCount);
    }

    @Test
    void failedSequenceClosesItsPersistentResource() {
        var state = new Sequence(27);
        var lease = state.claimExecution(0);
        var recurrentState = new CloseableState();
        state.setRecurrentState(lease, recurrentState);

        state.markFailed(lease, new IllegalStateException("failure"));

        assertEquals(1, recurrentState.closeCount);
    }

    @Test
    void cancellationRacingWithClaimCannotCommitPosition() throws Exception {
        var state = new Sequence(22);
        var start = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(2)) {
            var claim = executor.submit(() -> {
                start.await();
                try {
                    return state.claimExecution(0);
                } catch (IllegalStateException cancelled) {
                    return null;
                }
            });
            var cancel = executor.submit(() -> {
                start.await();
                state.cancel();
                return null;
            });
            start.countDown();
            var lease = claim.get(5, TimeUnit.SECONDS);
            cancel.get(5, TimeUnit.SECONDS);
            if (lease != null) assertTrue(state.releaseExecutionAndCheckCancellation(lease, 1));
            assertEquals(Sequence.TerminalState.CANCELLED, state.terminalState());
            assertEquals(0, state.currentTokenPosition());
            assertFalse(state.isExecutionClaimed());
        }
    }

    @Test
    void cancellationBeforeClaimCannotOverwriteEarlierFailure() {
        var state = new Sequence(24);
        var lease = state.claimExecution(0);
        state.cancel();
        var failure = new IllegalStateException("prior quantum failed");
        state.markFailed(lease, failure);
        state.markCancelledBeforeClaim();
        assertEquals(Sequence.TerminalState.FAILED, state.terminalState());
        assertEquals(failure, state.terminalFailure());
    }

    @Test
    void concurrentClaimersPublishExactlyOneLease() throws Exception {
        var state = new Sequence(23);
        var start = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(16)) {
            List<java.util.concurrent.Future<Sequence.ExecutionLease>> claims = new ArrayList<>();
            for (int index = 0; index < 16; index++) {
                claims.add(executor.submit(() -> {
                    start.await();
                    try {
                        return state.claimExecution(0);
                    } catch (IllegalStateException alreadyClaimed) {
                        return null;
                    }
                }));
            }
            start.countDown();
            int winners = 0;
            Sequence.ExecutionLease winner = null;
            for (var claim : claims) {
                var lease = claim.get(5, TimeUnit.SECONDS);
                if (lease != null) {
                    winners++;
                    winner = lease;
                }
            }
            assertEquals(1, winners);
            state.releaseExecution(winner, 1);
        }
    }

    @Test
    void terminalCleanupRetriesFailuresWithoutReclosingReleasedState() {
        for (boolean failSequence : List.of(false, true)) {
            var state = new Sequence(31);
            var lease = state.claimExecution(0);
            var attempts = new java.util.concurrent.atomic.AtomicInteger();
            var kv = new CloseableState();
            state.setKvCacheState(lease, kv);
            state.setRecurrentState(lease, (AutoCloseable) () -> {
                if (attempts.incrementAndGet() <= 2) throw new IllegalStateException("transient free failure");
            });
            state.releaseExecution(lease, 1);
            if (failSequence) state.markFailedAfterRelease(new IllegalStateException("execution failed"));
            else assertThrows(IllegalStateException.class, state::cancel);
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
