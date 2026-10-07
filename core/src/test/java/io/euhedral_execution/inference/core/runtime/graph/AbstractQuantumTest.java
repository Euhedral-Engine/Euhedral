package io.euhedral_execution.inference.core.runtime.graph;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CancellationException;
import org.junit.jupiter.api.Test;

class AbstractQuantumTest {

    /// A lake that keeps what it is given.
    private static final class Kept implements FrameLake {
        final List<AbstractFrame> frames = new ArrayList<>();
        boolean refuse;

        @Override
        public void publish(AbstractFrame frame) {
            if (this.refuse) throw new IllegalStateException("closed");
            this.frames.add(frame);
        }

        @Override
        public void publishFromCallback(AbstractFrame frame) {
            publish(frame);
        }

        @Override
        public void admit() {}

        @Override
        public void admitDuringDrain() {}

        @Override
        public void terminated() {}
    }

    private static final class Probe extends AbstractQuantum {
        final List<String> calls = new ArrayList<>();
        RuntimeException commitFailure;

        @Override
        protected void commit() {
            this.calls.add("commit");
            if (this.commitFailure != null) throw this.commitFailure;
        }

        @Override
        protected void release() {
            this.calls.add("release");
        }

        @Override
        protected void published() {
            this.calls.add("published");
        }
    }

    private static final class Told extends AbstractFrame implements AbstractQuantum.Continuation {
        AbstractQuantum quantum;
        int executions;

        Told() {
            super(FrameSeeds.ID_HASH);
        }

        @Override
        public void concluded(AbstractQuantum quantum) {
            this.quantum = quantum;
        }

        @Override
        public void execute() {
            this.executions++;
        }
    }

    @Test
    void theContinuationIsToldAndThrownOnceAfterPublication() {
        var lake = new Kept();
        var quantum = new Probe();
        var next = new Told();
        quantum.continueWith(lake, next);
        quantum.retire(null);
        quantum.publishOutcome();
        quantum.publishOutcome();
        assertEquals(List.of(next), lake.frames, "thrown once, never run inline");
        assertSame(quantum, next.quantum);
        assertEquals(0, next.executions);
        assertEquals(List.of("release", "commit", "published", "published"), quantum.calls);
        assertNull(quantum.outcome());
    }

    @Test
    void releaseAlwaysRunsAndCommitOnlyWithoutFailure() {
        var quantum = new Probe();
        var device = new IllegalStateException("device");
        quantum.retire(device);
        assertEquals(List.of("release"), quantum.calls);
        assertSame(device, quantum.outcome());
    }

    @Test
    void aCancelledQuantumEndsInCancellation() {
        var quantum = new Probe();
        quantum.cancel();
        assertTrue(quantum.stopRequested());
        quantum.retire(null);
        assertInstanceOf(CancellationException.class, quantum.outcome());
        assertEquals(List.of("release"), quantum.calls);
    }

    @Test
    void aFailedCommitIsTheOutcome() {
        var quantum = new Probe();
        quantum.commitFailure = new IllegalStateException("commit");
        quantum.retire(null);
        assertSame(quantum.commitFailure, quantum.outcome());
    }

    @Test
    void laterFailuresAreSuppressedIntoTheFirstAndIgnoredOnceSealed() {
        var quantum = new Probe();
        var first = new IllegalStateException("first");
        var second = new IllegalStateException("second");
        quantum.fail(first);
        quantum.fail(second);
        assertSame(first, quantum.failure());
        assertEquals(List.of(second), List.of(first.getSuppressed()));
        var sealed = new Probe();
        assertTrue(sealed.seal());
        sealed.fail(new IllegalStateException("late"));
        assertNull(sealed.failure());
    }

    @Test
    void aClosedLakeRunsTheContinuationHere() {
        var lake = new Kept();
        lake.refuse = true;
        var quantum = new Probe();
        var next = new Told();
        quantum.continueWith(lake, next);
        quantum.retire(null);
        quantum.publishOutcome();
        assertEquals(1, next.executions, "a lake that refuses must not strand the chain");
    }
}
