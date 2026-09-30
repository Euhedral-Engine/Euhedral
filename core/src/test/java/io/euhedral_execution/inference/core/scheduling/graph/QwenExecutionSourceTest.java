package io.euhedral_execution.inference.core.scheduling.graph;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/// The Qwen source deals only in ready frames: it knows nothing of stages, instructions, or graphs.
class QwenExecutionSourceTest {

    private final QwenExecutionSource source = new QwenExecutionSource();
    private final Receiver receiver = new Receiver();

    @Test
    void readyStorageHoldsArbitraryFramesAndNotQwenScheduleState() {
        for (Field field : QwenExecutionSource.class.getDeclaredFields()) {
            if (Modifier.isStatic(field.getModifiers())) continue;
            String type = field.getType().getName();
            assertFalse(type.contains(".scheduling."), "source field " + field.getName() + " is " + type);
        }
        Plain first = new Plain();
        Plain second = new Plain();
        this.source.publish(first);
        this.source.publish(second);
        List<AbstractFrame> pulled = new ArrayList<>();
        assertEquals(2, this.source.pull(pulled::add, frame -> false, 10));
        assertEquals(List.of(first, second), pulled);
    }

    @Test
    void pullStopsBeforeAFrameItMayNotTakeAndLeavesItQueued() {
        Plain first = new Plain();
        this.source.publish(first);
        List<AbstractFrame> pulled = new ArrayList<>();
        assertEquals(0, this.source.pull(pulled::add, frame -> true, 10));
        assertTrue(pulled.isEmpty());
        assertEquals(1, this.source.pull(pulled::add, frame -> false, 10));
        assertEquals(List.of(first), pulled);
    }

    @Test
    void pullDeliversOnlyToItsConsumerAndNeverPushes() {
        this.source.addDownstream(this.receiver);
        this.source.publishFromCallback(new Plain());
        List<AbstractFrame> pulled = new ArrayList<>();
        assertEquals(1, this.source.pull(pulled::add, frame -> false, 10));
        assertTrue(this.receiver.pushed.isEmpty());
    }

    @Test
    void requestWithNothingReadyLeavesDemandThatALaterPublicationServes() {
        this.source.addDownstream(this.receiver);
        this.source.request(2);
        assertTrue(this.receiver.pushed.isEmpty(), "an empty request pushes nothing");

        Plain ready = new Plain();
        this.source.publish(ready);
        assertEquals(List.of(ready), this.receiver.pushed, "the earlier demand is served on publication");
        this.source.publish(new Plain());
        this.source.publish(new Plain());
        assertEquals(2, this.receiver.pushed.size(), "only the requested demand is pushed");
        List<AbstractFrame> pulled = new ArrayList<>();
        assertEquals(1, this.source.pull(pulled::add, frame -> false, 10), "the rest waits for Euhedral");
    }

    @Test
    void driverCallbackPublicationNeverRunsOrPushesAFrame() {
        this.source.addDownstream(this.receiver);
        this.source.request(Long.MAX_VALUE);
        Plain retired = new Plain();
        this.source.publishFromCallback(retired);
        assertTrue(this.receiver.pushed.isEmpty(), "a callback thread only enqueues");
        this.source.request(1);
        assertEquals(List.of(retired), this.receiver.pushed, "the next Euhedral call delivers it");
    }

    @Test
    void framesMadeReadyDuringAPullAreDeliveredByThatPullWithoutRecursion() {
        AtomicInteger depth = new AtomicInteger();
        List<Integer> depths = new ArrayList<>();
        Chain third = new Chain(null, depth, depths);
        Chain second = new Chain(third, depth, depths);
        Chain first = new Chain(second, depth, depths);
        this.source.addDownstream(this.receiver);
        this.source.request(Long.MAX_VALUE);
        this.receiver.executing = true;
        this.source.publish(first);
        assertEquals(List.of(1, 1, 1), depths, "each frame runs after its predecessor returned");
    }

    @Test
    void sourceCompletesOnlyAfterEveryAdmittedGraphTerminates() {
        this.source.addDownstream(this.receiver);
        this.source.admit();
        this.source.admit();
        this.source.completeGracefully();
        assertThrows(IllegalStateException.class, this.source::admit, "admission is closed");
        assertFalse(this.source.isComplete());
        this.source.terminated();
        assertFalse(this.source.isComplete());
        assertEquals(0, this.receiver.completions);
        this.source.terminated();
        assertTrue(this.source.isComplete());
        assertEquals(1, this.receiver.completions);
        assertFalse(this.source.isAttached());
        this.source.awaitTermination();
    }

    @Test
    void secondDownstreamIsRejected() {
        this.source.addDownstream(this.receiver);
        Receiver other = new Receiver();
        this.source.addDownstream(other);
        assertEquals(1, other.errors);
    }

    private static final class Plain extends AbstractFrame {
        Plain() {
            super(0L);
        }
    }

    /// A frame that publishes its successor when it finishes, like a stage.
    private final class Chain extends AbstractFrame {
        private final Chain next;
        private final AtomicInteger depth;
        private final List<Integer> depths;

        Chain(Chain next, AtomicInteger depth, List<Integer> depths) {
            super(0L);
            this.next = next;
            this.depth = depth;
            this.depths = depths;
        }

        @Override
        public void execute() {
            this.depths.add(this.depth.get());
        }

        @Override
        public void doFinally() {
            if (this.next != null) QwenExecutionSourceTest.this.source.publish(this.next);
        }
    }

    /// A receiver that either records pushes or runs them inline, as Euhedral's execution terminal does.
    private static final class Receiver implements LatticeReceiver {
        final List<AbstractFrame> pushed = Collections.synchronizedList(new ArrayList<>());
        final AtomicInteger inside = new AtomicInteger();
        boolean executing;
        int completions;
        int errors;

        @Override
        public void push(AbstractFrame frame) {
            this.pushed.add(frame);
            if (!this.executing) return;
            if (frame instanceof Chain chain) {
                chain.depth.set(this.inside.incrementAndGet());
                try {
                    frame.execute();
                    frame.doFinally();
                } finally {
                    chain.depth.set(this.inside.decrementAndGet());
                }
            }
        }

        @Override
        public void onComplete() {
            this.completions++;
        }

        @Override
        public void onError(Throwable failure) {
            this.errors++;
        }

        @Override
        public void addUpstream(LatticeSource upstream) {}
    }
}
