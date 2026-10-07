package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Function;

/// A lake for driving graphs by hand in tests: published frames wait in arrival order until a test pulls them.
/// Frames that a pulled frame publishes are appended and delivered by the same pull.
final class TestLake implements FrameLake {

    private final ConcurrentLinkedQueue<AbstractFrame> ready = new ConcurrentLinkedQueue<>();
    private final AtomicInteger active = new AtomicInteger();

    @Override
    public void publish(AbstractFrame frame) {
        this.ready.add(frame);
    }

    @Override
    public void publishFromCallback(AbstractFrame frame) {
        this.ready.add(frame);
    }

    @Override
    public void admit() {
        this.active.incrementAndGet();
    }

    @Override
    public void admitDuringDrain() {
        this.active.incrementAndGet();
    }

    @Override
    public void terminated() {
        if (this.active.decrementAndGet() < 0) throw new IllegalStateException("termination exceeded admission");
    }

    /// Admitted units that have not terminated.
    int activeGraphs() {
        return this.active.get();
    }

    /// Hands up to `demand` ready frames to `consumer`, stopping before the first frame `stop` accepts.
    long pull(Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stop, long demand) {
        long delivered = 0;
        while (delivered < demand) {
            AbstractFrame frame = this.ready.peek();
            if (frame == null || stop.apply(frame)) break;
            this.ready.poll();
            delivered++;
            consumer.accept(frame);
        }
        return delivered;
    }
}
