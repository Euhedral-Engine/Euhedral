package io.euhedral_execution.inference.core.scheduling.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/// The fan-in edge of frames whose number is decided when they are published: the counterpart of a
/// stage's incoming edges ([StageFrame]) for work a stage spawns at run time. The producer
/// [#expect]s the frames before it publishes them; each arrives once when it ends, successfully or
/// not, and the last arrival publishes the continuation. A failure is recorded for the continuation
/// to read; the first one wins.
///
/// Reusable: [#expect] begins the next round once the continuation of the previous one has run.
public final class Join {

    private final AtomicInteger pending = new AtomicInteger();
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private final FrameLake lake;
    private final AbstractFrame continuation;

    public Join(FrameLake lake, AbstractFrame continuation) {
        this.lake = Objects.requireNonNull(lake, "lake");
        this.continuation = Objects.requireNonNull(continuation, "continuation");
    }

    /// Begins a round of `arrivals` frames. Before any of them is published.
    public void expect(int arrivals) {
        if (arrivals < 1) throw new IllegalArgumentException("a join needs at least one arrival");
        this.failure.set(null);
        this.pending.set(arrivals);
    }

    /// One frame of the round ended. The last arrival publishes the continuation.
    public void arrive() {
        if (this.pending.decrementAndGet() == 0) this.lake.publish(this.continuation);
    }

    /// One frame of the round failed, and ends.
    public void fail(Throwable cause) {
        this.failure.compareAndSet(null, cause);
        arrive();
    }

    /// The first failure of the round, or null. Read by the continuation.
    public Throwable failure() {
        return this.failure.get();
    }
}
