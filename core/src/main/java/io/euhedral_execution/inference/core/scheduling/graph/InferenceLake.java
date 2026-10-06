package io.euhedral_execution.inference.core.scheduling.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.core.ingest.QueueIngestSink;
import io.euhedral_execution.data_structures.queues.PartitionedMpscQueue;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/// The pool of ready work of an inference runtime, as several ingest sinks that are each an
/// upstream source of the lattice.
///
/// Producers throw frames into the lake from anywhere: a stage that made its successors ready, a
/// driver callback, a request thread. A frame goes into the sink its routing hash selects, so the
/// frames of one lane (equal hashes) keep to one sink and the frames that may run in parallel
/// spread over all of them. Every sink is a queue ingest sink with partitioned producer queues, and
/// every one is attached to the lattice on its own: the workers pull from many sources at once
/// instead of all queuing at one, which keeps producers and consumers off a common cache line and
/// lets the cores work side by side.
///
/// The lake counts the units it carries (quanta, host tasks). [#completeGracefully] closes
/// admission; once every admitted unit terminated, the sinks complete and the lattice detaches
/// them.
public final class InferenceLake implements FrameLake {

    private static final int CLOSED = Integer.MIN_VALUE;
    private static final int QUEUE_CAPACITY = 8192;

    private final LatticeTerminal lattice;
    private final QueueIngestSink[] sinks;
    private volatile boolean attached;
    private final AtomicInteger accepted = new AtomicInteger();
    private final AtomicBoolean finished = new AtomicBoolean();
    private final CompletableFuture<Void> termination = new CompletableFuture<>();

    /// A lake of `sinks` ingest sinks of `partitions` producer partitions each. Its sinks attach to
    /// `lattice` when the first unit is admitted or the first frame published, not before.
    public InferenceLake(LatticeTerminal lattice, int sinks, int partitions) {
        this.lattice = Objects.requireNonNull(lattice, "lattice");
        if (sinks < 1) throw new IllegalArgumentException("a lake needs at least one sink");
        if (partitions < 1) throw new IllegalArgumentException("a sink needs at least one partition");
        this.sinks = new QueueIngestSink[sinks];
        for (int i = 0; i < sinks; i++)
            this.sinks[i] = new QueueIngestSink(new PartitionedMpscQueue<>(partitions, QUEUE_CAPACITY));
    }

    /// Attaches every sink to the lattice, once. A lake that completed before anything used it
    /// never attaches.
    private void attach() {
        synchronized (this.sinks) {
            if (this.attached || this.finished.get()) return;
            for (QueueIngestSink sink : this.sinks) this.lattice.addUpstream(sink.getDelegate());
            this.attached = true;
        }
    }

    @Override
    public void publish(AbstractFrame frame) {
        if (!this.attached) attach();
        if (!sinkOf(frame).offer(frame)) throw new IllegalStateException("the inference lake is closed");
    }

    /// Offering is enqueueing only, so a driver callback publishes the same way.
    @Override
    public void publishFromCallback(AbstractFrame frame) {
        publish(frame);
    }

    private QueueIngestSink sinkOf(AbstractFrame frame) {
        return this.sinks[(int) Long.remainderUnsigned(frame.getRoutingHash(), this.sinks.length)];
    }

    @Override
    public void admit() {
        if (!this.attached) attach();
        while (true) {
            int count = this.accepted.get();
            if (count < 0) throw new IllegalStateException("inference admission is closed");
            if (count == Integer.MAX_VALUE) throw new IllegalStateException("too many active units");
            if (this.accepted.compareAndSet(count, count + 1)) return;
        }
    }

    @Override
    public void admitDuringDrain() {
        if (!this.attached) attach();
        while (true) {
            int count = this.accepted.get();
            if ((count & Integer.MAX_VALUE) == 0) throw new IllegalStateException("inference admission is closed");
            if ((count & Integer.MAX_VALUE) == Integer.MAX_VALUE)
                throw new IllegalStateException("too many active units");
            if (this.accepted.compareAndSet(count, count + 1)) return;
        }
    }

    @Override
    public void terminated() {
        int remaining = this.accepted.decrementAndGet();
        if (remaining == CLOSED) signalComplete();
        else if (remaining == -1 || remaining == Integer.MAX_VALUE)
            throw new IllegalStateException("unit termination exceeded admission");
    }

    /// Units admitted and not yet terminated.
    public int active() {
        return this.accepted.get() & Integer.MAX_VALUE;
    }

    /// Closes admission; the lake completes after every admitted unit terminated.
    public void completeGracefully() {
        int old = this.accepted.getAndUpdate(value -> value | CLOSED);
        if ((old & Integer.MAX_VALUE) == 0) signalComplete();
    }

    private void signalComplete() {
        if (!this.finished.compareAndSet(false, true)) return;
        RuntimeException failure = null;
        for (QueueIngestSink sink : this.sinks) {
            try {
                sink.complete();
            } catch (RuntimeException completionFailure) {
                if (failure == null) failure = completionFailure;
                else if (failure != completionFailure) failure.addSuppressed(completionFailure);
            }
        }
        if (failure != null) {
            this.termination.completeExceptionally(failure);
            throw failure;
        }
        this.termination.complete(null);
    }

    /// Waits until every sink completed and the lattice detached it.
    public void awaitTermination() {
        this.termination.join();
    }

    public boolean isComplete() {
        return this.finished.get();
    }

    /// Whether any sink is still attached to the lattice.
    public boolean isAttached() {
        if (!this.attached) return false;
        for (QueueIngestSink sink : this.sinks) if (!sink.isComplete()) return true;
        return false;
    }

    /// The number of ingest sinks (upstream sources of the lattice).
    public int sinks() {
        return this.sinks.length;
    }
}
