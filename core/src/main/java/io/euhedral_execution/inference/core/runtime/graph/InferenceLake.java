package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.core.ingest.AbstractIngestSink;
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
/// spread over all of them. Every sink is a queue ingest sink, attached to the lattice on its own: the
/// workers pull from many sources at once instead of all queuing at one, which keeps producers and
/// consumers off a common cache line and lets the cores work side by side.
///
/// A sink is like a GPU stream. With one producer partition it keeps the order frames were published
/// in, whatever mixture of ordered and unordered frames it carries, and an ordered frame runs in its
/// own chain's order (frames of one `idHash`), not in queue order: two chains in one sink may run at
/// the same time. A sink of several partitions keeps no order across producers. More sinks, not more
/// partitions, reduce contention: partitions only spread the producers.
///
/// The lake counts the units it carries (quanta, host tasks). [#completeGracefully] closes
/// admission; once every admitted unit terminated, the sinks complete and the lattice detaches
/// them.
///
/// A worker polls every source attached to the lattice for as long as it is attached, and an idle worker
/// looks again within microseconds, so sinks that stayed attached would keep a core busy after the last
/// request. The sinks are therefore attached for a busy period: beside them the lake attaches an idle watch,
/// a source the workers poll like the sinks, and when no unit has been admitted for the idle time (one second
/// by default) the watch detaches the sinks and itself, leaving the lattice with nothing to poll. The next
/// unit attaches a fresh set. A frame is published only while a unit it belongs to is admitted, so none is
/// in flight when the sinks go.
public final class InferenceLake implements FrameLake {

    private static final int CLOSED = Integer.MIN_VALUE;
    private static final int QUEUE_CAPACITY = 8192;

    /// How long no unit may have been admitted before the sinks detach, in milliseconds
    /// (`euhedral.lake.idle-detach-ms`); a value of 0 or less keeps them attached.
    public static final String IDLE_DETACH_PROPERTY = "euhedral.lake.idle-detach-ms";

    private static final long DEFAULT_IDLE_DETACH_MILLIS = 1_000;

    private final LatticeTerminal lattice;
    private final int sinkCount;
    private final int partitions;
    private final long idleDetachNanos;
    /// The sinks attached now with their watch, or null while none are. Changed under the lock of [#attach].
    private volatile Generation generation;
    /// When the last unit terminated (the lake is idle since).
    private volatile long idleSince = System.nanoTime();
    private final java.util.List<io.euhedral_execution.core.generics.LatticeSource> attachedSources =
            new java.util.concurrent.CopyOnWriteArrayList<>();
    private final AtomicInteger accepted = new AtomicInteger();
    private final AtomicBoolean finished = new AtomicBoolean();
    private final CompletableFuture<Void> termination = new CompletableFuture<>();

    /// A lake of `sinks` ingest sinks of `partitions` producer partitions each. Its sinks attach to `lattice`
    /// when the first unit is admitted or the first frame published, not before, and detach after the idle
    /// time ([#IDLE_DETACH_PROPERTY]).
    public InferenceLake(LatticeTerminal lattice, int sinks, int partitions) {
        this(
                lattice,
                sinks,
                partitions,
                java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(
                        Long.getLong(IDLE_DETACH_PROPERTY, DEFAULT_IDLE_DETACH_MILLIS)));
    }

    /// As above, with the idle time in nanoseconds (0 or less: the sinks stay attached).
    public InferenceLake(LatticeTerminal lattice, int sinks, int partitions, long idleDetachNanos) {
        this.lattice = Objects.requireNonNull(lattice, "lattice");
        if (sinks < 1) throw new IllegalArgumentException("a lake needs at least one sink");
        if (partitions < 1) throw new IllegalArgumentException("a sink needs at least one partition");
        this.sinkCount = sinks;
        this.partitions = partitions;
        this.idleDetachNanos = idleDetachNanos;
    }

    /// The sinks attached for one busy period and the watch that detaches them.
    private final class Generation {
        final QueueIngestSink[] sinks = new QueueIngestSink[InferenceLake.this.sinkCount];
        final IdleWatch watch = new IdleWatch(this);

        Generation() {
            for (int i = 0; i < this.sinks.length; i++)
                this.sinks[i] =
                        new QueueIngestSink(new PartitionedMpscQueue<>(InferenceLake.this.partitions, QUEUE_CAPACITY));
        }

        /// The sinks drain what they hold and complete; the watch ends.
        void finish() {
            for (QueueIngestSink sink : this.sinks) sink.completeGracefully();
            this.watch.complete();
        }
    }

    /// A source the workers poll beside the sinks: it notices that the lake has been idle long enough.
    private final class IdleWatch extends AbstractIngestSink {
        private final Generation owner;
        private final Delegate delegate = new Delegate();

        IdleWatch(Generation owner) {
            this.owner = owner;
        }

        private void poll() {
            if (InferenceLake.this.idleDetachNanos <= 0) return;
            if ((InferenceLake.this.accepted.get() & Integer.MAX_VALUE) != 0) return;
            if (System.nanoTime() - InferenceLake.this.idleSince < InferenceLake.this.idleDetachNanos) return;
            detach(this.owner);
        }

        @Override
        public io.euhedral_execution.core.generics.LatticeSource getDelegate() {
            return this.delegate;
        }

        @Override
        public void complete() {
            this.delegate.complete();
        }

        @Override
        public boolean isComplete() {
            return this.delegate.isComplete();
        }

        private final class Delegate extends AbstractIngestSink.Delegate {
            @Override
            public long hookOnPull(
                    java.util.function.Consumer<AbstractFrame> consumer,
                    java.util.function.Function<AbstractFrame, Boolean> stopCondition,
                    long demand) {
                poll();
                return 0;
            }

            @Override
            public void hookOnRequest(io.euhedral_execution.core.generics.LatticeReceiver terminal, long demand) {
                poll();
            }
        }
    }

    /// Attaches `source` (a serial source of work the lake does not carry) to the lattice, and
    /// completes it with the lake.
    public void attach(io.euhedral_execution.core.generics.LatticeSource source) {
        Objects.requireNonNull(source, "source");
        this.lattice.addUpstream(source);
        this.attachedSources.add(source);
    }

    /// The sinks attached now; attaches a fresh set (and the watch) when there are none. A lake that completed
    /// before anything used it never attaches.
    private Generation attach() {
        Generation current = this.generation;
        if (current != null) return current;
        synchronized (this.attachedSources) {
            current = this.generation;
            if (current != null || this.finished.get()) return current;
            current = new Generation();
            for (QueueIngestSink sink : current.sinks) this.lattice.addUpstream(sink.getDelegate());
            this.lattice.addUpstream(current.watch.getDelegate());
            this.generation = current;
            return current;
        }
    }

    /// Detaches `from`'s sinks when no unit is admitted: the watch found the lake idle.
    private void detach(Generation from) {
        synchronized (this.attachedSources) {
            if (this.generation != from || this.finished.get()) return;
            // A unit admitted after this check attaches a fresh set: it counts itself before it asks for one.
            if ((this.accepted.get() & Integer.MAX_VALUE) != 0) return;
            this.generation = null;
        }
        from.finish();
    }

    @Override
    public void publish(AbstractFrame frame) {
        Generation current = this.generation;
        if (current == null) current = attach();
        // A set detached by the watch refuses the offer: the frame goes to the set attached since.
        while (current == null || !sinkOf(current, frame).offer(frame)) {
            if (this.finished.get()) throw new IllegalStateException("the inference lake is closed");
            Generation next = attach();
            if (next == current) throw new IllegalStateException("the inference lake is closed");
            current = next;
        }
    }

    /// Offering is enqueueing only, so a driver callback publishes the same way.
    @Override
    public void publishFromCallback(AbstractFrame frame) {
        publish(frame);
    }

    private QueueIngestSink sinkOf(Generation generation, AbstractFrame frame) {
        return generation.sinks[(int) Long.remainderUnsigned(frame.getRoutingHash(), generation.sinks.length)];
    }

    @Override
    public void admit() {
        while (true) {
            int count = this.accepted.get();
            if (count < 0) throw new IllegalStateException("inference admission is closed");
            if (count == Integer.MAX_VALUE) throw new IllegalStateException("too many active units");
            if (this.accepted.compareAndSet(count, count + 1)) break;
        }
        // Counted first: a watch that finds the lake idle after this cannot detach the set this returns.
        attach();
    }

    @Override
    public void admitDuringDrain() {
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
        if ((remaining & Integer.MAX_VALUE) == 0) this.idleSince = System.nanoTime();
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
        for (io.euhedral_execution.core.generics.LatticeSource source : this.attachedSources) {
            try {
                source.complete();
            } catch (RuntimeException completionFailure) {
                failure = completionFailure;
            }
        }
        Generation attached;
        synchronized (this.attachedSources) {
            attached = this.generation;
            this.generation = null;
        }
        if (attached != null) {
            for (QueueIngestSink sink : attached.sinks) {
                try {
                    sink.complete();
                } catch (RuntimeException completionFailure) {
                    if (failure == null) failure = completionFailure;
                    else if (failure != completionFailure) failure.addSuppressed(completionFailure);
                }
            }
            attached.watch.complete();
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
        Generation current = this.generation;
        if (current == null) return false;
        for (QueueIngestSink sink : current.sinks) if (!sink.isComplete()) return true;
        return false;
    }

    /// The number of ingest sinks (upstream sources of the lattice) while attached.
    public int sinks() {
        return this.sinkCount;
    }
}
