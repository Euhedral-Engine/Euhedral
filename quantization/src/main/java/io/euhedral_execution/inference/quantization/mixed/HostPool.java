package io.euhedral_execution.inference.quantization.mixed;

import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.ingest.AbstractIngestSink;
import io.euhedral_execution.inference.core.runtime.HostTasks;
import io.euhedral_execution.inference.core.runtime.graph.FrameLake;
import io.euhedral_execution.inference.core.runtime.graph.InferenceLake;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

/// One kind of host work for the search to run beside the GPU, as frames on the process's lattice: sinks of its
/// own (as many as the lattice has workers) and the engine's [HostTasks] publishing each task into them as one
/// frame. A sink never runs frames in `pull()`, where a worker would take the sink over and run its queue
/// itself: it hands frames to the lattice only through requests, so whichever worker is free takes each. A search
/// keeps one pool per kind of work, so no kind queues behind another. [#submit] returns the task's future;
/// nothing waits. [#close] waits for the admitted tasks and detaches the sinks.
public final class HostPool implements FrameLake, Ensemble.Frames, AutoCloseable {

    private final Sink[] sinks;
    /// The lake a pool that attaches only while it has work publishes into (null for request-only sinks).
    private final InferenceLake lake;
    private final HostTasks tasks;

    private HostPool(int sinks) {
        this.sinks = new Sink[sinks];
        for (int i = 0; i < sinks; i++) this.sinks[i] = new Sink();
        this.lake = null;
        this.tasks = new HostTasks(this);
    }

    private HostPool(InferenceLake lake) {
        this.sinks = null;
        this.lake = lake;
        this.tasks = new HostTasks(this);
    }

    /// A pool for occasional work beside the engine: `sinks` sinks of an [InferenceLake], which attaches them when a
    /// task arrives and detaches them once none has run for its idle time, so between tasks the lattice's workers
    /// have no idle sinks of this pool to cycle through.
    public static HostPool startAttachingOnDemand(int sinks) {
        if (sinks < 1) throw new IllegalArgumentException("a host pool needs a sink");
        return new HostPool(new InferenceLake(ControlPlaneLattice.getOrCreate(), sinks));
    }

    /// A pool of `sinks` sinks on the process's lattice.
    public static HostPool start(int sinks) {
        if (sinks < 1) throw new IllegalArgumentException("a host pool needs a sink");
        HostPool pool = new HostPool(sinks);
        ControlPlaneLattice lattice = ControlPlaneLattice.getOrCreate();
        for (Sink sink : pool.sinks) lattice.addUpstream(sink);
        return pool;
    }

    /// Runs `task` as one frame; the future ends with its result or its failure, on the worker that ran it.
    @Override
    public <T> CompletableFuture<T> submit(Supplier<T> task) {
        return tasks.onWorker(task);
    }

    @Override
    public void publish(AbstractFrame frame) {
        if (lake != null) {
            lake.publish(frame);
            return;
        }
        sinks[(int) Long.remainderUnsigned(frame.getRoutingHash(), sinks.length)].offer(frame);
    }

    @Override
    public void publishFromCallback(AbstractFrame frame) {
        publish(frame);
    }

    // HostTasks counts its own tasks and waits for them at close; the pool has nothing more to count.
    @Override
    public void admit() {
        if (lake != null) lake.admit();
    }

    @Override
    public void admitDuringDrain() {
        if (lake != null) lake.admitDuringDrain();
    }

    @Override
    public void terminated() {
        if (lake != null) lake.terminated();
    }

    @Override
    public void close() {
        tasks.close();
        if (lake != null) {
            lake.completeGracefully();
            lake.awaitTermination();
        } else for (Sink sink : sinks) sink.complete();
    }

    /// A queue of frames handed to the lattice only against its requests.
    private static final class Sink extends AbstractIngestSink {
        private final ConcurrentLinkedQueue<AbstractFrame> queue = new ConcurrentLinkedQueue<>();
        private final Delegate delegate = new Delegate();

        void offer(AbstractFrame frame) {
            queue.add(frame);
            delegate.deliver();
        }

        @Override
        public LatticeSource getDelegate() {
            return delegate;
        }

        @Override
        public void complete() {
            delegate.complete();
        }

        @Override
        public boolean isComplete() {
            return delegate.isComplete();
        }

        private final class Delegate extends AbstractIngestSink.Delegate {
            private final AtomicBoolean delivering = new AtomicBoolean();
            // A request is a wakeup: set before taking ownership, so a releasing owner or the requester serves it.
            private final AtomicBoolean requested = new AtomicBoolean();

            @Override
            public long hookOnPull(
                    Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stopCondition, long demand) {
                return 0;
            }

            @Override
            public void hookOnRequest(LatticeReceiver terminal, long demand) {
                requested.set(true);
                deliver();
            }

            /// Pushes queued frames to the lattice up to its outstanding demand, by whoever takes ownership.
            void deliver() {
                while ((requested.get() || (!queue.isEmpty() && demand.getAcquire() > 0))
                        && delivering.compareAndSet(false, true)) {
                    try {
                        requested.set(false);
                        LatticeReceiver terminal = getDownstream();
                        if (isComplete() || terminal == null) return;
                        long allowed = demand.getAcquire();
                        long pushed = 0;
                        for (AbstractFrame frame; pushed < allowed && (frame = queue.poll()) != null; pushed++)
                            terminal.push(frame);
                        if (pushed > 0) addAndGetDemand(-pushed);
                    } finally {
                        delivering.set(false);
                    }
                }
            }
        }
    }
}
