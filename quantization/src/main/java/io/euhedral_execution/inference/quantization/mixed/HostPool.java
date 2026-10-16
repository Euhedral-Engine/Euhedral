package io.euhedral_execution.inference.quantization.mixed;

import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.ingest.AbstractIngestSink;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

/// Host work for the search to run beside the GPU: ingest sinks registered on the lattice, each running one
/// queued task per poll on whichever worker polls it (the lattice polls a source one thread at a time).
/// [#submit] queues a task and returns its future; nothing waits. [#close] stops the sinks taking tasks, lets
/// the running ones end, and removes the sinks from the lattice.
public final class HostPool implements AutoCloseable {

    /// One sink: a queue of tasks, the lattice's polls run them.
    private final class Work extends AbstractIngestSink {
        private final int index;
        private final Delegate delegate = new Delegate();

        Work(int index) {
            this.index = index;
        }

        private void poll() {
            if (index >= active) return;
            Runnable task = tasks.poll();
            if (task != null) task.run();
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
            @Override
            public long hookOnPull(
                    Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stopCondition, long demand) {
                poll();
                return 0;
            }

            @Override
            public void hookOnRequest(LatticeReceiver terminal, long demand) {
                poll();
            }
        }
    }

    private final List<Work> sinks = new ArrayList<>();
    /// The tasks every sink takes from: whichever enabled sink polls next runs the oldest.
    private final ConcurrentLinkedQueue<Runnable> tasks = new ConcurrentLinkedQueue<>();
    /// The sinks that take tasks: the first `active` ones.
    private volatile int active;
    private final ConcurrentLinkedQueue<CompletableFuture<?>> submitted = new ConcurrentLinkedQueue<>();
    private volatile boolean closed;

    /// `sinks` sinks registered on the process's lattice.
    public static HostPool start(int sinks) {
        ControlPlaneLattice lattice = ControlPlaneLattice.getOrCreate();
        if (sinks < 1) throw new IllegalArgumentException("a host pool needs a sink");
        HostPool pool = new HostPool();
        pool.active = sinks;
        for (int i = 0; i < sinks; i++) {
            Work work = pool.new Work(i);
            pool.sinks.add(work);
            lattice.addUpstream(work);
        }
        return pool;
    }

    private HostPool() {}

    /// Queues `task` on a sink (in turn); the future ends with its result, its failure, or a cancellation if
    /// the pool closed before it ran. Called from the owner's thread and from tasks.
    public <T> CompletableFuture<T> submit(Supplier<T> task) {
        CompletableFuture<T> future = new CompletableFuture<>();
        submitted.add(future);
        tasks.add(() -> {
            if (closed) {
                future.cancel(false);
                return;
            }
            try {
                future.complete(task.get());
            } catch (Throwable failure) {
                future.completeExceptionally(failure);
            }
        });
        return future;
    }

    /// Only the first sink takes tasks: work beside the engine's frames keeps to one owner worker, so the
    /// stage frames on every other worker run unhindered.
    public void narrow() {
        active = 1;
    }

    /// Every sink takes tasks again.
    public void widen() {
        active = sinks.size();
    }

    @Override
    public void close() {
        closed = true;
        // Tasks still queued are cancelled by the next poll; the running ones end by themselves.
        for (CompletableFuture<?> future; (future = submitted.poll()) != null; ) {
            try {
                future.handle((value, failure) -> null).get();
            } catch (CancellationException | ExecutionException ignored) {
                // Outcome not needed.
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        for (Work work : sinks) work.complete();
    }
}
