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

/// One kind of host work for the search to run beside the GPU: ingest sinks registered on the lattice, sharing
/// one queue, each running one queued task per poll on whichever worker polls it (the lattice polls a source one
/// thread at a time). A search keeps one pool per kind of work, so no kind queues behind another; as many sinks
/// as workers lets every worker take the pool's work.
/// [#submit] queues a task and returns its future; nothing waits. [#close] stops the sinks taking tasks, lets
/// the running ones end, and removes the sinks from the lattice.
public final class HostPool implements AutoCloseable {

    /// One sink: the lattice's polls run the pool's queued tasks.
    private final class Work extends AbstractIngestSink {
        private final Delegate delegate = new Delegate();

        private void poll() {
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
    private final ConcurrentLinkedQueue<CompletableFuture<?>> submitted = new ConcurrentLinkedQueue<>();
    private volatile boolean closed;

    /// `sinks` sinks registered on the process's lattice.
    public static HostPool start(int sinks) {
        ControlPlaneLattice lattice = ControlPlaneLattice.getOrCreate();
        if (sinks < 1) throw new IllegalArgumentException("a host pool needs a sink");
        HostPool pool = new HostPool();
        for (int i = 0; i < sinks; i++) {
            Work work = pool.new Work();
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
