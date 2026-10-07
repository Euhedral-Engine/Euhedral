package io.euhedral_execution.inference.core.runtime;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.generics.LatticeTerminal;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.locks.LockSupport;

/// Test stand-in for Euhedral: one worker per attached source pulls and runs ready frames, parking
/// briefly after an empty pull, until the source completes. Frames that a CUDA driver callback only
/// enqueued are therefore delivered, as Euhedral's own workers deliver them.
public final class PullingLattice implements LatticeTerminal, AutoCloseable {

    private final List<Thread> workers = new CopyOnWriteArrayList<>();
    private final RuntimeException completionFailure;
    private volatile boolean closed;

    public PullingLattice() {
        this(null);
    }

    /// A lattice whose downstream throws `completionFailure` when a source completes.
    public PullingLattice(RuntimeException completionFailure) {
        this.completionFailure = completionFailure;
    }

    @Override
    public void addUpstream(LatticeSource source) {
        source.addDownstream(new LatticeReceiver() {
            @Override
            public void push(AbstractFrame frame) {
                run(frame);
            }

            @Override
            public void onComplete() {
                if (PullingLattice.this.completionFailure != null) throw PullingLattice.this.completionFailure;
            }

            @Override
            public void onError(Throwable failure) {
                throw new AssertionError(failure);
            }

            @Override
            public void addUpstream(LatticeSource upstream) {}
        });
        Thread worker = Thread.ofPlatform().daemon().name("pulling-lattice").start(() -> {
            while (!this.closed && !source.isComplete()) {
                if (source.pull(PullingLattice::run, frame -> false, 4096) == 0) LockSupport.parkNanos(20_000L);
            }
        });
        this.workers.add(worker);
    }

    /// What Euhedral's execution terminal does with one delivered frame.
    public static void run(AbstractFrame frame) {
        try {
            frame.execute();
        } catch (Exception failure) {
            frame.doFinallyWithError(failure);
            return;
        }
        frame.doFinally();
    }

    @Override
    public void close() throws InterruptedException {
        this.closed = true;
        for (Thread worker : this.workers) worker.join();
    }
}
