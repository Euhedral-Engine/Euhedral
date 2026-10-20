package io.euhedral_execution.inference.core.runtime;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.generics.LatticeTerminal;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;

/// A lattice that runs nothing by itself: the test thread takes the frames its sources hold ready and
/// runs them in waves, each wave the frames the previous one published, so the frames of one wave were
/// all runnable at once whatever the number of cores. Records the wave and route of the frame it is
/// running ([#running]).
public final class SteppingLattice implements LatticeTerminal {
    private static final ThreadLocal<long[]> RUNNING = new ThreadLocal<>();
    private final List<LatticeSource> sources = new CopyOnWriteArrayList<>();
    private final Queue<AbstractFrame> pushed = new ConcurrentLinkedQueue<>();
    private long wave;

    @Override
    public void addUpstream(LatticeSource source) {
        source.addDownstream(new LatticeReceiver() {
            @Override
            public void push(AbstractFrame frame) {
                SteppingLattice.this.pushed.add(frame);
            }

            @Override
            public void onComplete() {}

            @Override
            public void onError(Throwable failure) {
                throw new AssertionError(failure);
            }

            @Override
            public void addUpstream(LatticeSource upstream) {}
        });
        this.sources.add(source);
    }

    /// {wave, route} of the frame running on this thread, or null outside a stepped frame.
    public static long[] running() {
        long[] running = RUNNING.get();
        return running == null ? null : running.clone();
    }

    /// The frames ready now, taken from the sources.
    public List<AbstractFrame> ready() {
        List<AbstractFrame> frames = new ArrayList<>();
        for (var frame = this.pushed.poll(); frame != null; frame = this.pushed.poll()) frames.add(frame);
        for (var source : this.sources) if (!source.isComplete()) source.pull(frames::add, frame -> false, 4096);
        return frames;
    }

    /// Runs `frames` as one wave.
    public void run(List<AbstractFrame> frames) {
        this.wave++;
        for (var frame : frames) {
            RUNNING.set(new long[] {this.wave, frame.getRoutingHash()});
            try {
                PullingLattice.run(frame);
            } finally {
                RUNNING.remove();
            }
        }
    }

    /// Runs the admissions until none is ready, and returns the other frames they made ready, unrun.
    public List<AbstractFrame> admitAll() {
        List<AbstractFrame> held = new ArrayList<>();
        while (true) {
            List<AbstractFrame> admissions = new ArrayList<>();
            for (var frame : ready()) {
                if (frame instanceof EuhedralInferenceRuntime.Admission) admissions.add(frame);
                else held.add(frame);
            }
            if (admissions.isEmpty()) return held;
            run(admissions);
        }
    }

    /// Runs waves until `done` holds; fails if the frames run out or `limit` waves pass first.
    public void runUntil(BooleanSupplier done, int limit) {
        for (int waves = 0; waves < limit && !done.getAsBoolean(); waves++) {
            List<AbstractFrame> frames = ready();
            assertFalse(frames.isEmpty(), "no frame is ready, and the work is not done");
            run(frames);
        }
        assertTrue(done.getAsBoolean(), "the work did not finish within " + limit + " waves");
    }

    /// Runs `close` on another thread while this one runs the frames it waits for.
    public void closeWhileStepping(Runnable close) throws Exception {
        var closed = CompletableFuture.runAsync(close);
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (!closed.isDone() && System.nanoTime() < deadline) {
            List<AbstractFrame> frames = ready();
            if (frames.isEmpty()) Thread.onSpinWait();
            else run(frames);
        }
        closed.get(1, TimeUnit.SECONDS);
    }
}
