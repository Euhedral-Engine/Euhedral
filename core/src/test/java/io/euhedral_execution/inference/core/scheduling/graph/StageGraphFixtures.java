package io.euhedral_execution.inference.core.scheduling.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/// Synthetic stages, a recording device stream, and a stand-in for Euhedral's frame lifecycle.
final class StageGraphFixtures {

    private StageGraphFixtures() {}

    /// What Euhedral's execution terminal does with one delivered frame.
    static void run(AbstractFrame frame) {
        try {
            frame.execute();
        } catch (Exception failure) {
            frame.doFinallyWithError(failure);
            return;
        }
        frame.doFinally();
    }

    /// Frames the source currently exposes, taken without running them.
    static List<AbstractFrame> take(QwenExecutionSource source) {
        List<AbstractFrame> frames = new ArrayList<>();
        source.pull(frames::add, frame -> false, Long.MAX_VALUE);
        return frames;
    }

    /// Pulls and runs every exposed frame, including those that running them makes ready.
    static int drain(QwenExecutionSource source) {
        return (int) source.pull(StageGraphFixtures::run, frame -> false, Long.MAX_VALUE);
    }

    static int[][] dependencies(int[]... stages) {
        return stages;
    }

    /// A device stream that records kernel order and holds each retirement boundary until released.
    static final class RecordingStream implements GpuStream {
        final List<String> kernels = Collections.synchronizedList(new ArrayList<>());
        final List<String> threads = Collections.synchronizedList(new ArrayList<>());
        final List<Registration> registrations = Collections.synchronizedList(new ArrayList<>());
        final ConcurrentHashMap<Long, Throwable> failures = new ConcurrentHashMap<>();
        final AtomicInteger recoveries = new AtomicInteger();
        final AtomicLong tickets = new AtomicLong();
        volatile boolean retireOnNotify;
        volatile RuntimeException registrationFailure;
        int selectedDepth;
        boolean closed;

        record Registration(long ticket, RetirementListener listener) {}

        void kernel(String name) {
            if (this.selectedDepth <= 0) throw new AssertionError("kernel launched without its stream selected");
            this.kernels.add(name);
            this.threads.add(Thread.currentThread().getName());
        }

        @Override
        public synchronized void submit(Runnable launches, boolean overlapPredecessor) {
            this.selectedDepth++;
            try {
                launches.run();
            } finally {
                this.selectedDepth--;
            }
        }

        @Override
        public long notifyRetired(RetirementListener listener) {
            RuntimeException failure = this.registrationFailure;
            if (failure != null) throw failure;
            long ticket = this.tickets.incrementAndGet();
            this.registrations.add(new Registration(ticket, listener));
            if (this.retireOnNotify) listener.retired(ticket, false);
            return ticket;
        }

        /// Announces the oldest unannounced boundary the way the CUDA driver would.
        void retireNext(boolean driverThread) {
            Registration next;
            synchronized (this.registrations) {
                if (this.registrations.isEmpty()) throw new AssertionError("no armed retirement boundary");
                next = this.registrations.removeFirst();
            }
            next.listener().retired(next.ticket(), driverThread);
        }

        int armed() {
            return this.registrations.size();
        }

        @Override
        public Throwable confirmRetired(long ticket) {
            return this.failures.remove(ticket);
        }

        @Override
        public void synchronize() {}

        @Override
        public void recover(Throwable failure) {
            this.recoveries.incrementAndGet();
        }

        @Override
        public void close() {
            this.closed = true;
        }
    }

    /// A stage that launches one named kernel on its quantum's stream.
    static final class TestStage extends StageFrame {
        final List<Boolean> retirements = Collections.synchronizedList(new ArrayList<>());
        volatile RuntimeException failure;
        volatile Runnable beforeLaunch;
        volatile int launches;

        TestStage(StageGraph graph, int stage) {
            super(graph, stage);
        }

        @Override
        protected void submit() {
            Runnable hook = this.beforeLaunch;
            if (hook != null) hook.run();
            this.launches++;
            ((RecordingStream) graph().stream()).kernel("k" + stage());
            RuntimeException injected = this.failure;
            if (injected != null) throw injected;
        }

        @Override
        protected void retired(boolean committed) {
            this.retirements.add(committed);
            ((TestQuantum) graph().quantum()).events.add((committed ? "commit:" : "release:") + stage());
        }
    }

    /// A quantum binding that records its terminal sequence.
    static final class TestQuantum implements StageQuantum {
        final List<String> events = Collections.synchronizedList(new ArrayList<>());
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        final CompletableFuture<String> outcome = new CompletableFuture<>();
        final AtomicInteger outcomes = new AtomicInteger();
        volatile boolean cancelled;
        volatile boolean overlap;
        volatile Runnable onOutcome;

        @Override
        public boolean stopRequested() {
            return this.cancelled || this.failure.get() != null;
        }

        @Override
        public void fail(Throwable cause) {
            if (!this.failure.compareAndSet(null, cause) && this.failure.get() != cause) {
                this.failure.get().addSuppressed(cause);
            }
        }

        @Override
        public boolean overlapLaunches() {
            return this.overlap;
        }

        @Override
        public void retire(Throwable deviceFailure) {
            if (deviceFailure != null) fail(deviceFailure);
            this.events.add("retire");
        }

        @Override
        public void publishOutcome() {
            this.outcomes.incrementAndGet();
            this.events.add("outcome");
            Runnable hook = this.onOutcome;
            if (hook != null) hook.run();
            Throwable cause = this.failure.get();
            this.outcome.complete(cause != null ? "FAILED" : this.cancelled ? "CANCELLED" : "SUCCESS");
        }
    }

    /// Records graphs returned for reuse.
    static final class Recycler implements StageGraph.Recycler {
        final List<StageGraph> recycled = Collections.synchronizedList(new ArrayList<>());

        @Override
        public void recycle(StageGraph graph) {
            this.recycled.add(graph);
        }
    }

    static StageGraph graph(
            StageTopology topology, RecordingStream stream, QwenExecutionSource source, Recycler recycler) {
        return new StageGraph(topology, TestStage::new, stream, source, recycler);
    }

    static TestStage stage(StageGraph graph, int stage) {
        return (TestStage) graph.stage(stage);
    }
}
