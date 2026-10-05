package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/// An asynchronous stream: operations queued by `submit` (host-to-device copies, marker recordings, marker waits,
/// retirement boundaries) run in order on a worker thread that plays the CUDA driver, so a boundary's listener
/// runs on that thread with `driverThread == true`, exactly as a driver callback would.
final class FakeStream implements GpuStream {
    private static final ThreadLocal<FakeStream> SELECTED = new ThreadLocal<>();
    private static final Runnable STOP = () -> {};

    private final LinkedBlockingQueue<Runnable> operations = new LinkedBlockingQueue<>();
    private final Thread worker;
    private final AtomicLong tickets = new AtomicLong();
    private static final AtomicLong MARKER_IDS = new AtomicLong();
    private static final Map<Long, Marker> MARKERS = new HashMap<>();
    private final Set<Long> closedMarkers = Collections.synchronizedSet(new HashSet<>());
    private final AtomicInteger failRetirements = new AtomicInteger();
    private final AtomicInteger failNotifications = new AtomicInteger();
    private final AtomicInteger recoveries = new AtomicInteger();
    private final AtomicBoolean confirmedOnWorker = new AtomicBoolean();
    private final AtomicBoolean closed = new AtomicBoolean();

    private static final class Marker {
        private int requested;
        private int recorded;
    }

    FakeStream() {
        this.worker = new Thread(this::run, "fake-stream");
        this.worker.setDaemon(true);
        this.worker.start();
    }

    static FakeStream selected() {
        return SELECTED.get();
    }

    private void run() {
        while (true) {
            Runnable operation;
            try {
                operation = this.operations.take();
            } catch (InterruptedException interrupted) {
                return;
            }
            if (operation == STOP) return;
            operation.run();
        }
    }

    void enqueue(Runnable operation) {
        this.operations.add(operation);
    }

    /// Makes the stream stop at its current position until `release` is counted down.
    void stall(CountDownLatch release) {
        enqueue(() -> {
            try {
                if (!release.await(20, TimeUnit.SECONDS)) throw new AssertionError("stall was never released");
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        });
    }

    /// The next `count` boundaries confirm as a device failure.
    void failNextRetirements(int count) {
        this.failRetirements.set(count);
    }

    /// The next `count` boundary registrations throw.
    void failNextNotifications(int count) {
        this.failNotifications.set(count);
    }

    boolean markerClosed(long marker) {
        return this.closedMarkers.contains(marker);
    }

    int recoveries() {
        return this.recoveries.get();
    }

    boolean confirmedOnWorker() {
        return this.confirmedOnWorker.get();
    }

    @Override
    public void submit(Runnable launches, boolean overlapPredecessor) {
        FakeStream previous = SELECTED.get();
        SELECTED.set(this);
        try {
            launches.run();
        } finally {
            if (previous == null) SELECTED.remove();
            else SELECTED.set(previous);
        }
    }

    @Override
    public long notifyRetired(RetirementListener listener) {
        if (this.failNotifications.getAndUpdate(count -> Math.max(0, count - 1)) > 0)
            throw new IllegalStateException("injected boundary registration failure");
        long ticket = this.tickets.incrementAndGet();
        enqueue(() -> listener.retired(ticket, true));
        return ticket;
    }

    @Override
    public Throwable confirmRetired(long ticket) {
        if (Thread.currentThread() == this.worker) this.confirmedOnWorker.set(true);
        if (this.failRetirements.getAndUpdate(count -> Math.max(0, count - 1)) > 0)
            return new IllegalStateException("injected device failure");
        return null;
    }

    @Override
    public long openMarker() {
        long id = MARKER_IDS.incrementAndGet();
        synchronized (MARKERS) {
            MARKERS.put(id, new Marker());
        }
        return id;
    }

    @Override
    public void mark(long marker) {
        Marker state = marker(marker);
        synchronized (state) {
            state.requested++;
        }
        enqueue(() -> {
            synchronized (state) {
                state.recorded++;
                state.notifyAll();
            }
        });
    }

    @Override
    public void await(long marker) {
        Marker state = marker(marker);
        int target;
        synchronized (state) {
            target = state.requested;
        }
        if (target == 0) return;
        enqueue(() -> {
            synchronized (state) {
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20);
                while (state.recorded < target) {
                    long remaining = deadline - System.nanoTime();
                    if (remaining <= 0) throw new AssertionError("marker " + marker + " was never recorded");
                    try {
                        TimeUnit.NANOSECONDS.timedWait(state, remaining);
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                }
            }
        });
    }

    @Override
    public void closeMarker(long marker) {
        if (!this.closedMarkers.add(marker)) throw new AssertionError("marker " + marker + " closed twice");
    }

    private Marker marker(long id) {
        synchronized (MARKERS) {
            Marker marker = MARKERS.get(id);
            if (marker == null) throw new IllegalArgumentException("unknown marker " + id);
            return marker;
        }
    }

    @Override
    public void synchronize() {
        CountDownLatch reached = new CountDownLatch(1);
        enqueue(reached::countDown);
        try {
            if (!reached.await(20, TimeUnit.SECONDS)) throw new AssertionError("stream did not drain");
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    @Override
    public void recover(Throwable failure) {
        this.recoveries.incrementAndGet();
        synchronize();
    }

    @Override
    public void close() {
        if (!this.closed.compareAndSet(false, true)) return;
        enqueue(STOP);
        try {
            this.worker.join(TimeUnit.SECONDS.toMillis(20));
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
    }
}
