package io.euhedral_execution.inference.core.host;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/// A [HostFrames] for tests that run no lattice: its tasks run on a small pool of daemon threads of the test's own.
/// Production code never has one; the point of [HostFrames] is that it does not.
public final class TestHostFrames implements HostFrames, AutoCloseable {

    /// Shared by tests that need no isolation.
    public static final TestHostFrames SHARED = new TestHostFrames(8);

    private final ExecutorService pool;

    public TestHostFrames(int threads) {
        AtomicInteger names = new AtomicInteger();
        this.pool = Executors.newFixedThreadPool(threads, runnable -> {
            Thread thread = new Thread(runnable, "test-host-frames-" + names.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        });
    }

    @Override
    public void run(Task task) {
        this.pool.execute(() -> {
            try {
                task.run();
            } catch (Throwable failure) {
                task.failed(failure);
            }
        });
    }

    @Override
    public void runFromCallback(Task task) {
        run(task);
    }

    @Override
    public void close() {
        this.pool.shutdown();
    }
}
