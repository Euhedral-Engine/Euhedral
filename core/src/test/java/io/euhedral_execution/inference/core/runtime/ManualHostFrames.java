package io.euhedral_execution.inference.core.runtime;

import java.util.ArrayDeque;

/// A [HostFrames] whose tasks wait until the test runs them, for deterministic orderings: nothing runs on its own.
public final class ManualHostFrames implements HostFrames {

    private final ArrayDeque<Task> queue = new ArrayDeque<>();
    private boolean closed;

    @Override
    public synchronized void run(Task task) {
        if (this.closed) throw new IllegalStateException("closed");
        this.queue.addLast(task);
    }

    @Override
    public void runFromCallback(Task task) {
        run(task);
    }

    /// Tasks waiting.
    public synchronized int pending() {
        return this.queue.size();
    }

    /// Runs the oldest waiting task on the calling thread; returns whether there was one.
    public boolean runOne() {
        Task task;
        synchronized (this) {
            task = this.queue.pollFirst();
        }
        if (task == null) return false;
        try {
            task.run();
        } catch (Throwable failure) {
            task.failed(failure);
        }
        return true;
    }

    /// Runs tasks (including those they publish) until none is left; returns how many ran.
    public int runAll() {
        int ran = 0;
        while (runOne()) ran++;
        return ran;
    }

    public synchronized void close() {
        this.closed = true;
    }
}
