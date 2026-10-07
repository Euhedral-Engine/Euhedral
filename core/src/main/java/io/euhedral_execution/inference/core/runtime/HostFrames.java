package io.euhedral_execution.inference.core.runtime;

/// Where host work goes to run: each piece becomes one frame that a lattice worker takes first come, first
/// served, or that the lattice routes by the frame's hash. A holder of `HostFrames` describes what is ready;
/// it never chooses a thread, owns one, or waits for the work it hands over.
///
/// Work runs on some worker later, never inline on the caller. A task is short: the synchronous CPU portion
/// of starting or finishing an operation, after which it returns and the operation's completion publishes the
/// next task. Backpressure is the bound of a resource that the requester stays within (a window of loads that
/// fits the staging slots), never a wait.
public interface HostFrames {

    /// One piece of host work.
    interface Task {

        /// Runs on a lattice worker.
        void run();

        /// The task did not complete: the lattice rejected it without running it (it is shutting down), or
        /// [#run] threw. Runs on the thread that found out, so it must not block.
        default void failed(Throwable cause) {
            throw new IllegalStateException("host work failed", cause);
        }
    }

    /// Makes `task` runnable from a worker or any ordinary thread.
    ///
    /// @throws IllegalStateException when the host work is closed; the task did not run
    void run(Task task);

    /// Makes `task` runnable from a CUDA driver callback thread, where nothing but enqueueing is allowed: the
    /// task is queued and runs on a worker.
    ///
    /// @throws IllegalStateException when the host work is closed; the task did not run
    void runFromCallback(Task task);

    /// A task that runs `work`; a failure is reported to `failure`.
    static Task of(Runnable work, java.util.function.Consumer<Throwable> failure) {
        return new Task() {
            @Override
            public void run() {
                work.run();
            }

            @Override
            public void failed(Throwable cause) {
                failure.accept(cause);
            }
        };
    }
}
