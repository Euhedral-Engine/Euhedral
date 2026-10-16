package io.euhedral_execution.inference.quantization.mixed;

import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.inference.core.runtime.HostTasks;
import io.euhedral_execution.inference.core.runtime.graph.InferenceLake;
import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;

/// One kind of host work for the search to run beside the GPU, as frames on the process's lattice: a lake of
/// its own (as many sinks as the lattice has workers, so every worker takes the pool's frames) and the
/// engine's [HostTasks] publishing each task into it as one frame, which whichever worker is free runs. A
/// search keeps one pool per kind of work, so no kind queues behind another. [#submit] returns the task's
/// future; nothing waits. [#close] waits for the admitted tasks and detaches the lake.
public final class HostPool implements AutoCloseable {

    private final InferenceLake lake;
    private final HostTasks tasks;

    private HostPool(InferenceLake lake, HostTasks tasks) {
        this.lake = lake;
        this.tasks = tasks;
    }

    /// A pool of `sinks` sinks on the process's lattice.
    public static HostPool start(int sinks) {
        if (sinks < 1) throw new IllegalArgumentException("a host pool needs a sink");
        InferenceLake lake = new InferenceLake(ControlPlaneLattice.getOrCreate(), sinks, 1);
        return new HostPool(lake, new HostTasks(lake));
    }

    /// Runs `task` as one frame; the future ends with its result or its failure, on the worker that ran it.
    public <T> CompletableFuture<T> submit(Supplier<T> task) {
        return tasks.onWorker(task);
    }

    @Override
    public void close() {
        tasks.close();
        lake.completeGracefully();
        lake.awaitTermination();
    }
}
