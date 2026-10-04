package io.euhedral_execution.inference.api.chat;

import java.util.Objects;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// Runs one request's delivery tasks one at a time, in submission order, on the backend's workers.
///
/// `execute` never blocks: it queues the task and, when no drain is in flight, hands one drain to the workers.
/// The drain runs every queued task, including tasks queued while it runs, so a task (a network write) may
/// block its worker without holding up whoever submits the next one; the other workers take over the rest of
/// the lattice's work meanwhile.
final class SerialTasks implements Executor {
    private static final Logger LOG = LoggerFactory.getLogger(SerialTasks.class);
    private final Executor workers;
    private final ConcurrentLinkedQueue<Runnable> tasks = new ConcurrentLinkedQueue<>();
    private final AtomicInteger pending = new AtomicInteger();

    SerialTasks(Executor workers) {
        this.workers = Objects.requireNonNull(workers, "workers");
    }

    @Override
    public void execute(Runnable task) {
        this.tasks.add(Objects.requireNonNull(task, "task"));
        if (this.pending.getAndIncrement() == 0) this.workers.execute(this::drain);
    }

    private void drain() {
        int missed = 1;
        while (true) {
            for (Runnable task = this.tasks.poll(); task != null; task = this.tasks.poll()) {
                try {
                    task.run();
                } catch (RuntimeException failure) {
                    // A task reports its own failures; one that escapes must not stop the tasks after it.
                    LOG.error("Chat delivery task failed", failure);
                }
            }
            missed = this.pending.addAndGet(-missed);
            if (missed == 0) return;
        }
    }
}
