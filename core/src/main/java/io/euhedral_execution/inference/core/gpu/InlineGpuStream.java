package io.euhedral_execution.inference.core.gpu;

import java.util.concurrent.atomic.AtomicLong;

/// Ordering for synchronous test GPUs: each submitted operation has finished when `submit` returns,
/// so a boundary is retired as soon as it is notified.
public class InlineGpuStream implements GpuStream {

    private final AtomicLong tickets = new AtomicLong();

    @Override
    public void submit(Runnable launches, boolean overlapPredecessor) {
        launches.run();
    }

    @Override
    public long notifyRetired(RetirementListener listener) {
        long ticket = this.tickets.incrementAndGet();
        listener.retired(ticket, false);
        return ticket;
    }

    @Override
    public Throwable confirmRetired(long ticket) {
        return null;
    }

    @Override
    public void synchronize() {}

    @Override
    public void recover(Throwable failure) {}

    @Override
    public void close() {}
}
