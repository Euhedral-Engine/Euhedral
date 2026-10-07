package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.data_structures.queues.MpscQueue;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// Parallel execution, ordered completion, as Euhedral-Execution's `FrameSequencer` and its Kafka offset collector do
/// it. Entries are offered in the order they must conclude and execute in any order. An entry that completes marks
/// itself [Entry#ready()] and calls [#drain()]. Whichever thread finds the head ready concludes the ready prefix, in
/// offer order, through [#drained]; a work-in-progress count makes exactly one thread drain at a time, without a
/// lock, and a completion that arrives during a drain makes that drain look again instead of waiting.
///
/// Any thread may offer (the queue is multi-producer); the order is the order of the offers, so the owner of the
/// order offers from one chain of work.
public abstract class Sequencer<E extends Sequencer.Entry> {

    private static final Logger LOG = LoggerFactory.getLogger(Sequencer.class);

    /// Something concluded in order.
    public interface Entry {
        /// Whether the entry completed and may conclude once everything offered before it concluded.
        boolean ready();
    }

    private final MpscQueue<E> order = new MpscQueue<>(16, 2);
    private final AtomicInteger wip = new AtomicInteger();

    /// Appends `entry` to the order.
    public final void offer(E entry) {
        if (!this.order.offer(entry)) throw new IllegalStateException("the sequencer refused an entry");
    }

    /// Whether every entry offered so far has concluded.
    public final boolean isEmpty() {
        return this.order.isEmpty();
    }

    /// Concludes the ready entries at the head, in order. Called after an entry became ready; returns at once when
    /// another thread is draining, which then concludes it.
    public final void drain() {
        if (this.wip.getAndIncrement() != 0) return;
        int missed = 1;
        while (true) {
            E head;
            while ((head = this.order.peek()) != null && head.ready()) {
                try {
                    drained(head);
                } catch (RuntimeException | Error failure) {
                    // An entry's failure is its own; the entries behind it still conclude.
                    LOG.error("an entry failed to conclude", failure);
                } finally {
                    this.order.poll();
                }
            }
            missed = this.wip.addAndGet(-missed);
            if (missed == 0) return;
        }
    }

    /// Concludes `entry`, after every entry offered before it. Runs on one thread at a time.
    protected abstract void drained(E entry);
}
