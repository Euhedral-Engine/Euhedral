package io.euhedral_execution.inference.core.scheduling.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.data_structures.queues.MpscQueue;
import java.util.ArrayDeque;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// A lattice source that generates its frames from its own state, in the lattice's `request` and
/// `pull` calls.
///
/// Euhedral calls `request` and `pull` on a registered source one thread at a time, and publishes
/// the state each call leaves to the next (the registration's work-in-progress counter is acquired
/// and released around every call). A subclass therefore keeps its state in plain fields, touched
/// only from [#handle], and needs no lock, atomic or volatile for it.
///
/// Everything else reaches that state by posting a message. [#post] is a lock-free enqueue that any
/// thread may do (a worker that finished a frame, a CUDA driver callback that may only enqueue);
/// the next `request` or `pull` drains the mailbox through [#handle], which updates the state and
/// calls [#ready] for each frame that became runnable. A frame that is ready is delivered by that
/// same call: pushed downstream against the demand a worker registered with `request` (so it is
/// routed by its hash, and runs outside this source's call), or handed to the pulling worker, which
/// leaves frames its stop condition names (the ordered ones) in the source for the routed path.
///
/// One source is one serial owner. Work that must run in parallel is split over several sources,
/// each attached to the lattice on its own, so that Euhedral's workers serve them side by side.
public abstract class SerialSource implements LatticeSource {

    private static final Logger LOG = LoggerFactory.getLogger(SerialSource.class);

    private final MpscQueue<Object> mailbox = new MpscQueue<>(256, 4);
    private final ArrayDeque<AbstractFrame> ready = new ArrayDeque<>();
    private final AtomicBoolean attached = new AtomicBoolean();
    private final AtomicBoolean finished = new AtomicBoolean();
    private volatile LatticeReceiver downstream;
    /// Demand registered by `request` and not yet served; serial.
    private long demand;

    /// Posts `message` for [#handle]. Any thread; never blocks.
    public final void post(Object message) {
        Objects.requireNonNull(message, "message");
        if (!this.mailbox.offer(message)) throw new IllegalStateException("a source mailbox rejected a message");
    }

    /// Updates the state for one message. Serial: called only from `request` and `pull`. Must not
    /// throw for a message it understands, never blocks, and calls [#ready] for each frame the
    /// message made runnable.
    protected abstract void handle(Object message);

    /// Marks `frame` runnable. Serial: called from [#handle].
    protected final void ready(AbstractFrame frame) {
        this.ready.addLast(frame);
    }

    /// Whether a frame is waiting for a worker.
    protected final boolean hasReady() {
        return !this.ready.isEmpty();
    }

    private void drain() {
        Object message;
        while ((message = this.mailbox.poll()) != null) {
            try {
                handle(message);
            } catch (RuntimeException | Error failure) {
                LOG.error("a serial source failed to handle {}", message, failure);
            }
        }
    }

    @Override
    public final long pull(
            Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stopCondition, long demand) {
        if (demand <= 0 || this.finished.get()) return 0;
        drain();
        long delivered = 0;
        while (delivered < demand) {
            AbstractFrame frame = this.ready.peekFirst();
            if (frame == null || stopCondition.apply(frame)) break;
            this.ready.pollFirst();
            delivered++;
            // The frame may run here; what it posts is handled before the next one is taken.
            consumer.accept(frame);
            drain();
        }
        return delivered;
    }

    @Override
    public final void request(long demand) {
        if (demand <= 0 || this.finished.get()) return;
        long next = this.demand + demand;
        this.demand = next < 0 ? Long.MAX_VALUE : next;
        LatticeReceiver receiver = this.downstream;
        if (receiver == null) return;
        drain();
        while (this.demand > 0) {
            AbstractFrame frame = this.ready.pollFirst();
            if (frame == null) break;
            if (this.demand != Long.MAX_VALUE) this.demand--;
            receiver.push(frame);
            drain();
        }
    }

    @Override
    public void addDownstream(LatticeReceiver receiver) {
        Objects.requireNonNull(receiver, "receiver");
        if (!this.attached.compareAndSet(false, true)) {
            receiver.onError(new IllegalStateException("a serial source has one downstream"));
            return;
        }
        this.downstream = receiver;
        if (this.finished.get()) completeDownstream();
    }

    /// Completes the source: the lattice detaches it. Idempotent; any thread.
    @Override
    public void complete() {
        if (!this.finished.compareAndSet(false, true)) return;
        completeDownstream();
    }

    private void completeDownstream() {
        LatticeReceiver receiver = this.downstream;
        this.downstream = null;
        if (receiver != null) receiver.onComplete();
    }

    @Override
    public boolean isComplete() {
        return this.finished.get();
    }

    /// Whether the lattice still holds this source.
    public boolean isAttached() {
        return this.downstream != null;
    }
}
