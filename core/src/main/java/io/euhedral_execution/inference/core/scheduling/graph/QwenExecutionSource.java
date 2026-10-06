package io.euhedral_execution.inference.core.scheduling.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.data_structures.queues.MpscQueue;
import java.lang.invoke.VarHandle;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Function;

/// The boundary between Qwen frame graphs and Euhedral.
///
/// Admission publishes a graph's root frames and each stage publishes the successors it made ready.
/// The source only holds ready frames: it does not know what a frame computes, which frames depend on
/// it, or how many stages a graph has. Workers take ready frames from it through `pull` and `request`;
/// nothing above them decides which worker takes which frame.
///
/// Euhedral calls `pull` and `request` one at a time. Publishers run concurrently on workers and on
/// driver callback threads, so the ready queue has one drain owner at a time: a worker-side publication
/// may deliver against outstanding `request` demand only while no other drain is active.
public final class QwenExecutionSource implements LatticeSource {

    private static final int CLOSED = Integer.MIN_VALUE;

    private final MpscQueue<AbstractFrame> ready = new MpscQueue<>(256, 4);
    private final AtomicInteger drain = new AtomicInteger();
    private final AtomicLong requested = new AtomicLong();
    private final AtomicInteger accepted = new AtomicInteger();
    private final AtomicReference<LatticeReceiver> downstream = new AtomicReference<>();
    private final AtomicBoolean attached = new AtomicBoolean();
    private final AtomicBoolean finished = new AtomicBoolean();
    private final CompletableFuture<Void> termination = new CompletableFuture<>();

    /// Accepts one graph for execution. The source cannot complete until that graph terminates.
    public void admit() {
        while (true) {
            int count = this.accepted.get();
            if (count < 0) throw new IllegalStateException("Qwen execution admission is closed");
            if (count == Integer.MAX_VALUE) throw new IllegalStateException("too many active quanta");
            if (this.accepted.compareAndSet(count, count + 1)) return;
        }
    }

    /// Accepts one more unit while the source drains: allowed only while an accepted unit is still running, whose
    /// continuation this is, so a closing source finishes the work it holds and then refuses everything.
    public void admitDuringDrain() {
        while (true) {
            int count = this.accepted.get();
            if ((count & Integer.MAX_VALUE) == 0) throw new IllegalStateException("Qwen execution admission is closed");
            if ((count & Integer.MAX_VALUE) == Integer.MAX_VALUE)
                throw new IllegalStateException("too many active quanta");
            if (this.accepted.compareAndSet(count, count + 1)) return;
        }
    }

    /// Accepted graphs that have not yet reached their terminal state.
    public int activeGraphs() {
        return this.accepted.get() & Integer.MAX_VALUE;
    }

    /// Records that an accepted graph reached its terminal state.
    public void terminated() {
        int remaining = this.accepted.decrementAndGet();
        if (remaining == CLOSED) signalComplete();
        // Open counts are non-negative and closed counts start at CLOSED: either underflow wraps.
        else if (remaining == -1 || remaining == Integer.MAX_VALUE) {
            throw new IllegalStateException("graph termination exceeded admission");
        }
    }

    /// Makes a ready frame available to Euhedral from a worker or admission thread.
    public void publish(AbstractFrame frame) {
        offer(frame);
        // Pairs with the demand increment in request(): one side observes the other.
        VarHandle.fullFence();
        if (this.requested.get() != 0L) deliver();
    }

    /// Makes a ready frame available from a driver callback thread, which must never run frames.
    public void publishFromCallback(AbstractFrame frame) {
        offer(frame);
    }

    private void offer(AbstractFrame frame) {
        Objects.requireNonNull(frame, "frame");
        if (!this.ready.offer(frame)) throw new IllegalStateException("Qwen ready queue rejected a frame");
    }

    @Override
    public void addDownstream(LatticeReceiver receiver) {
        Objects.requireNonNull(receiver, "receiver");
        if (!this.attached.compareAndSet(false, true)) {
            receiver.onError(new IllegalStateException("Qwen source already has a downstream"));
        } else if (this.finished.get()) {
            notifyComplete(receiver);
        } else {
            this.downstream.set(receiver);
            if (this.finished.get() && this.downstream.compareAndSet(receiver, null)) notifyComplete(receiver);
        }
    }

    @Override
    public long pull(Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stopCondition, long demand) {
        Objects.requireNonNull(consumer, "consumer");
        Objects.requireNonNull(stopCondition, "stopCondition");
        if (demand <= 0 || this.finished.get()) return 0;
        // A worker-side publication is delivering against request demand; it owns the queue.
        if (this.drain.getAndIncrement() != 0) return 0;
        long delivered = 0;
        try {
            while (delivered < demand) {
                AbstractFrame frame = this.ready.peek();
                if (frame == null || stopCondition.apply(frame)) break;
                this.ready.poll();
                delivered++;
                // Frames that this one makes ready are appended to the queue and delivered by this loop.
                consumer.accept(frame);
            }
        } finally {
            // Publications during this pull were either consumed above or remain queued for the next
            // Euhedral call; pull never pushes.
            this.drain.set(0);
        }
        return delivered;
    }

    @Override
    public void request(long demand) {
        if (demand <= 0 || this.finished.get()) return;
        while (true) {
            long current = this.requested.get();
            long next = current + demand;
            if (next < 0) next = Long.MAX_VALUE;
            if (this.requested.compareAndSet(current, next)) break;
        }
        deliver();
    }

    /// Pushes ready frames against outstanding request demand while this thread owns the drain.
    /// Euhedral's receiver does not throw on push.
    private void deliver() {
        if (this.drain.getAndIncrement() != 0) return;
        int missed = 1;
        while (true) {
            pushReady();
            missed = this.drain.addAndGet(-missed);
            if (missed == 0) return;
        }
    }

    private void pushReady() {
        LatticeReceiver receiver = this.downstream.get();
        if (receiver == null) return;
        while (true) {
            long demand = this.requested.get();
            if (demand == 0L) return;
            AbstractFrame frame = this.ready.poll();
            if (frame == null) return;
            if (demand != Long.MAX_VALUE) this.requested.decrementAndGet();
            receiver.push(frame);
        }
    }

    @Override
    public void complete() {
        completeGracefully();
    }

    /// Closes admission; the source completes after every accepted graph has terminated.
    public void completeGracefully() {
        int old = this.accepted.getAndUpdate(value -> value | CLOSED);
        if ((old & Integer.MAX_VALUE) == 0) signalComplete();
    }

    private void signalComplete() {
        if (this.finished.compareAndSet(false, true)) {
            LatticeReceiver receiver = this.downstream.getAndSet(null);
            notifyComplete(receiver);
        }
    }

    private void notifyComplete(LatticeReceiver receiver) {
        try {
            if (receiver != null) receiver.onComplete();
        } catch (RuntimeException | Error failure) {
            this.termination.completeExceptionally(failure);
            throw failure;
        }
        this.termination.complete(null);
    }

    /// Waits until the downstream completion callback has returned.
    public void awaitTermination() {
        this.termination.join();
    }

    @Override
    public boolean isComplete() {
        return this.finished.get();
    }

    /// Whether this source currently retains its lattice receiver.
    public boolean isAttached() {
        return this.downstream.get() != null;
    }
}
