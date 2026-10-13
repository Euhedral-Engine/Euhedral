package io.euhedral_execution.inference.quantization.lattice;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.ingest.AbstractIngestSink;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import java.util.ArrayDeque;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.Consumer;
import java.util.function.Function;

/// A lattice source of continuations that wait for the GPU: [#await] registers a frame behind a recorded CUDA event,
/// and the lattice's polls (one thread at a time) emit the frame once the event has completed. Nothing waits: a poll
/// queries each pending event once (cuEventQuery) and returns. The pattern of the engine's AsyncReads, with CUDA
/// events in place of io_uring completions.
public final class GpuCompletions extends AbstractIngestSink {

    private record Pending(long event, AbstractFrame frame) {}

    private final Cuda cuda;
    private final ConcurrentLinkedQueue<Pending> incoming = new ConcurrentLinkedQueue<>();
    /// Only the polling thread touches it.
    private final ArrayDeque<Pending> waiting = new ArrayDeque<>();
    private final Delegate delegate = new Delegate();

    public GpuCompletions(Cuda cuda) {
        this.cuda = cuda;
    }

    /// Emits `frame` into the lattice once the work recorded before `event` has completed.
    public void await(long event, AbstractFrame frame) {
        incoming.add(new Pending(event, frame));
    }

    private long emit(Consumer<AbstractFrame> consumer, long demand) {
        for (Pending pending; (pending = incoming.poll()) != null; ) waiting.add(pending);
        if (waiting.isEmpty()) return 0;
        cuda.bindThread();
        long emitted = 0;
        for (var it = waiting.iterator(); it.hasNext() && emitted < demand; ) {
            Pending pending = it.next();
            if (!cuda.eventDone(pending.event())) continue;
            it.remove();
            consumer.accept(pending.frame());
            emitted++;
        }
        return emitted;
    }

    @Override
    public LatticeSource getDelegate() {
        return delegate;
    }

    @Override
    public void complete() {
        delegate.complete();
    }

    @Override
    public boolean isComplete() {
        return delegate.isComplete();
    }

    private final class Delegate extends AbstractIngestSink.Delegate {
        /// The terminal's push, made once: a request comes with every poll.
        private LatticeReceiver pushing;
        private Consumer<AbstractFrame> push;

        @Override
        public long hookOnPull(
                Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stopCondition, long demand) {
            return emit(consumer, demand);
        }

        @Override
        public void hookOnRequest(LatticeReceiver terminal, long demand) {
            if (terminal != pushing) {
                push = terminal::push;
                pushing = terminal;
            }
            long emitted = emit(push, demand);
            if (emitted > 0) addAndGetDemand(-emitted);
        }
    }
}
