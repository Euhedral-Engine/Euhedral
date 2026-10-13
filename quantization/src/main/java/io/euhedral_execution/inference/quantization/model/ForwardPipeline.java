package io.euhedral_execution.inference.quantization.model;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.runtime.graph.FrameLake;
import io.euhedral_execution.inference.core.runtime.graph.FrameSeeds;
import io.euhedral_execution.inference.quantization.gpu.Device;
import io.euhedral_execution.inference.quantization.lattice.GpuCompletions;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

/// A layer's pass over every batch of sequences as lattice frames, each frame's finalization publishing the next:
///
/// - `Start` (main stream) copies the batch in and runs [LayerForward#head], records an event, and publishes one
///   `Sequence` frame per sequence;
/// - `Sequence` (its slot's own stream, behind that event) runs [LayerForward#sequence]: the GDN recurrences or
///   attention of different sequences overlap on the device; the last to finish publishes `Finish`;
/// - `Finish` (main stream, behind every slot's event) runs [LayerForward#tail], copies the batch back, and publishes
///   the next batch's `Start`, or hands `Done` to [GpuCompletions] behind the pass's last event.
///
/// No frame waits: frames only submit device work, ordered across streams by events; the one wait for the device is
/// the completion source's, which emits `Done` when the GPU is through.
public final class ForwardPipeline implements AutoCloseable {

    private final Device device;
    private final LayerForward forward;
    private final FrameLake lake;
    private final GpuCompletions completions;
    private final long[] slotStreams, slotEvents;
    private final long startEvent, finalEvent;
    private final FrameSeeds seeds = new FrameSeeds();

    public ForwardPipeline(Device device, LayerForward forward, FrameLake lake, GpuCompletions completions, int slots) {
        this.device = device;
        this.forward = forward;
        this.lake = lake;
        this.completions = completions;
        this.slotStreams = new long[slots];
        this.slotEvents = new long[slots];
        for (int s = 0; s < slots; s++) {
            slotStreams[s] = device.newStream();
            slotEvents[s] = device.cuda.createEvent();
        }
        this.startEvent = device.cuda.createEvent();
        this.finalEvent = device.cuda.createEvent();
        forward.prepareSlots(slots);
    }

    /// One pass of `layer` over `sequences` sequences of `length` tokens whose hidden states are pinned host memory
    /// at `states`, `batch` sequences at a time through the device buffer `batchStates`; with `write` the results
    /// go back. Completes on a lattice worker once the device has finished.
    public CompletableFuture<Void> run(
            LayerWeights layer,
            long states,
            long batchStates,
            int sequences,
            int length,
            int batch,
            LayerForward.Observer observer,
            boolean write) {
        if (batch > slotStreams.length) throw new IllegalArgumentException("more sequences per batch than slots");
        var pass = new Pass(layer, states, batchStates, sequences, length, batch, observer, write);
        lake.admit(); // released by Done (or a failure): the lake stays attached while the pass waits on the GPU
        lake.publish(pass.start());
        return pass.future;
    }

    private final class Pass {
        final LayerWeights layer;
        final long states, batchStates, sequenceBytes;
        final int sequences, length, batch;
        final LayerForward.Observer observer;
        final boolean write;
        final CompletableFuture<Void> future = new CompletableFuture<>();
        final AtomicInteger remaining = new AtomicInteger();

        Pass(
                LayerWeights layer,
                long states,
                long batchStates,
                int sequences,
                int length,
                int batch,
                LayerForward.Observer observer,
                boolean write) {
            this.layer = layer;
            this.states = states;
            this.batchStates = batchStates;
            this.sequences = sequences;
            this.length = length;
            this.batch = batch;
            this.observer = observer;
            this.write = write;
            this.sequenceBytes = 2L * forward.hidden() * length;
        }

        AbstractFrame start() {
            return new Start(0);
        }

        int count(int first) {
            return Math.min(batch, sequences - first);
        }

        void fail(Throwable failure) {
            if (future.completeExceptionally(failure)) lake.terminated();
        }

        /// A frame of this pass: runs `work`, then publishes `next()` unless the pass failed.
        abstract class Step extends AbstractFrame {
            Step() {
                super(FrameSeeds.ID_HASH);
                randomizeHash(seeds.next());
            }

            abstract void work();

            abstract void next();

            @Override
            public void execute() {
                if (future.isDone()) return;
                try {
                    work();
                } catch (Throwable failure) {
                    fail(failure);
                }
            }

            @Override
            public void doFinally() {
                if (!future.isDone()) next();
            }

            @Override
            public void doFinallyWithError(Throwable failure) {
                fail(failure);
            }
        }

        final class Start extends Step {
            final int first;

            Start(int first) {
                this.first = first;
            }

            @Override
            void work() {
                int count = count(first);
                device.select(device.stream);
                device.gpu.copyHostToDeviceAsync(batchStates, states + first * sequenceBytes, count * sequenceBytes);
                forward.head(layer, batchStates, count * length, observer);
                device.cuda.recordEvent(startEvent, device.stream);
            }

            @Override
            void next() {
                int count = count(first);
                remaining.set(count);
                for (int s = 0; s < count; s++) lake.publish(new Sequence(first, s));
            }
        }

        final class Sequence extends Step {
            final int first, slot;

            Sequence(int first, int slot) {
                this.first = first;
                this.slot = slot;
            }

            @Override
            void work() {
                long stream = slotStreams[slot];
                device.select(stream);
                device.cuda.waitEvent(stream, startEvent);
                forward.sequence(layer, slot, length);
                device.cuda.recordEvent(slotEvents[slot], stream);
            }

            @Override
            void next() {
                if (remaining.decrementAndGet() == 0) lake.publish(new Finish(first));
            }
        }

        final class Finish extends Step {
            final int first;

            Finish(int first) {
                this.first = first;
            }

            @Override
            void work() {
                int count = count(first);
                device.select(device.stream);
                for (int s = 0; s < count; s++) device.cuda.waitEvent(device.stream, slotEvents[s]);
                forward.tail(layer, batchStates, count * length, observer);
                if (write) {
                    device.gpu.copyDeviceToHostAsync(
                            states + first * sequenceBytes, batchStates, count * sequenceBytes);
                }
                if (first + count >= sequences) device.cuda.recordEvent(finalEvent, device.stream);
            }

            @Override
            void next() {
                int following = first + count(first);
                if (following < sequences) lake.publish(new Start(following));
                else completions.await(finalEvent, new Done());
            }
        }

        final class Done extends Step {
            @Override
            void work() {}

            @Override
            void next() {
                if (future.complete(null)) lake.terminated();
            }
        }
    }

    @Override
    public void close() {
        for (long event : slotEvents) device.cuda.destroyEvent(event);
        device.cuda.destroyEvent(startEvent);
        device.cuda.destroyEvent(finalEvent);
    }
}
