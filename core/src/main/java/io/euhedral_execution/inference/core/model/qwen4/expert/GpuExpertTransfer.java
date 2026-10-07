package io.euhedral_execution.inference.core.model.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.Objects;
import java.util.concurrent.atomic.LongAdder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// Copies expert records into device memory on a fixed set of copy streams (see [ExpertTransfer]):
/// no lock, no queue, no completion thread. A copy's retirement boundary is a driver callback that
/// enqueues; the load that submitted it takes it from there.
///
/// A [DeviceFence] is honoured on the device: before the copy is submitted, the copy stream is
/// ordered behind the fence ([DeviceFence#awaitOn]), so a refill of a slot never overtakes the
/// kernels that were reading it. The host never waits for the fence.
public final class GpuExpertTransfer implements ExpertTransfer {
    private static final Logger LOG = LoggerFactory.getLogger(GpuExpertTransfer.class);

    private final ExecutionGpu gpu;
    private final GpuStream marks;
    private final GpuStream[] copyStreams;
    private final LongAdder submitted = new LongAdder();
    private final java.util.concurrent.atomic.AtomicInteger turn = new java.util.concurrent.atomic.AtomicInteger();
    private boolean closed;

    /// A transfer with `streams` copy streams from `gpu`.
    public GpuExpertTransfer(ExecutionGpu gpu, int streams) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (streams < 1) throw new IllegalArgumentException("streams must be positive");
        // Markers are device-wide events: any stream opens them.
        this.marks = gpu.openStream();
        this.copyStreams = new GpuStream[streams];
        try {
            for (int stream = 0; stream < streams; stream++) this.copyStreams[stream] = gpu.openStream();
        } catch (RuntimeException | Error failure) {
            closeStreams();
            throw failure;
        }
    }

    @Override
    public int nextStream() {
        return Math.floorMod(this.turn.getAndIncrement(), this.copyStreams.length);
    }

    @Override
    public int streams() {
        return this.copyStreams.length;
    }

    @Override
    public long openMarker() {
        return this.marks.openMarker();
    }

    @Override
    public void closeMarker(long marker) {
        this.marks.closeMarker(marker);
    }

    @Override
    public void stream(
            int copyStream,
            HostRecord record,
            long deviceAddress,
            DeviceFence waitFor,
            long marker,
            GpuStream.RetirementListener retired) {
        GpuStream stream = this.copyStreams[copyStream];
        if (waitFor != null) waitFor.awaitOn(stream);
        stream.submit(
                () -> this.gpu.copyHostWeightsToDevice(deviceAddress, record.hostAddress(), record.byteSize()), false);
        stream.mark(marker);
        stream.notifyRetired(retired);
        this.submitted.increment();
    }

    @Override
    public Throwable confirm(int copyStream, long ticket) {
        return this.copyStreams[copyStream].confirmRetired(ticket);
    }

    @Override
    public void recover(int copyStream, Throwable failure) {
        this.copyStreams[copyStream].recover(failure);
    }

    /// Copies submitted so far.
    public long submittedCopies() {
        return this.submitted.sum();
    }

    private void closeStreams() {
        for (int index = 0; index < this.copyStreams.length; index++) {
            GpuStream stream = this.copyStreams[index];
            if (stream == null) continue;
            this.copyStreams[index] = null;
            try {
                stream.close();
            } catch (RuntimeException failure) {
                LOG.warn("closing an expert copy stream failed", failure);
            }
        }
        try {
            if (this.marks != null) this.marks.close();
        } catch (RuntimeException failure) {
            LOG.warn("closing the expert marker stream failed", failure);
        }
    }

    /// Waits on the device for the copies in flight, then closes the streams. The owner has seen
    /// every copy retire.
    @Override
    public void close() {
        if (this.closed) return;
        this.closed = true;
        for (GpuStream stream : this.copyStreams) if (stream != null) stream.synchronize();
        closeStreams();
    }
}
