package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.Objects;
import java.util.concurrent.atomic.LongAdder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// Copies expert records into device memory, each lane on a copy stream of its own (see
/// [ExpertTransfer]). Nothing is shared between lanes but counters: no lock, no queue, no
/// completion thread. A copy's retirement boundary is a driver callback that enqueues; whoever owns
/// the lane takes it from there.
///
/// A [DeviceFence] is honoured on the device: before the copy is submitted, the lane's stream is
/// ordered behind the fence ([DeviceFence#awaitOn]), so a refill of a slot never overtakes the
/// kernels that were reading it. The host never waits for the fence.
public final class GpuExpertTransfer implements ExpertTransfer {
    private static final Logger LOG = LoggerFactory.getLogger(GpuExpertTransfer.class);

    private final ExecutionGpu gpu;
    private final GpuStream marks;
    private final GpuStream[] laneStreams;
    private final LongAdder submitted = new LongAdder();
    private boolean closed;

    /// A transfer with `lanes` copy streams from `gpu`.
    public GpuExpertTransfer(ExecutionGpu gpu, int lanes) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (lanes < 1) throw new IllegalArgumentException("lanes must be positive");
        // Markers are device-wide events: any stream opens them.
        this.marks = gpu.openStream();
        this.laneStreams = new GpuStream[lanes];
        try {
            for (int lane = 0; lane < lanes; lane++) this.laneStreams[lane] = gpu.openStream();
        } catch (RuntimeException | Error failure) {
            closeStreams();
            throw failure;
        }
    }

    @Override
    public int lanes() {
        return this.laneStreams.length;
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
            int lane,
            HostRecord record,
            long deviceAddress,
            DeviceFence waitFor,
            long marker,
            GpuStream.RetirementListener retired) {
        GpuStream stream = this.laneStreams[lane];
        if (waitFor != null) waitFor.awaitOn(stream);
        stream.submit(
                () -> this.gpu.copyHostWeightsToDevice(deviceAddress, record.hostAddress(), record.byteSize()), false);
        stream.mark(marker);
        stream.notifyRetired(retired);
        this.submitted.increment();
    }

    @Override
    public Throwable confirm(int lane, long ticket) {
        return this.laneStreams[lane].confirmRetired(ticket);
    }

    @Override
    public void recover(int lane, Throwable failure) {
        this.laneStreams[lane].recover(failure);
    }

    /// Copies submitted so far.
    public long submittedCopies() {
        return this.submitted.sum();
    }

    private void closeStreams() {
        for (int lane = 0; lane < this.laneStreams.length; lane++) {
            GpuStream stream = this.laneStreams[lane];
            if (stream == null) continue;
            this.laneStreams[lane] = null;
            try {
                stream.close();
            } catch (RuntimeException failure) {
                LOG.warn("closing an expert copy lane failed", failure);
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
        for (GpuStream lane : this.laneStreams) if (lane != null) lane.synchronize();
        closeStreams();
    }
}
