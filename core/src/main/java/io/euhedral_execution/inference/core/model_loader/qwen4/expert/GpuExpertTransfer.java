package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.host.HostFrames;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// Copies expert records into device memory on one dedicated copy stream, so transfers overlap the compute
/// streams and are ordered among themselves.
///
/// A transfer is one asynchronous host-to-device copy of the pinned record, followed by a retirement boundary
/// on the stream. The boundary's listener runs on a CUDA driver thread, where it may only enqueue: it
/// publishes the transfer's completion as a host frame ([HostFrames#runFromCallback]), and the worker that
/// runs it confirms the boundary, closes the host record (a staging slot may be reused only after the copy
/// that reads it retired), signals the completion. The transfer owns no thread: device retirement becomes
/// runnable work, and no driver thread ever calls CUDA, blocks, or runs cache code.
///
/// Submissions are serialized by a lock, since one stream's order is the order of submission. The number of
/// copies in flight is not limited here: every copy holds a pinned staging record until its completion frame
/// closes it, so the staging pool the requester draws from is the bound, and a requester that keeps its
/// outstanding loads within it never waits. [#startAsync] never blocks and never queues.
///
/// A [DeviceFence] is honoured on the device: before the copy is submitted, the copy stream is ordered behind
/// the fence ([DeviceFence#awaitOn]), so a refill of a slot never overtakes the kernels that were reading it.
/// The host never waits for the fence.
public final class GpuExpertTransfer implements ExpertTransfer {
    private static final Logger LOG = LoggerFactory.getLogger(GpuExpertTransfer.class);
    private static final long CLOSE_TIMEOUT_SECONDS = 30;

    private final ExecutionGpu gpu;
    private final GpuStream copyStream;
    private final HostFrames frames;
    private final AtomicInteger inFlight = new AtomicInteger();
    private final AtomicInteger inFlightHighWater = new AtomicInteger();
    private final Object submission = new Object();
    private final Object idle = new Object();
    private final AtomicBoolean closed = new AtomicBoolean();
    private final LongAdder submitted = new LongAdder();
    private final LongAdder busyNanos = new LongAdder();
    private boolean finished;

    /// One copy from request to completion. As the host task published by the boundary's listener it
    /// completes itself.
    private final class Transfer implements HostFrames.Task {
        private final HostRecord record;
        private final long deviceAddress;
        private final DeviceFence waitFor;
        private final Completion done;
        private volatile long ticket;
        private long submittedNanos;

        Transfer(HostRecord record, long deviceAddress, DeviceFence waitFor, Completion done) {
            this.record = record;
            this.deviceAddress = deviceAddress;
            this.waitFor = waitFor;
            this.done = done;
        }

        /// Runs on a worker once the copy's boundary retired.
        @Override
        public void run() {
            Throwable failure = null;
            try {
                failure = GpuExpertTransfer.this.copyStream.confirmRetired(this.ticket);
            } catch (Throwable confirmFailure) {
                failure = confirmFailure;
            }
            GpuExpertTransfer.this.busyNanos.add(System.nanoTime() - this.submittedNanos);
            closeRecord(this.record);
            GpuExpertTransfer.this.finished();
            signal(this.done, failure);
        }

        /// The lattice rejected or lost the completion: the copy's outcome is unknown, so its record stays
        /// held.
        @Override
        public void failed(Throwable cause) {
            GpuExpertTransfer.this.finished();
            signal(this.done, cause);
        }
    }

    /// A transfer with its own copy stream from `gpu`, whose completions run as host work on `frames`.
    public GpuExpertTransfer(ExecutionGpu gpu, HostFrames frames) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.frames = Objects.requireNonNull(frames, "frames");
        this.copyStream = gpu.openStream();
    }

    @Override
    public void startAsync(HostRecord record, long deviceAddress, DeviceFence waitFor, Completion done) {
        Objects.requireNonNull(record, "record");
        Objects.requireNonNull(done, "done");
        if (this.closed.get()) {
            closeRecord(record);
            signal(done, new IllegalStateException("the expert transfer is closed"));
            return;
        }
        submitOrReport(new Transfer(record, deviceAddress, waitFor, done), false);
    }

    /// Submits the copy and returns; the same as [#startAsync] for callers that expect a thrown failure (the
    /// cache's blocking acquire, tests).
    @Override
    public void start(HostRecord record, long deviceAddress, DeviceFence waitFor, Completion done) {
        Objects.requireNonNull(record, "record");
        Objects.requireNonNull(done, "done");
        try {
            if (this.closed.get()) throw new IllegalStateException("the expert transfer is closed");
        } catch (Throwable failure) {
            closeRecord(record);
            throw failure;
        }
        submitOrReport(new Transfer(record, deviceAddress, waitFor, done), true);
    }

    /// Submits the copy of a transfer. A failure before the boundary is armed closes the record; with
    /// `throwing` it is rethrown to a blocked caller, otherwise reported through the completion. Returns
    /// whether the copy is under way.
    private boolean submitOrReport(Transfer transfer, boolean throwing) {
        boolean touchedStream = false;
        boolean armed = false;
        int now = this.inFlight.incrementAndGet();
        this.inFlightHighWater.accumulateAndGet(now, Math::max);
        try {
            synchronized (this.submission) {
                if (this.closed.get()) throw new IllegalStateException("the expert transfer is closed");
                touchedStream = true;
                if (transfer.waitFor != null) transfer.waitFor.awaitOn(this.copyStream);
                this.copyStream.submit(
                        () -> this.gpu.copyHostWeightsToDevice(
                                transfer.deviceAddress, transfer.record.hostAddress(), transfer.record.byteSize()),
                        false);
                transfer.submittedNanos = System.nanoTime();
                this.copyStream.notifyRetired((ticket, driverThread) -> {
                    transfer.ticket = ticket;
                    try {
                        this.frames.runFromCallback(transfer);
                    } catch (RuntimeException | Error lost) {
                        LOG.error("an expert transfer's completion could not be published", lost);
                    }
                });
                armed = true;
            }
            this.submitted.increment();
            return true;
        } catch (Throwable failure) {
            if (!armed) {
                // Without a boundary the copy may or may not have been queued: prove the stream stopped reading
                // the record before releasing it.
                if (touchedStream) this.copyStream.recover(failure);
                closeRecord(transfer.record);
                finished();
                if (throwing) throw failure;
                signal(transfer.done, failure);
            }
            return false;
        }
    }

    private static void signal(Completion done, Throwable failure) {
        try {
            done.complete(failure);
        } catch (Throwable doneFailure) {
            LOG.error("an expert completion callback failed", doneFailure);
        }
    }

    /// Closes the record unless the GPU could not prove that its copy stopped reading it: a poisoned GPU
    /// retains everything its streams may still touch.
    private void closeRecord(HostRecord record) {
        if (!this.gpu.completionProven()) return;
        try {
            record.close();
        } catch (RuntimeException failure) {
            LOG.warn("closing an expert record failed", failure);
        }
    }

    /// Copies submitted so far.
    public long submittedCopies() {
        return this.submitted.sum();
    }

    /// Time the copy stream held copies, from submission to retirement, summed over copies (copies queued
    /// behind one another overlap in this sum).
    public long copyNanos() {
        return this.busyNanos.sum();
    }

    /// The most copies that were in flight at once.
    public int inFlightHighWater() {
        return this.inFlightHighWater.get();
    }

    /// A transfer ended: its record is closed and its completion is signalled.
    private void finished() {
        if (this.inFlight.decrementAndGet() == 0) {
            synchronized (this.idle) {
                this.idle.notifyAll();
            }
        }
    }

    /// Rejects new transfers, waits for those in flight to complete, and closes the copy stream.
    ///
    /// @throws IllegalStateException when transfers did not complete within 30 seconds; the stream is kept open
    @Override
    public synchronized void close() {
        if (this.finished) return;
        this.closed.set(true);
        boolean drained = false;
        try {
            // No transfer in flight means every transfer has been signalled.
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(CLOSE_TIMEOUT_SECONDS);
            synchronized (this.idle) {
                while (this.inFlight.get() != 0) {
                    long remaining = deadline - System.nanoTime();
                    if (remaining <= 0) break;
                    this.idle.wait(remaining / 1_000_000L, (int) (remaining % 1_000_000L));
                }
                drained = this.inFlight.get() == 0;
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
        if (!drained)
            throw new IllegalStateException("expert transfers are still in flight; the copy stream is kept open");
        this.finished = true;
        this.copyStream.synchronize();
        this.copyStream.close();
    }
}
