package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// Copies expert records into device memory on one dedicated copy stream, so transfers overlap the compute
/// streams and are ordered among themselves.
///
/// A transfer is one asynchronous host-to-device copy of the pinned record, followed by a retirement boundary on
/// the stream. The boundary's listener may run on a CUDA driver thread, where it may only enqueue: it hands the
/// transfer to a completion thread, which confirms the boundary, closes the host record (a staging slot may be
/// reused only after the copy that reads it retired) and signals the completion. No driver thread ever calls
/// CUDA, blocks, or runs cache code.
///
/// Submissions are serialized by a lock, since one stream's order is the order of submission. At most
/// `maxInFlight` transfers are outstanding; [#start] waits for a permit when they are, which bounds the
/// completion queue without dropping anything.
///
/// A [DeviceFence] is honoured on the device: before the copy is submitted, the copy stream is ordered behind
/// the fence ([DeviceFence#awaitOn]), so a refill of a slot never overtakes the kernels that were reading it.
/// The host never waits for the fence.
public final class GpuExpertTransfer implements ExpertTransfer {
    private static final Logger LOG = LoggerFactory.getLogger(GpuExpertTransfer.class);
    public static final int DEFAULT_MAX_IN_FLIGHT = 16;
    private static final long CLOSE_TIMEOUT_SECONDS = 30;

    private final ExecutionGpu gpu;
    private final GpuStream copyStream;
    private final int maxInFlight;
    private final Semaphore permits;
    private final BlockingQueue<Pending> retired;
    private final Pending stop = new Pending(null, null);
    private final Thread completionThread;
    private final Object submission = new Object();
    private final AtomicBoolean closed = new AtomicBoolean();
    private boolean finished;

    /// One copy from submission to completion.
    private static final class Pending {
        private final HostRecord record;
        private final Completion done;
        private volatile long ticket;

        private Pending(HostRecord record, Completion done) {
            this.record = record;
            this.done = done;
        }
    }

    public GpuExpertTransfer(ExecutionGpu gpu) {
        this(gpu, DEFAULT_MAX_IN_FLIGHT);
    }

    /// A transfer with its own copy stream from `gpu`, allowing `maxInFlight` copies outstanding at once.
    public GpuExpertTransfer(ExecutionGpu gpu, int maxInFlight) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (maxInFlight < 1) throw new IllegalArgumentException("maxInFlight must be positive");
        this.maxInFlight = maxInFlight;
        this.permits = new Semaphore(maxInFlight);
        // A listener only enqueues, and never more than the permits allow, plus the stop marker.
        this.retired = new ArrayBlockingQueue<>(maxInFlight + 1);
        this.copyStream = gpu.openStream();
        this.completionThread = new Thread(this::complete, "expert-transfer-completion");
        this.completionThread.setDaemon(true);
        this.completionThread.start();
    }

    @Override
    public void start(HostRecord record, long deviceAddress, DeviceFence waitFor, Completion done)
            throws InterruptedException {
        Objects.requireNonNull(record, "record");
        Objects.requireNonNull(done, "done");
        boolean permitted = false;
        boolean touchedStream = false;
        boolean armed = false;
        try {
            if (this.closed.get()) throw new IllegalStateException("the expert transfer is closed");
            this.permits.acquire();
            permitted = true;
            Pending pending = new Pending(record, done);
            synchronized (this.submission) {
                if (this.closed.get()) throw new IllegalStateException("the expert transfer is closed");
                touchedStream = true;
                if (waitFor != null) waitFor.awaitOn(this.copyStream);
                this.copyStream.submit(
                        () -> this.gpu.copyHostWeightsToDevice(deviceAddress, record.hostAddress(), record.byteSize()),
                        false);
                this.copyStream.notifyRetired((ticket, driverThread) -> {
                    pending.ticket = ticket;
                    // Capacity covers every permit, so this never fails and never blocks.
                    if (!this.retired.offer(pending)) LOG.error("the expert completion queue overflowed");
                });
                armed = true;
            }
        } catch (Throwable failure) {
            if (!armed) {
                // Without a boundary the copy may or may not have been queued: prove the stream stopped reading
                // the record before releasing it.
                if (touchedStream) this.copyStream.recover(failure);
                closeRecord(record);
                if (permitted) this.permits.release();
            }
            throw failure;
        }
    }

    /// Runs on the completion thread: the only caller of `confirmRetired`.
    private void complete() {
        while (true) {
            Pending pending;
            try {
                pending = this.retired.take();
            } catch (InterruptedException interrupted) {
                // Only close() ends this thread, with the stop marker.
                continue;
            }
            if (pending == this.stop) return;
            Throwable failure = null;
            try {
                failure = this.copyStream.confirmRetired(pending.ticket);
            } catch (Throwable confirmFailure) {
                failure = confirmFailure;
            }
            closeRecord(pending.record);
            this.permits.release();
            try {
                pending.done.complete(failure);
            } catch (Throwable doneFailure) {
                LOG.error("an expert completion callback failed", doneFailure);
            }
        }
    }

    /// Closes the record unless the GPU could not prove that its copy stopped reading it: a poisoned GPU retains
    /// everything its streams may still touch.
    private void closeRecord(HostRecord record) {
        if (!this.gpu.completionProven()) return;
        try {
            record.close();
        } catch (RuntimeException failure) {
            LOG.warn("closing an expert record failed", failure);
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
            // Every permit back means every transfer has been signalled.
            drained = this.permits.tryAcquire(this.maxInFlight, CLOSE_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
        if (!drained)
            throw new IllegalStateException("expert transfers are still in flight; the copy stream is kept open");
        this.finished = true;
        // A start that was already waiting for a permit wakes, finds the transfer closed and fails.
        this.permits.release(this.maxInFlight);
        this.retired.offer(this.stop);
        try {
            this.completionThread.join();
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
        this.copyStream.synchronize();
        this.copyStream.close();
    }
}
