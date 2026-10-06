package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

/// Moves expert records from pinned host memory into device memory, asynchronously.
public interface ExpertTransfer extends AutoCloseable {

    /// Receives the outcome of one transfer.
    @FunctionalInterface
    interface Completion {
        /// Signals that the transfer is over: `failure` is null when every byte of the record is in
        /// device memory, otherwise the reason it is not. May be called from any thread, including
        /// the one that called [ExpertTransfer#start] before it returns.
        void complete(Throwable failure);
    }

    /// Begins copying `record` to `deviceAddress`, behind `waitFor` (null for nothing) on the
    /// device. Calls `done` exactly once, when the bytes are in device memory or the copy failed.
    ///
    /// The transfer owns `record` from the call on and closes it, once the copy no longer reads it,
    /// including when this method throws. A method that throws signals no completion and leaves
    /// nothing running that writes to `deviceAddress`. It may block (interruptibly) for capacity.
    ///
    /// @throws InterruptedException when interrupted while waiting for capacity
    void start(HostRecord record, long deviceAddress, DeviceFence waitFor, Completion done) throws InterruptedException;

    /// As [#start], without blocking the caller: a transfer that must wait for capacity parks as a
    /// continuation. The outcome, including a failure to start, is reported only through `done`,
    /// exactly once; `record` is closed once the copy no longer reads it. May call `done` on the
    /// calling thread.
    ///
    /// The default adapts a transfer whose [#start] does not block.
    default void startAsync(HostRecord record, long deviceAddress, DeviceFence waitFor, Completion done) {
        try {
            start(record, deviceAddress, waitFor, done);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            done.complete(interrupted);
        } catch (RuntimeException | Error failure) {
            done.complete(failure);
        }
    }

    /// Whether this transfer supports the streaming form ([#stream]): a copy submitted on a lane's
    /// own copy stream whose completion is a device marker, so that nothing on the host waits for
    /// the bytes to arrive.
    default boolean streams() {
        return false;
    }

    /// Opens a device marker that [#stream] records behind a copy.
    default long openMarker() {
        throw new UnsupportedOperationException("this transfer does not stream");
    }

    default void closeMarker(long marker) {}

    /// Submits the copy of `record` to `deviceAddress` on the copy stream of `lane`, behind
    /// `waitFor` (null for nothing) on the device, records `marker` behind it, and arms `retired`
    /// on that stream: it is called (on a driver thread, where it may only enqueue) once the copy
    /// retired, and its ticket is then given to [#confirm]. Work that reads the destination submits
    /// a wait for `marker` on its own stream and need not wait on the host.
    ///
    /// A lane is used by one caller at a time, from `stream` until its `confirm`, which is what
    /// makes a lock unnecessary: the lane's stream is the order of its copies. The caller owns
    /// `record` and closes it after `confirm`. When this method throws nothing is armed and the
    /// copy may or may not have been queued: the caller must [#recover] the lane before releasing
    /// `record`.
    default void stream(
            int lane,
            HostRecord record,
            long deviceAddress,
            DeviceFence waitFor,
            long marker,
            io.euhedral_execution.inference.core.gpu.GpuStream.RetirementListener retired) {
        throw new UnsupportedOperationException("this transfer does not stream");
    }

    /// The device failure of the copy whose boundary retired as `ticket`, or null.
    default Throwable confirm(int lane, long ticket) {
        throw new UnsupportedOperationException("this transfer does not stream");
    }

    /// Proves the lane's copy stream idle after a failed [#stream].
    default void recover(int lane, Throwable failure) {}

    /// Waits for the transfers in flight, then releases what the transfer holds.
    @Override
    void close();
}
