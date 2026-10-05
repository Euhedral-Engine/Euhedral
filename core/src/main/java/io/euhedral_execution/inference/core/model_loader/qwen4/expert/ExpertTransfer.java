package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

/// Moves expert records from pinned host memory into device memory, asynchronously.
public interface ExpertTransfer extends AutoCloseable {

    /// Receives the outcome of one transfer.
    @FunctionalInterface
    interface Completion {
        /// Signals that the transfer is over: `failure` is null when every byte of the record is in device
        /// memory, otherwise the reason it is not. May be called from any thread, including the one that
        /// called [ExpertTransfer#start] before it returns.
        void complete(Throwable failure);
    }

    /// Begins copying `record` to `deviceAddress`, behind `waitFor` (null for nothing) on the device. Calls
    /// `done` exactly once, when the bytes are in device memory or the copy failed.
    ///
    /// The transfer owns `record` from the call on and closes it, once the copy no longer reads it, including
    /// when this method throws. A method that throws signals no completion and leaves nothing running that
    /// writes to `deviceAddress`. It may block (interruptibly) for capacity.
    ///
    /// @throws InterruptedException when interrupted while waiting for capacity
    void start(HostRecord record, long deviceAddress, DeviceFence waitFor, Completion done) throws InterruptedException;

    /// Waits for the transfers in flight, then releases what the transfer holds.
    @Override
    void close();
}
