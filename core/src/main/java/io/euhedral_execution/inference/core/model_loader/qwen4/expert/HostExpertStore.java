package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.host.HostFrames;
import java.io.IOException;
import java.util.concurrent.TimeoutException;

/// A host-side source of expert records: it makes the bytes of any `(bank, expert)` addressable in pinned
/// host memory, where a copy to the device reads them. A bank's ordinal is its index in [#banks].
///
/// All methods may be called from many threads at once. Records must be closed before the store is: closing the
/// store releases its pinned memory once the last open record has been closed.
public interface HostExpertStore extends AutoCloseable {

    /// `timeoutNanos` value for "wait as long as it takes".
    long NO_TIMEOUT = -1L;

    /// The banks this store serves; a bank's ordinal is its index.
    ExpertBank[] banks();

    /// Opens a record, waiting as long as staging needs.
    ///
    /// @throws IOException when the record cannot be read
    /// @throws InterruptedException when interrupted while waiting for staging space
    default HostRecord open(int bank, int expert) throws IOException, InterruptedException {
        try {
            return open(bank, expert, NO_TIMEOUT);
        } catch (TimeoutException unreachable) {
            throw new IllegalStateException("an untimed open timed out", unreachable);
        }
    }

    /// Opens a record, waiting up to `timeoutNanos` (negative: without limit) for staging space.
    ///
    /// @throws TimeoutException when no staging space became free in time
    /// @throws IndexOutOfBoundsException when `bank` or `expert` is out of range
    HostRecord open(int bank, int expert, long timeoutNanos) throws IOException, InterruptedException, TimeoutException;

    /// Receives the outcome of [#openAsync]: the record (the receiver must close it) or the failure.
    @FunctionalInterface
    interface OpenListener {
        void opened(int tag, HostRecord record, Throwable failure);
    }

    /// Opens a record without blocking the caller: `listener` runs exactly once, on the calling thread when the
    /// record is at hand, otherwise later on a worker. A store that must wait for staging space parks a
    /// continuation instead of a thread, and a store that must read performs the (synchronous) read as its own host
    /// frame through `frames`, so that simultaneous opens read on different workers.
    ///
    /// The default adapts a store whose [#open] never blocks.
    default void openAsync(int bank, int expert, HostFrames frames, OpenListener listener, int tag) {
        HostRecord record;
        try {
            record = open(bank, expert);
        } catch (IOException | RuntimeException | Error failure) {
            listener.opened(tag, null, failure);
            return;
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            listener.opened(tag, null, interrupted);
            return;
        }
        listener.opened(tag, record, null);
    }

    /// Bytes read from the artifact file so far.
    long bytesRead();

    /// Records opened so far.
    long recordOpens();

    @Override
    void close();
}
