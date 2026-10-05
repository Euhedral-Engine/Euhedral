package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

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

    /// Bytes read from the artifact file so far.
    long bytesRead();

    /// Records opened so far.
    long recordOpens();

    @Override
    void close();
}
