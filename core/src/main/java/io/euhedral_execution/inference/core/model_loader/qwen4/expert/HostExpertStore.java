package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.io.IOException;

/// A host-side source of expert records: it makes the bytes of any `(bank, expert)` addressable in
/// pinned host memory, where a copy to the device reads them. A bank's ordinal is its index in
/// [#banks].
///
/// Records are opened on a lane. A lane is a pinned staging slot that one caller uses at a time:
/// its record is valid from [#open] until the lane's next `open`, and nothing is released or
/// pooled, so there is nothing to lock. A caller that opens on a lane only after the copy that
/// reads the lane's previous record retired never overwrites live bytes.
public interface HostExpertStore extends AutoCloseable {

    /// The banks this store serves; a bank's ordinal is its index.
    ExpertBank[] banks();

    /// The lanes this store stages records on: the most records that are open at once. A store that
    /// stages nothing (every record already addressable) has no limit.
    int lanes();

    /// Makes the record addressable in pinned host memory, reading it into the staging of `lane`
    /// when the store stages. The read is synchronous: callers run it in a frame of its own.
    ///
    /// @throws IOException when the record cannot be read
    /// @throws InterruptedException when the reading thread was interrupted; the store is
    ///     unaffected
    /// @throws IndexOutOfBoundsException when `bank`, `expert` or `lane` is out of range
    HostRecord open(int bank, int expert, int lane) throws IOException, InterruptedException;

    /// Bytes read from the artifact file so far.
    long bytesRead();

    /// Records opened so far.
    long recordOpens();

    @Override
    void close();
}
