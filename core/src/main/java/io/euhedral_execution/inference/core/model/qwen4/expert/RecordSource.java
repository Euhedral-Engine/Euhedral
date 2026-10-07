package io.euhedral_execution.inference.core.model.qwen4.expert;

import io.euhedral_execution.inference.core.runtime.graph.AsyncReads;
import java.io.IOException;
import java.lang.foreign.MemorySegment;

/// Where a [FileExpertStore] gets the bytes of a record. The store owns the pinned staging slots; a source only
/// fills them. A tier between the artifact and the staging slots (a host-side record cache) is a source that
/// answers from its own memory before asking the file.
public interface RecordSource extends AutoCloseable {

    /// Fills `destination`, which is exactly `bank.recordBytes(expert)` bytes, with the record. May be called
    /// from many threads at once, and must stay usable when a read is interrupted.
    ///
    /// @throws InterruptedException when the reading thread was interrupted; the source is unaffected
    void read(ExpertBank bank, int expert, MemorySegment destination) throws IOException, InterruptedException;

    /// Whether [#readRange] reads part of a record, so that one record's read can be split across
    /// callers.
    default boolean ranged() {
        return false;
    }

    /// Fills `destination` with the `destination.byteSize()` bytes of the record that start `from`
    /// bytes into it. Only for a source that is [#ranged].
    ///
    /// @throws InterruptedException when the reading thread was interrupted; the source is unaffected
    default void readRange(ExpertBank bank, int expert, long from, MemorySegment destination)
            throws IOException, InterruptedException {
        throw new UnsupportedOperationException("this source reads whole records");
    }

    /// Submits the read of the `destination.byteSize()` bytes of the record that start `from` bytes into it
    /// and returns at once; `done` is emitted when they arrived or the read failed. Returns false, having
    /// submitted nothing, when this source cannot read asynchronously: the caller reads with [#readRange].
    default boolean readRangeAsync(
            ExpertBank bank, int expert, long from, MemorySegment destination, AsyncReads.Read done) {
        return false;
    }

    /// Fills `destination` with records `first` to `first + count - 1` of `bank`, one after another. For a loading
    /// thread that may block (the startup fill); a source whose records lie end to end in its file reads them in
    /// one go.
    default void readRun(ExpertBank bank, int first, int count, MemorySegment destination)
            throws IOException, InterruptedException {
        long at = 0;
        for (int expert = first; expert < first + count; expert++) {
            long size = bank.recordBytes(expert);
            read(bank, expert, destination.asSlice(at, size));
            at += size;
        }
    }

    /// Bytes this source has read from the artifact file.
    long bytesRead();

    @Override
    void close();
}
