package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.scheduling.graph.AsyncReads;
import java.io.IOException;

/// A host-side source of expert records: it makes the bytes of any `(bank, expert)` addressable in
/// pinned host memory, where a copy to the device reads them. A bank's ordinal is its index in
/// [#banks].
///
/// Records are opened into a staging buffer: pinned memory that one load holds from [#acquireStaging]
/// until the device copy that reads it retired ([#releaseStaging]). The buffers are a fixed pool of
/// physical memory; taking one never waits (it fails when none is free).
public interface HostExpertStore extends AutoCloseable {

    /// The banks this store serves; a bank's ordinal is its index.
    ExpertBank[] banks();

    /// Takes a free staging buffer, or returns -1 when none is free: never waits. Its holder writes one
    /// record into it and gives it back with [#releaseStaging] once nothing reads it any more.
    int acquireStaging();

    /// Gives back a buffer [#acquireStaging] returned.
    void releaseStaging(int buffer);

    /// Staging buffers pinned.
    int stagingBuffers();

    /// Makes the record addressable in pinned host memory, reading it into staging buffer `buffer`. The read is
    /// synchronous: callers run it in a frame of its own.
    ///
    /// @throws IOException when the record cannot be read
    /// @throws InterruptedException when the reading thread was interrupted; the store is
    ///     unaffected
    /// @throws IndexOutOfBoundsException when `bank`, `expert` or `buffer` is out of range
    HostRecord open(int bank, int expert, int buffer) throws IOException, InterruptedException;

    /// As [#open(int, int, int)], as the host tier's `directive` says: a hit copies the record out of
    /// the tier, a fill reads the artifact into the tier's slot and copies it out, anything else reads
    /// into the staging slot. A store without a tier ignores the directive.
    default HostRecord open(int bank, int expert, int buffer, TierDirective directive)
            throws IOException, InterruptedException {
        return open(bank, expert, buffer);
    }

    /// How many parts one record's artifact read can be split into, each read by a caller of its own
    /// ([#readPart]); 1 when it cannot be split.
    default int readParts() {
        return 1;
    }

    /// Reads part `part` of `parts` of the record's bytes from the artifact into where the
    /// `directive` puts them (the tier slot of a fill, which also copies its range into the staging
    /// buffer; otherwise the staging buffer itself). The parts
    /// are disjoint, so callers read them side by side; when all have returned,
    /// [#completeOpen] makes the record addressable. Only for a directive that reads the artifact
    /// and a store with `readParts() > 1`.
    default void readPart(int bank, int expert, int buffer, TierDirective directive, int part, int parts)
            throws IOException, InterruptedException {
        throw new UnsupportedOperationException("this store reads whole records");
    }

    /// Whether a record's artifact read can be made in parts ([#readPart], [#readPartAsync]).
    default boolean rangedReads() {
        return false;
    }

    /// Submits the read of part `part` of `parts` and returns at once; `done` is emitted when its bytes
    /// arrived, and its frame then calls [#completePart]. Returns false, having submitted nothing, when the
    /// part is empty or cannot be read asynchronously: the caller then calls [#readPart].
    default boolean readPartAsync(
            int bank, int expert, int buffer, TierDirective directive, int part, int parts, AsyncReads.Read done) {
        return false;
    }

    /// Finishes part `part` of `parts` once its asynchronous read arrived (a fill's part copies its range into
    /// the staging buffer).
    default void completePart(
            int bank, int expert, int buffer, TierDirective directive, int part, int parts, long readNanos) {}

    /// The record read by [#readPart] in all its parts, addressable in pinned host memory (a fill's
    /// parts copied their ranges into staging as they read them).
    default HostRecord completeOpen(int bank, int expert, int buffer, TierDirective directive) {
        throw new UnsupportedOperationException("this store reads whole records");
    }

    /// The host tier's bookkeeping for the device cache shard `shard`, owned by that shard's source, or
    /// null when the store has no tier.
    default RamTierShard tier(int shard) {
        return null;
    }

    /// Bytes read from the artifact file so far.
    long bytesRead();

    /// Records opened so far.
    long recordOpens();

    @Override
    void close();
}
