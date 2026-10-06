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

    /// As [#open(int, int, int)], as the host tier's `directive` says: a hit copies the record out of
    /// the tier, a fill reads the artifact into the tier's slot and copies it out, anything else reads
    /// into the staging slot. A store without a tier ignores the directive.
    default HostRecord open(int bank, int expert, int lane, TierDirective directive)
            throws IOException, InterruptedException {
        return open(bank, expert, lane);
    }

    /// How many parts one record's artifact read can be split into, each read by a caller of its own
    /// ([#readPart]); 1 when it cannot be split.
    default int readParts() {
        return 1;
    }

    /// Reads part `part` of `parts` of the record's bytes from the artifact into where the
    /// `directive` puts them (the tier slot of a fill, otherwise the lane's staging slot). The parts
    /// are disjoint, so callers read them side by side; when all have returned,
    /// [#completeOpen] makes the record addressable. Only for a directive that reads the artifact
    /// and a store with `readParts() > 1`.
    default void readPart(int bank, int expert, int lane, TierDirective directive, int part, int parts)
            throws IOException, InterruptedException {
        throw new UnsupportedOperationException("this store reads whole records");
    }

    /// The record read by [#readPart] in all its parts, addressable in pinned host memory (a fill
    /// copies it out of its tier slot).
    default HostRecord completeOpen(int bank, int expert, int lane, TierDirective directive) {
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
