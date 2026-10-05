package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

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

    /// Bytes this source has read from the artifact file.
    long bytesRead();

    @Override
    void close();
}
