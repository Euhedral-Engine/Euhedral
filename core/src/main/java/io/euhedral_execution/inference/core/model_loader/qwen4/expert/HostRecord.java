package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.lang.foreign.MemorySegment;

/// One expert record addressable in pinned host memory, valid from [HostExpertStore#open] until [#close]: the
/// device's copy engines read `byteSize` bytes at `hostAddress` directly. Closing releases whatever staging the
/// store holds for the record, so it must follow the end of the copy that reads it. Closing is idempotent.
public interface HostRecord extends AutoCloseable {

    long hostAddress();

    long byteSize();

    /// The record's bytes as a segment.
    default MemorySegment segment() {
        return MemorySegment.ofAddress(hostAddress()).reinterpret(byteSize());
    }

    @Override
    void close();
}
