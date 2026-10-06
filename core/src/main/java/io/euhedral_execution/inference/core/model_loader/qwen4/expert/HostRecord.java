package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.lang.foreign.MemorySegment;

/// One expert record addressable in pinned host memory, valid from [HostExpertStore#open] until the
/// lane's next open: the device's copy engines read `byteSize` bytes at `hostAddress` directly.
public interface HostRecord {

    long hostAddress();

    long byteSize();

    /// The record's bytes as a segment.
    default MemorySegment segment() {
        return MemorySegment.ofAddress(hostAddress()).reinterpret(byteSize());
    }
}
