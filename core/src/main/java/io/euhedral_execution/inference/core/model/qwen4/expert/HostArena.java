package io.euhedral_execution.inference.core.model.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuMemory;

/// One pinned host allocation. Its owner frees it once, when no copy reads it any more.
final class HostArena {
    private final GpuMemory memory;
    private final long address;
    private final long byteSize;
    private boolean freed;

    HostArena(GpuMemory memory, long byteSize) {
        this.memory = memory;
        this.byteSize = byteSize;
        this.address = memory.allocateHostWeights(byteSize);
    }

    long address() {
        return this.address;
    }

    long byteSize() {
        return this.byteSize;
    }

    /// Frees the memory (once).
    void close() {
        if (this.freed) return;
        this.freed = true;
        this.memory.freeHostWeights(this.address);
    }
}
