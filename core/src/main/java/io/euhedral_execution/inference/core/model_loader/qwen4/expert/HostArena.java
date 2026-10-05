package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuMemory;

/// One pinned host allocation with a release that waits for its users: closing it frees the memory at once
/// when nothing is borrowed, otherwise when the last borrower returns. The memory is freed exactly once, and
/// never while a record that points into it is still open.
final class HostArena {
    private final GpuMemory memory;
    private final long address;
    private final long byteSize;
    private int borrowed;
    private boolean closed;
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

    /// Registers one more user of the memory.
    synchronized void borrow() {
        if (this.closed) throw new IllegalStateException("the host arena is closed");
        this.borrowed++;
    }

    synchronized void giveBack() {
        if (--this.borrowed == 0 && this.closed) free();
    }

    synchronized boolean isClosed() {
        return this.closed;
    }

    /// Frees the memory now, or when the last borrower returns.
    synchronized void close() {
        this.closed = true;
        if (this.borrowed == 0) free();
    }

    private void free() {
        if (this.freed) return;
        this.freed = true;
        this.memory.freeHostWeights(this.address);
    }
}
