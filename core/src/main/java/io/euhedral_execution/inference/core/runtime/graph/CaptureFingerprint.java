package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.inference.core.gpu.GpuMemory;

/// Mixes the device allocations and sizes a quantum's submissions depend on into one 64-bit value, part of
/// its capture key: a quantum whose state was reallocated, or whose session's state took over a closed
/// session's addresses, keys differently from the one that was captured.
public final class CaptureFingerprint {
    private long value = 0x9e3779b97f4a7c15L;

    public CaptureFingerprint add(long word) {
        long x = this.value ^ (word + 0x9e3779b97f4a7c15L + (this.value << 6) + (this.value >>> 2));
        x ^= x >>> 30;
        x *= 0xbf58476d1ce4e5b9L;
        x ^= x >>> 27;
        x *= 0x94d049bb133111ebL;
        this.value = x ^ (x >>> 31);
        return this;
    }

    /// Mixes an address with the identity of the allocation behind it: a capture references allocations,
    /// not addresses, and a freed address may come back as another allocation.
    public CaptureFingerprint add(GpuMemory memory, long address) {
        return add(address).add(address == 0 ? 0 : memory.allocationId(address));
    }

    public long value() {
        return this.value;
    }
}
