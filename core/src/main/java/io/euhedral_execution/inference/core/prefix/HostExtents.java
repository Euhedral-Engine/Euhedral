package io.euhedral_execution.inference.core.prefix;

import java.util.Map;
import java.util.TreeMap;

/// First-fit allocator of byte extents over a fixed range, in units of [#ALIGNMENT]. It owns no memory; it
/// hands out offsets into whatever arena the caller maps over the range. Thread-safe.
public final class HostExtents {
    public static final long ALIGNMENT = 4096;

    private final long totalBytes;
    // Free extents by offset, never adjacent: neighbours are merged when an extent is freed.
    private final TreeMap<Long, Long> free = new TreeMap<>();
    private long freeBytes;

    public HostExtents(long totalBytes) {
        if (totalBytes < 0) throw new IllegalArgumentException("totalBytes must not be negative");
        this.totalBytes = totalBytes - totalBytes % ALIGNMENT;
        this.freeBytes = this.totalBytes;
        if (this.totalBytes > 0) this.free.put(0L, this.totalBytes);
    }

    /// Reserves `bytes`, rounded up to the alignment, and returns its offset, or -1 when no free extent is
    /// large enough.
    public synchronized long allocate(long bytes) {
        if (bytes <= 0) throw new IllegalArgumentException("bytes must be positive");
        long size = roundUp(bytes);
        for (Map.Entry<Long, Long> entry : this.free.entrySet()) {
            if (entry.getValue() < size) continue;
            long offset = entry.getKey();
            long remaining = entry.getValue() - size;
            this.free.remove(offset);
            if (remaining > 0) this.free.put(offset + size, remaining);
            this.freeBytes -= size;
            return offset;
        }
        return -1;
    }

    /// Returns an extent that [#allocate] handed out, with the size it was allocated with.
    public synchronized void free(long offset, long bytes) {
        long size = roundUp(bytes);
        if (offset < 0 || offset % ALIGNMENT != 0 || size <= 0 || offset + size > this.totalBytes)
            throw new IllegalArgumentException("extent lies outside the range");
        Map.Entry<Long, Long> before = this.free.floorEntry(offset);
        if (before != null && before.getKey() + before.getValue() > offset)
            throw new IllegalArgumentException("extent is already free");
        Map.Entry<Long, Long> after = this.free.ceilingEntry(offset);
        if (after != null && after.getKey() < offset + size)
            throw new IllegalArgumentException("extent is already free");
        long start = offset;
        long length = size;
        if (before != null && before.getKey() + before.getValue() == offset) {
            start = before.getKey();
            length += before.getValue();
            this.free.remove(start);
        }
        if (after != null && after.getKey() == offset + size) {
            length += after.getValue();
            this.free.remove(after.getKey());
        }
        this.free.put(start, length);
        this.freeBytes += size;
    }

    public synchronized long freeBytes() {
        return this.freeBytes;
    }

    public synchronized long usedBytes() {
        return this.totalBytes - this.freeBytes;
    }

    public long totalBytes() {
        return this.totalBytes;
    }

    private static long roundUp(long bytes) {
        return (bytes + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    }
}
