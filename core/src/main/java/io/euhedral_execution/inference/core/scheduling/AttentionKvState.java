package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/// Sequence-owned NVFP4 pages for one full-attention layer. Existing KV payloads
/// never move. The sequence lease serializes reservation with attention; close
/// runs after GPU completion. Each D256 row holds 128 code and 16 scale bytes.
public final class AttentionKvState implements AutoCloseable {
    public static final int PAGE_TOKENS = 256;
    public static final int HEAD_ROW_BYTES = 144;
    private final ExecutionGpu gpu;
    private final long planePageBytes;
    private final List<Long> pages = new ArrayList<>();
    private final List<Long> retiredTables = new ArrayList<>();
    private long table;
    private int tableSlots;
    private int capacity;
    private int length;
    private long decodeScratch;
    private int scratchHeads;
    private boolean closed;

    public AttentionKvState(ExecutionGpu gpu, int keyValueWidth) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (keyValueWidth <= 0 || keyValueWidth % 256 != 0)
            throw new IllegalArgumentException("NVFP4 KV requires complete D256 heads");
        this.planePageBytes = Math.multiplyExact((long) PAGE_TOKENS * HEAD_ROW_BYTES, keyValueWidth / 256);
    }

    /// Reserves contiguous append rows without publishing them to consumers.
    public void prepareAppend(long startPosition, int tokenCount) {
        ensureOpen();
        if (startPosition != this.length || tokenCount <= 0)
            throw new IllegalArgumentException("KV append must begin at the current sequence length");
        int required = Math.toIntExact(Math.addExact(startPosition, tokenCount));
        if (required <= this.capacity) return;
        int count = Math.toIntExact(((long) required + PAGE_TOKENS - 1) / PAGE_TOKENS);
        int nextCapacity = Math.toIntExact((long) count * PAGE_TOKENS);
        // Retain ownership of successful allocations if a later operation fails.
        while (this.pages.size() < count) this.pages.add(allocate(2 * this.planePageBytes));
        int slots = Math.max(1, this.tableSlots);
        while (slots < count) slots = Math.multiplyExact(slots, 2);
        long nextTable = this.table;
        if (slots != this.tableSlots) nextTable = allocate((long) slots * 2 * Long.BYTES);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment entries = arena.allocate((long) slots * 2 * Long.BYTES, Long.BYTES);
            for (int i = 0; i < count; i++) {
                entries.setAtIndex(ValueLayout.JAVA_LONG, i, this.pages.get(i));
                entries.setAtIndex(ValueLayout.JAVA_LONG, slots + i, this.pages.get(i) + this.planePageBytes);
            }
            // Synchronous transfer, not queued upload-buffer submission. No device
            // reader can still use this layer's table during reservation.
            this.gpu.copyHostToDevice(nextTable, entries, entries.byteSize());
        } catch (RuntimeException | Error failure) {
            if (nextTable != this.table) this.retiredTables.add(nextTable);
            throw failure;
        }
        if (nextTable != this.table && this.table != 0) this.retiredTables.add(this.table);
        this.table = nextTable;
        this.tableSlots = slots;
        this.capacity = nextCapacity;
        releaseRetired();
    }

    /// Called by the frame's GPU-completion edge, never merely after submission.
    public void commitAppend(int tokenCount) {
        ensureOpen();
        if (tokenCount <= 0 || (long) this.length + tokenCount > this.capacity)
            throw new IllegalArgumentException("KV append exceeds reserved capacity");
        this.length = Math.addExact(this.length, tokenCount);
    }

    /// Device pointer to the K page-address table, not to BF16 payloads.
    public long keyCacheAddress() {
        ensureOpen();
        if (this.table == 0) throw new IllegalStateException("KV cache has not been allocated");
        return this.table;
    }

    /// Device pointer to the V page-address table.
    public long valueCacheAddress() {
        return Math.addExact(keyCacheAddress(), (long) this.tableSlots * Long.BYTES);
    }

    /// Split-KV scratch is sequence-owned and allocated only for decode.
    public long decodeScratchAddress(int queryHeads) {
        ensureOpen();
        if (queryHeads <= 0) throw new IllegalArgumentException("queryHeads must be positive");
        if (this.decodeScratch == 0) {
            this.decodeScratch = allocate(Math.multiplyExact((long) queryHeads, 64L * 258 * Float.BYTES));
            this.scratchHeads = queryHeads;
        }
        if (queryHeads != this.scratchHeads) throw new IllegalArgumentException("decode head geometry changed");
        return this.decodeScratch;
    }

    public int capacity() {
        ensureOpen();
        return this.capacity;
    }

    public int length() {
        ensureOpen();
        return this.length;
    }

    @Override
    public void close() {
        if (this.closed) return;
        Throwable failure = null;
        try {
            releaseRetired();
        } catch (Throwable error) {
            failure = combine(failure, error);
        }
        for (int i = this.pages.size() - 1; i >= 0; i--) {
            try {
                this.gpu.free(this.pages.get(i));
                this.pages.remove(i);
            } catch (Throwable error) {
                failure = combine(failure, error);
            }
        }
        if (this.table != 0) {
            try {
                this.gpu.free(this.table);
                this.table = 0;
            } catch (Throwable error) {
                failure = combine(failure, error);
            }
        }
        if (this.decodeScratch != 0) {
            try {
                this.gpu.free(this.decodeScratch);
                this.decodeScratch = 0;
            } catch (Throwable error) {
                failure = combine(failure, error);
            }
        }
        this.closed =
                this.pages.isEmpty() && this.retiredTables.isEmpty() && this.table == 0 && this.decodeScratch == 0;
        if (failure != null) throw propagate(failure);
    }

    private long allocate(long bytes) {
        long address = this.gpu.allocate(bytes);
        if (address == 0) throw new IllegalStateException("GPU returned a null attention KV allocation");
        return address;
    }

    private void releaseRetired() {
        Throwable failure = null;
        for (int i = this.retiredTables.size() - 1; i >= 0; i--) {
            try {
                this.gpu.free(this.retiredTables.get(i));
                this.retiredTables.remove(i);
            } catch (Throwable error) {
                failure = combine(failure, error);
            }
        }
        if (failure != null) throw propagate(failure);
    }

    private void ensureOpen() {
        if (this.closed) throw new IllegalStateException("attention KV state is closed");
    }

    private static Throwable combine(Throwable first, Throwable next) {
        if (first == null) return next;
        first.addSuppressed(next);
        return first;
    }

    private static RuntimeException propagate(Throwable failure) {
        if (failure instanceof RuntimeException runtimeException) return runtimeException;
        if (failure instanceof Error error) throw error;
        return new IllegalStateException("failed to release attention KV state", failure);
    }
}
