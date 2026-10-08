package io.euhedral_execution.inference.core.state;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.CaptureFingerprint;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/// Sequence-owned NVFP4 pages for one full-attention layer. Existing KV payloads
/// never move. The sequence takes one state-sharing quantum at a time, and a prompt quantum's chunks append in
/// order (their carried-state edges), which orders reservation with attention; close runs after GPU completion.
/// Each D256 row holds 128 code and 16 scale bytes.
///
/// Reservation runs inside the owning quantum with its stream selected, and allocates in stream order on
/// it. A grown page table is uploaded from pinned staging by a copy queued on that stream, ahead of the
/// stages that read the table, so reservation never waits for the device. The outgrown table is freed in
/// stream order right after the upload: its readers belong to quanta that already retired. The staging
/// stays owned until the append is committed or discarded, which happens only after the quantum retired.
///
/// Rows move through three frontiers. `capacity` is reserved: backed by pages. The submitted
/// frontier covers rows whose writes are queued on the owning quantum's stream; later stages of that
/// quantum may read them, because stream order runs those reads after the writes. `length` is the
/// committed frontier, published only after the quantum's device work retired; other quanta and
/// external readers see nothing beyond it.
public final class AttentionKvState implements AutoCloseable {
    public static final int PAGE_TOKENS = 256;
    public static final int HEAD_ROW_BYTES = 144;
    private final ExecutionGpu gpu;
    private final long planePageBytes;
    private final List<Long> pages = new ArrayList<>();
    /// Table uploads' staging, each with the frontier its append reaches: released once that append settled, so a
    /// later quantum's queued upload keeps its staging while an earlier quantum commits.
    private final List<Pending> pendingStaging = new ArrayList<>();

    private record Pending(long upTo, ExecutionGpu.UploadBuffer staging) {}

    private long table;
    private int tableSlots;
    private int capacity;
    private int submittedLength;
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

    /// Reserves contiguous append rows without publishing them to consumers. The rows continue the submitted
    /// frontier: a quantum's first append starts at the committed length, and a prompt's later chunks each start
    /// where the previous chunk's append ended. A prompt quantum reserves all its rows once, at admission, so its
    /// chunks find them reserved.
    public void prepareAppend(long startPosition, int tokenCount) {
        ensureOpen();
        if (startPosition != this.submittedLength || tokenCount <= 0)
            throw new IllegalArgumentException("KV append must continue the submitted frontier at "
                    + this.submittedLength + ", not " + startPosition);
        int required = Math.toIntExact(Math.addExact(startPosition, tokenCount));
        if (required <= this.capacity) return;
        int count = Math.toIntExact(((long) required + PAGE_TOKENS - 1) / PAGE_TOKENS);
        int nextCapacity = Math.toIntExact((long) count * PAGE_TOKENS);
        // Retain ownership of successful allocations if a later operation fails.
        while (this.pages.size() < count) this.pages.add(allocate(2 * this.planePageBytes));
        int slots = Math.max(1, this.tableSlots);
        while (slots < count) slots = Math.multiplyExact(slots, 2);
        long tableBytes = (long) slots * 2 * Long.BYTES;
        long nextTable = this.table;
        if (slots != this.tableSlots) nextTable = allocate(tableBytes);
        try {
            ExecutionGpu.UploadBuffer staging = this.gpu.allocateUploadBuffer(tableBytes);
            // Owned until retirement from here on: a queued copy may read it even if this call fails.
            this.pendingStaging.add(new Pending(required, staging));
            MemorySegment entries = staging.segment();
            entries.fill((byte) 0);
            for (int i = 0; i < count; i++) {
                entries.setAtIndex(ValueLayout.JAVA_LONG_UNALIGNED, i, this.pages.get(i));
                entries.setAtIndex(ValueLayout.JAVA_LONG_UNALIGNED, slots + i, this.pages.get(i) + this.planePageBytes);
            }
            // No reader of this layer's table precedes this stage in the quantum, and an earlier quantum's readers
            // precede it through the carried KV edge, so the stream orders the rewrite.
            this.gpu.copyUploadToDevice(nextTable, staging);
        } catch (RuntimeException | Error failure) {
            if (nextTable != this.table) this.gpu.freeAsync(nextTable);
            throw failure;
        }
        // Readers of the outgrown table precede this stage (earlier stages, or an earlier quantum's through the
        // carried KV edge); the stream frees it after this upload.
        if (nextTable != this.table && this.table != 0) this.gpu.freeAsync(this.table);
        this.table = nextTable;
        this.tableSlots = slots;
        this.capacity = nextCapacity;
    }

    /// Whether appending `tokenCount` rows at `startPosition` fits the reserved pages, so that reservation
    /// neither allocates nor uploads a page table.
    public boolean reserves(long startPosition, int tokenCount) {
        return !this.closed && startPosition + tokenCount <= this.capacity;
    }

    public void fingerprint(CaptureFingerprint fingerprint) {
        fingerprint.add(this.gpu, this.table).add(this.tableSlots).add(this.gpu, this.decodeScratch);
    }

    /// Records that the writes for `tokenCount` reserved rows after the submitted frontier were submitted to the
    /// owning quantum's stream (a prompt's chunks, one after another).
    public void appendSubmitted(int tokenCount) {
        ensureOpen();
        if (tokenCount <= 0 || (long) this.submittedLength + tokenCount > this.capacity)
            throw new IllegalArgumentException("KV append exceeds reserved capacity");
        this.submittedLength += tokenCount;
    }

    /// Rows that a later stage of the submitting quantum may read.
    public int submittedLength() {
        ensureOpen();
        return this.submittedLength;
    }

    /// Publishes the submitted frontier. Called only after the quantum's device work retired.
    public void commitSubmitted() {
        ensureOpen();
        this.length = this.submittedLength;
        releaseRetired(this.length);
    }

    /// Publishes the `rows` rows a quantum appended, every one of them, once its device work retired. Rows a later
    /// quantum of the sequence appended meanwhile stay submitted.
    public void commitAppended(int rows) {
        ensureOpen();
        if (rows < 0 || (long) this.length + rows > this.submittedLength)
            throw new IllegalArgumentException("committed rows exceed the submitted frontier");
        this.length += rows;
        releaseRetired(this.length);
    }

    /// Publishes only the first `rows` submitted rows (a speculative verification's accepted prefix);
    /// the rest stay beyond the committed frontier, where the next append overwrites them.
    public void commitSubmitted(int rows) {
        ensureOpen();
        if (rows < 0 || (long) this.length + rows > this.submittedLength)
            throw new IllegalArgumentException("committed rows exceed the submitted frontier");
        long submitted = this.submittedLength;
        this.length += rows;
        this.submittedLength = this.length;
        releaseRetired(submitted);
    }

    /// Drops a submitted frontier that must not become visible, such as a failed quantum's. Called
    /// after the quantum's device work retired, so it also releases what the reservation retired.
    public void discardSubmitted() {
        if (this.closed) return;
        long submitted = this.submittedLength;
        this.submittedLength = this.length;
        releaseRetired(submitted);
    }

    /// As [#discardSubmitted()] for a quantum whose append ends at `end`: a later quantum of the sequence may still
    /// be in flight, so only staging of appends up to `end` is released.
    public void discardSubmitted(long end) {
        if (this.closed) return;
        this.submittedLength = this.length;
        releaseRetired(end);
    }

    /// Drops committed rows from `length` on (speculative MTP draft rows), between quanta. The rows stay
    /// in their pages until the next append overwrites them; no reader sees past the frontier.
    public void truncate(int length) {
        ensureOpen();
        if (this.submittedLength != this.length) throw new IllegalStateException("an append is in flight");
        if (length < 0 || length > this.length) throw new IllegalArgumentException("can only truncate committed rows");
        this.length = length;
        this.submittedLength = length;
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

    /// Device pages, K plane first then V plane in each (tests compare committed rows).
    public List<Long> pageAddresses() {
        ensureOpen();
        return List.copyOf(this.pages);
    }

    /// Bytes of one plane (K or V) of one page: PAGE_TOKENS token rows of all KV heads.
    public long planePageBytes() {
        return this.planePageBytes;
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
            releaseRetired(Long.MAX_VALUE);
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
        this.closed = this.pages.isEmpty() && this.table == 0 && this.decodeScratch == 0;
        if (failure != null) throw propagate(failure);
    }

    private long allocate(long bytes) {
        long address = this.gpu.allocateAsync(bytes);
        if (address == 0) throw new IllegalStateException("GPU returned a null attention KV allocation");
        return address;
    }

    /// Releases the staging of appends reaching at most `upTo`, whose quanta retired, so no queued work references
    /// it. A GPU that cannot prove its submitted work stopped keeps the staging.
    private void releaseRetired(long upTo) {
        if (this.pendingStaging.isEmpty() || !this.gpu.completionProven()) return;
        var settled = this.pendingStaging.iterator();
        while (settled.hasNext()) {
            Pending pending = settled.next();
            if (pending.upTo() > upTo) continue;
            pending.staging().close();
            settled.remove();
        }
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
