package io.euhedral_execution.inference.core.state;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.CaptureFingerprint;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/// Sequence-owned pages for one full-attention layer, in the sequence's [KvFormat]. Existing KV payloads
/// never move. The sequence takes one state-sharing quantum at a time, and a prompt quantum's chunks append in
/// order (their carried-state edges), which orders reservation with attention; close runs after GPU completion.
/// Each D256 row holds the format's [KvFormat#headRowBytes] (NVFP4: 128 code and 16 scale bytes).
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
///
/// Two sides write it, each one at a time: appends (`prepareAppend`, `appendSubmitted`), which the carried KV edges
/// order across a sequence's quanta, own the pages, the table, the submitted frontier and the table staging;
/// conclusions (the commits and discards), which the sequence runs in admission order, own `length`. A later
/// quantum's append may run while an earlier one concludes, so a conclusion touches nothing the appends own
/// except, between quanta, the submitted frontier.
///
/// With a [HostRun], only the first `devicePages` pages are device memory. The pages past them live in pinned host
/// memory, and the table addresses them in the run's device region instead: an append first copies the host pages
/// the quantum reads into the region ([#stageIn]) and afterwards copies the pages it wrote back ([#stageOut]), both
/// on its stream, so the attention after it reads every row through the table as for device pages. Quanta of up to
/// [#STAGED_TAIL_ROWS] rows (the captured ones) copy every reserved host page in and the last two back, so what they
/// queue changes only with the reservation, which the fingerprint carries.
public final class AttentionKvState implements AutoCloseable {
    /// Rows of the quanta whose host page copies depend only on the reservation: decode, verification and draft
    /// quanta, which append at the end of the reserved rows and touch at most its last two pages.
    public static final int STAGED_TAIL_ROWS = 8;

    /// Host pages of one cache: the first `devicePages` pages are device memory, the next `pages` pinned host memory
    /// at `host`, read and written through the device region at `region` (one page of [#planePageBytes] K then V
    /// bytes each).
    public record HostRun(int devicePages, int pages, long host, long region) {
        public HostRun {
            if (devicePages <= 0 || pages <= 0 || host == 0 || region == 0)
                throw new IllegalArgumentException("a host run needs device pages, host pages and addresses");
        }
    }

    public static final int PAGE_TOKENS = 256;
    /// Bytes of an NVFP4 head row ([KvFormat#NVFP4]).
    public static final int HEAD_ROW_BYTES = 144;
    private final ExecutionGpu gpu;
    private final KvFormat format;
    private final long planePageBytes;
    private final List<Long> pages = new ArrayList<>();
    private final HostRun host;
    /// Table uploads' staging, each with the frontier its append reaches. The appends release what a committed
    /// append queued; the close releases the rest.
    private final List<Pending> pendingStaging = new ArrayList<>();

    private record Pending(long upTo, ExecutionGpu.UploadBuffer staging) {}

    private long table;
    private int tableSlots;
    private int capacity;
    private int submittedLength;
    /// Written by conclusions, read by appends.
    private volatile int length;
    private long decodeScratch;
    private int scratchHeads;
    private boolean closed;

    public AttentionKvState(ExecutionGpu gpu, int keyValueWidth) {
        this(gpu, keyValueWidth, KvFormat.NVFP4);
    }

    public AttentionKvState(ExecutionGpu gpu, int keyValueWidth, KvFormat format) {
        this(gpu, keyValueWidth, format, null);
    }

    /// A cache whose pages past `host`'s device pages live in its host run (null: every page on the device).
    public AttentionKvState(ExecutionGpu gpu, int keyValueWidth, KvFormat format, HostRun host) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.format = Objects.requireNonNull(format, "format");
        if (keyValueWidth <= 0 || keyValueWidth % 256 != 0)
            throw new IllegalArgumentException("the KV cache requires complete D256 heads");
        this.planePageBytes = planePageBytes(keyValueWidth, format);
        this.host = host;
    }

    /// Bytes of one plane (K or V) of one page of `keyValueWidth`-wide rows in `format`.
    public static long planePageBytes(int keyValueWidth, KvFormat format) {
        return Math.multiplyExact((long) PAGE_TOKENS * format.headRowBytes, keyValueWidth / 256);
    }

    /// The format the cache stores its rows in.
    public KvFormat format() {
        return this.format;
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
        releaseRetired(this.length);
        if (required <= this.capacity) return;
        int count = Math.toIntExact(((long) required + PAGE_TOKENS - 1) / PAGE_TOKENS);
        int nextCapacity = Math.toIntExact((long) count * PAGE_TOKENS);
        int device = count;
        if (this.host != null) {
            if (count > this.host.devicePages() + this.host.pages())
                throw new IllegalStateException("an attention cache of " + required + " rows is longer than the "
                        + (this.host.devicePages() + this.host.pages()) * PAGE_TOKENS + " rows planned for it");
            device = Math.min(count, this.host.devicePages());
        }
        // Retain ownership of successful allocations if a later operation fails.
        while (this.pages.size() < device) this.pages.add(allocate(2 * this.planePageBytes));
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
                long page =
                        i < device ? this.pages.get(i) : this.host.region() + (i - device) * 2 * this.planePageBytes;
                entries.setAtIndex(ValueLayout.JAVA_LONG_UNALIGNED, i, page);
                entries.setAtIndex(ValueLayout.JAVA_LONG_UNALIGNED, slots + i, page + this.planePageBytes);
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
        fingerprint
                .add(this.gpu, this.table)
                .add(this.tableSlots)
                .add(this.gpu, this.decodeScratch)
                .add(this.format.ordinal())
                .add(reservedHostPages());
    }

    /// Host pages the reserved rows reach (0 without a host run).
    public int reservedHostPages() {
        if (this.host == null) return 0;
        return Math.max(0, this.capacity / PAGE_TOKENS - this.host.devicePages());
    }

    /// Queues on the selected stream the copy of the host pages an append of `rows` rows at `start` and the
    /// attention after it read into the device region: every reserved one for a quantum of at most
    /// [#STAGED_TAIL_ROWS] rows, otherwise those holding the rows before `start + rows`. Nothing without host
    /// pages.
    public void stageIn(long start, int rows) {
        ensureOpen();
        int reserved = reservedHostPages();
        if (reserved == 0) return;
        int pages = rows <= STAGED_TAIL_ROWS
                ? reserved
                : Math.min(
                        reserved,
                        Math.toIntExact((start + rows + PAGE_TOKENS - 1) / PAGE_TOKENS) - this.host.devicePages());
        if (pages <= 0) return;
        this.gpu.copyHostToDeviceAsync(this.host.region(), this.host.host(), pages * 2 * this.planePageBytes);
    }

    /// Queues on the selected stream the copy back to host memory of the host pages an append of `rows` rows
    /// at `start` wrote: the last two reserved ones for a quantum of at most [#STAGED_TAIL_ROWS] rows (an
    /// append at the end of the reservation touches no other), otherwise exactly those holding its rows.
    public void stageOut(long start, int rows) {
        ensureOpen();
        int reserved = reservedHostPages();
        if (reserved == 0) return;
        int device = this.host.devicePages();
        int first;
        int end;
        if (rows <= STAGED_TAIL_ROWS) {
            end = reserved;
            first = Math.max(0, end - 2);
        } else {
            first = Math.max(0, Math.toIntExact(start / PAGE_TOKENS) - device);
            end = Math.min(reserved, Math.toIntExact((start + rows + PAGE_TOKENS - 1) / PAGE_TOKENS) - device);
        }
        if (end <= first) return;
        long offset = first * 2 * this.planePageBytes;
        this.gpu.copyDeviceToHostAsync(
                this.host.host() + offset, this.host.region() + offset, (end - first) * 2 * this.planePageBytes);
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
    }

    /// Publishes the `rows` rows a quantum appended, every one of them, once its device work retired. Rows a later
    /// quantum of the sequence appended meanwhile stay submitted.
    public void commitAppended(int rows) {
        ensureOpen();
        if (rows < 0 || (long) this.length + rows > this.submittedLength)
            throw new IllegalArgumentException("committed rows exceed the submitted frontier");
        this.length += rows;
    }

    /// Publishes only the first `rows` submitted rows (a speculative verification's accepted prefix);
    /// the rest stay beyond the committed frontier, where the next append overwrites them.
    public void commitSubmitted(int rows) {
        ensureOpen();
        if (rows < 0 || (long) this.length + rows > this.submittedLength)
            throw new IllegalArgumentException("committed rows exceed the submitted frontier");
        this.length += rows;
        this.submittedLength = this.length;
    }

    /// Drops a submitted frontier that must not become visible, such as a failed quantum's. Called after the
    /// quantum's device work retired. A later quantum of the sequence may still be appending: it fails behind this
    /// one, so its rows never become visible either.
    public void discardSubmitted() {
        if (this.closed) return;
        this.submittedLength = this.length;
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

    /// Device pages, K plane first then V plane in each (tests compare committed rows); without the host pages.
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
        // Freed in stream order on the calling thread's stream (the legacy default stream on a worker), which the
        // lanes' streams
        // are not ordered against: safe because every quantum that read this memory retired before the close.
        for (int i = this.pages.size() - 1; i >= 0; i--) {
            try {
                this.gpu.freeAsync(this.pages.get(i));
                this.pages.remove(i);
            } catch (Throwable error) {
                failure = combine(failure, error);
            }
        }
        if (this.table != 0) {
            try {
                this.gpu.freeAsync(this.table);
                this.table = 0;
            } catch (Throwable error) {
                failure = combine(failure, error);
            }
        }
        if (this.decodeScratch != 0) {
            try {
                this.gpu.freeAsync(this.decodeScratch);
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
