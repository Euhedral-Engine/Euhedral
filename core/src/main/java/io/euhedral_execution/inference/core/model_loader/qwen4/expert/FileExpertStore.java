package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.data_structures.queues.MpmcQueue;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.scheduling.graph.AsyncReads;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

/// Cold storage: records stay in the artifact file, with an optional [RamTier] of ordinary memory in
/// front of it, and a pinned staging buffer holds one between the read and the end of its device
/// copy.
///
/// Every buffer is as large as the biggest record rounded up to a page; there are as many as the plan
/// pins, a fixed amount of physical memory. A load takes one with [#acquireStaging] when the cache's
/// owner admits it and gives it back with [#releaseStaging] once the device copy that read it retired;
/// when none is free, the load is not started yet (its fetch tries again). [#open] fills a buffer as the load's
/// [TierDirective] says: from the artifact (the one host copy of the record is from the page cache into
/// pinned memory, with no heap array in between), from a tier slot (one copy of the record out of RAM),
/// or from the artifact through a tier slot that keeps the record for the next miss. The tier's
/// bookkeeping is not here: it belongs to the cache's owner, who decides the directive.
public final class FileExpertStore implements HostExpertStore {
    private static final long PAGE = 4096;

    private final ExpertBank[] banks;
    private final ExpertKeys keys;
    private final RecordSource source;
    private final GpuMemory memory;
    private final long slotBytes;
    /// The buffers pinned so far, by index; an index is published to a load (through the free list, or by
    /// the thread that pinned it) only after its address is written.
    private final HostArena[] buffers = new HostArena[MAX_BUFFERS];
    private final long[] addresses = new long[MAX_BUFFERS];
    private final Integer[] indices = new Integer[MAX_BUFFERS];
    private final AtomicInteger pinned = new AtomicInteger();
    private final MpmcQueue<Integer> free = new MpmcQueue<>(64, 4);
    private final RamTier tier;
    private final int readParts;
    private final LongAdder ramCopyBytes = new LongAdder();
    private final LongAdder ramCopyNanos = new LongAdder();
    private final LongAdder opens = new LongAdder();

    /// The most staging buffers a store pins: far more than loads can be in flight on any machine.
    static final int MAX_BUFFERS = 4096;

    /// A store that reads `file` directly, with `buffers` staging buffers pinned up front.
    public FileExpertStore(GpuMemory memory, Path file, ExpertBank[] banks, int buffers) throws IOException {
        this(memory, new FileRecordSource(file, banks), banks, buffers);
    }

    /// A store over `source`, which it owns from here on: it is closed with the store, and also
    /// when this constructor fails.
    public FileExpertStore(GpuMemory memory, RecordSource source, ExpertBank[] banks, int buffers) {
        this(memory, source, null, banks, buffers);
    }

    /// A store over `source` with `tier` in front of it (or none, when null). The store owns both from
    /// here on, and closes them, also when this constructor fails.
    public FileExpertStore(GpuMemory memory, RecordSource source, RamTier tier, ExpertBank[] banks, int buffers) {
        this(memory, source, tier, banks, buffers, 1);
    }

    /// As above, with each artifact read split into up to `readParts` parts when the source can read
    /// ranges. `buffers` staging buffers are pinned up front; more are pinned when a load finds none
    /// free.
    public FileExpertStore(
            GpuMemory memory, RecordSource source, RamTier tier, ExpertBank[] banks, int buffers, int readParts) {
        Objects.requireNonNull(memory, "memory");
        if (readParts < 1) throw new IllegalArgumentException("readParts must be positive");
        this.readParts = source.ranged() ? readParts : 1;
        this.source = Objects.requireNonNull(source, "source");
        this.tier = tier;
        this.memory = memory;
        try {
            if (buffers < 1 || buffers > MAX_BUFFERS)
                throw new IllegalArgumentException("buffers must be 1 to " + MAX_BUFFERS);
            this.banks = banks.clone();
            this.keys = new ExpertKeys(this.banks);
            this.slotBytes = ExpertFiles.alignUp(ExpertFiles.maxRecordBytes(this.banks), PAGE);
            for (int i = 0; i < buffers; i++) this.free.offer(pin());
        } catch (RuntimeException | Error failure) {
            try {
                freeBuffers();
                source.close();
                if (tier != null) tier.close();
            } catch (RuntimeException closeFailure) {
                failure.addSuppressed(closeFailure);
            }
            throw failure;
        }
    }

    @Override
    public ExpertBank[] banks() {
        return this.banks.clone();
    }

    /// Pins one more staging buffer and returns its index.
    private Integer pin() {
        int index = this.pinned.getAndIncrement();
        if (index >= MAX_BUFFERS) {
            this.pinned.decrementAndGet();
            throw new IllegalStateException("more than " + MAX_BUFFERS + " expert records staged at once");
        }
        HostArena arena = new HostArena(this.memory, this.slotBytes);
        this.buffers[index] = arena;
        this.addresses[index] = arena.address();
        this.indices[index] = index;
        return this.indices[index];
    }

    @Override
    public int acquireStaging() {
        Integer buffer = this.free.poll();
        return buffer != null ? buffer : -1;
    }

    @Override
    public void releaseStaging(int buffer) {
        if (!this.free.offer(this.indices[buffer])) throw new IllegalStateException("a staging buffer was lost");
    }

    /// Staging buffers pinned.
    @Override
    public int stagingBuffers() {
        return this.pinned.get();
    }

    private long staging(int buffer) {
        java.util.Objects.checkIndex(buffer, this.pinned.get());
        return this.addresses[buffer];
    }

    @Override
    public HostRecord open(int bank, int expert, int buffer) throws IOException, InterruptedException {
        return stage(bank, expert, buffer, null);
    }

    @Override
    public HostRecord open(int bank, int expert, int buffer, TierDirective directive)
            throws IOException, InterruptedException {
        return stage(bank, expert, buffer, directive);
    }

    private HostRecord stage(int bank, int expert, int buffer, TierDirective directive)
            throws IOException, InterruptedException {
        this.keys.key(bank, expert);
        long size = this.banks[bank].recordBytes(expert);
        TierDirective.Mode mode = directive == null ? TierDirective.Mode.NONE : directive.mode();
        if (!stagesThrough(directive)) {
            // A pinned tier is read by the device's copy in place.
            if (mode == TierDirective.Mode.FILL)
                this.source.read(
                        this.banks[bank],
                        expert,
                        MemorySegment.ofAddress(directive.address()).reinterpret(size));
            this.opens.increment();
            return new StagedRecord(directive.address(), size);
        }
        long address = staging(buffer);
        MemorySegment staging = MemorySegment.ofAddress(address).reinterpret(size);
        switch (mode) {
            case HIT -> copyOut(directive, staging, size);
            case FILL -> {
                this.source.read(
                        this.banks[bank],
                        expert,
                        MemorySegment.ofAddress(directive.address()).reinterpret(size));
                copyOut(directive, staging, size);
            }
            case NONE, BYPASS -> this.source.read(this.banks[bank], expert, staging);
        }
        this.opens.increment();
        return new StagedRecord(address, size);
    }

    /// One copy of the record from its tier slot into the staging slot.
    private void copyOut(TierDirective directive, MemorySegment staging, long size) {
        copyRange(staging, MemorySegment.ofAddress(directive.address()).reinterpret(size));
    }

    private void copyRange(MemorySegment staging, MemorySegment tier) {
        long begin = System.nanoTime();
        staging.copyFrom(tier);
        this.ramCopyBytes.add(tier.byteSize());
        this.ramCopyNanos.add(System.nanoTime() - begin);
    }

    @Override
    public int readParts() {
        return this.readParts;
    }

    @Override
    public void readPart(int bank, int expert, int buffer, TierDirective directive, int part, int parts)
            throws IOException, InterruptedException {
        long size = this.banks[bank].recordBytes(expert);
        long chunk = partChunk(size, parts);
        long from = Math.min(size, chunk * part);
        long length = Math.min(size, from + chunk) - from;
        if (length <= 0) return;
        this.source.readRange(this.banks[bank], expert, from, partDestination(buffer, directive, from, length));
        completePart(bank, expert, buffer, directive, part, parts, 0);
    }

    @Override
    public boolean rangedReads() {
        return this.source.ranged();
    }

    @Override
    public boolean readPartAsync(
            int bank, int expert, int buffer, TierDirective directive, int part, int parts, AsyncReads.Read done) {
        long size = this.banks[bank].recordBytes(expert);
        long chunk = partChunk(size, parts);
        long from = Math.min(size, chunk * part);
        long length = Math.min(size, from + chunk) - from;
        if (length <= 0) return false;
        return this.source.readRangeAsync(
                this.banks[bank], expert, from, partDestination(buffer, directive, from, length), done);
    }

    @Override
    public void completePart(
            int bank, int expert, int buffer, TierDirective directive, int part, int parts, long readNanos) {
        if (readNanos > 0 && this.source instanceof FileRecordSource file) file.asyncReadEnded(readNanos);
        if (directive.mode() != TierDirective.Mode.FILL || !stagesThrough(directive)) return;
        long size = this.banks[bank].recordBytes(expert);
        long chunk = partChunk(size, parts);
        long from = Math.min(size, chunk * part);
        long length = Math.min(size, from + chunk) - from;
        if (length <= 0) return;
        // A fill keeps what it read in the tier slot and stages it too: each part copies its own range,
        // so the copies run side by side as the reads do.
        copyRange(
                MemorySegment.ofAddress(staging(buffer) + from).reinterpret(length),
                MemorySegment.ofAddress(directive.address() + from).reinterpret(length));
    }

    /// Parts start on page boundaries; the last ones may be short or empty.
    private static long partChunk(long size, int parts) {
        return ExpertFiles.alignUp((size + parts - 1) / parts, PAGE);
    }

    /// Where a part's bytes are read to: the tier slot of a fill, otherwise the staging buffer.
    private MemorySegment partDestination(int buffer, TierDirective directive, long from, long length) {
        long base = directive.mode() == TierDirective.Mode.FILL ? directive.address() : staging(buffer);
        return MemorySegment.ofAddress(base + from).reinterpret(length);
    }

    @Override
    public HostRecord completeOpen(int bank, int expert, int buffer, TierDirective directive) {
        long size = this.banks[bank].recordBytes(expert);
        long address = stagesThrough(directive) ? staging(buffer) : directive.address();
        this.opens.increment();
        return new StagedRecord(address, size);
    }

    /// Whether a load with `directive` passes through a staging buffer: always, but for a record whose tier slot
    /// is pinned memory (a hit or a fill of a pinned tier), which the device's copy reads in place.
    @Override
    public boolean stagesThrough(TierDirective directive) {
        if (this.tier == null || !this.tier.pinned() || directive == null) return true;
        return directive.mode() != TierDirective.Mode.HIT && directive.mode() != TierDirective.Mode.FILL;
    }

    @Override
    public RamTierShard tier(int shard) {
        return this.tier == null ? null : this.tier.shard(shard);
    }

    /// The host tier, or null.
    public RamTier ramTier() {
        return this.tier;
    }

    @Override
    public long bytesRead() {
        return this.source.bytesRead();
    }

    @Override
    public long recordOpens() {
        return this.opens.sum();
    }

    /// Bytes copied from the host tier into staging slots.
    public long ramCopyBytes() {
        return this.ramCopyBytes.sum();
    }

    /// Time in those copies, summed over copies (parallel copies add up).
    public long ramCopyNanos() {
        return this.ramCopyNanos.sum();
    }

    /// Bytes of one staging buffer.
    public long slotBytes() {
        return this.slotBytes;
    }

    /// Closes the source and the tier and frees the staging buffers. No buffer may be in use.
    @Override
    public void close() {
        try {
            this.source.close();
        } finally {
            try {
                if (this.tier != null) this.tier.close();
            } finally {
                freeBuffers();
            }
        }
    }

    private record StagedRecord(long hostAddress, long byteSize) implements HostRecord {}

    private void freeBuffers() {
        int count = Math.min(this.pinned.get(), MAX_BUFFERS);
        for (int i = 0; i < count; i++) {
            if (this.buffers[i] != null) this.buffers[i].close();
        }
    }
}
