package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.atomic.LongAdder;

/// Cold storage: records stay in the artifact file, with an optional [RamTier] of ordinary memory in
/// front of it, and a lane's pinned staging slot holds one between the read and the end of its
/// device copy.
///
/// All slots are carved from one pinned arena, each as large as the biggest record rounded up to a
/// page. [#open] fills the lane's slot as its [TierDirective] says: from the artifact (the one host
/// copy of the record is from the page cache into pinned memory, with no heap array in between),
/// from a tier slot (one copy of the record out of RAM), or from the artifact through a tier slot
/// that keeps the record for the next miss. The lane owns its slot, so there is no pool, no wait and
/// no lock; memory is bounded by `lanes * slotBytes` however many records are requested. The tier's
/// bookkeeping is not here: its shards belong to the cache shards' owners, who decide the directive.
public final class FileExpertStore implements HostExpertStore {
    private static final long PAGE = 4096;

    private final ExpertBank[] banks;
    private final ExpertKeys keys;
    private final RecordSource source;
    private final HostArena arena;
    private final long slotBytes;
    private final int slotCount;
    private final RamTier tier;
    private final LongAdder ramCopyBytes = new LongAdder();
    private final LongAdder ramCopyNanos = new LongAdder();
    private final LongAdder opens = new LongAdder();

    /// A store that reads `file` directly.
    public FileExpertStore(GpuMemory memory, Path file, ExpertBank[] banks, int lanes) throws IOException {
        this(memory, new FileRecordSource(file, banks), banks, lanes);
    }

    /// A store over `source`, which it owns from here on: it is closed with the store, and also
    /// when this constructor fails.
    public FileExpertStore(GpuMemory memory, RecordSource source, ExpertBank[] banks, int lanes) {
        this(memory, source, null, banks, lanes);
    }

    /// A store over `source` with `tier` in front of it (or none, when null). The store owns both from
    /// here on, and closes them, also when this constructor fails.
    public FileExpertStore(GpuMemory memory, RecordSource source, RamTier tier, ExpertBank[] banks, int lanes) {
        Objects.requireNonNull(memory, "memory");
        this.source = Objects.requireNonNull(source, "source");
        this.tier = tier;
        try {
            if (lanes < 1) throw new IllegalArgumentException("lanes must be positive");
            this.banks = banks.clone();
            this.keys = new ExpertKeys(this.banks);
            this.slotCount = lanes;
            this.slotBytes = ExpertFiles.alignUp(ExpertFiles.maxRecordBytes(this.banks), PAGE);
            this.arena = new HostArena(memory, Math.multiplyExact(this.slotBytes, (long) lanes));
        } catch (RuntimeException | Error failure) {
            try {
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

    @Override
    public int lanes() {
        return this.slotCount;
    }

    @Override
    public HostRecord open(int bank, int expert, int lane) throws IOException, InterruptedException {
        return stage(bank, expert, lane, null);
    }

    @Override
    public HostRecord open(int bank, int expert, int lane, TierDirective directive)
            throws IOException, InterruptedException {
        return stage(bank, expert, lane, directive);
    }

    private HostRecord stage(int bank, int expert, int lane, TierDirective directive)
            throws IOException, InterruptedException {
        this.keys.key(bank, expert);
        java.util.Objects.checkIndex(lane, this.slotCount);
        long size = this.banks[bank].recordBytes(expert);
        long address = this.arena.address() + this.slotBytes * lane;
        MemorySegment staging = MemorySegment.ofAddress(address).reinterpret(size);
        switch (directive == null ? TierDirective.Mode.NONE : directive.mode()) {
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
        long begin = System.nanoTime();
        staging.copyFrom(MemorySegment.ofAddress(directive.address()).reinterpret(size));
        this.ramCopyBytes.add(size);
        this.ramCopyNanos.add(System.nanoTime() - begin);
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

    /// Bytes of one staging slot.
    public long slotBytes() {
        return this.slotBytes;
    }

    /// Closes the source and the tier and frees the staging arena. No lane may be in use.
    @Override
    public void close() {
        try {
            this.source.close();
        } finally {
            try {
                if (this.tier != null) this.tier.close();
            } finally {
                this.arena.close();
            }
        }
    }

    private record StagedRecord(long hostAddress, long byteSize) implements HostRecord {}
}
