package io.euhedral_execution.inference.core.model.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/// Loads an expert into the cache from one test thread, doing by hand what a serial source does
/// with its messages: the shard is single-owner, and a test that is its only caller owns it.
public final class ExpertTestSupport {

    private ExpertTestSupport() {}

    /// A store of `lanes` staging slots over `file` with a host tier of `ramSlots` records in `shards`
    /// shards in front of it, preloaded when the tier holds every record.
    public static FileExpertStore ramStore(
            GpuMemory memory, Path file, ExpertBank[] banks, int lanes, int ramSlots, int shards) throws IOException {
        FileRecordSource source = new FileRecordSource(file, banks);
        RamTier tier = new RamTier(banks, ramSlots, shards, ReplacementPolicy.BANK_PARTITIONED);
        try {
            if (tier.isResident()) tier.preload(source, 4);
        } catch (InterruptedException interrupted) {
            source.close();
            tier.close();
            Thread.currentThread().interrupt();
            throw new IOException(interrupted);
        } catch (IOException | RuntimeException | Error failure) {
            source.close();
            tier.close();
            throw failure;
        }
        return new FileExpertStore(memory, source, tier, banks, lanes);
    }

    /// A lease on the expert, loading it (read into staging buffer 0, copy on stream 0, wait for the
    /// copy) when it is not resident.
    public static ExpertLease acquire(ExpertCache cache, int bank, int expert) {
        int shardIndex = cache.shardOf(bank, expert);
        ExpertCacheShard shard = cache.shard(shardIndex);
        ExpertCacheShard.Ticket ticket = new ExpertCacheShard.Ticket();
        shard.claim(bank, expert, true, ticket);
        if (ticket.waiting())
            throw new IllegalStateException("every expert slot of shard " + shardIndex + " is in use");
        if (ticket.lease() != null) return ticket.lease();
        ExpertCacheShard.Load load = ticket.load();
        int buffer = 0;
        int stream = 0;
        HostRecord record = null;
        RamTierShard tier = cache.store().tier(shardIndex);
        TierDirective directive = new TierDirective();
        if (tier != null) tier.plan(bank, expert, directive);
        try {
            record = cache.store().open(bank, expert, buffer, directive);
            settle(tier, directive, true);
            CompletableFuture<Long> retired = new CompletableFuture<>();
            cache.transfer().stream(
                    stream,
                    record,
                    load.deviceAddress(),
                    load.fence(),
                    load.readyMarker(),
                    (ticketId, driverThread) -> retired.complete(ticketId));
            ExpertLease lease = load.submitted();
            Throwable failure = cache.transfer().confirm(stream, retired.get(20, TimeUnit.SECONDS));
            load.retired(failure);
            if (failure != null) throw new ExpertTransferException("expert " + expert + " failed to load", failure);
            return lease;
        } catch (ExpertTransferException failure) {
            throw failure;
        } catch (Exception failure) {
            if (record == null) {
                settle(tier, directive, false);
                load.failed(failure);
            }
            throw new IllegalStateException("expert " + expert + " of bank " + bank + " could not be loaded", failure);
        }
    }

    private static void settle(RamTierShard tier, TierDirective directive, boolean read) {
        if (tier == null) return;
        switch (directive.mode()) {
            case HIT -> tier.used(directive);
            case FILL -> {
                if (read) tier.filled(directive);
                else tier.abandoned(directive);
            }
            case NONE, BYPASS -> {}
        }
        directive.clear();
    }
}
