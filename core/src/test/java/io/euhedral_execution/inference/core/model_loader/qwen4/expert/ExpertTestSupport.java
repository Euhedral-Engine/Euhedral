package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/// Loads an expert into the cache from one test thread, doing by hand what a serial source does
/// with its messages: the shard is single-owner, and a test that is its only caller owns it.
public final class ExpertTestSupport {

    private ExpertTestSupport() {}

    /// A lease on the expert, loading it (read, copy on the shard's first lane, wait for the copy)
    /// when it is not resident.
    public static ExpertLease acquire(ExpertCache cache, int bank, int expert) {
        int shardIndex = cache.shardOf(bank, expert);
        ExpertCacheShard shard = cache.shard(shardIndex);
        ExpertCacheShard.Ticket ticket = new ExpertCacheShard.Ticket();
        shard.claim(bank, expert, true, ticket);
        if (ticket.waiting())
            throw new IllegalStateException("every expert slot of shard " + shardIndex + " is in use");
        if (ticket.lease() != null) return ticket.lease();
        ExpertCacheShard.Load load = ticket.load();
        int lane = cache.laneBase(shardIndex);
        HostRecord record = null;
        try {
            record = cache.store().open(bank, expert, lane);
            CompletableFuture<Long> retired = new CompletableFuture<>();
            cache.transfer().stream(
                    lane,
                    record,
                    load.deviceAddress(),
                    load.fence(),
                    load.readyMarker(),
                    (ticketId, driverThread) -> retired.complete(ticketId));
            ExpertLease lease = load.submitted();
            Throwable failure = cache.transfer().confirm(lane, retired.get(20, TimeUnit.SECONDS));
            load.retired(failure);
            if (failure != null) throw new ExpertTransferException("expert " + expert + " failed to load", failure);
            return lease;
        } catch (ExpertTransferException failure) {
            throw failure;
        } catch (Exception failure) {
            if (record == null) load.failed(failure);
            throw new IllegalStateException("expert " + expert + " of bank " + bank + " could not be loaded", failure);
        }
    }
}
