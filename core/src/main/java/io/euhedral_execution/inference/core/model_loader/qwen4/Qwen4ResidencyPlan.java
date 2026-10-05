package io.euhedral_execution.inference.core.model_loader.qwen4;

import java.util.Locale;
import java.util.Map;

/// Where every part of a Flash-Next model lives for one device, one host and one maximum context: the outcome of
/// [Qwen4ResidencyPlanner]. Placement is computed, never configured.
///
/// `placements` holds the storage class of every fixed object by name (device resident, host mapped, host staged
/// or deferred); every expert bank is device cached (`cachedBanks`) or deferred; n-gram shards are host staged.
public record Qwen4ResidencyPlan(
        int maxContextTokens,
        Qwen4Mode mode,
        boolean fits,
        String explanation,
        Map<String, Placement> placements,
        Map<String, StorageClass> banks,
        Device device,
        Host host,
        ExpertCacheGeometry expertCache,
        ExpertStoreMode expertStore,
        NgramMode ngram) {

    /// Storage of one fixed object; `move` is set for objects that left the device.
    public record Placement(StorageClass storage, Qwen4Priority.Move move) {}

    /// How the routed experts are held on the host behind the device cache.
    public enum ExpertStoreMode {
        /// All expert records in one pinned huge-page arena.
        PINNED_ARENA,
        /// Records stay in the artifact file (and the OS page cache) and pass through pinned staging slots.
        FILE_BACKED
    }

    /// How the n-gram tables are held on the host.
    public enum NgramMode {
        PINNED_ARENA,
        /// The artifact file mapped read-only; rows are gathered from it.
        MAPPED_FILE
    }

    /// Device bytes. `expertCacheBytes` is the whole slots of the cache; `slackBytes` what is left below one slot.
    public record Device(
            long freeBytes,
            long kvBytes,
            long indexerBytes,
            long gdnStateBytes,
            long workspaceBytes,
            long runtimeReserveBytes,
            long fixedResidentBytes,
            long stagingRingBytes,
            long expertCacheBytes,
            long slackBytes) {

        public long contextBytes() {
            return kvBytes + indexerBytes + gdnStateBytes;
        }

        /// Everything the plan places on the device.
        public long plannedBytes() {
            return contextBytes()
                    + workspaceBytes
                    + runtimeReserveBytes
                    + fixedResidentBytes
                    + stagingRingBytes
                    + expertCacheBytes;
        }
    }

    /// Host bytes by use. Pinned bytes are page-locked; file-backed ones are the artifact's own pages.
    public record Host(
            long mappedBytes,
            long stagedBytes,
            long expertStorePinnedBytes,
            long expertStoreFileBytes,
            long ngramPinnedBytes,
            long ngramFileBytes,
            long deferredMtpBytes,
            long deferredVisionBytes) {

        public long pinnedBytes() {
            return mappedBytes + stagedBytes + expertStorePinnedBytes + ngramPinnedBytes;
        }
    }

    public StorageClass storageOf(String object) {
        Placement placement = this.placements.get(object);
        if (placement != null) return placement.storage();
        StorageClass bank = this.banks.get(object);
        if (bank != null) return bank;
        throw new IllegalArgumentException("no such object in the plan: " + object);
    }

    /// The startup report: what was requested, what the device has, and where everything goes.
    public String report() {
        Device d = this.device;
        Host h = this.host;
        StringBuilder out = new StringBuilder("Flash-Next residency\n\n");
        line(out, "requested context", Integer.toString(this.maxContextTokens));
        line(out, "device memory available", mib(d.freeBytes()));
        line(
                out,
                "context/state reserve",
                mib(d.contextBytes() + d.workspaceBytes() + d.runtimeReserveBytes())
                        + " (KV " + mib(d.kvBytes()) + ", indexer " + mib(d.indexerBytes()) + ", GDN "
                        + mib(d.gdnStateBytes())
                        + ", workspace " + mib(d.workspaceBytes()) + ", runtime " + mib(d.runtimeReserveBytes()) + ")");
        line(out, "fixed resident", mib(d.fixedResidentBytes()));
        line(
                out,
                "fixed host-backed",
                mib(h.stagedBytes()) + " staged through a " + mib(d.stagingRingBytes()) + " ring");
        line(out, "host mapped", mib(h.mappedBytes()));
        line(out, "expert cache", mib(d.expertCacheBytes()) + " of " + mib(d.freeBytes()));
        line(
                out,
                "expert slots",
                this.expertCache.slotCount() + " of " + this.expertCache.totalExperts()
                        + " experts (slot "
                        + String.format(Locale.ROOT, "%.2f", this.expertCache.slotBytes() / 1048576.0)
                        + " MiB, minimum " + this.expertCache.minimumSlots() + ")");
        line(
                out,
                "expert host store",
                this.expertStore + ", " + mib(h.expertStorePinnedBytes()) + " pinned, " + mib(h.expertStoreFileBytes())
                        + " file backed");
        line(out, "n-gram host storage", this.ngram + ", " + mib(h.ngramPinnedBytes() + h.ngramFileBytes()));
        line(out, "deferred MTP", mib(h.deferredMtpBytes()));
        line(out, "deferred vision", mib(h.deferredVisionBytes()));
        line(out, "pinned host memory", mib(h.pinnedBytes()));
        if (!this.fits) line(out, "does not fit", this.explanation);
        return out.toString();
    }

    private static void line(StringBuilder out, String label, String value) {
        out.append(String.format(Locale.ROOT, "%-26s%s%n", label + ":", value));
    }

    private static String mib(long bytes) {
        return (bytes >> 20) + " MiB";
    }
}
