package io.euhedral_execution.inference.core.model.qwen4.loader;

import java.util.Locale;
import java.util.Map;

/// Where every part of a Flash-Next model lives for one device, one host and one maximum context: the outcome
/// of [ResidencyPlanner]. Placement is computed, never configured.
///
/// `placements` holds the storage class of every fixed object by name (device resident, host mapped, host
/// staged or deferred); every expert bank is device cached (`cachedBanks`) or deferred; n-gram shards are
/// host staged.
public record ResidencyPlan(
        int maxContextTokens,
        Mode mode,
        boolean fits,
        String explanation,
        Map<String, Placement> placements,
        Map<String, StorageClass> banks,
        Device device,
        Host host,
        ExpertCacheGeometry expertCache,
        ExpertStoreMode expertStore,
        NgramMode ngram,
        int prefillChunkTokens,
        HostBudget budget) {

    /// Storage of one fixed object; `move` is set for objects that left the device.
    public record Placement(StorageClass storage, Priority.Move move) {}

    /// How the routed experts are held on the host behind the device cache. In every mode records reach the
    /// device through a small pool of pinned staging slots; what differs is where a record comes from when a
    /// slot is filled.
    public enum ExpertStoreMode {
        /// Every record sits in ordinary (pageable) memory, loaded once at startup: no routed expert is read
        /// from the artifact during inference.
        RAM_RESIDENT,
        /// A bounded pageable cache of records, filled lazily from the artifact, replaced by a layer-aware
        /// policy.
        RAM_CACHED,
        /// No RAM tier: every GPU miss reads its record from the artifact (and the OS page cache).
        FILE_BACKED
    }

    /// How the n-gram tables are held on the host.
    public enum NgramMode {
        PINNED_ARENA,
        /// The artifact file mapped read-only; rows are gathered from it.
        MAPPED_FILE
    }

    /// Device bytes. `expertCacheBytes` is the whole slots of the cache; `slackBytes` what is left below one
    /// slot.
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

    /// Host bytes by use. Pinned bytes are page-locked; the expert tier is ordinary memory; file-backed bytes
    /// are the artifact's own pages. The tier holds `expertRamSlots` equal records.
    public record Host(
            long mappedBytes,
            long stagedBytes,
            long expertStagingPinnedBytes,
            long expertRamBytes,
            int expertRamSlots,
            long expertFileBytes,
            long ngramPinnedBytes,
            long ngramFileBytes,
            long deferredMtpBytes,
            long deferredVisionBytes) {

        public long pinnedBytes() {
            return mappedBytes + stagedBytes + expertStagingPinnedBytes + ngramPinnedBytes;
        }

        /// Ordinary memory the plan holds in all: the expert tier and everything pinned (pinned memory is
        /// ordinary memory that cannot be paged out).
        public long residentBytes() {
            return pinnedBytes() + expertRamBytes;
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
                        + ", workspace " + mib(d.workspaceBytes()) + ", runtime " + mib(d.runtimeReserveBytes())
                        + ", of which "
                        + mib(ResidencyPlanner.SYSTEM_RESERVE_BYTES) + " left free for the system)");
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
                this.expertStore + ", " + mib(h.expertRamBytes()) + " RAM (" + h.expertRamSlots() + " records), "
                        + mib(h.expertStagingPinnedBytes()) + " pinned staging, " + mib(h.expertFileBytes())
                        + " in the artifact");
        line(out, "n-gram host storage", this.ngram + ", " + mib(h.ngramPinnedBytes() + h.ngramFileBytes()));
        line(out, "deferred MTP", mib(h.deferredMtpBytes()));
        line(out, "deferred vision", mib(h.deferredVisionBytes()));
        line(out, "prefill chunk", this.prefillChunkTokens + " tokens");
        line(out, "pinned host memory", mib(h.pinnedBytes()));
        line(out, "ordinary host memory", mib(h.residentBytes()) + " of a budget of " + this.budget.describe());
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
