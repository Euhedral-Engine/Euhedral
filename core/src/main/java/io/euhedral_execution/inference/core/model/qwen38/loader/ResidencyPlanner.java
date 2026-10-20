package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightStaging;
import io.euhedral_execution.inference.core.artifact.format.LinearFormats;
import io.euhedral_execution.inference.core.artifact.format.TrellisSettings;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.offload.DiskWindow;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import io.euhedral_execution.inference.core.state.KvFormat;
import java.util.Set;

/// Decides which weights stay on the device so that a context of `maxContextTokens` fits in the memory
/// the device has free. Nothing is offloaded when everything fits; otherwise the smallest selection
/// of [HostWeightSelection] that makes room is taken, because every offloaded byte costs decode time. The
/// selection goes to pinned host memory up to the host budget and the rest stays in the artifact file (disk).
///
/// The device budget is the resident weights, the staging ring for host-backed projections, the KV pages
/// of every sequence of `maxContextTokens`, a fixed reserve for the first sequence's GDN state, the
/// execution workspaces, the shared scratch, the prefill scratch of the largest trellis object (zero for other
/// artifacts) and the kernel modules loaded after the model.
///
/// The LM head (the one object too large for the staging slots) leaves the device last: it is mapped into host
/// memory and read in place over the bus, once per pass whatever the sequence count, when everything else
/// offloadable is offloaded and the plan still does not fit. The draft head stays resident: every draft of every
/// sequence reads it.
public final class ResidencyPlanner {

    /// Everything outside weights and KV that a running sequence needs on the device.
    static final long RESERVE_BYTES = 1280L << 20;

    /// Most rows of one row-exact verification that an NVFP4 linear runs on the tensor-core decode kernels (8 rows
    /// in 4 groups: `EUHEDRAL_DECODE_MAX_ROWS * DECODE_GROUPS_MAX` in `nvfp4_linear.c`); beyond it row-exact
    /// verification runs the scalar reference, which sweeps the weights once per row.
    static final int NVFP4_ROW_EXACT_MAX_ROWS = 32;

    private ResidencyPlanner() {}

    /// The outcome: the objects to keep in host memory and those left in the artifact file, the device bytes the
    /// plan needs, the pinned host bytes of the host-backed objects (`hostArenaBytes`) and of the disk window
    /// (`windowBytes`, 0 without disk-backed objects), the attention cache pages each sequence keeps in pinned host
    /// memory per attention layer (`kvHostPages`, 0 when every page is on the device) and their pinned bytes over
    /// every sequence and layer (`kvHostBytes`), and whether it fits.
    public record Plan(
            Set<String> hostBacked,
            Set<String> diskBacked,
            long deviceBytes,
            long hostArenaBytes,
            long windowBytes,
            int kvHostPages,
            long kvHostBytes,
            boolean fits) {

        public Plan(
                Set<String> hostBacked,
                Set<String> diskBacked,
                long deviceBytes,
                long hostArenaBytes,
                long windowBytes,
                boolean fits) {
            this(hostBacked, diskBacked, deviceBytes, hostArenaBytes, windowBytes, 0, 0L, fits);
        }

        public Plan(Set<String> hostBacked, long deviceBytes, boolean fits) {
            this(hostBacked, Set.of(), deviceBytes, 0L, 0L, fits);
        }
    }

    /// No attention cache page leaves the device: the planners' callers outside the engine.
    public static final int ALL_KV_ON_DEVICE = Integer.MAX_VALUE;

    /// NVFP4 KV bytes of one sequence of `tokens` positions, in whole pages.
    public static long kvBytes(Qwen38Config config, int tokens) {
        return kvBytes(config, tokens, KvFormat.NVFP4);
    }

    /// KV bytes of one sequence of `tokens` positions in `format`, in whole pages.
    public static long kvBytes(Qwen38Config config, int tokens, KvFormat format) {
        long layers = 0;
        for (LayerType type : config.layerTypes()) if (type == LayerType.FULL_ATTENTION) layers++;
        long pages = (tokens + 255L) / 256L;
        int width = config.numKeyValueHeads() * config.attentionHeadDim();
        return Math.multiplyExact(layers * pages, 2 * AttentionKvState.planePageBytes(width, format));
    }

    /// Plans residency for one sequence; see the overload for several.
    public static Plan plan(
            Artifact artifact,
            ArtifactProfile profile,
            long freeBytes,
            int maxContextTokens,
            long hostBytes,
            KvFormat kvFormat) {
        return plan(artifact, profile, freeBytes, maxContextTokens, hostBytes, kvFormat, 1);
    }

    /// Plans residency for `sequences` concurrent sequences of up to `maxContextTokens` each, with at most
    /// `hostBytes` of pinned host memory and a KV cache in `kvFormat`. Each sequence holds its KV, and each one
    /// beyond the first the state of [SequenceFootprint] on top of the reserve. Objects leave the device in
    /// [HostWeightSelection]'s order, as many as the device needs; they go to host memory in that order while it
    /// holds them beside the disk window, and the rest stay in the artifact file. The plan does not fit when the
    /// device cannot make room, or when host memory cannot hold the token embedding (a gather, read in place) and
    /// the window.
    public static Plan plan(
            Artifact artifact,
            ArtifactProfile profile,
            long freeBytes,
            int maxContextTokens,
            long hostBytes,
            KvFormat kvFormat,
            int sequences) {
        return plan(artifact, profile, freeBytes, maxContextTokens, hostBytes, kvFormat, sequences, ALL_KV_ON_DEVICE);
    }

    /// As above, with `kvDeviceMinimumTokens` of each sequence's attention cache kept on the device, a workspace of
    /// one prefill chunk and the default trellis thresholds.
    public static Plan plan(
            Artifact artifact,
            ArtifactProfile profile,
            long freeBytes,
            int maxContextTokens,
            long hostBytes,
            KvFormat kvFormat,
            int sequences,
            int kvDeviceMinimumTokens) {
        return plan(
                artifact,
                profile,
                freeBytes,
                maxContextTokens,
                hostBytes,
                kvFormat,
                sequences,
                kvDeviceMinimumTokens,
                InferenceConfig.PREFILL_CHUNK_TOKENS,
                TrellisSettings.defaults());
    }

    /// As above, with `kvDeviceMinimumTokens` of each sequence's attention cache kept on the device, a workspace of
    /// `workspaceRows` rows and the device's `trellis` thresholds (which size the trellis scratch). When the
    /// resident weights and every sequence's whole cache do not fit, each sequence's pages past that many
    /// tokens go to pinned host memory first (each attention layer's are copied to one device region of the
    /// sequence's before the layer reads them, [KvHostPages]); only when the weights and the device share
    /// still do not fit do weights leave the device, as above. The host pages come out of the host memory
    /// before any weight does.
    public static Plan plan(
            Artifact artifact,
            ArtifactProfile profile,
            long freeBytes,
            int maxContextTokens,
            long hostBytes,
            KvFormat kvFormat,
            int sequences,
            int kvDeviceMinimumTokens,
            int workspaceRows,
            TrellisSettings trellis) {
        if (sequences < 1) throw new IllegalArgumentException("sequences must be at least 1");
        if (kvDeviceMinimumTokens <= 0) throw new IllegalArgumentException("kvDeviceMinimumTokens must be positive");
        long extra = sequences == 1
                ? 0
                : SequenceFootprint.extraSequenceBytes(
                        artifact.config(),
                        kvFormat,
                        maxContextTokens,
                        1 + profile.speculativeDepth(),
                        profile.speculation() == ArtifactProfile.Speculation.MTP);
        long fixed = Math.addExact(
                Math.addExact(
                        Math.multiplyExact(sequences, kvBytes(artifact.config(), maxContextTokens, kvFormat)),
                        RESERVE_BYTES),
                Math.multiplyExact(sequences - 1L, extra));
        fixed = Math.addExact(fixed, trellisScratchBytes(artifact, profile, workspaceRows, trellis));
        Plan resident = deviceNeed(artifact, profile, Set.of());
        if (resident.deviceBytes() + fixed <= freeBytes) return tiers(artifact, resident, fixed, hostBytes, true);
        // The cache past the device minimum moves to host memory before any weight leaves the device.
        int pages = Math.toIntExact((maxContextTokens + 255L) / 256L);
        int devicePages = (int) Math.min(pages, (kvDeviceMinimumTokens + 255L) / 256L);
        int hostPages = pages - devicePages;
        long kvHost = 0;
        long region = Math.multiplyExact((long) hostPages, pageBytes(artifact.config(), kvFormat));
        long spilled = Math.multiplyExact(Math.multiplyExact(region, attentionLayers(artifact.config())), sequences);
        // Host memory that cannot hold the pages leaves the cache on the device and the weights to make room.
        if (hostPages > 0 && spilled <= hostBytes) {
            kvHost = spilled;
            long whole = Math.multiplyExact(sequences, kvBytes(artifact.config(), maxContextTokens, kvFormat));
            long kept = Math.multiplyExact(
                    sequences, Math.addExact(kvBytes(artifact.config(), devicePages * 256, kvFormat), region));
            fixed = Math.addExact(Math.subtractExact(fixed, whole), kept);
            hostBytes -= kvHost;
        } else {
            hostPages = 0;
        }
        return withKvHost(
                offload(artifact, profile, freeBytes, sequences, hostBytes, fixed, resident), hostPages, kvHost);
    }

    /// Bytes of one attention cache page of a full-attention layer (its K plane then its V plane) in `format`.
    public static long pageBytes(Qwen38Config config, KvFormat format) {
        return 2 * AttentionKvState.planePageBytes(config.numKeyValueHeads() * config.attentionHeadDim(), format);
    }

    /// Full-attention layers of the base model: the layers whose caches may keep pages in host memory.
    public static int attentionLayers(Qwen38Config config) {
        int layers = 0;
        for (LayerType type : config.layerTypes()) if (type == LayerType.FULL_ATTENTION) layers++;
        return layers;
    }

    private static Plan withKvHost(Plan plan, int hostPages, long kvHost) {
        if (kvHost == 0) return plan;
        return new Plan(
                plan.hostBacked(),
                plan.diskBacked(),
                plan.deviceBytes(),
                plan.hostArenaBytes(),
                plan.windowBytes(),
                hostPages,
                kvHost,
                plan.fits());
    }

    /// The weight offload of [#plan]: grows the offloaded selection until the device need beside `fixed` fits.
    private static Plan offload(
            Artifact artifact,
            ArtifactProfile profile,
            long freeBytes,
            int sequences,
            long hostBytes,
            long fixed,
            Plan resident) {
        TensorDescriptor head = headOf(artifact);
        // A mapped NVFP4 head is swept over the bus once per row when a verification outgrows the decode kernels.
        boolean mappable = head != null
                && !(head.format() == WeightFormat.NVFP4
                        && (long) sequences * (1 + profile.speculativeDepth()) > NVFP4_ROW_EXACT_MAX_ROWS);
        if (resident.deviceBytes() + fixed <= freeBytes) return tiers(artifact, resident, fixed, hostBytes, true);
        long all = artifactBytes(artifact, profile);
        // Grow the offloaded bytes until the device need fits. The ring grows with the largest selected
        // tensor, so each step re-evaluates the whole selection.
        long step = 64L << 20;
        for (long bytes = step; ; bytes += step) {
            Set<String> selection = HostWeightSelection.select(artifact, bytes);
            Plan candidate = deviceNeed(artifact, profile, selection);
            if (candidate.deviceBytes() + fixed <= freeBytes) return tiers(artifact, candidate, fixed, hostBytes, true);
            if (bytes > all) {
                // Everything else offloadable is offloaded: the LM head goes last.
                if (!mappable) return tiers(artifact, candidate, fixed, hostBytes, false);
                Set<String> mapped = withHead(head, selection);
                if (mapped.size() == selection.size()) return tiers(artifact, candidate, fixed, hostBytes, false);
                Plan last = deviceNeed(artifact, profile, mapped);
                return tiers(artifact, last, fixed, hostBytes, last.deviceBytes() + fixed <= freeBytes);
            }
        }
    }

    /// The largest count of sequences, at most `cap`, whose plan fits and offloads (host- plus disk-backed) no more
    /// weight bytes than one sequence's plan: more sequences never evict resident weights. 1 when even one does not
    /// fit.
    public static int sequencesFor(
            Artifact artifact,
            ArtifactProfile profile,
            long freeBytes,
            int maxContextTokens,
            long hostBytes,
            KvFormat kvFormat,
            int cap) {
        return sequencesFor(artifact, profile, freeBytes, maxContextTokens, hostBytes, kvFormat, cap, ALL_KV_ON_DEVICE);
    }

    /// As above, with `kvDeviceMinimumTokens` of each sequence's cache kept on the device ([#plan]).
    public static int sequencesFor(
            Artifact artifact,
            ArtifactProfile profile,
            long freeBytes,
            int maxContextTokens,
            long hostBytes,
            KvFormat kvFormat,
            int cap,
            int kvDeviceMinimumTokens) {
        return sequencesFor(
                artifact,
                profile,
                freeBytes,
                maxContextTokens,
                hostBytes,
                kvFormat,
                cap,
                kvDeviceMinimumTokens,
                InferenceConfig.PREFILL_CHUNK_TOKENS,
                TrellisSettings.defaults());
    }

    /// As above, with a workspace of `workspaceRows` rows and the device's `trellis` thresholds ([#plan]).
    public static int sequencesFor(
            Artifact artifact,
            ArtifactProfile profile,
            long freeBytes,
            int maxContextTokens,
            long hostBytes,
            KvFormat kvFormat,
            int cap,
            int kvDeviceMinimumTokens,
            int workspaceRows,
            TrellisSettings trellis) {
        Plan single = plan(
                artifact,
                profile,
                freeBytes,
                maxContextTokens,
                hostBytes,
                kvFormat,
                1,
                kvDeviceMinimumTokens,
                workspaceRows,
                trellis);
        if (!single.fits()) return 1;
        long offloaded = offloadedBytes(artifact, single);
        int low = 1;
        int high = cap;
        // Fitting and the offloaded bytes only worsen with the count: search for the largest that is no worse.
        while (low < high) {
            int middle = (low + high + 1) >>> 1;
            Plan candidate = plan(
                    artifact,
                    profile,
                    freeBytes,
                    maxContextTokens,
                    hostBytes,
                    kvFormat,
                    middle,
                    kvDeviceMinimumTokens,
                    workspaceRows,
                    trellis);
            if (candidate.fits() && offloadedBytes(artifact, candidate) <= offloaded) low = middle;
            else high = middle - 1;
        }
        return low;
    }

    /// The scratch the prefill route of the largest trellis object needs at the workspace's `rows` (decoded weight,
    /// rotated activations and outputs) with the `trellis` thresholds; 0 for an artifact with no trellis object.
    static long trellisScratchBytes(Artifact artifact, ArtifactProfile profile, int rows, TrellisSettings trellis) {
        long largest = 0;
        for (TensorDescriptor tensor : artifact.tensors()) {
            if (tensor.format() != WeightFormat.TRELLIS
                    || !CompactWeightLoader.uploads(tensor.name(), profile.speculation())) continue;
            largest = Math.max(largest, LinearFormats.of(tensor).planScratchBytes(tensor, rows, trellis));
        }
        return largest;
    }

    /// The LM head's descriptor, or null when the artifact has none.
    private static TensorDescriptor headOf(Artifact artifact) {
        for (TensorDescriptor tensor : artifact.tensors())
            if (tensor.name().equals(HostWeightSelection.OUTPUT_HEAD)) return tensor;
        return null;
    }

    /// `selection` followed by the LM head `head`.
    private static Set<String> withHead(TensorDescriptor head, Set<String> selection) {
        Set<String> result = new java.util.LinkedHashSet<>(selection);
        result.add(head.name());
        return result;
    }

    private static long offloadedBytes(Artifact artifact, Plan plan) {
        long bytes = 0;
        for (TensorDescriptor tensor : artifact.tensors())
            if (plan.hostBacked().contains(tensor.name()) || plan.diskBacked().contains(tensor.name()))
                bytes += tensor.byteSize();
        return bytes;
    }

    /// Plans residency for `artifact` on a device with `freeBytes` free, with unlimited host memory, or returns the
    /// plan that comes closest (`fits` false) when even the largest host-backed selection does not make room.
    public static Plan plan(Artifact artifact, ArtifactProfile profile, long freeBytes, int maxContextTokens) {
        return plan(artifact, profile, freeBytes, maxContextTokens, Long.MAX_VALUE, KvFormat.NVFP4);
    }

    /// Splits `plan`'s offloaded objects (its `hostBacked`, in selection order) between host memory and the
    /// artifact file. The embedding and the LM head stay in host memory (read in place); then each object goes to
    /// host memory while the arena and the window that the remaining objects would need fit in `hostBytes`, and
    /// from the first that does not, every object goes to disk.
    private static Plan tiers(Artifact artifact, Plan plan, long fixed, long hostBytes, boolean deviceFits) {
        java.util.Map<String, TensorDescriptor> tensors = new java.util.HashMap<>();
        for (TensorDescriptor tensor : artifact.tensors()) tensors.put(tensor.name(), tensor);
        java.util.List<String> order = java.util.List.copyOf(plan.hostBacked());
        long total = 0;
        for (String name : order) total += tensors.get(name).byteSize();
        if (total <= hostBytes)
            return new Plan(plan.hostBacked(), Set.of(), plan.deviceBytes() + fixed, total, 0L, deviceFits);
        // The largest object at or after each position: the window the objects from there on would need.
        long[] largestFrom = new long[order.size() + 1];
        for (int index = order.size() - 1; index >= 0; index--) {
            String name = order.get(index);
            long size = HostWeightSelection.mapped(name) ? 0 : tensors.get(name).byteSize();
            largestFrom[index] = Math.max(largestFrom[index + 1], size);
        }
        Set<String> host = new java.util.LinkedHashSet<>();
        Set<String> disk = new java.util.LinkedHashSet<>();
        // The LM head's bytes are in the arena whatever comes before it.
        long head = plan.hostBacked().contains(HostWeightSelection.OUTPUT_HEAD)
                ? tensors.get(HostWeightSelection.OUTPUT_HEAD).byteSize()
                : 0;
        long arena = 0;
        long largestDisk = 0;
        for (int index = 0; index < order.size(); index++) {
            String name = order.get(index);
            long size = tensors.get(name).byteSize();
            boolean mapped = HostWeightSelection.mapped(name);
            long reserved = mapped ? 0 : head;
            if (mapped
                    || (disk.isEmpty() && arena + size + windowBytes(largestFrom[index + 1]) + reserved <= hostBytes)) {
                host.add(name);
                arena += size;
            } else {
                disk.add(name);
                largestDisk = Math.max(largestDisk, size);
            }
        }
        long window = windowBytes(largestDisk);
        boolean fits = deviceFits && arena + window <= hostBytes;
        return new Plan(
                java.util.Collections.unmodifiableSet(host),
                java.util.Collections.unmodifiableSet(disk),
                plan.deviceBytes() + fixed,
                arena,
                window,
                fits);
    }

    /// Pinned bytes of a disk window whose largest object has `largest` bytes; 0 for none.
    private static long windowBytes(long largest) {
        return largest == 0 ? 0 : DiskWindow.SLOTS * DiskWindow.slotBytesFor(largest);
    }

    private static long artifactBytes(Artifact artifact, ArtifactProfile profile) {
        long total = 0;
        for (TensorDescriptor tensor : artifact.tensors())
            if (CompactWeightLoader.uploads(tensor.name(), profile.speculation())) total += tensor.byteSize();
        return total;
    }

    private static Plan deviceNeed(Artifact artifact, ArtifactProfile profile, Set<String> host) {
        long resident = 0;
        long largest = 0;
        for (TensorDescriptor tensor : artifact.tensors()) {
            if (!CompactWeightLoader.uploads(tensor.name(), profile.speculation())) continue;
            if (DFlash2Inventory.MAPPED.contains(tensor.name())) continue;
            boolean hostBacked = host.contains(tensor.name());
            if (hostBacked && HostWeightSelection.mapped(tensor.name())) continue;
            if (hostBacked) largest = Math.max(largest, tensor.byteSize());
            else resident += tensor.byteSize();
        }
        if (largest > 0) resident += WeightStaging.slotBytesFor(largest) * WeightStaging.SLOTS;
        return new Plan(host, resident, true);
    }
}
