package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightStaging;
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
/// execution workspaces, the shared scratch and the kernel modules loaded after the model.
public final class ResidencyPlanner {

    /// Everything outside weights and KV that a running sequence needs on the device.
    static final long RESERVE_BYTES = 1280L << 20;

    private ResidencyPlanner() {}

    /// The outcome: the objects to keep in host memory and those left in the artifact file, the device bytes the
    /// plan needs, the pinned host bytes of the host-backed objects (`hostArenaBytes`) and of the disk window
    /// (`windowBytes`, 0 without disk-backed objects), and whether it fits.
    public record Plan(
            Set<String> hostBacked,
            Set<String> diskBacked,
            long deviceBytes,
            long hostArenaBytes,
            long windowBytes,
            boolean fits) {

        public Plan(Set<String> hostBacked, long deviceBytes, boolean fits) {
            this(hostBacked, Set.of(), deviceBytes, 0L, 0L, fits);
        }
    }

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
        if (sequences < 1) throw new IllegalArgumentException("sequences must be at least 1");
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
        Plan resident = deviceNeed(artifact, profile, Set.of());
        if (resident.deviceBytes() + fixed <= freeBytes) return tiers(artifact, resident, fixed, hostBytes, true);
        long all = artifactBytes(artifact, profile);
        // Grow the offloaded bytes until the device need fits. The ring grows with the largest selected
        // tensor, so each step re-evaluates the whole selection.
        long step = 64L << 20;
        for (long bytes = step; ; bytes += step) {
            Plan candidate = deviceNeed(artifact, profile, HostWeightSelection.select(artifact, bytes));
            if (candidate.deviceBytes() + fixed <= freeBytes) return tiers(artifact, candidate, fixed, hostBytes, true);
            if (bytes > all) return tiers(artifact, candidate, fixed, hostBytes, false);
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
        Plan single = plan(artifact, profile, freeBytes, maxContextTokens, hostBytes, kvFormat, 1);
        if (!single.fits()) return 1;
        long offloaded = offloadedBytes(artifact, single);
        int low = 1;
        int high = cap;
        // Fitting and the offloaded bytes only worsen with the count: search for the largest that is no worse.
        while (low < high) {
            int middle = (low + high + 1) >>> 1;
            Plan candidate = plan(artifact, profile, freeBytes, maxContextTokens, hostBytes, kvFormat, middle);
            if (candidate.fits() && offloadedBytes(artifact, candidate) <= offloaded) low = middle;
            else high = middle - 1;
        }
        return low;
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
    /// artifact file. The embedding stays in host memory; then each object goes to host memory while the arena and
    /// the window that the remaining objects would need fit in `hostBytes`, and from the first that does not, every
    /// object goes to disk.
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
            long size = name.equals(HostWeightSelection.EMBEDDING)
                    ? 0
                    : tensors.get(name).byteSize();
            largestFrom[index] = Math.max(largestFrom[index + 1], size);
        }
        Set<String> host = new java.util.LinkedHashSet<>();
        Set<String> disk = new java.util.LinkedHashSet<>();
        long arena = 0;
        long largestDisk = 0;
        for (int index = 0; index < order.size(); index++) {
            String name = order.get(index);
            long size = tensors.get(name).byteSize();
            boolean embedding = name.equals(HostWeightSelection.EMBEDDING);
            if (disk.isEmpty() && (embedding || arena + size + windowBytes(largestFrom[index + 1]) <= hostBytes)) {
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
            if (hostBacked && tensor.name().equals(HostWeightSelection.EMBEDDING)) continue;
            if (hostBacked) largest = Math.max(largest, tensor.byteSize());
            else resident += tensor.byteSize();
        }
        if (largest > 0) resident += WeightStaging.slotBytesFor(largest) * WeightStaging.SLOTS;
        return new Plan(host, resident, true);
    }
}
