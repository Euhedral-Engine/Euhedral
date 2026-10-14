package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightStaging;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import io.euhedral_execution.inference.core.state.KvFormat;
import java.util.Set;

/// Decides which weights stay on the device so that a context of `maxContextTokens` fits in the memory
/// the device has free. Nothing is host-backed when everything fits; otherwise the smallest selection
/// of [HostWeightSelection] that makes room is taken, because every host-backed byte costs decode time.
///
/// The device budget is the resident weights, the staging ring for host-backed projections, the KV pages
/// of one sequence of `maxContextTokens` and a fixed reserve for the per-sequence GDN state, the
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

    /// Plans residency with at most `hostBytes` of pinned host memory and a KV cache in `kvFormat`: objects
    /// beyond what the device and that host memory hold are left in the artifact file. The disk tier is planned
    /// by the offload work; until then, nothing is disk-backed and the host budget is not enforced.
    public static Plan plan(
            Artifact artifact,
            ArtifactProfile profile,
            long freeBytes,
            int maxContextTokens,
            long hostBytes,
            KvFormat kvFormat) {
        return plan(artifact, profile, freeBytes, maxContextTokens, kvFormat);
    }

    /// Plans residency for `artifact` on a device with `freeBytes` free, or returns the plan that comes
    /// closest (`fits` false) when even the largest host-backed selection does not make room.
    public static Plan plan(Artifact artifact, ArtifactProfile profile, long freeBytes, int maxContextTokens) {
        return plan(artifact, profile, freeBytes, maxContextTokens, KvFormat.NVFP4);
    }

    private static Plan plan(
            Artifact artifact, ArtifactProfile profile, long freeBytes, int maxContextTokens, KvFormat kvFormat) {
        long fixed = kvBytes(artifact.config(), maxContextTokens, kvFormat) + RESERVE_BYTES;
        Plan resident = deviceNeed(artifact, profile, Set.of());
        if (resident.deviceBytes() + fixed <= freeBytes) return withFixed(artifact, resident, fixed, true);
        long all = artifactBytes(artifact, profile);
        // Grow the host-backed bytes until the device need fits. The ring grows with the largest selected
        // tensor, so each step re-evaluates the whole selection.
        long step = 64L << 20;
        for (long bytes = step; ; bytes += step) {
            Plan candidate = deviceNeed(artifact, profile, HostWeightSelection.select(artifact, bytes));
            if (candidate.deviceBytes() + fixed <= freeBytes) return withFixed(artifact, candidate, fixed, true);
            if (bytes > all) return withFixed(artifact, candidate, fixed, false);
        }
    }

    private static Plan withFixed(Artifact artifact, Plan plan, long fixed, boolean fits) {
        long host = 0;
        for (TensorDescriptor tensor : artifact.tensors())
            if (plan.hostBacked().contains(tensor.name())) host += tensor.byteSize();
        return new Plan(plan.hostBacked(), Set.of(), plan.deviceBytes() + fixed, host, 0L, fits);
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
