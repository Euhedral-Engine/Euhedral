package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import java.util.Set;

/// Decides which weights stay on the device so that a context of `maxContextTokens` fits in the memory
/// the device has free. Nothing is host-backed when everything fits; otherwise the smallest selection
/// of [HostWeightSelection] that makes room is taken, because every host-backed byte costs decode time.
///
/// The device budget is the resident weights, the staging ring for host-backed projections, the KV pages
/// of one sequence of `maxContextTokens` and a fixed reserve for the per-sequence GDN state, the
/// execution workspaces, the shared scratch and the kernel modules loaded after the model.
public final class ResidencyPlanner {

    /// Bytes of KV cache per token of one NVFP4 attention layer: K and V rows of 144 bytes per 256 values.
    private static final long KV_BYTES_PER_LAYER_TOKEN = 2L * 144;

    /// Everything outside weights and KV that a running sequence needs on the device.
    static final long RESERVE_BYTES = 1280L << 20;

    private ResidencyPlanner() {}

    /// The outcome: the objects to keep in host memory, the device bytes the plan needs, and whether it fits.
    public record Plan(Set<String> hostBacked, long deviceBytes, boolean fits) {}

    /// KV bytes of one sequence of `tokens` positions, in whole pages.
    public static long kvBytes(QwenConfig config, int tokens) {
        long layers = 0;
        for (QwenLayerType type : config.layerTypes()) if (type == QwenLayerType.FULL_ATTENTION) layers++;
        long pages = (tokens + 255L) / 256L;
        long widthUnits = (long) config.numKeyValueHeads() * config.attentionHeadDim() / 256L;
        return layers * pages * 256L * KV_BYTES_PER_LAYER_TOKEN * widthUnits;
    }

    /// Plans residency for `artifact` on a device with `freeBytes` free, or returns the plan that comes
    /// closest (`fits` false) when even the largest host-backed selection does not make room.
    public static Plan plan(QwenArtifact artifact, ArtifactProfile profile, long freeBytes, int maxContextTokens) {
        long fixed = kvBytes(artifact.config(), maxContextTokens) + RESERVE_BYTES;
        Plan resident = deviceNeed(artifact, profile, Set.of());
        if (resident.deviceBytes() + fixed <= freeBytes) return withFixed(resident, fixed, true);
        long all = artifactBytes(artifact, profile);
        // Grow the host-backed bytes until the device need fits. The ring grows with the largest selected
        // tensor, so each step re-evaluates the whole selection.
        long step = 64L << 20;
        for (long bytes = step; ; bytes += step) {
            Plan candidate = deviceNeed(artifact, profile, HostWeightSelection.select(artifact, bytes));
            if (candidate.deviceBytes() + fixed <= freeBytes) return withFixed(candidate, fixed, true);
            if (bytes > all) return withFixed(candidate, fixed, false);
        }
    }

    private static Plan withFixed(Plan plan, long fixed, boolean fits) {
        return new Plan(plan.hostBacked(), plan.deviceBytes() + fixed, fits);
    }

    private static long artifactBytes(QwenArtifact artifact, ArtifactProfile profile) {
        long total = 0;
        for (TensorDescriptor tensor : artifact.tensors())
            if (QwenCompactWeightLoader.uploads(tensor.name(), profile.speculation())) total += tensor.byteSize();
        return total;
    }

    private static Plan deviceNeed(QwenArtifact artifact, ArtifactProfile profile, Set<String> host) {
        long resident = 0;
        long largest = 0;
        for (TensorDescriptor tensor : artifact.tensors()) {
            if (!QwenCompactWeightLoader.uploads(tensor.name(), profile.speculation())) continue;
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
