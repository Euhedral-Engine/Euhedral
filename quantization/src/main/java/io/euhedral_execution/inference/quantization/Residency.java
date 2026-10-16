package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.loader.HostWeightSelection;
import io.euhedral_execution.inference.core.model.qwen38.loader.ResidencyPlanner;
import io.euhedral_execution.inference.core.state.KvFormat;
import java.util.Locale;
import java.util.Set;

/// Which weights a command keeps in host memory: as the engine plans it ([ResidencyPlanner]), none when the
/// artifact, the KV of every sequence the command runs at once, their state and the reserve fit in the
/// memory free on the device, otherwise the smallest selection that makes room. An explicit `--host-mib`
/// overrides the plan.
public final class Residency {

    /// `--host-mib` left unset.
    public static final long PLANNED = -1;

    /// Device memory of one sequence beyond the first, which the engine's reserve already holds: its GDN state
    /// (144 MiB), its rows in the batched quanta's workspaces and its drafter state. Measured: `generate` peaks
    /// 1986 MiB higher with 8 sequences than with 2.
    private static final long SEQUENCE_BYTES = 336L << 20;

    private Residency() {}

    /// The attention cache format the engine gives `artifact`'s sequences: its profile's.
    public static KvFormat kvFormat(Artifact artifact) {
        return ArtifactProfile.kvFormat(null, ArtifactProfile.of(artifact));
    }

    /// The host-backed objects for `sequences` sequences of up to `tokens` tokens each, or the `hostBytes`
    /// given.
    public static Set<String> hostBacked(
            Artifact artifact,
            ArtifactProfile.Speculation speculation,
            CudaGpuMemory gpu,
            long hostBytes,
            int sequences,
            int tokens) {
        if (hostBytes != PLANNED) return HostWeightSelection.select(artifact, hostBytes);
        ArtifactProfile found = ArtifactProfile.of(artifact);
        var profile = new ArtifactProfile(found.quantization(), found.compressed(), speculation);
        long free = gpu.deviceMemoryInfo().freeBytes();
        int context = Math.toIntExact(Math.min(Integer.MAX_VALUE, (long) sequences * tokens));
        var plan = ResidencyPlanner.plan(
                artifact,
                profile,
                free - (sequences - 1) * SEQUENCE_BYTES,
                context,
                Long.MAX_VALUE,
                kvFormat(artifact));
        if (!plan.fits())
            throw new IllegalStateException(sequences + " sequences of " + tokens + " tokens do not fit in the "
                    + (free >> 20) + " MiB free on the GPU, even with the weights in host memory; lower --batch");
        long host = 0;
        for (var tensor : artifact.tensors()) if (plan.hostBacked().contains(tensor.name())) host += tensor.byteSize();
        System.out.printf(
                Locale.ROOT,
                "residency: %d MiB of the %d MiB free on the GPU, %d MiB of weights in host memory%n",
                plan.deviceBytes() >> 20,
                free >> 20,
                host >> 20);
        return plan.hostBacked();
    }
}
