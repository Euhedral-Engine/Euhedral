package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.loader.HostWeightSelection;
import io.euhedral_execution.inference.core.model.qwen38.loader.ResidencyPlanner;
import io.euhedral_execution.inference.core.model.qwen38.loader.WeightLoader;
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

    /// The device bytes the plan for one sequence of `tokens` tokens needs, and what each sequence beyond the first
    /// adds (its KV and [#SEQUENCE_BYTES]), when the device has `gpu`'s free bytes; the first is -1 when even
    /// host-backed weights leave one sequence no room.
    static long[] sequenceFigures(
            Artifact artifact, ArtifactProfile.Speculation speculation, CudaGpuMemory gpu, int tokens) {
        return sequenceFigures(artifact, speculation, gpu.deviceMemoryInfo().freeBytes(), tokens);
    }

    /// As above, for a device with `freeBytes` free.
    static long[] sequenceFigures(
            Artifact artifact, ArtifactProfile.Speculation speculation, long freeBytes, int tokens) {
        ArtifactProfile found = ArtifactProfile.of(artifact);
        var profile = new ArtifactProfile(found.quantization(), found.compressed(), speculation, found.draftDepth());
        KvFormat format = kvFormat(artifact);
        var single = ResidencyPlanner.plan(artifact, profile, freeBytes, tokens, Long.MAX_VALUE, format);
        return new long[] {
            single.fits() ? single.deviceBytes() : -1,
            ResidencyPlanner.kvBytes(artifact.config(), tokens, format) + SEQUENCE_BYTES
        };
    }

    /// The device bytes a candidate adds beyond its weights when all of them are resident: `sequences` sequences'
    /// KV, the first one's state, the workspace and the reserve ([ResidencyPlanner]'s figures), and
    /// [#SEQUENCE_BYTES] for each sequence beyond the first. The weights of the objects `speculation` loads are
    /// counted by the caller, as the cache holds them.
    static long otherBytes(Artifact artifact, ArtifactProfile.Speculation speculation, int sequences, int tokens) {
        var plan = resident(artifact, speculation, 1L << 50, sequences, tokens);
        long weights = 0;
        for (var tensor : artifact.tensors())
            if (WeightLoader.uploads(tensor.name(), speculation)) weights += tensor.byteSize();
        return Math.max(0, plan.deviceBytes() - weights) + (sequences - 1L) * SEQUENCE_BYTES;
    }

    /// Whether `sequences` sequences of up to `tokens` tokens fit in `free` bytes with every weight resident.
    static boolean fitsResident(
            Artifact artifact, ArtifactProfile.Speculation speculation, long free, int sequences, int tokens) {
        var plan = resident(artifact, speculation, free, sequences, tokens);
        return plan.fits() && plan.hostBacked().isEmpty();
    }

    private static ResidencyPlanner.Plan resident(
            Artifact artifact, ArtifactProfile.Speculation speculation, long free, int sequences, int tokens) {
        ArtifactProfile found = ArtifactProfile.of(artifact);
        var profile = new ArtifactProfile(found.quantization(), found.compressed(), speculation);
        int context = Math.toIntExact(Math.min(Integer.MAX_VALUE, (long) sequences * tokens));
        return ResidencyPlanner.plan(
                artifact,
                profile,
                free - (sequences - 1L) * SEQUENCE_BYTES,
                context,
                Long.MAX_VALUE,
                kvFormat(artifact));
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
        return hostBacked(artifact, speculation, gpu.deviceMemoryInfo().freeBytes(), hostBytes, sequences, tokens);
    }

    /// As above, for a device with `free` bytes free: a process that scores one artifact after another plans each
    /// against what the device had before the first, since the memory the first left in use (kernel modules, the
    /// runtime's per-process state) is what the planner's reserve already stands for.
    public static Set<String> hostBacked(
            Artifact artifact,
            ArtifactProfile.Speculation speculation,
            long free,
            long hostBytes,
            int sequences,
            int tokens) {
        if (hostBytes != PLANNED) return HostWeightSelection.select(artifact, hostBytes);
        ArtifactProfile found = ArtifactProfile.of(artifact);
        var profile = new ArtifactProfile(found.quantization(), found.compressed(), speculation);
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

    /// The residency of `artifact` with at most `hostBytes` of weights in host memory ([#PLANNED]: no limit):
    /// the planner's host-backed objects, and, when they do not all fit in host memory, the objects left in the
    /// artifact file and read from disk.
    public static ResidencyPlanner.Plan plan(
            Artifact artifact,
            ArtifactProfile.Speculation speculation,
            CudaGpuMemory gpu,
            long hostBytes,
            int sequences,
            int tokens) {
        ArtifactProfile found = ArtifactProfile.of(artifact);
        var profile = new ArtifactProfile(found.quantization(), found.compressed(), speculation, found.draftDepth());
        long free = gpu.deviceMemoryInfo().freeBytes();
        int context = Math.toIntExact(Math.min(Integer.MAX_VALUE, (long) sequences * tokens));
        var plan = ResidencyPlanner.plan(
                artifact,
                profile,
                free - (sequences - 1) * SEQUENCE_BYTES,
                context,
                hostBytes == PLANNED ? Long.MAX_VALUE : hostBytes,
                kvFormat(artifact));
        if (!plan.fits())
            throw new IllegalStateException(sequences + " sequences of " + tokens + " tokens do not fit in the "
                    + (free >> 20) + " MiB free on the GPU; lower --batch");
        long host = 0, disk = 0;
        for (var tensor : artifact.tensors()) {
            if (plan.hostBacked().contains(tensor.name())) host += tensor.byteSize();
            if (plan.diskBacked().contains(tensor.name())) disk += tensor.byteSize();
        }
        System.out.printf(
                Locale.ROOT,
                "residency: %d MiB of the %d MiB free on the GPU, %d MiB of weights in host memory, %d MiB read from"
                        + " disk%n",
                plan.deviceBytes() >> 20,
                free >> 20,
                host >> 20,
                disk >> 20);
        return plan;
    }
}
