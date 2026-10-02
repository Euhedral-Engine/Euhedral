package io.euhedral_execution.inference.benchmark.run;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.HostWeightSelection;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.WeightResidency;
import io.euhedral_execution.inference.core.model_loader.WeightStaging;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;

/// Refuses to load the model when the device lacks room for it, so a benchmark never competes with
/// another process for the memory that process already holds. Checked immediately before loading.
public final class GpuCapacity {
    private GpuCapacity() {}

    /// Returns null when free device memory covers the artifact plus headroom, otherwise the reason.
    public static String check(Path cudaLibrary, Path artifact, long headroomMiB) throws IOException {
        return check(cudaLibrary, artifact, headroomMiB, WeightResidency.ALL);
    }

    /// As [#check(Path, Path, long)], counting only the objects `residency` uploads.
    public static String check(Path cudaLibrary, Path artifact, long headroomMiB, WeightResidency residency)
            throws IOException {
        return check(cudaLibrary, artifact, headroomMiB, residency, 0L, QwenModel.DEFAULT_STAGING_SLOTS);
    }

    /// As [#check(Path, Path, long, WeightResidency)], with `hostWeightBytes` of weights held in host
    /// memory: the host-mapped embedding takes no device memory, and staged projections count only as
    /// the staging ring that holds `stagingSlots` of the largest.
    public static String check(
            Path cudaLibrary,
            Path artifact,
            long headroomMiB,
            WeightResidency residency,
            long hostWeightBytes,
            int stagingSlots)
            throws IOException {
        long weights;
        if (residency == WeightResidency.ALL && hostWeightBytes == 0) weights = Files.size(artifact);
        else {
            QwenArtifact read = QwenArtifactReader.read(artifact);
            Set<String> host = HostWeightSelection.select(read, hostWeightBytes);
            weights = 0;
            long largest = 0;
            for (TensorDescriptor tensor : read.tensors()) {
                if (!residency.uploads(tensor.name())) continue;
                if (tensor.name().equals(HostWeightSelection.EMBEDDING) && host.contains(tensor.name())) continue;
                if (host.contains(tensor.name())) largest = Math.max(largest, tensor.byteSize());
                else weights += tensor.byteSize();
            }
            if (largest > 0) weights += WeightStaging.slotBytesFor(largest) * stagingSlots;
        }
        long required = weights + headroomMiB * 1024L * 1024L;
        try (var gpu = new CudaGpuMemory(cudaLibrary)) {
            var info = gpu.deviceMemoryInfo();
            return evaluate(info.freeBytes(), info.totalBytes(), required);
        }
    }

    /// Returns null when `free` covers `required`, otherwise a human-readable refusal.
    static String evaluate(long free, long total, long required) {
        if (free >= required) return null;
        return String.format(
                java.util.Locale.ROOT,
                "GPU has %.1f GiB free of %.1f GiB; loading needs %.1f GiB (artifact plus gpuHeadroomMiB)",
                free / 1073741824.0,
                total / 1073741824.0,
                required / 1073741824.0);
    }
}
