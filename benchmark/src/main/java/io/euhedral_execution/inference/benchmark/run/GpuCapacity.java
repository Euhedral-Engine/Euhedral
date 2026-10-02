package io.euhedral_execution.inference.benchmark.run;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.WeightResidency;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

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
        long weights = residency == WeightResidency.ALL
                ? Files.size(artifact)
                : residency.residentBytes(QwenArtifactReader.read(artifact));
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
