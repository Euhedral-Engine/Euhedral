package io.euhedral_execution.inference.core.settings;

/// Streaming of weights from disk, for models larger than the memory they run in.
///
/// @param concurrentReads artifact records read at once
/// @param readSplit parts one record read is split into
/// @param copyStreams CUDA streams copying expert weights to the GPU
/// @param gpuStreams GPU streams a step runs on
public record WeightStreamingSettings(int concurrentReads, int readSplit, int copyStreams, int gpuStreams) {

    public WeightStreamingSettings {
        if (concurrentReads < 1) throw new IllegalArgumentException("concurrent-reads must be positive");
        if (readSplit < 1) throw new IllegalArgumentException("read-split must be positive");
        if (copyStreams < 1) throw new IllegalArgumentException("copy-streams must be positive");
        if (gpuStreams < 1) throw new IllegalArgumentException("gpu-streams must be positive");
    }

    public static WeightStreamingSettings defaults() {
        return new WeightStreamingSettings(8, 1, 4, 2);
    }
}
