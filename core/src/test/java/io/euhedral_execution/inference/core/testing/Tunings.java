package io.euhedral_execution.inference.core.testing;

import io.euhedral_execution.inference.core.settings.BatchingSettings;
import io.euhedral_execution.inference.core.settings.HostSettings;
import io.euhedral_execution.inference.core.settings.MemorySettings;
import io.euhedral_execution.inference.core.settings.Tuning;
import java.time.Duration;
import java.util.OptionalLong;

/// The default tuning with one setting changed, as tests state it.
public final class Tunings {

    private Tunings() {}

    /// A workspace of `rows` rows (`batching.prefill-chunk-tokens`).
    public static Tuning prefillChunkTokens(int rows) {
        Tuning defaults = Tuning.defaults();
        BatchingSettings batching = defaults.batching();
        return new Tuning(
                new BatchingSettings(
                        batching.maxSequences(), batching.maxSequencesPerPass(), batching.maxTokensPerPass(), rows),
                defaults.memory(),
                defaults.host(),
                defaults.kernels());
    }

    /// `memory.gpu-limit` and `memory.host-weights-limit`, each empty for automatic.
    public static Tuning limits(OptionalLong gpuLimit, OptionalLong hostWeightsLimit) {
        Tuning defaults = Tuning.defaults();
        MemorySettings memory = defaults.memory();
        return new Tuning(
                defaults.batching(),
                new MemorySettings(
                        memory.hostLimit(),
                        hostWeightsLimit,
                        gpuLimit,
                        memory.gpuReserve(),
                        memory.pinnedBufferPool(),
                        memory.weightStreaming()),
                defaults.host(),
                defaults.kernels());
    }

    /// `memory.host-weights-limit` of `bytes`.
    public static Tuning hostWeightsLimit(long bytes) {
        return limits(OptionalLong.empty(), OptionalLong.of(bytes));
    }

    /// `host.idle-release` of `idle`.
    public static Tuning idleRelease(Duration idle) {
        Tuning defaults = Tuning.defaults();
        HostSettings host = defaults.host();
        return new Tuning(
                defaults.batching(),
                defaults.memory(),
                new HostSettings(host.cpus(), host.taskParallelism(), idle, host.nativeCache()),
                defaults.kernels());
    }
}
