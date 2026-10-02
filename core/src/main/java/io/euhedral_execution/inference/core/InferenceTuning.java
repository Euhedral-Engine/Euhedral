package io.euhedral_execution.inference.core;

import io.euhedral_execution.inference.core.gpu.Q3DispatchMode;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.WeightResidency;
import io.euhedral_execution.inference.core.scheduling.QwenGenerationSession;
import java.util.BitSet;
import java.util.Objects;

/// Immutable engine tuning for controlled experiments. Each field is an independent axis.
///
/// `workerProcessorIds` are the resolved logical processor IDs handed to Euhedral; the engine
/// validates them against [ProcessorTopology] before loading any model resource. Use
/// [WorkerProcessorSelection] to derive them from topology. The set is copied on input and output.
/// `prefillChunkTokens` bounds the prompt tokens submitted per prefill quantum and therefore the
/// per-quantum GPU workspace. It does not change the token sequence. `weightResidency` selects which
/// artifact objects are placed on the GPU. `hostWeightBytes` keeps at least that many bytes of layer
/// projections in pinned host memory, staged to the device through `stagingSlots` slots on each use
/// (see [io.euhedral_execution.inference.core.model_loader.HostWeightSelection]); it trades transfer
/// time for device memory and does not change results.
public record InferenceTuning(
        BitSet workerProcessorIds,
        int prefillChunkTokens,
        Q3DispatchMode q3DispatchMode,
        int q3SmallRowThreshold,
        WeightResidency weightResidency,
        long hostWeightBytes,
        int stagingSlots) {
    public static final int DEFAULT_PREFILL_CHUNK_TOKENS = QwenGenerationSession.DEFAULT_PREFILL_CHUNK_TOKENS;

    public InferenceTuning {
        workerProcessorIds = (BitSet)
                Objects.requireNonNull(workerProcessorIds, "workerProcessorIds").clone();
        if (workerProcessorIds.isEmpty()) throw new IllegalArgumentException("workerProcessorIds must not be empty");
        if (prefillChunkTokens <= 0) throw new IllegalArgumentException("prefillChunkTokens must be positive");
        Objects.requireNonNull(q3DispatchMode, "q3DispatchMode");
        if (q3SmallRowThreshold < 0) throw new IllegalArgumentException("Q3 threshold must not be negative");
        Objects.requireNonNull(weightResidency, "weightResidency");
        if (hostWeightBytes < 0) throw new IllegalArgumentException("hostWeightBytes must not be negative");
        if (stagingSlots < 2) throw new IllegalArgumentException("stagingSlots must be at least 2");
    }

    public InferenceTuning(
            BitSet workerProcessorIds,
            int prefillChunkTokens,
            Q3DispatchMode q3DispatchMode,
            int q3SmallRowThreshold,
            WeightResidency weightResidency) {
        this(
                workerProcessorIds,
                prefillChunkTokens,
                q3DispatchMode,
                q3SmallRowThreshold,
                weightResidency,
                0L,
                QwenModel.DEFAULT_STAGING_SLOTS);
    }

    public InferenceTuning(
            BitSet workerProcessorIds, int prefillChunkTokens, Q3DispatchMode q3DispatchMode, int q3SmallRowThreshold) {
        this(workerProcessorIds, prefillChunkTokens, q3DispatchMode, q3SmallRowThreshold, WeightResidency.ALL);
    }

    public InferenceTuning(BitSet workerProcessorIds, int prefillChunkTokens) {
        this(workerProcessorIds, prefillChunkTokens, Q3DispatchMode.AUTO, Q3DispatchMode.DEFAULT_SMALL_ROW_THRESHOLD);
    }

    /// Default tuning for the given workers: the engine's existing prefill chunk size.
    public static InferenceTuning defaults(BitSet workerProcessorIds) {
        return new InferenceTuning(workerProcessorIds, DEFAULT_PREFILL_CHUNK_TOKENS);
    }

    /// Default tuning for a resolved selection.
    public static InferenceTuning defaults(WorkerProcessorSelection workers) {
        return defaults(Objects.requireNonNull(workers, "workers").processorIds());
    }

    public InferenceTuning withWorkerProcessorIds(BitSet ids) {
        return new InferenceTuning(
                ids,
                this.prefillChunkTokens,
                this.q3DispatchMode,
                this.q3SmallRowThreshold,
                this.weightResidency,
                this.hostWeightBytes,
                this.stagingSlots);
    }

    public InferenceTuning withPrefillChunkTokens(int tokens) {
        return new InferenceTuning(
                this.workerProcessorIds,
                tokens,
                this.q3DispatchMode,
                this.q3SmallRowThreshold,
                this.weightResidency,
                this.hostWeightBytes,
                this.stagingSlots);
    }

    public InferenceTuning withQ3Dispatch(Q3DispatchMode mode, int smallRowThreshold) {
        return new InferenceTuning(
                this.workerProcessorIds,
                this.prefillChunkTokens,
                mode,
                smallRowThreshold,
                this.weightResidency,
                this.hostWeightBytes,
                this.stagingSlots);
    }

    public InferenceTuning withWeightResidency(WeightResidency residency) {
        return new InferenceTuning(
                this.workerProcessorIds,
                this.prefillChunkTokens,
                this.q3DispatchMode,
                this.q3SmallRowThreshold,
                residency,
                this.hostWeightBytes,
                this.stagingSlots);
    }

    public InferenceTuning withHostWeights(long bytes, int slots) {
        return new InferenceTuning(
                this.workerProcessorIds,
                this.prefillChunkTokens,
                this.q3DispatchMode,
                this.q3SmallRowThreshold,
                this.weightResidency,
                bytes,
                slots);
    }

    @Override
    public BitSet workerProcessorIds() {
        return (BitSet) this.workerProcessorIds.clone();
    }
}
