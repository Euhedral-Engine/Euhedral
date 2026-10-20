package io.euhedral_execution.inference.core.settings;

import io.euhedral_execution.inference.core.artifact.format.TrellisSettings;
import java.util.Objects;
import java.util.OptionalInt;

/// Thresholds of the trellis linear kernels.
///
/// @param tensorCoreMaxTokens most rows of a launch other than a verification that stay on the tensor-core route; at
///     least {@link TrellisSettings#MIN_MMA_ROWS}, the rows of a decode step
/// @param targetBlocks thread blocks the K split aims for, or empty for 8 per streaming multiprocessor
/// @param wholeKMinTokens fewest rows for blocks that run every K split of their columns themselves
/// @param wideBlockMinTokens fewest rows for blocks of 64 rows instead of 32
public record TrellisKernelSettings(
        int tensorCoreMaxTokens, OptionalInt targetBlocks, int wholeKMinTokens, int wideBlockMinTokens) {

    public TrellisKernelSettings {
        if (tensorCoreMaxTokens < TrellisSettings.MIN_MMA_ROWS)
            throw new IllegalArgumentException(
                    "tensor-core-max-tokens must be at least " + TrellisSettings.MIN_MMA_ROWS);
        Objects.requireNonNull(targetBlocks, "targetBlocks");
        if (targetBlocks.isPresent() && targetBlocks.getAsInt() < 1)
            throw new IllegalArgumentException("target-blocks must be positive");
        if (wholeKMinTokens < 1) throw new IllegalArgumentException("whole-k-min-tokens must be positive");
        if (wideBlockMinTokens < 1) throw new IllegalArgumentException("wide-block-min-tokens must be positive");
    }

    public static TrellisKernelSettings defaults() {
        return new TrellisKernelSettings(512, OptionalInt.empty(), 64, 128);
    }
}
