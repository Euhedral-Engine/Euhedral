package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.settings.TrellisKernelSettings;

/// The thresholds of the trellis linear route on one device ([#of]: `kernels.trellis` of the settings).
///
/// The defaults ([TrellisKernelSettings#defaults]) are measured on an RTX 5070 Ti (70 streaming multiprocessors):
///
/// - 512 rows on the tensor-core route: with Swift-1.5 at 3 bits it prefills 512-row chunks at 2773 tokens/s, the
///   prefill route at 490, so the tensor-core route serves every chunk the default workspace holds.
/// - 8 target blocks per multiprocessor (560): over the 3-bit model's object shapes, weights rotated past L2, the
///   trellis linears of one decode step take 14.41, 13.20, 13.01, 13.07 and 13.20 ms at 140, 280, 560, 840 and 1120
///   target blocks.
/// - whole-K blocks from 64 rows: the trellis linears of one step with split partials and with whole-K blocks of 32
///   rows take 48.6 and 59.4 ms at 48 rows, 56.5 and 64.9 ms at 56, 64.4 and 60.4 ms at 64.
/// - 64-row blocks from 128 rows: whole-K blocks of 32 and of 64 rows take 74.4 and 77.6 ms at 80 rows, 77.6 and
///   83.7 ms at 96, 102.9 and 101.7 ms at 128, 364.2 and 319.1 ms at 512.
///
/// @param mmaMaxRows the most rows a launch other than a verification keeps on the tensor-core route
///     (`kernels.trellis.tensor-core-max-tokens`); more rows decode the object to BF16 and multiply on the row-exact
///     GEMM
/// @param targetCtas the thread blocks the K split of the tensor-core route aims for
///     (`kernels.trellis.target-blocks`)
/// @param wholeKRows the fewest rows for which each block of the tensor-core route runs every K split of its columns
///     itself, with no split partials (`kernels.trellis.whole-k-min-tokens`): a verification's results are bitwise
///     the same, other launches accumulate runs of k16 tiles in FP16
/// @param wholeKWideRows the fewest rows for which those launches use blocks of 64 rows instead of 32
///     (`kernels.trellis.wide-block-min-tokens`); the results are bitwise the same
public record TrellisSettings(int mmaMaxRows, int targetCtas, int wholeKRows, int wholeKWideRows) {

    /// Decode-type quanta (decode, draft) have up to eight rows and never take the prefill route.
    public static final int MIN_MMA_ROWS = 8;

    /// Target blocks per streaming multiprocessor when the settings state none.
    static final int BLOCKS_PER_MULTIPROCESSOR = 8;

    /// The multiprocessors of the card the defaults were measured on, for a device that reports none.
    public static final int DEFAULT_MULTIPROCESSORS = 70;

    public TrellisSettings {
        if (mmaMaxRows < MIN_MMA_ROWS)
            throw new IllegalArgumentException("kernels.trellis.tensor-core-max-tokens must be at least " + MIN_MMA_ROWS
                    + ", the rows of a decode-type quantum: " + mmaMaxRows);
        if (targetCtas < 1)
            throw new IllegalArgumentException("kernels.trellis.target-blocks must be positive: " + targetCtas);
        if (wholeKRows < 1)
            throw new IllegalArgumentException("kernels.trellis.whole-k-min-tokens must be positive: " + wholeKRows);
        if (wholeKWideRows < 1)
            throw new IllegalArgumentException(
                    "kernels.trellis.wide-block-min-tokens must be positive: " + wholeKWideRows);
    }

    /// The thresholds `settings` state on a device of `multiprocessors` streaming multiprocessors: the target blocks
    /// are 8 per multiprocessor unless stated.
    public static TrellisSettings of(TrellisKernelSettings settings, int multiprocessors) {
        if (multiprocessors < 1) throw new IllegalArgumentException("multiprocessors must be positive");
        return new TrellisSettings(
                settings.tensorCoreMaxTokens(),
                settings.targetBlocks().orElse(BLOCKS_PER_MULTIPROCESSOR * multiprocessors),
                settings.wholeKMinTokens(),
                settings.wideBlockMinTokens());
    }

    /// The default thresholds on the card they were measured on.
    public static TrellisSettings defaults() {
        return of(TrellisKernelSettings.defaults(), DEFAULT_MULTIPROCESSORS);
    }
}
