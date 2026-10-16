package io.euhedral_execution.inference.core.artifact.format;

/// The machine-tuned knobs of the trellis linear route, read once from system properties.
///
/// @param mmaMaxRows the most rows a launch other than a verification keeps on the tensor-core route
///     (`euhedral.trellis.mma-max-rows`); more rows decode the object to BF16 and multiply on the row-exact GEMM
/// @param targetCtas the thread blocks the K split of the tensor-core route aims for
///     (`euhedral.trellis.target-ctas`)
/// @param wholeKRows the fewest rows for which each block of the tensor-core route runs every K split of its columns
///     itself, with no split partials (`euhedral.trellis.whole-k-rows`); the results are bitwise the same
public record TrellisSettings(int mmaMaxRows, int targetCtas, int wholeKRows) {

    public static final String MMA_MAX_ROWS = "euhedral.trellis.mma-max-rows";
    public static final String TARGET_CTAS = "euhedral.trellis.target-ctas";
    public static final String WHOLE_K_ROWS = "euhedral.trellis.whole-k-rows";

    /// Decode-type quanta (decode, draft) have up to eight rows and never take the prefill route.
    private static final int MIN_MMA_ROWS = 8;

    /// Measured on an RTX 5070 Ti with Swift-1.5 at 3 bits: the tensor-core route prefills 512-row chunks at 996
    /// tokens/s, the prefill route at 490, so the tensor-core route serves every chunk the default workspace holds.
    private static final int DEFAULT_MMA_MAX_ROWS = 512;
    /// Measured on an RTX 5070 Ti over the 3-bit model's object shapes, weights rotated past L2: the trellis linears
    /// of one decode step take 14.41, 13.20, 13.01, 13.07 and 13.20 ms at 140, 280, 560, 840 and 1120 target
    /// blocks.
    private static final int DEFAULT_TARGET_CTAS = 560;
    /// Measured on an RTX 5070 Ti over the 3-bit model's object shapes, the trellis linears of one step with split
    /// partials and with whole-K blocks: 64.5 and 67.8 ms at 64 rows, 83.0 and 78.6 ms at 80, 108.4 and 89.7 ms at
    /// 96.
    private static final int DEFAULT_WHOLE_K_ROWS = 80;

    public TrellisSettings {
        if (mmaMaxRows < MIN_MMA_ROWS)
            throw new IllegalArgumentException(MMA_MAX_ROWS + " must be at least " + MIN_MMA_ROWS
                    + ", the rows of a decode-type quantum: " + mmaMaxRows);
        if (targetCtas < 1) throw new IllegalArgumentException(TARGET_CTAS + " must be positive: " + targetCtas);
        if (wholeKRows < 1) throw new IllegalArgumentException(WHOLE_K_ROWS + " must be positive: " + wholeKRows);
    }

    /// The settings the system properties name, or the machine defaults.
    public static TrellisSettings fromProperties() {
        return new TrellisSettings(
                Integer.getInteger(MMA_MAX_ROWS, DEFAULT_MMA_MAX_ROWS),
                Integer.getInteger(TARGET_CTAS, DEFAULT_TARGET_CTAS),
                Integer.getInteger(WHOLE_K_ROWS, DEFAULT_WHOLE_K_ROWS));
    }
}
