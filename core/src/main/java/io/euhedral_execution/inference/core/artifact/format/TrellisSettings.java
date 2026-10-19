package io.euhedral_execution.inference.core.artifact.format;

/// The machine-tuned knobs of the trellis linear route, read once from system properties.
///
/// @param mmaMaxRows the most rows a launch other than a verification keeps on the tensor-core route
///     (`euhedral.trellis.mma-max-rows`); more rows decode the object to BF16 and multiply on the row-exact GEMM
/// @param targetCtas the thread blocks the K split of the tensor-core route aims for
///     (`euhedral.trellis.target-ctas`)
/// @param wholeKRows the fewest rows for which each block of the tensor-core route runs every K split of its columns
///     itself, with no split partials (`euhedral.trellis.whole-k-rows`): a verification's results are bitwise the
///     same, other launches accumulate runs of k16 tiles in FP16
/// @param wholeKWideRows the fewest rows for which those launches use blocks of 64 rows instead of 32
///     (`euhedral.trellis.whole-k-wide-rows`); the results are bitwise the same
public record TrellisSettings(int mmaMaxRows, int targetCtas, int wholeKRows, int wholeKWideRows) {

    public static final String MMA_MAX_ROWS = "euhedral.trellis.mma-max-rows";
    public static final String TARGET_CTAS = "euhedral.trellis.target-ctas";
    public static final String WHOLE_K_ROWS = "euhedral.trellis.whole-k-rows";
    public static final String WHOLE_K_WIDE_ROWS = "euhedral.trellis.whole-k-wide-rows";

    /// Decode-type quanta (decode, draft) have up to eight rows and never take the prefill route.
    private static final int MIN_MMA_ROWS = 8;

    /// Measured on an RTX 5070 Ti with Swift-1.5 at 3 bits: the tensor-core route prefills 512-row chunks at 1372
    /// tokens/s, the prefill route at 490, so the tensor-core route serves every chunk the default workspace holds.
    private static final int DEFAULT_MMA_MAX_ROWS = 512;
    /// Measured on an RTX 5070 Ti over the 3-bit model's object shapes, weights rotated past L2: the trellis linears
    /// of one decode step take 14.41, 13.20, 13.01, 13.07 and 13.20 ms at 140, 280, 560, 840 and 1120 target
    /// blocks.
    private static final int DEFAULT_TARGET_CTAS = 560;
    /// Measured on an RTX 5070 Ti over the 3-bit model's object shapes, the trellis linears of one step with split
    /// partials and with whole-K blocks of 32 rows: 48.6 and 59.4 ms at 48 rows, 56.5 and 64.9 ms at 56, 64.4 and
    /// 60.4 ms at 64.
    private static final int DEFAULT_WHOLE_K_ROWS = 64;
    /// Measured on the same card and shapes, whole-K blocks of 32 and of 64 rows: 74.4 and 77.6 ms at 80 rows, 77.6
    /// and 83.7 ms at 96, 102.9 and 101.7 ms at 128, 364.2 and 319.1 ms at 512.
    private static final int DEFAULT_WHOLE_K_WIDE_ROWS = 128;

    public TrellisSettings {
        if (mmaMaxRows < MIN_MMA_ROWS)
            throw new IllegalArgumentException(MMA_MAX_ROWS + " must be at least " + MIN_MMA_ROWS
                    + ", the rows of a decode-type quantum: " + mmaMaxRows);
        if (targetCtas < 1) throw new IllegalArgumentException(TARGET_CTAS + " must be positive: " + targetCtas);
        if (wholeKRows < 1) throw new IllegalArgumentException(WHOLE_K_ROWS + " must be positive: " + wholeKRows);
        if (wholeKWideRows < 1)
            throw new IllegalArgumentException(WHOLE_K_WIDE_ROWS + " must be positive: " + wholeKWideRows);
    }

    /// The settings the system properties name, or the machine defaults.
    public static TrellisSettings fromProperties() {
        return new TrellisSettings(
                Integer.getInteger(MMA_MAX_ROWS, DEFAULT_MMA_MAX_ROWS),
                Integer.getInteger(TARGET_CTAS, DEFAULT_TARGET_CTAS),
                Integer.getInteger(WHOLE_K_ROWS, DEFAULT_WHOLE_K_ROWS),
                Integer.getInteger(WHOLE_K_WIDE_ROWS, DEFAULT_WHOLE_K_WIDE_ROWS));
    }
}
