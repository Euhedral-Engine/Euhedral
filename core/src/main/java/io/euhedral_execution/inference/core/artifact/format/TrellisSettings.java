package io.euhedral_execution.inference.core.artifact.format;

/// The machine-tuned knobs of the trellis linear route, read once from system properties.
///
/// @param mmaMaxRows the most rows a launch other than a verification keeps on the tensor-core route
///     (`euhedral.trellis.mma-max-rows`); more rows decode the object to BF16 and multiply on the row-exact GEMM
/// @param targetCtas the thread blocks the K split of the tensor-core route aims for
///     (`euhedral.trellis.target-ctas`)
public record TrellisSettings(int mmaMaxRows, int targetCtas) {

    public static final String MMA_MAX_ROWS = "euhedral.trellis.mma-max-rows";
    public static final String TARGET_CTAS = "euhedral.trellis.target-ctas";

    private static final int DEFAULT_MMA_MAX_ROWS = 64;
    private static final int DEFAULT_TARGET_CTAS = 140;

    public TrellisSettings {
        if (mmaMaxRows < 1) throw new IllegalArgumentException(MMA_MAX_ROWS + " must be positive: " + mmaMaxRows);
        if (targetCtas < 1) throw new IllegalArgumentException(TARGET_CTAS + " must be positive: " + targetCtas);
    }

    /// The settings the system properties name, or the machine defaults.
    public static TrellisSettings fromProperties() {
        return new TrellisSettings(
                Integer.getInteger(MMA_MAX_ROWS, DEFAULT_MMA_MAX_ROWS),
                Integer.getInteger(TARGET_CTAS, DEFAULT_TARGET_CTAS));
    }
}
