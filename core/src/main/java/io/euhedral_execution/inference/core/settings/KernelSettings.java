package io.euhedral_execution.inference.core.settings;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;

/// Kernel thresholds.
///
/// @param trellis the trellis linear kernels
/// @param rmsNormDecodeSlices slices of the output pass of a one-row RMS norm, or empty for {@value
///     #DEFAULT_RMS_NORM_DECODE_SLICES} (the value chosen with the sliced kernel, used on every card)
/// @param fusedVerifyFrom the fewest keys at which the fused verifier kernel serves each of 0 to 8 verified rows
///     ({@value #NEVER} is never), or empty for {@link #DEFAULT_FUSED_VERIFY_FROM}
public record KernelSettings(
        TrellisKernelSettings trellis, OptionalInt rmsNormDecodeSlices, Optional<List<Long>> fusedVerifyFrom) {

    public static final int DEFAULT_RMS_NORM_DECODE_SLICES = 40;

    /// The most slices: they become the y extent of a launch grid, which the device limits to 65535.
    public static final int MAX_RMS_NORM_DECODE_SLICES = 65535;

    /// The key count that never starts the fused verifier kernel.
    public static final long NEVER = 0xFFFFFFFFL;

    public static final int FUSED_VERIFY_ROWS = 9;

    /// Measured on an RTX 5070 Ti (6 query heads per KV head): rows that fill their 8-column mma tiles win from 2048
    /// keys, the others only on longer caches.
    public static final List<Long> DEFAULT_FUSED_VERIFY_FROM =
            List.of(NEVER, NEVER, 49152L, 16384L, 4096L, 2048L, 16384L, 16384L, 12288L);

    public KernelSettings {
        Objects.requireNonNull(trellis, "trellis");
        Objects.requireNonNull(rmsNormDecodeSlices, "rmsNormDecodeSlices");
        if (rmsNormDecodeSlices.isPresent()
                && (rmsNormDecodeSlices.getAsInt() < 1 || rmsNormDecodeSlices.getAsInt() > MAX_RMS_NORM_DECODE_SLICES))
            throw new IllegalArgumentException("rms-norm-decode-slices must be 1 to " + MAX_RMS_NORM_DECODE_SLICES);
        Objects.requireNonNull(fusedVerifyFrom, "fusedVerifyFrom");
        fusedVerifyFrom = fusedVerifyFrom.map(List::copyOf);
        fusedVerifyFrom.ifPresent(table -> {
            if (table.size() != FUSED_VERIFY_ROWS)
                throw new IllegalArgumentException("fused-verify-from must list " + FUSED_VERIFY_ROWS + " entries");
            for (long keys : table)
                if (keys < 1 || keys > NEVER)
                    throw new IllegalArgumentException("fused-verify-from entries must be 1 to " + NEVER);
        });
    }

    public KernelSettings(TrellisKernelSettings trellis) {
        this(trellis, OptionalInt.empty(), Optional.empty());
    }

    public static KernelSettings defaults() {
        return new KernelSettings(TrellisKernelSettings.defaults());
    }

    /// The slices of a one-row RMS norm's output pass: the setting, or its default.
    public int rmsNormSlices() {
        return rmsNormDecodeSlices.orElse(DEFAULT_RMS_NORM_DECODE_SLICES);
    }

    /// The fused verifier table: the setting, or its default.
    public List<Long> fusedVerifyFromOrDefault() {
        return fusedVerifyFrom.orElse(DEFAULT_FUSED_VERIFY_FROM);
    }
}
