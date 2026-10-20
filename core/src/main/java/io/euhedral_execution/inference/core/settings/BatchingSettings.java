package io.euhedral_execution.inference.core.settings;

import java.util.Objects;
import java.util.OptionalInt;

/// How sequences share engine passes.
///
/// @param maxSequences sequences held at once; 0 holds as many as fit
/// @param maxSequencesPerPass sequences in one shared decode or verification pass, or empty for no bound
/// @param maxTokensPerPass rows in one shared pass, or empty for the prefill chunk
/// @param prefillChunkTokens rows of the longest prefill chunk
public record BatchingSettings(
        int maxSequences, OptionalInt maxSequencesPerPass, OptionalInt maxTokensPerPass, int prefillChunkTokens) {

    public BatchingSettings {
        if (maxSequences < 0) throw new IllegalArgumentException("max-sequences must not be negative");
        Objects.requireNonNull(maxSequencesPerPass, "maxSequencesPerPass");
        Objects.requireNonNull(maxTokensPerPass, "maxTokensPerPass");
        if (maxSequencesPerPass.isPresent() && maxSequencesPerPass.getAsInt() < 1)
            throw new IllegalArgumentException("max-sequences-per-pass must be positive");
        if (maxTokensPerPass.isPresent() && maxTokensPerPass.getAsInt() < 1)
            throw new IllegalArgumentException("max-tokens-per-pass must be positive");
        if (prefillChunkTokens < 1) throw new IllegalArgumentException("prefill-chunk-tokens must be positive");
    }

    public static BatchingSettings defaults() {
        return new BatchingSettings(1, OptionalInt.empty(), OptionalInt.empty(), 512);
    }
}
