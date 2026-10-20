package io.euhedral_execution.inference.core.settings;

/// Sampling for the fields a request leaves out. The defaults are the Qwen3.8 checkpoints' recommended values.
///
/// @param temperature the temperature; 0 is greedy decoding
/// @param topK the most likely tokens kept; 0 keeps every token
/// @param topP the probability mass kept, in (0, 1]
public record SamplingSettings(float temperature, int topK, float topP) {

    public SamplingSettings {
        if (!Float.isFinite(temperature) || temperature < 0.0f)
            throw new IllegalArgumentException("sampling.temperature must be finite and not negative");
        if (topK < 0) throw new IllegalArgumentException("sampling.top-k must not be negative");
        if (!Float.isFinite(topP) || topP <= 0.0f || topP > 1.0f)
            throw new IllegalArgumentException("sampling.top-p must be in (0, 1]");
    }

    public static SamplingSettings defaults() {
        return new SamplingSettings(1.0f, 20, 0.95f);
    }
}
