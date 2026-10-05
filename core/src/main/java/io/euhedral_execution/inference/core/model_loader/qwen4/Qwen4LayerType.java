package io.euhedral_execution.inference.core.model_loader.qwen4;

/// The token mixer of a text layer.
public enum Qwen4LayerType {
    /// Gated DeltaNet (`linear_attention` in the source configuration).
    GATED_DELTA_NET,
    /// Qwen sparse attention: attention over the keys an indexer selects (`full_attention` in the source).
    SPARSE_ATTENTION;

    static Qwen4LayerType parse(String name) {
        return switch (name) {
            case "linear_attention" -> GATED_DELTA_NET;
            case "full_attention" -> SPARSE_ATTENTION;
            default -> throw new IllegalArgumentException("unknown layer type '" + name + "'");
        };
    }

    String sourceName() {
        return this == GATED_DELTA_NET ? "linear_attention" : "full_attention";
    }
}
