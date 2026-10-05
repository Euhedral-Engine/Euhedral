package io.euhedral_execution.inference.core.model_loader.qwen4;

/// The logical component an artifact object belongs to. The ordinal is the `group` field of the version 3
/// tables, so the order is part of the format. The residency planner places objects by group and priority, not by
/// their names.
public enum ComponentGroup {
    /// Always-used text weights that belong to no other group (the n-gram projections of the first layer).
    FIXED_TEXT,
    /// Routed experts of the text layers, held in expert banks.
    ROUTED_EXPERT,
    /// The shared expert of every layer and its gate.
    SHARED_EXPERT,
    /// The per-layer routing matrices.
    ROUTER,
    /// Gated DeltaNet projections and convolution kernels.
    GDN,
    /// Sparse-attention projections and the indexer's projection.
    QSA,
    /// Hyper-connection mixers, inject weights and norms.
    HYPER_CONNECTION,
    TOKEN_EMBEDDING,
    OUTPUT_HEAD,
    /// Norm scales and per-head control vectors: small, read by every layer.
    NORM_SMALL_STATE,
    /// The n-gram embedding tables.
    NGRAM,
    /// The multi-token-prediction layer, its routed experts included.
    MTP,
    VISION,
    OTHER
}
