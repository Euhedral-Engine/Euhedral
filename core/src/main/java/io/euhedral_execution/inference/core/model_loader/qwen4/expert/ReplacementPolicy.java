package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

/// How a cache of experts that the model visits layer by layer shares its slots among the layers.
///
/// The model visits layer 0 to the last and starts again, so the reuse distance of an expert is a whole pass
/// and a global recency order evicts exactly what the next pass needs.
public enum ReplacementPolicy {
    /// Each layer (bank) has a quota of the slots in proportion to its experts. A layer under its quota that
    /// needs a slot takes it from the layer furthest over its own, a layer at its quota replaces its own
    /// least recently used expert, and slots no layer is using are lent freely.
    BANK_PARTITIONED,
    /// One recency order over every expert: the baseline.
    GLOBAL_LRU
}
