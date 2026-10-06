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
    GLOBAL_LRU,
    /// The victim is the least requested of a sample of held records (counts halved every ten requests per slot), the
    /// least recent among equals. With admission, only a prefill's records must have been requested more often than
    /// their victim: a decode step's records always enter. For the host tier, which sees only the device's misses,
    /// from which recency is already filtered (docs/FLASH_NEXT_CACHE.md).
    FREQUENCY
}
