package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertLease;

/// The MoE block an expert source serves: which experts it needs, and where the leases go. Any
/// thread may call `arrive`; the rest is called from the source.
public interface ExpertBlock {

    /// The bank (layer) whose experts the block needs.
    int bank();

    /// The expert at `position` of the block's flattened list.
    int expertAt(int position);

    /// Whether the block's quantum stopped (failed or was cancelled): nothing more is loaded for
    /// it.
    boolean stopped();

    /// The expert at `position` is taken: `lease` holds it (carrying the marker of its copy when
    /// the copy was only submitted), or null when it will not come (the quantum stopped, or the
    /// load failed).
    void arrive(int position, ExpertLease lease);

    /// A load failed: the quantum fails.
    void failed(Throwable failure);

    /// A source finished its share of the block: all its experts were taken and every copy it
    /// submitted retired.
    void shardDone();
}
