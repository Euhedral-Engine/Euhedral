package io.euhedral_execution.inference.core.model.qwen38.prefix;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.AttentionStates;
import io.euhedral_execution.inference.core.model.qwen38.GdnStates;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.prefix.PrefixNode;

/// The prefix checkpoints a prompt graph takes: for each wanted position, the node that holds the state there, a
/// node the cache reserved for it (copied by the graph, published when the prompt commits) or one it already
/// stored. The chain runs from `parent`, where the prompt's earlier state is stored. Reserved by an owner frame
/// before the prompt is admitted ([PrefixCache#reserveCheckpoints]), copied by the graph's checkpoint stages, and
/// settled by an owner frame after it retired ([PrefixCache#settleCheckpoints]).
public final class PromptCheckpoints {

    private final PrefixCache cache;
    private final PrefixNode parent;
    private final int[] positions;
    private final PrefixNode[] nodes;
    private final boolean[] reserved;

    PromptCheckpoints(PrefixCache cache, PrefixNode parent, int[] positions, PrefixNode[] nodes, boolean[] reserved) {
        this.cache = cache;
        this.parent = parent;
        this.positions = positions;
        this.nodes = nodes;
        this.reserved = reserved;
    }

    /// Checkpoints in the chain.
    public int count() {
        return this.positions.length;
    }

    /// The position of checkpoint `index`: the end of the chunk after which the graph copies it.
    public int position(int index) {
        return this.positions[index];
    }

    public PrefixNode node(int index) {
        return this.nodes[index];
    }

    /// Whether the graph copies checkpoint `index` (the cache reserved it), rather than finding it stored.
    public boolean copies(int index) {
        return this.reserved[index];
    }

    PrefixNode parent() {
        return this.parent;
    }

    /// Queues checkpoint `index`'s copies, from `sequence`'s state into the node's extent, on the stream the calling
    /// stage has selected. Nothing for a checkpoint the cache already holds.
    public void copy(int index, Sequence sequence, ExecutionGpu gpu) {
        if (!this.reserved[index]) return;
        if (this.cache.isClosed()) throw new IllegalStateException("the prefix cache is closed");
        var copies = this.cache
                .layout()
                .captureCopies(this.nodes[index], (GdnStates) sequence.recurrentState(), (AttentionStates)
                        sequence.kvCacheState());
        long arena = this.cache.arenaAddress();
        for (PrefixLayout.Copy copy : copies)
            gpu.copyDeviceToHostAsync(arena + copy.hostOffset(), copy.deviceAddress(), copy.bytes());
    }
}
