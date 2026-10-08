package io.euhedral_execution.inference.core.runtime.graph;

import java.util.Arrays;

/// One sequence's record of the last users of each piece of its carried state ([GraphShape#carriedState]): the
/// graph of the sequence admitted last that touched a key, and its exits of it. A later graph of the sequence orders
/// its first users of a key behind them ([WorkspaceOwner#bind]), so it may be admitted while the earlier graph is
/// still in flight; graphs of other sequences are not ordered by it.
///
/// Confined to the workspace's owner, whose frames bind every graph.
public final class SequenceOwner {

    private StageFrame[][] last = new StageFrame[0][];

    /// The exits of `key` of the graph that last touched it, or null.
    StageFrame[] last(int key) {
        return key < this.last.length ? this.last[key] : null;
    }

    void record(int key, StageFrame[] exits) {
        if (key >= this.last.length) this.last = Arrays.copyOf(this.last, key + 1);
        this.last[key] = exits;
    }
}
