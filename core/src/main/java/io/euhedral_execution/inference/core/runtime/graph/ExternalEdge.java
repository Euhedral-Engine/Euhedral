package io.euhedral_execution.inference.core.runtime.graph;

/// One external edge: `target`, a stage of a later graph, waits for `source`, a last accessor of a workspace
/// buffer in an earlier graph. `slot` is the target graph's await slot that receives the marker to await.
final class ExternalEdge {
    final StageFrame source;
    final StageFrame target;
    int slot;
    /// The next edge registered on the same source.
    ExternalEdge next;

    ExternalEdge(StageFrame source, StageFrame target) {
        this.source = source;
        this.target = target;
    }
}
