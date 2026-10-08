package io.euhedral_execution.inference.core.runtime.graph;

/// For each workspace buffer of a shape, the stages that touch it first (entries: no other accessor among their
/// ancestors) and last (exits: no other accessor among their descendants). A graph admitted after another orders
/// its entries of a buffer behind the other's exits of it. Computed once per shape.
public final class WorkspaceUse {

    private final int[][] entries;
    private final int[][] exits;

    private WorkspaceUse(int[][] entries, int[][] exits) {
        this.entries = entries;
        this.exits = exits;
    }

    public static WorkspaceUse of(GraphShape shape) {
        int[][][] analysis =
                AccessAnalysis.of(shape.topology(), shape::workspaceBuffers, shape.workspaceBufferCount());
        return new WorkspaceUse(analysis[0], analysis[1]);
    }

    public int[] entries(int buffer) {
        return this.entries[buffer];
    }

    public int[] exits(int buffer) {
        return this.exits[buffer];
    }
}
