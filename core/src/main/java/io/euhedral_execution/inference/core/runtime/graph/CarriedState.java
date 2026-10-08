package io.euhedral_execution.inference.core.runtime.graph;

/// For each carried-state key of a shape ([GraphShape#carriedState]), the stages that touch it first (entries) and
/// last (exits), as [WorkspaceUse] does for workspace buffers. A later chunk's entries of a key follow an earlier
/// chunk's exits of it. Computed once per shape, in time linear in the shape's stages and edges.
public final class CarriedState {

    private final int[][] entries;
    private final int[][] exits;

    private CarriedState(int[][] entries, int[][] exits) {
        this.entries = entries;
        this.exits = exits;
    }

    public static CarriedState of(GraphShape shape) {
        int[][][] analysis = AccessAnalysis.of(shape.topology(), shape::carriedState, shape.carriedStateCount());
        return new CarriedState(analysis[0], analysis[1]);
    }

    public int[] entries(int key) {
        return this.entries[key];
    }

    public int[] exits(int key) {
        return this.exits[key];
    }
}
