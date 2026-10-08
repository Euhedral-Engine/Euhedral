package io.euhedral_execution.inference.core.runtime.graph;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

/// For each workspace buffer of a shape, the stages that touch it first (entries: no other accessor among their
/// ancestors) and last (exits: no other accessor among their descendants). A graph admitted after another orders
/// its entries of a buffer behind the other's exits of it. Computed once per shape.
public final class WorkspaceUse {

    private static final int[] NONE = new int[0];

    private final int[][] entries;
    private final int[][] exits;

    private WorkspaceUse(int[][] entries, int[][] exits) {
        this.entries = entries;
        this.exits = exits;
    }

    public static WorkspaceUse of(GraphShape shape) {
        StageTopology topology = shape.topology();
        int stages = topology.size();
        int buffers = shape.workspaceBufferCount();
        int[][] entries = new int[buffers][];
        int[][] exits = new int[buffers][];
        if (buffers == 0) return new WorkspaceUse(entries, exits);
        // Descendants of every stage; stage indexes are topological, so successors come later.
        BitSet[] below = new BitSet[stages];
        for (int stage = stages - 1; stage >= 0; stage--) {
            BitSet set = new BitSet(stages);
            for (int next : topology.submittedSuccessors(stage)) {
                set.set(next);
                set.or(below[next]);
            }
            for (int next : topology.retiredSuccessors(stage)) {
                set.set(next);
                set.or(below[next]);
            }
            below[stage] = set;
        }
        List<List<Integer>> accessors = new ArrayList<>(buffers);
        for (int b = 0; b < buffers; b++) accessors.add(new ArrayList<>());
        for (int stage = 0; stage < stages; stage++)
            for (int b : shape.workspaceBuffers(stage)) accessors.get(b).add(stage);
        for (int b = 0; b < buffers; b++) {
            List<Integer> touching = accessors.get(b);
            if (touching.isEmpty()) {
                entries[b] = NONE;
                exits[b] = NONE;
                continue;
            }
            BitSet all = new BitSet(stages);
            for (int stage : touching) all.set(stage);
            BitSet reached = new BitSet(stages);
            for (int stage : touching) reached.or(below[stage]);
            List<Integer> first = new ArrayList<>(), last = new ArrayList<>();
            for (int stage : touching) {
                if (!reached.get(stage)) first.add(stage);
                if (!below[stage].intersects(all)) last.add(stage);
            }
            entries[b] = first.stream().mapToInt(Integer::intValue).toArray();
            exits[b] = last.stream().mapToInt(Integer::intValue).toArray();
        }
        return new WorkspaceUse(entries, exits);
    }

    public int[] entries(int buffer) {
        return this.entries[buffer];
    }

    public int[] exits(int buffer) {
        return this.exits[buffer];
    }
}
