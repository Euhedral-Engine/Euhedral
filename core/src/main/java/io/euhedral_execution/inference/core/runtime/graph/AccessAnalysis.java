package io.euhedral_execution.inference.core.runtime.graph;

import java.util.function.IntFunction;

/// For each key a shape's stages declare (a workspace buffer, a piece of carried state), the stages that touch it
/// first (entries: no other accessor among their ancestors) and last (exits: no other accessor among their
/// descendants). Linear per key in the shape's stages and edges: stage indexes are topological, so one forward and
/// one backward pass decide both.
final class AccessAnalysis {

    private static final int[] NONE = new int[0];

    private AccessAnalysis() {}

    /// `[0][key]` the entries and `[1][key]` the exits of each of `keys` keys, given each stage's keys.
    static int[][][] of(StageTopology topology, IntFunction<int[]> keysOf, int keys) {
        int[][] entries = new int[keys][];
        int[][] exits = new int[keys][];
        if (keys == 0) return new int[][][] {entries, exits};
        int stages = topology.size();
        // Each key's accessors, in stage order.
        int[] counts = new int[keys];
        int[][] declared = new int[stages][];
        for (int stage = 0; stage < stages; stage++) {
            declared[stage] = keysOf.apply(stage);
            for (int key : declared[stage]) counts[key]++;
        }
        int[][] accessors = new int[keys][];
        for (int key = 0; key < keys; key++) accessors[key] = new int[counts[key]];
        java.util.Arrays.fill(counts, 0);
        for (int stage = 0; stage < stages; stage++)
            for (int key : declared[stage]) accessors[key][counts[key]++] = stage;
        boolean[] touches = new boolean[stages];
        boolean[] above = new boolean[stages];
        boolean[] below = new boolean[stages];
        for (int key = 0; key < keys; key++) {
            int[] touching = accessors[key];
            if (touching.length == 0) {
                entries[key] = NONE;
                exits[key] = NONE;
                continue;
            }
            for (int stage : touching) touches[stage] = true;
            int first = touching[0], last = touching[touching.length - 1];
            // Forward from the first accessor: whether an accessor precedes each stage.
            for (int stage = first; stage <= last; stage++) {
                if (!touches[stage] && !above[stage]) continue;
                for (int next : topology.submittedSuccessorsView(stage)) if (next <= last) above[next] = true;
                for (int next : topology.retiredSuccessorsView(stage)) if (next <= last) above[next] = true;
            }
            // Backward from the last accessor: whether an accessor follows each stage.
            for (int stage = last; stage >= first; stage--) {
                boolean after = false;
                for (int next : topology.submittedSuccessorsView(stage))
                    if (next <= last && (touches[next] || below[next])) {
                        after = true;
                        break;
                    }
                if (!after)
                    for (int next : topology.retiredSuccessorsView(stage))
                        if (next <= last && (touches[next] || below[next])) {
                            after = true;
                            break;
                        }
                below[stage] = after;
            }
            int entryCount = 0, exitCount = 0;
            for (int stage : touching) {
                if (!above[stage]) entryCount++;
                if (!below[stage]) exitCount++;
            }
            int[] keyEntries = new int[entryCount], keyExits = new int[exitCount];
            entryCount = 0;
            exitCount = 0;
            for (int stage : touching) {
                if (!above[stage]) keyEntries[entryCount++] = stage;
                if (!below[stage]) keyExits[exitCount++] = stage;
            }
            entries[key] = keyEntries;
            exits[key] = keyExits;
            for (int stage = first; stage <= last; stage++) {
                touches[stage] = false;
                above[stage] = false;
                below[stage] = false;
            }
        }
        return new int[][][] {entries, exits};
    }
}
