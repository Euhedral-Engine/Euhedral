package io.euhedral_execution.inference.core.scheduling.graph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/// Immutable execution-stage DAG. Stages are numbered in topological order: every dependency names an
/// earlier stage. Each edge states the boundary its producer must reach before the edge is satisfied.
public final class StageTopology {

    /// The producer boundary an edge waits for.
    public enum Boundary {
        /// The producer's device work entered the quantum's stream. Stream order, not the host, then
        /// sequences the consumer's device work after the producer's. This is the ordinary GPU edge.
        SUBMITTED,
        /// The producer's device work has retired. Reserved for host consumption, state publication,
        /// storage release, or crossing into another device-ordering domain.
        RETIRED
    }

    private static final int[] NONE = new int[0];

    private final int[] inDegree;
    private final int[][] submittedSuccessors;
    private final int[][] retiredSuccessors;
    private final int[] roots;

    private StageTopology(int[][] dependencies, Boundary[][] boundaries) {
        int count = dependencies.length;
        if (count == 0) throw new IllegalArgumentException("a stage graph needs at least one stage");
        this.inDegree = new int[count];
        List<List<Integer>> submitted = new ArrayList<>(count);
        List<List<Integer>> retired = new ArrayList<>(count);
        for (int stage = 0; stage < count; stage++) {
            submitted.add(new ArrayList<>());
            retired.add(new ArrayList<>());
        }
        List<Integer> rootStages = new ArrayList<>();
        for (int stage = 0; stage < count; stage++) {
            int[] incoming = Objects.requireNonNull(dependencies[stage], "dependencies");
            Boundary[] kinds = boundaries == null ? null : Objects.requireNonNull(boundaries[stage], "boundaries");
            if (kinds != null && kinds.length != incoming.length) {
                throw new IllegalArgumentException("stage " + stage + " has mismatched edge boundaries");
            }
            for (int edge = 0; edge < incoming.length; edge++) {
                int producer = incoming[edge];
                if (producer < 0 || producer >= stage) {
                    throw new IllegalArgumentException("stage dependencies must name earlier stages");
                }
                for (int earlier = 0; earlier < edge; earlier++) {
                    if (incoming[earlier] == producer) throw new IllegalArgumentException("duplicate stage edge");
                }
                Boundary boundary = kinds == null ? Boundary.SUBMITTED : Objects.requireNonNull(kinds[edge]);
                (boundary == Boundary.SUBMITTED ? submitted : retired)
                        .get(producer)
                        .add(stage);
            }
            this.inDegree[stage] = incoming.length;
            if (incoming.length == 0) rootStages.add(stage);
        }
        this.submittedSuccessors = toArrays(submitted);
        this.retiredSuccessors = toArrays(retired);
        this.roots = rootStages.stream().mapToInt(Integer::intValue).toArray();
    }

    /// Builds a topology whose edges are all ordinary submission edges.
    public static StageTopology submitted(int[][] dependencies) {
        return new StageTopology(copy(dependencies), null);
    }

    /// Builds a topology with an explicit boundary for each dependency edge.
    public static StageTopology of(int[][] dependencies, Boundary[][] boundaries) {
        Objects.requireNonNull(boundaries, "boundaries");
        if (boundaries.length != dependencies.length) {
            throw new IllegalArgumentException("every stage needs its edge boundaries");
        }
        Boundary[][] copied = new Boundary[boundaries.length][];
        for (int stage = 0; stage < boundaries.length; stage++) copied[stage] = boundaries[stage].clone();
        return new StageTopology(copy(dependencies), copied);
    }

    public int size() {
        return this.inDegree.length;
    }

    public int inDegree(int stage) {
        return this.inDegree[stage];
    }

    public int[] roots() {
        return this.roots.clone();
    }

    /// Stages whose edge from `stage` is satisfied when `stage` is submitted.
    public int[] submittedSuccessors(int stage) {
        return this.submittedSuccessors[stage].clone();
    }

    /// Stages whose edge from `stage` is satisfied only when `stage`'s device work has retired.
    public int[] retiredSuccessors(int stage) {
        return this.retiredSuccessors[stage].clone();
    }

    private static int[][] copy(int[][] dependencies) {
        Objects.requireNonNull(dependencies, "dependencies");
        int[][] copied = new int[dependencies.length][];
        for (int stage = 0; stage < dependencies.length; stage++) {
            copied[stage] =
                    Objects.requireNonNull(dependencies[stage], "dependencies").clone();
        }
        return copied;
    }

    private static int[][] toArrays(List<List<Integer>> lists) {
        int[][] arrays = new int[lists.size()][];
        for (int index = 0; index < arrays.length; index++) {
            List<Integer> list = lists.get(index);
            arrays[index] = list.isEmpty()
                    ? NONE
                    : list.stream().mapToInt(Integer::intValue).toArray();
        }
        return arrays;
    }

    @Override
    public String toString() {
        return "StageTopology[stages=" + size() + ", roots=" + Arrays.toString(this.roots) + "]";
    }
}
