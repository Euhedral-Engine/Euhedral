package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.inference.core.runtime.graph.StageTopology.Boundary;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/// Declares a shape's stages and the edges between them, and builds its [StageTopology]. A stage is named by the index
/// [#stage] returns; an edge runs from an earlier stage to a later one, at the boundary it waits for: the producer's
/// submission ([#submitted], ordinary stream order) or its retirement ([#retired], host consumption or another
/// ordering domain). Every shape of both models is built here.
public final class ShapeBuilder<S> {

    private final List<S> specs = new ArrayList<>();
    private final List<List<Integer>> producers = new ArrayList<>();
    private final List<List<Boundary>> boundaries = new ArrayList<>();
    /// [#reaches]' visit marks (the search that last visited each stage), its stack, and its search count.
    private int[] visited = new int[0];
    private int[] pending = new int[0];
    private int search;

    /// Adds a stage and returns its index.
    public int stage(S spec) {
        this.specs.add(Objects.requireNonNull(spec, "spec"));
        this.producers.add(new ArrayList<>());
        this.boundaries.add(new ArrayList<>());
        return this.specs.size() - 1;
    }

    /// `to` waits for `from`'s work to enter the stream.
    public void submitted(int from, int to) {
        edge(from, to, Boundary.SUBMITTED);
    }

    /// `to` waits for `from`'s work to retire.
    public void retired(int from, int to) {
        edge(from, to, Boundary.RETIRED);
    }

    /// Orders `stages` (ascending) one after another, as the users of one resource: each waits for the previous
    /// one's submission unless an edge path already orders them.
    public void chain(int[] stages) {
        for (int i = 1; i < stages.length; i++)
            if (!reaches(stages[i - 1], stages[i])) submitted(stages[i - 1], stages[i]);
    }

    /// Whether an edge path runs from `from` to `to`: a backward search from `to` over stages after `from`. The
    /// visit marks and the stack are kept between searches, so a long chain allocates nothing per link.
    private boolean reaches(int from, int to) {
        if (this.visited.length < this.specs.size()) {
            this.visited = java.util.Arrays.copyOf(this.visited, this.specs.size());
            this.pending = java.util.Arrays.copyOf(this.pending, this.specs.size());
        }
        int search = ++this.search;
        int depth = 0;
        this.pending[depth++] = to;
        while (depth > 0) {
            for (int producer : this.producers.get(this.pending[--depth])) {
                if (producer == from) return true;
                if (producer > from && this.visited[producer] != search) {
                    this.visited[producer] = search;
                    this.pending[depth++] = producer;
                }
            }
        }
        return false;
    }

    public int size() {
        return this.specs.size();
    }

    public List<S> specs() {
        return List.copyOf(this.specs);
    }

    /// The topology; rejects a duplicate edge between the same two stages.
    public StageTopology build() {
        int count = this.specs.size();
        int[][] dependencies = new int[count][];
        Boundary[][] kinds = new Boundary[count][];
        for (int stage = 0; stage < count; stage++) {
            dependencies[stage] = this.producers.get(stage).stream()
                    .mapToInt(Integer::intValue)
                    .toArray();
            kinds[stage] = this.boundaries.get(stage).toArray(Boundary[]::new);
        }
        return StageTopology.of(dependencies, kinds);
    }

    private void edge(int from, int to, Boundary boundary) {
        if (from < 0 || to >= this.specs.size() || from >= to)
            throw new IllegalArgumentException("an edge runs from an earlier stage to a later one: " + from + "→" + to);
        this.producers.get(to).add(from);
        this.boundaries.get(to).add(boundary);
    }
}
