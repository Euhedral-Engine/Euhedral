package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.StageTopology.Boundary;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/// A prompt's DAG: C copies of a model's chunk template in one graph. Chunks 0 to C-2 use the `full` template and
/// the last chunk the `last` one (a smaller row count may select a different view). Within a chunk the edges are the
/// template's. Between chunks there are only the edges the templates' declarations give: for every workspace buffer
/// and every carried-state key, chunk c's first users follow chunk c-1's last users. A stage that touches nothing
/// shared (a token upload, a weight transfer) has no edge from the previous chunk and runs as soon as its inputs
/// exist.
///
/// The edges between chunks come from each template's [WorkspaceUse] and [CarriedState], once, so a shape of many
/// chunks builds in time linear in its size.
public final class ChunkedShape implements GraphShape {

    /// Builds the frames and the storage of a chunked shape's graphs.
    public interface Chunks {
        /// The frame of stage `stage`: stage `templateStage` of chunk `chunk`.
        StageFrame create(StageGraph graph, int stage, int chunk, int templateStage, ExecutionGpu gpu);

        /// The storage one graph of the shape owns.
        GraphStorage newStorage(ExecutionGpu gpu);
    }

    private final GraphShape full;
    private final GraphShape last;
    private final int chunks;
    private final Chunks factory;
    private final int fullSize;
    private final StageTopology topology;

    public ChunkedShape(GraphShape full, GraphShape last, int chunks, Chunks factory) {
        this.full = Objects.requireNonNull(full, "full");
        this.last = Objects.requireNonNull(last, "last");
        this.factory = Objects.requireNonNull(factory, "factory");
        if (chunks < 1) throw new IllegalArgumentException("a chunked shape has at least one chunk");
        if (full.workspaceBufferCount() != last.workspaceBufferCount()
                || full.carriedStateCount() != last.carriedStateCount())
            throw new IllegalArgumentException("the templates must declare the same buffers and carried state");
        this.chunks = chunks;
        this.fullSize = full.topology().size();
        this.topology = build();
    }

    private StageTopology build() {
        int size = this.fullSize * (this.chunks - 1) + this.last.topology().size();
        List<List<Integer>> producers = new ArrayList<>(size);
        List<List<Boundary>> boundaries = new ArrayList<>(size);
        for (int stage = 0; stage < size; stage++) {
            producers.add(new ArrayList<>(2));
            boundaries.add(new ArrayList<>(2));
        }
        WorkspaceUse fullBuffers = WorkspaceUse.of(this.full);
        CarriedState fullCarried = CarriedState.of(this.full);
        WorkspaceUse lastBuffers = this.last == this.full ? fullBuffers : WorkspaceUse.of(this.last);
        CarriedState lastCarried = this.last == this.full ? fullCarried : CarriedState.of(this.last);
        for (int chunk = 0; chunk < this.chunks; chunk++) {
            GraphShape template = template(chunk);
            StageTopology edges = template.topology();
            int offset = chunk * this.fullSize;
            for (int stage = 0; stage < edges.size(); stage++) {
                for (int next : edges.submittedSuccessorsView(stage)) {
                    producers.get(offset + next).add(offset + stage);
                    boundaries.get(offset + next).add(Boundary.SUBMITTED);
                }
                for (int next : edges.retiredSuccessorsView(stage)) {
                    producers.get(offset + next).add(offset + stage);
                    boundaries.get(offset + next).add(Boundary.RETIRED);
                }
            }
            if (chunk == 0) continue;
            int previous = offset - this.fullSize;
            boolean lastChunk = chunk == this.chunks - 1;
            WorkspaceUse buffers = lastChunk ? lastBuffers : fullBuffers;
            CarriedState carried = lastChunk ? lastCarried : fullCarried;
            for (int buffer = 0; buffer < this.full.workspaceBufferCount(); buffer++)
                link(producers, boundaries, previous, fullBuffers.exits(buffer), offset, buffers.entries(buffer));
            for (int key = 0; key < this.full.carriedStateCount(); key++)
                link(producers, boundaries, previous, fullCarried.exits(key), offset, carried.entries(key));
        }
        int[][] dependencies = new int[size][];
        Boundary[][] kinds = new Boundary[size][];
        for (int stage = 0; stage < size; stage++) {
            dependencies[stage] =
                    producers.get(stage).stream().mapToInt(Integer::intValue).toArray();
            kinds[stage] = boundaries.get(stage).toArray(Boundary[]::new);
        }
        return StageTopology.of(dependencies, kinds);
    }

    /// Edges from the previous chunk's `exits` to this chunk's `entries`, once each.
    private static void link(
            List<List<Integer>> producers,
            List<List<Boundary>> boundaries,
            int previous,
            int[] exits,
            int offset,
            int[] entries) {
        for (int entry : entries) {
            List<Integer> incoming = producers.get(offset + entry);
            for (int exit : exits) {
                if (incoming.contains(previous + exit)) continue;
                incoming.add(previous + exit);
                boundaries.get(offset + entry).add(Boundary.SUBMITTED);
            }
        }
    }

    private GraphShape template(int chunk) {
        return chunk == this.chunks - 1 ? this.last : this.full;
    }

    /// The stage of `chunk` that is `templateStage` of its template.
    public int stage(int chunk, int templateStage) {
        return chunk * this.fullSize + templateStage;
    }

    public int chunkOf(int stage) {
        return Math.min(stage / this.fullSize, this.chunks - 1);
    }

    public int templateStageOf(int stage) {
        return stage - chunkOf(stage) * this.fullSize;
    }

    public int chunks() {
        return this.chunks;
    }

    @Override
    public StageTopology topology() {
        return this.topology;
    }

    @Override
    public StageFrame createStage(StageGraph graph, int stage, ExecutionGpu gpu) {
        return this.factory.create(graph, stage, chunkOf(stage), templateStageOf(stage), gpu);
    }

    @Override
    public GraphStorage newStorage(ExecutionGpu gpu) {
        return this.factory.newStorage(gpu);
    }

    @Override
    public int[] workspaceBuffers(int stage) {
        return template(chunkOf(stage)).workspaceBuffers(templateStageOf(stage));
    }

    @Override
    public int workspaceBufferCount() {
        return this.full.workspaceBufferCount();
    }

    @Override
    public int[] carriedState(int stage) {
        return template(chunkOf(stage)).carriedState(templateStageOf(stage));
    }

    @Override
    public int carriedStateCount() {
        return this.full.carriedStateCount();
    }
}
