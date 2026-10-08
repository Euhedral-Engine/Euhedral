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

    private final GraphShape[] templates;
    private final Chunks factory;
    /// The first stage of each chunk, and one past the last stage.
    private final int[] offsets;
    private final StageTopology topology;

    /// Chunks 0 to `chunks - 2` of `full`, then one of `last`.
    public ChunkedShape(GraphShape full, GraphShape last, int chunks, Chunks factory) {
        this(templates(full, last, chunks), factory);
    }

    /// One chunk of each of `templates`, in order: a chunk that also takes a prefix checkpoint has a template of
    /// its own.
    public ChunkedShape(GraphShape[] templates, Chunks factory) {
        this.factory = Objects.requireNonNull(factory, "factory");
        if (templates.length < 1) throw new IllegalArgumentException("a chunked shape has at least one chunk");
        this.templates = templates.clone();
        for (GraphShape template : this.templates) {
            Objects.requireNonNull(template, "template");
            if (template.workspaceBufferCount() != this.templates[0].workspaceBufferCount()
                    || template.carriedStateCount() != this.templates[0].carriedStateCount())
                throw new IllegalArgumentException("the templates must declare the same buffers and carried state");
        }
        this.offsets = new int[this.templates.length + 1];
        for (int chunk = 0; chunk < this.templates.length; chunk++)
            this.offsets[chunk + 1] =
                    this.offsets[chunk] + this.templates[chunk].topology().size();
        this.topology = build();
    }

    private static GraphShape[] templates(GraphShape full, GraphShape last, int chunks) {
        Objects.requireNonNull(full, "full");
        Objects.requireNonNull(last, "last");
        if (chunks < 1) throw new IllegalArgumentException("a chunked shape has at least one chunk");
        GraphShape[] templates = new GraphShape[chunks];
        java.util.Arrays.fill(templates, full);
        templates[chunks - 1] = last;
        return templates;
    }

    private StageTopology build() {
        int size = this.offsets[this.templates.length];
        List<List<Integer>> producers = new ArrayList<>(size);
        List<List<Boundary>> boundaries = new ArrayList<>(size);
        for (int stage = 0; stage < size; stage++) {
            producers.add(new ArrayList<>(2));
            boundaries.add(new ArrayList<>(2));
        }
        // Each distinct template's analyses, once.
        java.util.IdentityHashMap<GraphShape, WorkspaceUse> buffers = new java.util.IdentityHashMap<>();
        java.util.IdentityHashMap<GraphShape, CarriedState> carried = new java.util.IdentityHashMap<>();
        for (GraphShape template : this.templates) {
            buffers.computeIfAbsent(template, WorkspaceUse::of);
            carried.computeIfAbsent(template, CarriedState::of);
        }
        int bufferCount = this.templates[0].workspaceBufferCount();
        int keyCount = this.templates[0].carriedStateCount();
        for (int chunk = 0; chunk < this.templates.length; chunk++) {
            GraphShape template = this.templates[chunk];
            StageTopology edges = template.topology();
            int offset = this.offsets[chunk];
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
            GraphShape before = this.templates[chunk - 1];
            int previous = this.offsets[chunk - 1];
            for (int buffer = 0; buffer < bufferCount; buffer++)
                link(
                        producers,
                        boundaries,
                        previous,
                        buffers.get(before).exits(buffer),
                        offset,
                        buffers.get(template).entries(buffer));
            for (int key = 0; key < keyCount; key++)
                link(
                        producers,
                        boundaries,
                        previous,
                        carried.get(before).exits(key),
                        offset,
                        carried.get(template).entries(key));
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

    /// The template of chunk `chunk`.
    public GraphShape template(int chunk) {
        return this.templates[chunk];
    }

    /// The stage of `chunk` that is `templateStage` of its template.
    public int stage(int chunk, int templateStage) {
        return this.offsets[chunk] + templateStage;
    }

    public int chunkOf(int stage) {
        int found = java.util.Arrays.binarySearch(this.offsets, stage);
        return found >= 0 ? Math.min(found, this.templates.length - 1) : -found - 2;
    }

    public int templateStageOf(int stage) {
        return stage - this.offsets[chunkOf(stage)];
    }

    public int chunks() {
        return this.templates.length;
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
        return this.templates[chunkOf(stage)].workspaceBuffers(templateStageOf(stage));
    }

    @Override
    public int workspaceBufferCount() {
        return this.templates[0].workspaceBufferCount();
    }

    @Override
    public int[] carriedState(int stage) {
        return this.templates[chunkOf(stage)].carriedState(templateStageOf(stage));
    }

    @Override
    public int carriedStateCount() {
        return this.templates[0].carriedStateCount();
    }
}
