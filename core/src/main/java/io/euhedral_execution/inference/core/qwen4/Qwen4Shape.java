package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.graph.GraphShape;
import io.euhedral_execution.inference.core.scheduling.graph.GraphStorage;
import io.euhedral_execution.inference.core.scheduling.graph.StageFrame;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraph;
import io.euhedral_execution.inference.core.scheduling.graph.StageTopology;
import io.euhedral_execution.inference.core.scheduling.graph.StageTopology.Boundary;
import java.util.ArrayList;
import java.util.List;

/// The static DAG of one chunk of Flash-Next, given to the runtime to instantiate (a [GraphShape]):
/// its stages, the edges between them, and the frame of each stage. The shape is the whole
/// description of the work; running it is the lattice's business.
///
/// ```
/// (n-gram ids -> gather x8 ─┐)
/// embed -> [PLE] -> attention block -> route ─┬─> shared expert ──────────────────────────────┐
///   ^                                         └─(route copy retires)─> plan ─┬─> fetch 0 ─> expert 0 ─┤
///   |                                                                        ├─> fetch 1 ─> expert 1 ─┤
///   |                                                                        └─> fetch n ─> expert n ─┴─> finish
///   └─────────────────────────────────────── next layer <───────────────────────────────────────────────┘ ... -> head
/// ```
///
/// A layer has a fetch and an expert stage for every expert its largest block can name; the plan stage
/// learns how many a block names, and the stages beyond that complete in place without a hop. A fetch is
/// a host stage routed to the cache's owner; it ends when its expert is held, which for a miss is when the
/// expert's copy was submitted. An expert stage runs the expert's kernels on a lane as soon as its own
/// expert is held, whatever the others are doing. The finish adds the experts' outputs in ascending expert
/// order (the order the sum needs) once all of them ran.
///
/// A diagnostic variant of the same shape (used while timings or observers are set) splits each
/// layer's attention block and puts a device-completion edge between the pieces, so the wall time
/// of each component and the state after each layer can be read; it exists for tests and
/// measurement.
final class Qwen4Shape implements GraphShape {

    enum Kind {
        EMBED,
        PLEIDS,
        PLEGATHER,
        PLE,
        MIX,
        ATTENTION,
        INJECT,
        BLOCK,
        MID,
        ROUTE,
        SHARED,
        PLAN,
        PREDICT,
        PREFETCH,
        FETCH,
        EXPERT,
        FINISH,
        ENDINJECT,
        OBSERVE,
        HEAD
    }

    /// One stage: what it does, for which layer, and which one of its kind in the layer (the part of a gather,
    /// the active expert of a fetch or an expert stage).
    record Spec(Kind kind, int layer, int index) {}

    private record Edge(int from, Boundary boundary) {}

    /// Stages that gather the n-gram rows of a chunk side by side.
    static final int PLE_PARTS = 8;

    private final Qwen4ExecutionPlan plan;
    private final Qwen4ExecutionPlan.ShapeKey key;
    private final List<Spec> specs = new ArrayList<>();
    private final List<Edge[]> incoming = new ArrayList<>();
    private final int expertCap;
    private final StageTopology topology;

    Qwen4Shape(Qwen4ExecutionPlan plan, Qwen4ExecutionPlan.ShapeKey key) {
        this.plan = plan;
        this.key = key;
        this.expertCap = plan.maxExperts(key.rows());
        build();
        int[][] dependencies = new int[this.incoming.size()][];
        Boundary[][] boundaries = new Boundary[this.incoming.size()][];
        for (int stage = 0; stage < dependencies.length; stage++) {
            Edge[] edges = this.incoming.get(stage);
            dependencies[stage] = new int[edges.length];
            boundaries[stage] = new Boundary[edges.length];
            for (int i = 0; i < edges.length; i++) {
                dependencies[stage][i] = edges[i].from();
                boundaries[stage][i] = edges[i].boundary();
            }
        }
        this.topology = StageTopology.of(dependencies, boundaries);
    }

    private static Edge after(int from) {
        return new Edge(from, Boundary.SUBMITTED);
    }

    private static Edge retired(int from) {
        return new Edge(from, Boundary.RETIRED);
    }

    private int add(Kind kind, int layer, int index, Edge... edges) {
        this.specs.add(new Spec(kind, layer, index));
        this.incoming.add(edges);
        return this.specs.size() - 1;
    }

    /// The edge from `from` to a device stage that starts the next piece: nothing at the first
    /// piece of the graph.
    private static Edge[] follow(int from, Boundary boundary) {
        if (from < 0) return new Edge[0];
        return new Edge[] {new Edge(from, boundary)};
    }

    private void build() {
        Qwen4ExecutionPlan.Range range = this.key.range();
        boolean diagnostic = this.key.diagnostic();
        // The boundary a piece waits for in its predecessor: a diagnostic shape waits for the device between pieces.
        Boundary between = diagnostic ? Boundary.RETIRED : Boundary.SUBMITTED;
        int previous = -1;
        Boundary edge = Boundary.SUBMITTED;
        if (range.embeds()) {
            previous = add(Kind.EMBED, -1, -1);
            edge = between;
        }
        for (int layer = range.firstLayer(); layer < range.endLayer(); layer++) {
            if (layer == this.plan.pleLayer()) {
                // The n-gram rows depend on the tokens alone: their ids and gather are roots of the graph, so
                // they run beside the embedding and the layers before this one, in parts on different workers.
                int ids = add(Kind.PLEIDS, layer, -1);
                int extra = previous < 0 ? 0 : 1;
                Edge[] inputs = new Edge[PLE_PARTS + extra];
                for (int part = 0; part < PLE_PARTS; part++)
                    inputs[part] = after(add(Kind.PLEGATHER, layer, part, after(ids)));
                if (extra == 1) inputs[PLE_PARTS] = new Edge(previous, edge);
                previous = add(Kind.PLE, layer, -1, inputs);
                edge = between;
            }
            if (diagnostic) {
                previous = add(Kind.MIX, layer, -1, follow(previous, edge));
                previous = add(Kind.ATTENTION, layer, -1, retired(previous));
                previous = add(Kind.INJECT, layer, -1, retired(previous));
                previous = add(Kind.MID, layer, -1, retired(previous));
            } else previous = add(Kind.BLOCK, layer, -1, follow(previous, edge));
            int route = add(Kind.ROUTE, layer, -1, after(previous));
            int shared = add(Kind.SHARED, layer, -1, after(route));
            // The plan reads the route copy on the host: it waits for the device to retire it.
            int planned = add(Kind.PLAN, layer, -1, retired(route));
            // A decode step predicts the experts of a later layer from this one's input, behind the route on its lane
            // so the plan does not wait for it, and its prefetch is a host stage after it retires: a side branch.
            if (this.key.rows() == 1
                    && !diagnostic
                    && ExpertCacheOwner.prefetchCandidates() > 0
                    && layer + ExpertCacheOwner.prefetchDistance() < this.plan.layers()) {
                int predict = add(Kind.PREDICT, layer, -1, after(route));
                add(Kind.PREFETCH, layer, -1, retired(predict));
            }
            // Every expert the block can name: its fetch, then its kernels. The finish's combine adds their outputs
            // in ascending expert order, after all of them.
            Edge[] combined = new Edge[this.expertCap + 1];
            for (int e = 0; e < this.expertCap; e++) {
                int fetch = add(Kind.FETCH, layer, e, after(planned));
                combined[e] = after(add(Kind.EXPERT, layer, e, after(fetch), after(planned)));
            }
            combined[this.expertCap] = after(shared);
            previous = add(Kind.FINISH, layer, -1, combined);
            if (diagnostic) {
                previous = add(Kind.ENDINJECT, layer, -1, retired(previous));
                previous = add(Kind.OBSERVE, layer, -1, retired(previous));
                edge = Boundary.SUBMITTED;
            } else edge = Boundary.SUBMITTED;
        }
        if (range.head()) add(Kind.HEAD, -1, -1, follow(previous, edge));
    }

    @Override
    public StageTopology topology() {
        return this.topology;
    }

    @Override
    public StageFrame createStage(StageGraph graph, int stage, ExecutionGpu gpu) {
        return Qwen4Stages.create(graph, stage, this, this.specs.get(stage));
    }

    @Override
    public GraphStorage newStorage(ExecutionGpu gpu) {
        return this.plan.leaseStorage(this.key.rows());
    }

    Qwen4ExecutionPlan plan() {
        return this.plan;
    }

    boolean diagnostic() {
        return this.key.diagnostic();
    }

    /// Whether a quantum of this shape advances its sequence when it commits.
    boolean advances() {
        return this.key.range().advances();
    }
}
