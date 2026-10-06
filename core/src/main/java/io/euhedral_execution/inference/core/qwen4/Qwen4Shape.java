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
/// embed -> [PLE] -> attention block -> route ─┬─> shared expert ───────────────────────────┐
///   ^                                         └─(route copy retires)─> plan                │
///   |                                                                   │                 v
///   |        wave 0:  load ─────────────────────────────────────────> wave 0 ──> wave 1 ... ──> finish
///   |        wave w:  load (after wave w-window was submitted) ─────> wave w                    │
///   └─────────────────────────────────── next layer <──────────────────────────────────────────┘ ... -> head
/// ```
///
/// A layer has as many wave pairs as its largest block can need; the plan stage learns how many a
/// block uses, and the load and wave stages beyond that complete in place without a hop. A load
/// stage is a host stage whose completion is asynchronous (it ends when the wave's experts are
/// resident), and a wave stage runs on a lane; the wave stages are chained so the leases they close
/// stand behind earlier waves' kernels. The window edges (a wave's load starts after the wave
/// `window` before it was submitted) bound the loads outstanding by the cache's slots and the
/// staging slots, with no queue: a graph cannot ask for more than they have.
///
/// A diagnostic variant of the same shape (used while timings or observers are set) splits each
/// layer's attention block and puts a device-completion edge between the pieces, so the wall time
/// of each component and the state after each layer can be read; it exists for tests and
/// measurement.
final class Qwen4Shape implements GraphShape {

    enum Kind {
        EMBED,
        PLE,
        MIX,
        ATTENTION,
        INJECT,
        BLOCK,
        MID,
        ROUTE,
        SHARED,
        PLAN,
        LOAD,
        WAVE,
        FINISH,
        ENDINJECT,
        OBSERVE,
        HEAD
    }

    /// One stage: what it does, for which layer, and (load and wave stages) for which wave.
    record Spec(Kind kind, int layer, int wave) {}

    private record Edge(int from, Boundary boundary) {}

    private final Qwen4ExecutionPlan plan;
    private final Qwen4ExecutionPlan.ShapeKey key;
    private final List<Spec> specs = new ArrayList<>();
    private final List<Edge[]> incoming = new ArrayList<>();
    private final int waveCap;
    private final int[] loadStages;
    private final StageTopology topology;

    Qwen4Shape(Qwen4ExecutionPlan plan, Qwen4ExecutionPlan.ShapeKey key) {
        this.plan = plan;
        this.key = key;
        this.waveCap = plan.maxWaves(key.rows());
        this.loadStages =
                new int[Math.max(1, key.range().endLayer() - key.range().firstLayer()) * this.waveCap];
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

    private int add(Kind kind, int layer, int wave, Edge... edges) {
        this.specs.add(new Spec(kind, layer, wave));
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
                previous = add(Kind.PLE, layer, -1, follow(previous, edge));
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
            int[] waves = new int[this.waveCap];
            for (int w = 0; w < this.waveCap; w++) {
                // The window's gates are not edges of the shape: the experts of a wave are items that the plan stage
                // spawns, and an item runs when the wave `window` before its own was submitted.
                int load = add(Kind.LOAD, layer, w, after(planned));
                this.loadStages[(layer - range.firstLayer()) * this.waveCap + w] = load;
                waves[w] = add(Kind.WAVE, layer, w, after(load), after(w == 0 ? route : waves[w - 1]));
            }
            previous = add(Kind.FINISH, layer, -1, after(waves[this.waveCap - 1]), after(shared));
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
        return this.plan.leaseStorage(gpu, this.key.rows());
    }

    /// The stage that collects the arrivals of wave `wave` of `layer`.
    int loadStage(int layer, int wave) {
        return this.loadStages[(layer - this.key.range().firstLayer()) * this.waveCap + wave];
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
