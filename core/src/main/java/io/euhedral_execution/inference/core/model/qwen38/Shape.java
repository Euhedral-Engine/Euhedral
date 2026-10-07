package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.GraphShape;
import io.euhedral_execution.inference.core.runtime.graph.GraphStorage;
import io.euhedral_execution.inference.core.runtime.graph.ShapeBuilder;
import io.euhedral_execution.inference.core.runtime.graph.StageFrame;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;
import io.euhedral_execution.inference.core.runtime.graph.StageTopology;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/// One view of an [ExecutionPlan] as a [GraphShape]: its instructions, which are the stage specs, the topology the
/// runtime instantiates, the buffers its workspace holds, and whether it stages weights or preloads the ring. The
/// plan owns the weights, the staging ring and its views; a shape names the plan it belongs to.
///
/// Every instruction dependency is a submission edge: the consumer's operation is ordered after the producer's by the
/// stream, so it may launch as soon as the producer's launch succeeded. Stages may run on different device lanes, so
/// the
/// topology also orders every pair of stages that touch the same storage (one of them writing) which the data
/// dependencies leave unordered ([ExecutionPlan#withStorageHazards]). A quantum's only device-completion boundary is
/// its
/// retirement.
public final class Shape implements GraphShape {

    /// Which view of its plan a shape is.
    public enum View {
        /// The plan's own topology: the production plan's unfused form, or a reference or test plan's only view.
        OWN,
        DECODE,
        /// The decode view without the transfers of its first ring slots, for a quantum that finds them loaded.
        DECODE_PRELOADED,
        SMALL_PREFILL,
        REGION_PREFILL,
        MTP_DRAFT,
        DFLASH_BLOCK,
        DFLASH_CONTEXT
    }

    private final View view;
    private final ExecutionPlan plan;
    private final List<ExecutionPlan.Instruction> instructions;
    private final List<List<Integer>> successors;
    private final StageTopology topology;
    private final List<Integer> projectionWidths;
    private final List<ExecutionPlan.BufferSpec> bufferSpecs;
    private final boolean firstLayer;
    private final boolean reuseStorage;
    private final boolean prefetchesRing;

    Shape(View view, ExecutionPlan plan, ExecutionPlan.PlanData data, boolean reuseStorage, boolean prefetchesRing) {
        this.view = Objects.requireNonNull(view, "view");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.prefetchesRing = prefetchesRing;
        this.instructions = List.copyOf(data.instructions());
        this.projectionWidths = List.copyOf(data.projectionWidths());
        this.bufferSpecs = List.copyOf(data.bufferSpecs());
        this.firstLayer = data.firstLayer();
        this.reuseStorage = reuseStorage;
        List<List<Integer>> edges = new ArrayList<>();
        for (int index = 0; index < this.instructions.size(); index++) edges.add(new ArrayList<>());
        for (ExecutionPlan.Instruction instruction : this.instructions) {
            for (int dependency : instruction.dependencies()) {
                if (dependency < 0 || dependency >= instruction.id())
                    throw new IllegalArgumentException("instruction dependencies must point to earlier work");
                edges.get(dependency).add(instruction.id());
            }
        }
        this.successors = edges.stream().map(List::copyOf).toList();
        int[][] dependencies = new int[this.instructions.size()][];
        for (ExecutionPlan.Instruction instruction : this.instructions)
            dependencies[instruction.id()] = instruction.dependencies().stream()
                    .mapToInt(Integer::intValue)
                    .toArray();
        int[][] ordered = ExecutionPlan.withStorageHazards(this.instructions, dependencies, reuseStorage);
        ShapeBuilder<ExecutionPlan.Instruction> builder = new ShapeBuilder<>();
        for (ExecutionPlan.Instruction instruction : this.instructions) builder.stage(instruction);
        for (int stage = 0; stage < ordered.length; stage++)
            for (int producer : ordered[stage]) builder.submitted(producer, stage);
        this.topology = builder.build();
    }

    public View view() {
        return this.view;
    }

    /// The plan this view belongs to: its weights, its staging ring and its other views.
    public ExecutionPlan plan() {
        return this.plan;
    }

    public List<ExecutionPlan.Instruction> instructions() {
        return this.instructions;
    }

    @Override
    public StageTopology topology() {
        return this.topology;
    }

    @Override
    public StageFrame createStage(StageGraph graph, int stage, ExecutionGpu gpu) {
        return InstructionFrame.create(graph, this.instructions.get(stage), gpu);
    }

    @Override
    public GraphStorage newStorage(ExecutionGpu gpu) {
        return new WorkspaceStorage(gpu);
    }

    public List<Integer> successors(int instructionId) {
        return this.successors.get(instructionId);
    }

    List<Integer> projectionWidths() {
        return this.projectionWidths;
    }

    public List<ExecutionPlan.BufferSpec> bufferSpecs() {
        return this.bufferSpecs;
    }

    public boolean hasFirstLayer() {
        return this.firstLayer;
    }

    boolean reusePrefillStorage() {
        return this.reuseStorage;
    }

    public int bufferWidth(ExecutionPlan.Buffer buffer) {
        return this.bufferSpecs.stream()
                .filter(spec -> spec.buffer() == buffer)
                .mapToInt(ExecutionPlan.BufferSpec::width)
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("buffer is not in this plan: " + buffer));
    }

    public ExecutionPlan.ElementType bufferElementType(ExecutionPlan.Buffer buffer) {
        return this.bufferSpecs.stream()
                .filter(spec -> spec.buffer() == buffer)
                .map(ExecutionPlan.BufferSpec::elementType)
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("buffer is not in this plan: " + buffer));
    }

    /// Whether this view copies host-backed weights into the staging ring.
    public boolean stagesWeights() {
        return this.instructions.stream().anyMatch(i -> i.kind() == ExecutionPlan.Kind.WEIGHT_TRANSFER);
    }

    /// Whether this view (a DFlash2 block) leaves the decode view's first ring slots loaded when it completes.
    public boolean prefetchesRing() {
        return this.prefetchesRing;
    }

    /// The view to run when the ring holds what [#prefetchesRing] loads: the decode view without the transfers of its
    /// first slots, or this view when it has no such variant.
    public Shape preloadedVariant() {
        return this.plan.preloadedVariant(this);
    }

    @Override
    public String toString() {
        return "Shape[" + this.view + ", " + this.instructions.size() + " stages]";
    }
}
