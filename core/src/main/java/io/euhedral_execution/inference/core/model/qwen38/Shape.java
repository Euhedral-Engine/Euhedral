package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightStaging;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.ScratchUse;
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
        /// The decode view's work for a verification of more than eight rows, whose quantized linears expand into the
        /// scratch; and its preloaded variant.
        VERIFY,
        VERIFY_PRELOADED,
        SMALL_PREFILL,
        REGION_PREFILL,
        MTP_DRAFT,
        /// The MTP draft's work over more than eight rows (a catch-up over a chunk's or a verification's rows), whose
        /// quantized linears expand into the scratch.
        MTP_CATCHUP,
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
    /// The workspace slots each stage reads or writes; computed on first use.
    private volatile int[][] workspaceBuffers;
    /// The scratch route each stage takes, or null for a stage that takes none.
    private final ScratchUse[] scratchUses;

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
        this.scratchUses = new ScratchUse[this.instructions.size()];
        boolean[] scratch = new boolean[this.instructions.size()];
        for (ExecutionPlan.Instruction instruction : this.instructions) {
            this.scratchUses[instruction.id()] = scratchUse(instruction, view);
            scratch[instruction.id()] = this.scratchUses[instruction.id()] != null;
        }
        int[][] ordered = ExecutionPlan.withStorageHazards(this.instructions, dependencies, reuseStorage, scratch);
        ShapeBuilder<ExecutionPlan.Instruction> builder = new ShapeBuilder<>();
        for (ExecutionPlan.Instruction instruction : this.instructions) builder.stage(instruction);
        for (int stage = 0; stage < ordered.length; stage++)
            for (int producer : ordered[stage]) builder.submitted(producer, stage);
        this.topology = builder.build();
    }

    public View view() {
        return this.view;
    }

    /// The sequence state stage `stage` carries from one chunk of a prompt to the next: `2L` for layer L's GDN
    /// convolution and recurrent state, `2L + 1` for its KV rows. Layer slots include the MTP layer's. A
    /// first-layer plan's stages name no layer: their state is layer 0's, the sequence's only one.
    @Override
    public int[] carriedState(int stage) {
        ExecutionPlan.Instruction instruction = this.instructions.get(stage);
        int layer = Math.max(instruction.layerIndex(), 0);
        return switch (instruction.kind()) {
            case GDN_CONVOLUTION, GDN_RECURRENCE -> new int[] {2 * layer};
            case ATTENTION_KV_APPEND, ATTENTION_CAUSAL -> new int[] {2 * layer + 1};
            default -> NO_BUFFERS;
        };
    }

    @Override
    public int carriedStateCount() {
        return 2 * (this.plan.weights().config().numHiddenLayers() + 1);
    }

    /// The runtime workspace's slots stage `stage` reads or writes: its instruction's buffers, through the region
    /// aliases when the view shares storage. The input record and the logits belong to the graph, not the
    /// workspace. A view without a first layer names its hidden, normalized and projection slots on every stage.
    @Override
    public int[] workspaceBuffers(int stage) {
        int[][] buffers = this.workspaceBuffers;
        if (buffers == null) this.workspaceBuffers = buffers = declareWorkspaceBuffers();
        return buffers[stage];
    }

    @Override
    public int workspaceBufferCount() {
        return SharedWorkspace.bufferCount(this.plan);
    }

    /// The scratch route stage `stage` takes, or null: a gate/up region always (its gate/up rows), a quantized
    /// linear in a view whose quanta have more than eight rows (its expansion or activations).
    public ScratchUse scratchUse(int stage) {
        return this.scratchUses[stage];
    }

    /// The staging slots `instruction` writes (a transfer) or reads (a staged weight): each a workspace buffer, so a
    /// transfer into a slot follows the slot's last reader, in this graph or the one admitted before.
    private void addStagingSlots(ExecutionPlan.Instruction instruction, java.util.Set<Integer> slots) {
        WeightStaging staging = this.plan.staging();
        if (staging == null) return;
        long end = staging.slotAddress(0) + (long) staging.slots() * staging.slotBytes();
        for (var weight : instruction.weights()) {
            long address = weight.deviceAddress();
            if (address < staging.slotAddress(0) || address >= end) continue;
            slots.add(SharedWorkspace.stagingBuffer(
                    this.plan, (int) ((address - staging.slotAddress(0)) / staging.slotBytes())));
        }
    }

    private static ScratchUse scratchUse(ExecutionPlan.Instruction instruction, View view) {
        boolean nvfp4 = !instruction.weights().isEmpty() && instruction.weightFormat() == WeightFormat.NVFP4;
        if (instruction.kind() == ExecutionPlan.Kind.Q3_GATE_UP_SWIGLU)
            return nvfp4 ? ScratchUse.NVFP4_GATE_UP : ScratchUse.Q3_GATE_UP;
        boolean multiRow =
                switch (view) {
                    case DECODE, DECODE_PRELOADED, MTP_DRAFT -> false;
                    default -> true;
                };
        if (!multiRow) return null;
        return switch (instruction.kind()) {
            case Q3_LINEAR -> nvfp4 ? ScratchUse.NVFP4_LINEAR : ScratchUse.Q3_LINEAR;
            case Q4_LINEAR, Q5_LINEAR -> nvfp4 ? ScratchUse.NVFP4_LINEAR : ScratchUse.MX_LINEAR;
            case DFLASH_LINEAR -> nvfp4 ? ScratchUse.NVFP4_LINEAR : null;
            case DFLASH_LM_HEAD -> nvfp4 ? ScratchUse.NVFP4_LINEAR : ScratchUse.Q3_LINEAR;
            default -> null;
        };
    }

    private int[][] declareWorkspaceBuffers() {
        int[][] buffers = new int[this.instructions.size()][];
        if (!this.firstLayer) {
            int[] all = new int[2 + this.projectionWidths.size()];
            all[0] = Workspace.HIDDEN_SLOT;
            all[1] = Workspace.NORMALIZED_SLOT;
            for (int index = 0; index < this.projectionWidths.size(); index++)
                all[2 + index] = Workspace.PROJECTION_SLOTS + index;
            java.util.Arrays.fill(buffers, all);
            return buffers;
        }
        java.util.Map<ExecutionPlan.Buffer, ExecutionPlan.Buffer> owners =
                new java.util.EnumMap<>(ExecutionPlan.Buffer.class);
        if (this.reuseStorage) for (var pair : ExecutionPlan.REGION_STORAGE) owners.put(pair.getKey(), pair.getValue());
        for (ExecutionPlan.Instruction instruction : this.instructions) {
            java.util.TreeSet<Integer> slots = new java.util.TreeSet<>();
            for (ExecutionPlan.Buffer buffer : instruction.inputBuffers())
                slots.add(owners.getOrDefault(buffer, buffer).ordinal());
            for (ExecutionPlan.Buffer buffer : instruction.outputBuffers())
                slots.add(owners.getOrDefault(buffer, buffer).ordinal());
            slots.remove(ExecutionPlan.Buffer.LOGITS.ordinal());
            if (this.scratchUses[instruction.id()] != null) slots.add(SharedWorkspace.scratchBuffer(this.plan));
            addStagingSlots(instruction, slots);
            buffers[instruction.id()] =
                    slots.stream().mapToInt(Integer::intValue).toArray();
        }
        return buffers;
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
        Stages.Stage frame = Stages.create(graph, this.instructions.get(stage), gpu);
        frame.scratchUse = this.scratchUses[stage];
        return frame;
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
