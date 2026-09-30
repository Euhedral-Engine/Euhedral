package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactDenseFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import io.euhedral_execution.inference.core.scheduling.graph.StageTopology;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/// Immutable operation instructions and dependency edges for the loaded Qwen text model.
/// Model weights are borrowed; sequence state and quantum workspace have separate owners.
public final class QwenExecutionPlan {

    public enum Kind {
        EMBEDDING,
        RMS_NORM,
        RMS_NORM_UNIT_OFFSET,
        Q3_LINEAR,
        Q4_LINEAR,
        Q5_LINEAR,
        BF16_LINEAR,
        GDN_CONTROL,
        GDN_CONVOLUTION,
        GDN_RECURRENCE,
        GDN_GATED_RMS_NORM,
        RESIDUAL_ADD,
        RESIDUAL_RMS_NORM,
        GDN_PROJECT_CONTROL,
        GDN_PROJECTIONS,
        Q3_GATE_UP_SWIGLU,
        Q3_FFN_DOWN,
        FFN_STREAMED,
        SWIGLU,
        ATTENTION_QK_NORM_ROPE,
        ATTENTION_KV_APPEND,
        ATTENTION_PRODUCERS,
        ATTENTION_CAUSAL
    }

    public enum Buffer {
        HIDDEN_STATE,
        MIXER_HIDDEN,
        FINAL_HIDDEN_STATE,
        INPUT_NORMALIZED,
        QK_PROJECTED,
        VALUE_Z_PROJECTED,
        A_PROJECTED,
        B_PROJECTED,
        GDN_ALPHA,
        GDN_BETA,
        GDN_CONVOLVED,
        GDN_RECURRENT,
        GDN_NORMALIZED,
        ATTENTION_QK_NORMALIZED,
        ATTENTION_CONTEXT,
        MIXER_DELTA,
        POST_MIXER_NORMALIZED,
        GATE_UP,
        SWIGLU,
        FFN_DELTA,
        FFN_STAGING,
        FFN_ACCUMULATORS,
        FINAL_NORMALIZED,
        LOGITS,
        SLICE_PROJECTION
    }

    public enum ElementType {
        BF16,
        FP32
    }

    public record BufferSpec(Buffer buffer, int width, ElementType elementType, int storageSlot) {
        public BufferSpec {
            Objects.requireNonNull(buffer, "buffer");
            Objects.requireNonNull(elementType, "elementType");
            if (width <= 0 || storageSlot < 0) {
                throw new IllegalArgumentException("buffer width must be positive");
            }
        }

        public BufferSpec(Buffer buffer, int width, ElementType elementType) {
            this(buffer, width, elementType, buffer.ordinal());
        }
    }

    public record Instruction(
            int id,
            Kind kind,
            List<Integer> dependencies,
            List<TensorHandle> weights,
            List<Buffer> inputBuffers,
            List<Buffer> outputBuffers,
            int inputWidth,
            int outputWidth,
            int outputBufferIndex,
            int layerIndex) {
        public Instruction {
            Objects.requireNonNull(kind, "kind");
            dependencies = List.copyOf(dependencies);
            weights = weights.stream().map(QwenExecutionPlan::copyHandle).toList();
            inputBuffers = List.copyOf(inputBuffers);
            outputBuffers = List.copyOf(outputBuffers);
            if (id < 0 || inputWidth < 0 || outputWidth < 0 || outputBufferIndex < -1 || layerIndex < -1) {
                throw new IllegalArgumentException("invalid instruction dimensions");
            }
        }

        public Instruction(
                int id,
                Kind kind,
                List<Integer> dependencies,
                List<TensorHandle> weights,
                List<Buffer> inputBuffers,
                List<Buffer> outputBuffers,
                int inputWidth,
                int outputWidth,
                int outputBufferIndex) {
            this(
                    id,
                    kind,
                    dependencies,
                    weights,
                    inputBuffers,
                    outputBuffers,
                    inputWidth,
                    outputWidth,
                    outputBufferIndex,
                    -1);
        }

        public Instruction(int id, Kind kind, List<Integer> dependencies, TensorHandle weight, int outputWidth) {
            this(
                    id,
                    kind,
                    dependencies,
                    weight == null ? List.of() : List.of(weight),
                    List.of(),
                    List.of(),
                    0,
                    outputWidth,
                    -1,
                    -1);
        }

        @Override
        public List<TensorHandle> weights() {
            return this.weights.stream().map(QwenExecutionPlan::copyHandle).toList();
        }

        /// Returns a defensive descriptor copy for inspection outside the instruction executor.
        public TensorHandle weight() {
            return this.weights.isEmpty() ? null : copyHandle(this.weights.getFirst());
        }

        /// Returns a borrowed device address without allocating a defensive `TensorHandle` copy.
        public long weightAddress() {
            return weightAddress(0);
        }

        public long weightAddress(int weightIndex) {
            return this.weights.get(weightIndex).deviceAddress();
        }

        /// Returns payload size without allocating a defensive `TensorHandle` copy.
        public long weightByteSize() {
            return weightByteSize(0);
        }

        public long weightByteSize(int weightIndex) {
            return this.weights.get(weightIndex).byteSize();
        }

        public String weightName(int weightIndex) {
            return this.weights.get(weightIndex).name();
        }
    }

    private record PlanData(
            List<Instruction> instructions,
            List<Integer> projectionWidths,
            List<BufferSpec> bufferSpecs,
            boolean firstLayer,
            boolean fullModel) {
        PlanData(
                List<Instruction> instructions,
                List<Integer> projectionWidths,
                List<BufferSpec> bufferSpecs,
                boolean firstLayer) {
            this(instructions, projectionWidths, bufferSpecs, firstLayer, false);
        }
    }

    /// Prefill specializations derived from the full-model reference topology.
    /// `SMALL` covers quanta below the 64-row producer tile; `STREAMED` exists only at the
    /// qualified streamed-FFN geometry.
    private enum PrefillView {
        SMALL,
        REGIONS,
        STREAMED
    }

    /// Row threshold for the attention producer and gate/up regions (one 64-row prefill tile).
    private static final int REGION_MIN_ROWS = 64;
    /// Exact geometry at which the streamed FFN region is qualified; elsewhere gate/up + down is used.
    private static boolean streamedFfnRows(int rows) {
        return rows == 64 || rows == 1024;
    }

    private static final int STREAMED_FFN_HIDDEN = 5120;
    private static final int STREAMED_FFN_INTERMEDIATE = 17408;
    /// Two bounded BF16 feature slots of 4096 features each.
    private static final int STREAMED_FFN_STAGING_WIDTH = 2 * 4096;

    private final QwenWeights weights;
    private final List<Instruction> instructions;
    private final List<List<Integer>> successors;
    private final StageTopology stageTopology;
    private final List<Integer> projectionWidths;
    private final List<BufferSpec> bufferSpecs;
    private final boolean firstLayer;
    private final boolean reuseStorage;
    private final QwenExecutionPlan owner;
    private final QwenExecutionPlan smallPrefill;
    private final QwenExecutionPlan regionPrefill;
    private final QwenExecutionPlan streamedPrefill;

    /// Builds the production plan. For a complete model, prefill quanta automatically select the
    /// retained region architecture by row count and geometry; decode keeps the reference topology.
    /// Staged plans for partial weights (embedding-only, layer zero) remain reference-only.
    public QwenExecutionPlan(QwenWeights weights) {
        this(Objects.requireNonNull(weights, "weights"), planFromLoadedWeights(weights), null, false);
    }

    /// Unfused reference topology for every execution kind. This is a correctness oracle for tests
    /// and is never selected by the runtime.
    public static QwenExecutionPlan reference(QwenWeights weights) {
        Objects.requireNonNull(weights, "weights");
        PlanData data = planFromLoadedWeights(weights);
        return new QwenExecutionPlan(
                weights,
                new PlanData(data.instructions(), data.projectionWidths(), data.bufferSpecs(), data.firstLayer()),
                null,
                false);
    }

    /// Owning plan whose views belong to one family; runners admit any view of their owner.
    QwenExecutionPlan executionOwner() {
        return this.owner;
    }

    /// Selects the qualified topology for a quantum. Any view requalifies through its owner.
    public QwenExecutionPlan forExecution(QwenExecutionContext.ExecutionKind kind, int rows) {
        Objects.requireNonNull(kind, "kind");
        if (rows <= 0) throw new IllegalArgumentException("rows must be positive");
        QwenExecutionPlan family = this.owner;
        if (kind == QwenExecutionContext.ExecutionKind.DECODE || family.regionPrefill == null) return family;
        if (rows < REGION_MIN_ROWS) return family.smallPrefill;
        if (streamedFfnRows(rows) && family.streamedPrefill != null) return family.streamedPrefill;
        return family.regionPrefill;
    }

    /// Builds an embedding-only plan for callers that intentionally validate only token lookup.
    public static QwenExecutionPlan embeddingOnly(QwenWeights weights) {
        Objects.requireNonNull(weights, "weights");
        int hiddenSize = weights.config().hiddenSize();
        int vocabularySize = weights.config().vocabSize();
        TensorHandle embedding =
                validateQuantized(weights.tokenEmbedding(), vocabularySize, hiddenSize, WeightFormat.Q3_G64_FP16);
        return new QwenExecutionPlan(weights, embeddingOnly(embedding, hiddenSize));
    }

    /// Builds a dependency-graph prefix ending after the requested real layer, for staged validation.
    public static QwenExecutionPlan prefix(QwenWeights weights, int layerCount) {
        Objects.requireNonNull(weights, "weights");
        if (layerCount <= 0 || layerCount > weights.config().numHiddenLayers()) {
            throw new IllegalArgumentException("layerCount must select a non-empty model prefix");
        }
        QwenExecutionPlan fullPlan = new QwenExecutionPlan(weights);
        int terminalId = -1;
        for (Instruction instruction : fullPlan.instructions) {
            if (instruction.layerIndex() == layerCount - 1
                    && instruction.kind() == Kind.RESIDUAL_ADD
                    && instruction.outputBuffers().contains(Buffer.FINAL_HIDDEN_STATE)) {
                terminalId = instruction.id();
            }
        }
        if (terminalId < 0) throw new IllegalArgumentException("requested layer prefix has no final residual");
        List<Instruction> instructions = List.copyOf(fullPlan.instructions.subList(0, terminalId + 1));
        boolean[] usedBuffers = new boolean[Buffer.values().length];
        for (Instruction instruction : instructions) {
            for (Buffer buffer : instruction.inputBuffers()) usedBuffers[buffer.ordinal()] = true;
            for (Buffer buffer : instruction.outputBuffers()) usedBuffers[buffer.ordinal()] = true;
        }
        List<BufferSpec> buffers = fullPlan.bufferSpecs.stream()
                .filter(spec -> usedBuffers[spec.buffer().ordinal()])
                .toList();
        return new QwenExecutionPlan(weights, new PlanData(instructions, List.of(), buffers, true));
    }

    /// A standalone operator slice retained for low-level operation validation.
    public QwenExecutionPlan(QwenWeights weights, TensorHandle normWeight, List<TensorHandle> projections) {
        this(Objects.requireNonNull(weights, "weights"), operatorSlice(weights, normWeight, projections));
    }

    private QwenExecutionPlan(QwenWeights weights, PlanData data) {
        this(weights, data, null, false);
    }

    private QwenExecutionPlan(QwenWeights weights, PlanData data, QwenExecutionPlan owner, boolean reuseStorage) {
        this.owner = owner == null ? this : owner;
        this.weights = weights;
        this.instructions = List.copyOf(data.instructions());
        this.projectionWidths = List.copyOf(data.projectionWidths());
        this.bufferSpecs = List.copyOf(data.bufferSpecs());
        this.firstLayer = data.firstLayer();
        this.reuseStorage = reuseStorage;
        List<List<Integer>> edges = new ArrayList<>();
        for (int index = 0; index < this.instructions.size(); index++) {
            edges.add(new ArrayList<>());
        }
        for (Instruction instruction : this.instructions) {
            for (int dependency : instruction.dependencies()) {
                if (dependency < 0 || dependency >= instruction.id()) {
                    throw new IllegalArgumentException("instruction dependencies must point to earlier work");
                }
                edges.get(dependency).add(instruction.id());
            }
        }
        this.successors = edges.stream().map(List::copyOf).toList();
        int[][] dependencies = new int[this.instructions.size()][];
        for (Instruction instruction : this.instructions) {
            dependencies[instruction.id()] = instruction.dependencies().stream()
                    .mapToInt(Integer::intValue)
                    .toArray();
        }
        this.stageTopology = StageTopology.submitted(dependencies);
        if (owner != null || !data.fullModel()) {
            this.smallPrefill = null;
            this.regionPrefill = null;
            this.streamedPrefill = null;
            return;
        }
        QwenConfig config = weights.config();
        this.smallPrefill = prefillPlan(weights, data, PrefillView.SMALL, this);
        this.regionPrefill = prefillPlan(weights, data, PrefillView.REGIONS, this);
        this.streamedPrefill =
                config.hiddenSize() == STREAMED_FFN_HIDDEN && config.intermediateSize() == STREAMED_FFN_INTERMEDIATE
                        ? prefillPlan(weights, data, PrefillView.STREAMED, this)
                        : null;
    }

    private static QwenExecutionPlan prefillPlan(
            QwenWeights weights, PlanData data, PrefillView view, QwenExecutionPlan owner) {
        PlanData selected = prefillView(data, view);
        // The named lifetime pairs are qualified only when the view contains both the
        // attention producer and a fused FFN, not for a shape that falls back to ordinary FFN.
        boolean hasAttentionProducer =
                selected.instructions().stream().anyMatch(i -> i.kind() == Kind.ATTENTION_PRODUCERS);
        boolean hasFusedFfn = selected.instructions().stream()
                .anyMatch(i -> i.kind() == Kind.Q3_GATE_UP_SWIGLU || i.kind() == Kind.FFN_STREAMED);
        return new QwenExecutionPlan(weights, selected, owner, hasAttentionProducer && hasFusedFfn);
    }

    /// Rewrites the reference layer DAG into the retained prefill regions:
    /// A residual+RMSNorm, D early joint GDN projection/control, and, from 64 rows,
    /// F attention producers with direct cache writes plus B gate/up+SwiGLU (or C streamed FFN).
    private static PlanData prefillView(PlanData data, PrefillView view) {
        boolean producers = view != PrefillView.SMALL;
        boolean streamed = view == PrefillView.STREAMED;
        List<Instruction> source = earlyControlOrder(data.instructions());
        List<Instruction> result = new ArrayList<>();
        int[] remapped = new int[source.size()];
        java.util.Arrays.fill(remapped, -1);
        for (int index = 0; index < source.size(); index++) {
            Instruction first = source.get(index);
            Instruction next = index + 1 < source.size() ? source.get(index + 1) : null;
            if (producers
                    && first.kind() == Kind.Q4_LINEAR
                    && index + 3 < source.size()
                    && source.get(index + 2).kind() == Kind.ATTENTION_QK_NORM_ROPE) {
                Instruction norm = source.get(index + 2), append = source.get(index + 3);
                if (next.kind() != Kind.Q5_LINEAR
                        || append.kind() != Kind.ATTENTION_KV_APPEND
                        || !next.dependencies().equals(first.dependencies())
                        || !norm.dependencies().equals(List.of(first.id()))
                        || !append.dependencies().equals(List.of(norm.id(), next.id())))
                    throw new IllegalStateException("unexpected attention producer topology");
                for (int offset = 0; offset < 4; offset++)
                    remapped[source.get(index + offset).id()] = result.size();
                result.add(new Instruction(
                        result.size(),
                        Kind.ATTENTION_PRODUCERS,
                        remapDependencies(first.dependencies(), remapped),
                        List.of(
                                first.weight(),
                                next.weight(),
                                norm.weights().get(0),
                                norm.weights().get(1)),
                        first.inputBuffers(),
                        List.of(Buffer.QK_PROJECTED, Buffer.VALUE_Z_PROJECTED),
                        first.inputWidth(),
                        first.outputWidth(),
                        -1,
                        first.layerIndex()));
                index += 3;
                continue;
            }
            if (first.kind() == Kind.Q4_LINEAR
                    && next != null
                    && next.kind() == Kind.Q5_LINEAR
                    && index + 2 < source.size()
                    && source.get(index + 2).kind() == Kind.GDN_CONVOLUTION) {
                if (!next.dependencies().equals(first.dependencies())
                        || !first.inputBuffers().equals(next.inputBuffers())
                        || !first.outputBuffers().equals(List.of(Buffer.QK_PROJECTED))
                        || !next.outputBuffers().equals(List.of(Buffer.VALUE_Z_PROJECTED)))
                    throw new IllegalStateException("unexpected GDN projection topology");
                remapped[first.id()] = result.size();
                remapped[next.id()] = result.size();
                result.add(new Instruction(
                        result.size(),
                        Kind.GDN_PROJECTIONS,
                        remapDependencies(first.dependencies(), remapped),
                        List.of(first.weight(), next.weight()),
                        first.inputBuffers(),
                        List.of(Buffer.QK_PROJECTED, Buffer.VALUE_Z_PROJECTED),
                        first.inputWidth(),
                        first.outputWidth(),
                        -1,
                        first.layerIndex()));
                index++;
                continue;
            }
            if (first.kind() == Kind.RESIDUAL_ADD
                    && next != null
                    && next.kind() == Kind.RMS_NORM_UNIT_OFFSET
                    && !next.outputBuffers().contains(Buffer.FINAL_NORMALIZED)
                    && next.dependencies().equals(List.of(first.id()))
                    && next.inputBuffers().equals(first.outputBuffers())) {
                remapped[first.id()] = result.size();
                remapped[next.id()] = result.size();
                result.add(new Instruction(
                        result.size(),
                        Kind.RESIDUAL_RMS_NORM,
                        remapDependencies(first.dependencies(), remapped),
                        next.weights(),
                        first.inputBuffers(),
                        List.of(
                                first.outputBuffers().getFirst(),
                                next.outputBuffers().getFirst()),
                        first.inputWidth(),
                        first.outputWidth(),
                        -1,
                        first.layerIndex()));
                index++;
                continue;
            }
            if (streamed
                    && first.kind() == Kind.Q3_LINEAR
                    && next != null
                    && first.outputBuffers().equals(List.of(Buffer.GATE_UP))
                    && next.kind() == Kind.SWIGLU
                    && first.inputWidth() == STREAMED_FFN_HIDDEN
                    && first.outputWidth() == 2 * STREAMED_FFN_INTERMEDIATE
                    && index + 2 < source.size()) {
                Instruction down = source.get(index + 2);
                if (down.kind() != Kind.Q3_LINEAR
                        || !down.dependencies().equals(List.of(next.id()))
                        || !next.dependencies().equals(List.of(first.id()))
                        || !down.outputBuffers().equals(List.of(Buffer.FFN_DELTA)))
                    throw new IllegalStateException("unexpected streamed FFN topology");
                remapped[first.id()] = result.size();
                remapped[next.id()] = result.size();
                remapped[down.id()] = result.size();
                result.add(new Instruction(
                        result.size(),
                        Kind.FFN_STREAMED,
                        remapDependencies(first.dependencies(), remapped),
                        List.of(first.weight(), down.weight()),
                        first.inputBuffers(),
                        List.of(Buffer.FFN_DELTA, Buffer.FFN_STAGING, Buffer.FFN_ACCUMULATORS),
                        first.inputWidth(),
                        down.outputWidth(),
                        -1,
                        first.layerIndex()));
                index += 2;
                continue;
            }
            if (producers
                    && first.kind() == Kind.Q3_LINEAR
                    && next != null
                    && first.outputBuffers().equals(List.of(Buffer.GATE_UP))
                    && next.kind() == Kind.SWIGLU
                    && next.dependencies().equals(List.of(first.id()))
                    && first.inputWidth() % 128 == 0
                    && first.outputWidth() % 32 == 0) {
                remapped[first.id()] = result.size();
                remapped[next.id()] = result.size();
                result.add(new Instruction(
                        result.size(),
                        Kind.Q3_GATE_UP_SWIGLU,
                        remapDependencies(first.dependencies(), remapped),
                        first.weights(),
                        first.inputBuffers(),
                        next.outputBuffers(),
                        first.inputWidth(),
                        first.outputWidth(),
                        -1,
                        first.layerIndex()));
                index++;
                continue;
            }
            if (first.kind() == Kind.BF16_LINEAR
                    && first.outputBuffers().equals(List.of(Buffer.A_PROJECTED))
                    && index + 2 < source.size()) {
                Instruction b = source.get(index + 1);
                Instruction last = source.get(index + 2);
                if (b.kind() != Kind.BF16_LINEAR
                        || !b.outputBuffers().equals(List.of(Buffer.B_PROJECTED))
                        || last.kind() != Kind.GDN_CONTROL
                        || !first.dependencies().equals(b.dependencies())
                        || !first.inputBuffers().equals(b.inputBuffers())
                        || !last.dependencies().equals(List.of(first.id(), b.id()))) {
                    throw new IllegalStateException("GDN projection/control region has unexpected topology");
                }
                remapped[first.id()] = result.size();
                remapped[b.id()] = result.size();
                remapped[last.id()] = result.size();
                List<TensorHandle> regionWeights = new ArrayList<>(first.weights());
                regionWeights.addAll(b.weights());
                regionWeights.addAll(last.weights());
                result.add(new Instruction(
                        result.size(),
                        Kind.GDN_PROJECT_CONTROL,
                        remapDependencies(first.dependencies(), remapped),
                        regionWeights,
                        first.inputBuffers(),
                        last.outputBuffers(),
                        first.inputWidth(),
                        first.outputWidth(),
                        -1,
                        first.layerIndex()));
                index += 2;
                continue;
            }
            List<Integer> dependencies = remapDependencies(first.dependencies(), remapped);
            remapped[first.id()] = result.size();
            boolean ffnDown = producers
                    && first.kind() == Kind.Q3_LINEAR
                    && first.inputBuffers().equals(List.of(Buffer.SWIGLU))
                    && first.outputBuffers().equals(List.of(Buffer.FFN_DELTA));
            result.add(new Instruction(
                    result.size(),
                    ffnDown ? Kind.Q3_FFN_DOWN : first.kind(),
                    dependencies,
                    first.weights(),
                    producers && first.kind() == Kind.ATTENTION_CAUSAL
                            ? List.of(Buffer.QK_PROJECTED, Buffer.VALUE_Z_PROJECTED)
                            : first.inputBuffers(),
                    first.outputBuffers(),
                    first.inputWidth(),
                    first.outputWidth(),
                    first.outputBufferIndex(),
                    first.layerIndex()));
        }
        boolean needsGateUp = result.stream().anyMatch(i -> i.outputBuffers().contains(Buffer.GATE_UP));
        boolean needsSwiGlu = result.stream().anyMatch(i -> i.outputBuffers().contains(Buffer.SWIGLU));
        List<BufferSpec> buffers = new ArrayList<>(data.bufferSpecs().stream()
                .filter(spec -> !producers || spec.buffer() != Buffer.ATTENTION_QK_NORMALIZED)
                .filter(spec -> needsSwiGlu || spec.buffer() != Buffer.SWIGLU)
                .filter(spec -> needsGateUp || spec.buffer() != Buffer.GATE_UP)
                .filter(spec -> spec.buffer() != Buffer.A_PROJECTED && spec.buffer() != Buffer.B_PROJECTED)
                .toList());
        if (result.stream().anyMatch(i -> i.kind() == Kind.FFN_STREAMED)) {
            buffers.add(spec(Buffer.FFN_STAGING, STREAMED_FFN_STAGING_WIDTH, ElementType.BF16));
            buffers.add(spec(Buffer.FFN_ACCUMULATORS, STREAMED_FFN_HIDDEN, ElementType.FP32));
        }
        return new PlanData(result, data.projectionWidths(), buffers, data.firstLayer());
    }

    private static List<Instruction> earlyControlOrder(List<Instruction> instructions) {
        List<Instruction> result = new ArrayList<>(instructions.size());
        for (int index = 0; index < instructions.size(); index++) {
            if (index + 4 < instructions.size()
                    && instructions.get(index).kind() == Kind.Q4_LINEAR
                    && instructions.get(index + 1).kind() == Kind.Q5_LINEAR
                    && instructions.get(index + 2).kind() == Kind.BF16_LINEAR
                    && instructions.get(index + 3).kind() == Kind.BF16_LINEAR
                    && instructions.get(index + 4).kind() == Kind.GDN_CONTROL) {
                result.addAll(instructions.subList(index + 2, index + 5));
                result.add(instructions.get(index));
                result.add(instructions.get(index + 1));
                index += 4;
            } else result.add(instructions.get(index));
        }
        return result;
    }

    private static List<Integer> remapDependencies(List<Integer> dependencies, int[] remapped) {
        return dependencies.stream()
                .map(id -> {
                    int mapped = remapped[id];
                    if (mapped < 0) throw new IllegalStateException("region depends on unpublished work");
                    return mapped;
                })
                .distinct()
                .toList();
    }

    public List<Instruction> instructions() {
        return this.instructions;
    }

    /// The stage DAG a reusable frame graph instantiates. Every instruction dependency is a submission
    /// edge: the consumer's operation is ordered after the producer's by the quantum's stream, so it
    /// may launch as soon as the producer's launch succeeded. A quantum's only device-completion
    /// boundary is its retirement, where sequence state is published and quantum storage released.
    public StageTopology stageTopology() {
        return this.stageTopology;
    }

    public List<Integer> successors(int instructionId) {
        return this.successors.get(instructionId);
    }

    List<Integer> projectionWidths() {
        return this.projectionWidths;
    }

    public List<BufferSpec> bufferSpecs() {
        return this.bufferSpecs;
    }

    /// The prefill views this plan owns besides its own topology; empty for a view or a reference plan.
    List<QwenExecutionPlan> executionVariants() {
        if (this.owner != this || this.regionPrefill == null) return List.of();
        return this.streamedPrefill == null
                ? List.of(this.smallPrefill, this.regionPrefill)
                : List.of(this.smallPrefill, this.regionPrefill, this.streamedPrefill);
    }

    boolean reusePrefillStorage() {
        return this.reuseStorage;
    }

    public boolean hasFirstLayer() {
        return this.firstLayer;
    }

    public int bufferWidth(Buffer buffer) {
        return this.bufferSpecs.stream()
                .filter(spec -> spec.buffer() == buffer)
                .mapToInt(BufferSpec::width)
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("buffer is not in this plan: " + buffer));
    }

    public ElementType bufferElementType(Buffer buffer) {
        return this.bufferSpecs.stream()
                .filter(spec -> spec.buffer() == buffer)
                .map(BufferSpec::elementType)
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("buffer is not in this plan: " + buffer));
    }

    public QwenWeights weights() {
        return this.weights;
    }

    private static PlanData planFromLoadedWeights(QwenWeights weights) {
        Objects.requireNonNull(weights.config(), "config");
        int hiddenSize = weights.config().hiddenSize();
        int vocabularySize = weights.config().vocabSize();
        if (hiddenSize <= 0 || vocabularySize <= 0) {
            throw new IllegalArgumentException("invalid model dimensions");
        }
        TensorHandle embedding =
                validateQuantized(weights.tokenEmbedding(), vocabularySize, hiddenSize, WeightFormat.Q3_G64_FP16);
        QwenLayerWeights[] layers = weights.layers();
        if (layers == null || layers.length == 0) {
            return embeddingOnly(embedding, hiddenSize);
        }
        if (layers.length == 1 && layers[0] != null && layers[0].index() == 0) {
            return firstLayer(weights, embedding);
        }
        if (layers.length != weights.config().numHiddenLayers()) {
            throw new IllegalArgumentException("loaded weights do not contain every declared text layer");
        }
        return fullModel(weights, embedding);
    }

    private static PlanData embeddingOnly(TensorHandle embedding, int hiddenSize) {
        return new PlanData(
                List.of(node(
                        0,
                        Kind.EMBEDDING,
                        List.of(),
                        List.of(embedding),
                        List.of(),
                        List.of(Buffer.HIDDEN_STATE),
                        0,
                        hiddenSize)),
                List.of(),
                List.of(),
                false);
    }

    private static PlanData firstLayer(QwenWeights weights, TensorHandle embedding) {
        QwenConfig config = weights.config();
        QwenLayerWeights[] layers = weights.layers();
        if (layers[0] == null || layers[0].index() != 0) {
            throw new IllegalArgumentException("loaded weights do not contain actual text layer zero");
        }
        QwenLayerType[] layerTypes = config.layerTypes();
        if (layerTypes == null || layerTypes.length == 0 || layerTypes[0] != QwenLayerType.GATED_DELTA_NET) {
            throw new IllegalArgumentException("layer zero is not the loaded GDN topology");
        }
        if (!(layers[0].mixer() instanceof QwenCompactGatedDeltaNetWeights gdn)
                || !(layers[0].ffn() instanceof QwenCompactDenseFfnWeights ffn)) {
            throw new IllegalArgumentException("layer zero must use compact GDN and dense FFN weights");
        }
        int hidden = config.hiddenSize();
        int intermediate = config.intermediateSize();
        int keyHeads = config.linearNumKeyHeads();
        int valueHeads = config.linearNumValueHeads();
        int keyWidth = config.linearKeyHeadDim();
        int valueWidthPerHead = config.linearValueHeadDim();
        int kernelWidth = config.linearConvKernelDim();
        if (hidden <= 0
                || intermediate <= 0
                || keyHeads <= 0
                || valueHeads <= 0
                || valueHeads % keyHeads != 0
                || keyWidth != 128
                || valueWidthPerHead != 128
                || kernelWidth < 2
                || !Double.isFinite(config.rmsNormEpsilon())
                || config.rmsNormEpsilon() <= 0) {
            throw new IllegalArgumentException("unsupported layer-zero GDN geometry");
        }
        int keyProjected = Math.multiplyExact(Math.multiplyExact(2, keyHeads), keyWidth);
        int valueProjected = Math.multiplyExact(valueHeads, valueWidthPerHead);
        int valueZProjected = Math.multiplyExact(2, valueProjected);
        int convolutionChannels = Math.addExact(keyProjected, valueProjected);

        TensorHandle inputNorm = validateNorm(layers[0].inputNorm(), hidden);
        TensorHandle postNorm = validateNorm(layers[0].postAttentionNorm(), hidden);
        TensorHandle aLog = validateFp32Vector(gdn.aLog(), valueHeads);
        TensorHandle dtBias = validateFp32Vector(gdn.dtBias(), valueHeads);
        TensorHandle convolution = validateBf16Matrix(gdn.convolution(), kernelWidth, convolutionChannels);
        TensorHandle aProjection = validateBf16Matrix(gdn.aProjection(), valueHeads, hidden);
        TensorHandle bProjection = validateBf16Matrix(gdn.bProjection(), valueHeads, hidden);
        TensorHandle queryKey = validateQuantized(gdn.queryKey(), keyProjected, hidden, WeightFormat.Q4_G64_FP16);
        TensorHandle valueZ = validateQuantized(gdn.valueZ(), valueZProjected, hidden, WeightFormat.Q5_G64_FP16);
        TensorHandle gdnNorm = validateNorm(gdn.norm(), valueWidthPerHead);
        TensorHandle gdnOutput = validateQuantized(gdn.output(), hidden, valueProjected, WeightFormat.Q3_G64_FP16);
        TensorHandle gateUp =
                validateQuantized(ffn.gateUp(), Math.multiplyExact(2, intermediate), hidden, WeightFormat.Q3_G64_FP16);
        TensorHandle down = validateQuantized(ffn.down(), hidden, intermediate, WeightFormat.Q3_G64_FP16);

        List<Instruction> nodes = new ArrayList<>();
        nodes.add(node(
                0, Kind.EMBEDDING, List.of(), List.of(embedding), List.of(), List.of(Buffer.HIDDEN_STATE), 0, hidden));
        nodes.add(node(
                1,
                Kind.RMS_NORM_UNIT_OFFSET,
                List.of(0),
                List.of(inputNorm),
                List.of(Buffer.HIDDEN_STATE),
                List.of(Buffer.INPUT_NORMALIZED),
                hidden,
                hidden));
        nodes.add(node(
                2,
                Kind.Q4_LINEAR,
                List.of(1),
                List.of(queryKey),
                List.of(Buffer.INPUT_NORMALIZED),
                List.of(Buffer.QK_PROJECTED),
                hidden,
                keyProjected));
        nodes.add(node(
                3,
                Kind.Q5_LINEAR,
                List.of(1),
                List.of(valueZ),
                List.of(Buffer.INPUT_NORMALIZED),
                List.of(Buffer.VALUE_Z_PROJECTED),
                hidden,
                valueZProjected));
        nodes.add(node(
                4,
                Kind.BF16_LINEAR,
                List.of(1),
                List.of(aProjection),
                List.of(Buffer.INPUT_NORMALIZED),
                List.of(Buffer.A_PROJECTED),
                hidden,
                valueHeads));
        nodes.add(node(
                5,
                Kind.BF16_LINEAR,
                List.of(1),
                List.of(bProjection),
                List.of(Buffer.INPUT_NORMALIZED),
                List.of(Buffer.B_PROJECTED),
                hidden,
                valueHeads));
        nodes.add(node(
                6,
                Kind.GDN_CONTROL,
                List.of(4, 5),
                List.of(aLog, dtBias),
                List.of(Buffer.A_PROJECTED, Buffer.B_PROJECTED),
                List.of(Buffer.GDN_ALPHA, Buffer.GDN_BETA),
                valueHeads,
                valueHeads));
        nodes.add(node(
                7,
                Kind.GDN_CONVOLUTION,
                List.of(2, 3),
                List.of(convolution),
                List.of(Buffer.QK_PROJECTED, Buffer.VALUE_Z_PROJECTED),
                List.of(Buffer.GDN_CONVOLVED),
                convolutionChannels,
                convolutionChannels));
        nodes.add(node(
                8,
                Kind.GDN_RECURRENCE,
                List.of(6, 7),
                List.of(),
                List.of(Buffer.GDN_CONVOLVED, Buffer.GDN_ALPHA, Buffer.GDN_BETA),
                List.of(Buffer.GDN_RECURRENT),
                convolutionChannels,
                valueProjected));
        nodes.add(node(
                9,
                Kind.GDN_GATED_RMS_NORM,
                List.of(8, 3),
                List.of(gdnNorm),
                List.of(Buffer.GDN_RECURRENT, Buffer.VALUE_Z_PROJECTED),
                List.of(Buffer.GDN_NORMALIZED),
                valueProjected,
                valueProjected));
        nodes.add(node(
                10,
                Kind.Q3_LINEAR,
                List.of(9),
                List.of(gdnOutput),
                List.of(Buffer.GDN_NORMALIZED),
                List.of(Buffer.MIXER_DELTA),
                valueProjected,
                hidden));
        nodes.add(node(
                11,
                Kind.RESIDUAL_ADD,
                List.of(0, 10),
                List.of(),
                List.of(Buffer.HIDDEN_STATE, Buffer.MIXER_DELTA),
                List.of(Buffer.MIXER_HIDDEN),
                hidden,
                hidden));
        nodes.add(node(
                12,
                Kind.RMS_NORM_UNIT_OFFSET,
                List.of(11),
                List.of(postNorm),
                List.of(Buffer.MIXER_HIDDEN),
                List.of(Buffer.POST_MIXER_NORMALIZED),
                hidden,
                hidden));
        nodes.add(node(
                13,
                Kind.Q3_LINEAR,
                List.of(12),
                List.of(gateUp),
                List.of(Buffer.POST_MIXER_NORMALIZED),
                List.of(Buffer.GATE_UP),
                hidden,
                2 * intermediate));
        nodes.add(node(
                14,
                Kind.SWIGLU,
                List.of(13),
                List.of(),
                List.of(Buffer.GATE_UP),
                List.of(Buffer.SWIGLU),
                2 * intermediate,
                intermediate));
        nodes.add(node(
                15,
                Kind.Q3_LINEAR,
                List.of(14),
                List.of(down),
                List.of(Buffer.SWIGLU),
                List.of(Buffer.FFN_DELTA),
                intermediate,
                hidden));
        nodes.add(node(
                16,
                Kind.RESIDUAL_ADD,
                List.of(11, 15),
                List.of(),
                List.of(Buffer.MIXER_HIDDEN, Buffer.FFN_DELTA),
                List.of(Buffer.FINAL_HIDDEN_STATE),
                hidden,
                hidden));

        List<BufferSpec> buffers = List.of(
                spec(Buffer.HIDDEN_STATE, hidden, ElementType.BF16),
                spec(Buffer.MIXER_HIDDEN, hidden, ElementType.BF16),
                spec(Buffer.FINAL_HIDDEN_STATE, hidden, ElementType.BF16),
                spec(Buffer.INPUT_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.QK_PROJECTED, keyProjected, ElementType.BF16),
                spec(Buffer.VALUE_Z_PROJECTED, valueZProjected, ElementType.BF16),
                spec(Buffer.A_PROJECTED, valueHeads, ElementType.FP32),
                spec(Buffer.B_PROJECTED, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_ALPHA, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_BETA, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_CONVOLVED, convolutionChannels, ElementType.BF16),
                spec(Buffer.GDN_RECURRENT, valueProjected, ElementType.BF16),
                spec(Buffer.GDN_NORMALIZED, valueProjected, ElementType.BF16),
                spec(Buffer.MIXER_DELTA, hidden, ElementType.BF16),
                spec(Buffer.POST_MIXER_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.GATE_UP, 2 * intermediate, ElementType.BF16),
                spec(Buffer.SWIGLU, intermediate, ElementType.BF16),
                spec(Buffer.FFN_DELTA, hidden, ElementType.BF16));
        return new PlanData(List.copyOf(nodes), List.of(), buffers, true);
    }

    private static PlanData fullModel(QwenWeights weights, TensorHandle embedding) {
        QwenConfig config = weights.config();
        QwenLayerType[] layerTypes = config.layerTypes();
        QwenLayerWeights[] layers = weights.layers();
        int hidden = config.hiddenSize();
        int intermediate = config.intermediateSize();
        int queryHeads = config.numAttentionHeads();
        int keyValueHeads = config.numKeyValueHeads();
        int attentionHeadDim = config.attentionHeadDim();
        int rotaryDim = (int) Math.round(attentionHeadDim * config.partialRotaryFactor());
        int keyHeads = config.linearNumKeyHeads();
        int valueHeads = config.linearNumValueHeads();
        int keyHeadDim = config.linearKeyHeadDim();
        int valueHeadDim = config.linearValueHeadDim();
        int kernelWidth = config.linearConvKernelDim();
        if (layerTypes == null
                || layerTypes.length != config.numHiddenLayers()
                || layers.length != config.numHiddenLayers()
                || hidden <= 0
                || hidden % 128 != 0
                || intermediate <= 0
                || queryHeads <= 0
                || keyValueHeads <= 0
                || queryHeads % keyValueHeads != 0
                || attentionHeadDim != 256
                || !config.attentionOutputGate()
                || !Double.isFinite(config.ropeTheta())
                || config.ropeTheta() <= 0
                || !Double.isFinite(config.partialRotaryFactor())
                || config.partialRotaryFactor() <= 0
                || config.partialRotaryFactor() > 1
                || rotaryDim <= 0
                || rotaryDim % 2 != 0
                || keyHeads <= 0
                || valueHeads <= 0
                || valueHeads % keyHeads != 0
                || keyHeadDim != 128
                || valueHeadDim != 128
                || kernelWidth < 2
                || kernelWidth > 32
                || !Double.isFinite(config.rmsNormEpsilon())
                || !Float.isFinite((float) config.rmsNormEpsilon())
                || (float) config.rmsNormEpsilon() <= 0) {
            throw new IllegalArgumentException("unsupported Qwen text model geometry");
        }

        int attentionQueryWidth = Math.multiplyExact(queryHeads, attentionHeadDim);
        int attentionKeyWidth = Math.multiplyExact(keyValueHeads, attentionHeadDim);
        int attentionQkWidth = Math.addExact(attentionQueryWidth, attentionKeyWidth);
        int attentionGateValueWidth = Math.addExact(attentionQueryWidth, attentionKeyWidth);
        int gdnQueryKeyWidth = Math.multiplyExact(Math.multiplyExact(2, keyHeads), keyHeadDim);
        int gdnValueWidth = Math.multiplyExact(valueHeads, valueHeadDim);
        int gdnValueZWidth = Math.multiplyExact(2, gdnValueWidth);
        int gdnConvolutionWidth = Math.addExact(gdnQueryKeyWidth, gdnValueWidth);

        TensorHandle finalNorm = validateNorm(weights.finalNorm(), hidden);
        TensorHandle outputHead =
                validateQuantized(weights.lmHead(), config.vocabSize(), hidden, WeightFormat.Q3_G64_FP16);
        List<Instruction> nodes = new ArrayList<>();
        nodes.add(node(
                0, Kind.EMBEDDING, List.of(), List.of(embedding), List.of(), List.of(Buffer.HIDDEN_STATE), 0, hidden));
        int precedingLayer = 0;

        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            QwenLayerWeights layer = layers[layerIndex];
            if (layer == null || layer.index() != layerIndex) {
                throw new IllegalArgumentException("loaded text layers must be indexed contiguously from zero");
            }
            if (!(layer.ffn() instanceof QwenCompactDenseFfnWeights ffn)) {
                throw new IllegalArgumentException("all loaded text layers must use compact dense FFN weights");
            }
            TensorHandle inputNorm = validateNorm(layer.inputNorm(), hidden);
            TensorHandle postNorm = validateNorm(layer.postAttentionNorm(), hidden);
            Buffer layerInput = layerIndex == 0 ? Buffer.HIDDEN_STATE : Buffer.FINAL_HIDDEN_STATE;
            int normId = addNode(
                    nodes,
                    Kind.RMS_NORM_UNIT_OFFSET,
                    layerIndex,
                    List.of(precedingLayer),
                    List.of(inputNorm),
                    List.of(layerInput),
                    List.of(Buffer.INPUT_NORMALIZED),
                    hidden,
                    hidden);

            int mixerProjectionId;
            if (layerTypes[layerIndex] == QwenLayerType.GATED_DELTA_NET) {
                if (!(layer.mixer() instanceof QwenCompactGatedDeltaNetWeights gdn)) {
                    throw new IllegalArgumentException("declared GDN layer has incompatible compact weights");
                }
                TensorHandle aLog = validateFp32Vector(gdn.aLog(), valueHeads);
                TensorHandle dtBias = validateFp32Vector(gdn.dtBias(), valueHeads);
                TensorHandle convolution = validateBf16Matrix(gdn.convolution(), kernelWidth, gdnConvolutionWidth);
                TensorHandle aProjection = validateBf16Matrix(gdn.aProjection(), valueHeads, hidden);
                TensorHandle bProjection = validateBf16Matrix(gdn.bProjection(), valueHeads, hidden);
                TensorHandle queryKey =
                        validateQuantized(gdn.queryKey(), gdnQueryKeyWidth, hidden, WeightFormat.Q4_G64_FP16);
                TensorHandle valueZ = validateQuantized(gdn.valueZ(), gdnValueZWidth, hidden, WeightFormat.Q5_G64_FP16);
                TensorHandle gdnNorm = validateNorm(gdn.norm(), valueHeadDim);
                TensorHandle gdnOutput =
                        validateQuantized(gdn.output(), hidden, gdnValueWidth, WeightFormat.Q3_G64_FP16);

                int queryKeyId = addNode(
                        nodes,
                        Kind.Q4_LINEAR,
                        layerIndex,
                        List.of(normId),
                        List.of(queryKey),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.QK_PROJECTED),
                        hidden,
                        gdnQueryKeyWidth);
                int valueZId = addNode(
                        nodes,
                        Kind.Q5_LINEAR,
                        layerIndex,
                        List.of(normId),
                        List.of(valueZ),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.VALUE_Z_PROJECTED),
                        hidden,
                        gdnValueZWidth);
                int aProjectionId = addNode(
                        nodes,
                        Kind.BF16_LINEAR,
                        layerIndex,
                        List.of(normId),
                        List.of(aProjection),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.A_PROJECTED),
                        hidden,
                        valueHeads);
                int bProjectionId = addNode(
                        nodes,
                        Kind.BF16_LINEAR,
                        layerIndex,
                        List.of(normId),
                        List.of(bProjection),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.B_PROJECTED),
                        hidden,
                        valueHeads);
                int controlId = addNode(
                        nodes,
                        Kind.GDN_CONTROL,
                        layerIndex,
                        List.of(aProjectionId, bProjectionId),
                        List.of(aLog, dtBias),
                        List.of(Buffer.A_PROJECTED, Buffer.B_PROJECTED),
                        List.of(Buffer.GDN_ALPHA, Buffer.GDN_BETA),
                        valueHeads,
                        valueHeads);
                int convolutionId = addNode(
                        nodes,
                        Kind.GDN_CONVOLUTION,
                        layerIndex,
                        List.of(queryKeyId, valueZId),
                        List.of(convolution),
                        List.of(Buffer.QK_PROJECTED, Buffer.VALUE_Z_PROJECTED),
                        List.of(Buffer.GDN_CONVOLVED),
                        gdnConvolutionWidth,
                        gdnConvolutionWidth);
                int recurrenceId = addNode(
                        nodes,
                        Kind.GDN_RECURRENCE,
                        layerIndex,
                        List.of(controlId, convolutionId),
                        List.of(),
                        List.of(Buffer.GDN_CONVOLVED, Buffer.GDN_ALPHA, Buffer.GDN_BETA),
                        List.of(Buffer.GDN_RECURRENT),
                        gdnConvolutionWidth,
                        gdnValueWidth);
                int gatedNormId = addNode(
                        nodes,
                        Kind.GDN_GATED_RMS_NORM,
                        layerIndex,
                        List.of(recurrenceId, valueZId),
                        List.of(gdnNorm),
                        List.of(Buffer.GDN_RECURRENT, Buffer.VALUE_Z_PROJECTED),
                        List.of(Buffer.GDN_NORMALIZED),
                        gdnValueWidth,
                        gdnValueWidth);
                mixerProjectionId = addNode(
                        nodes,
                        Kind.Q3_LINEAR,
                        layerIndex,
                        List.of(gatedNormId),
                        List.of(gdnOutput),
                        List.of(Buffer.GDN_NORMALIZED),
                        List.of(Buffer.MIXER_DELTA),
                        gdnValueWidth,
                        hidden);
            } else if (layerTypes[layerIndex] == QwenLayerType.FULL_ATTENTION) {
                if (!(layer.mixer() instanceof QwenCompactAttentionWeights attention)) {
                    throw new IllegalArgumentException(
                            "declared full-attention layer has incompatible compact weights");
                }
                TensorHandle queryKey =
                        validateQuantized(attention.queryKey(), attentionQkWidth, hidden, WeightFormat.Q4_G64_FP16);
                TensorHandle gateValue = validateQuantized(
                        attention.gateValue(), attentionGateValueWidth, hidden, WeightFormat.Q5_G64_FP16);
                TensorHandle queryNorm = validateNorm(attention.queryNorm(), attentionHeadDim);
                TensorHandle keyNorm = validateNorm(attention.keyNorm(), attentionHeadDim);
                TensorHandle attentionOutput =
                        validateQuantized(attention.output(), hidden, attentionQueryWidth, WeightFormat.Q3_G64_FP16);

                int queryKeyId = addNode(
                        nodes,
                        Kind.Q4_LINEAR,
                        layerIndex,
                        List.of(normId),
                        List.of(queryKey),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.QK_PROJECTED),
                        hidden,
                        attentionQkWidth);
                int gateValueId = addNode(
                        nodes,
                        Kind.Q5_LINEAR,
                        layerIndex,
                        List.of(normId),
                        List.of(gateValue),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.VALUE_Z_PROJECTED),
                        hidden,
                        attentionGateValueWidth);
                int qkNormRopeId = addNode(
                        nodes,
                        Kind.ATTENTION_QK_NORM_ROPE,
                        layerIndex,
                        List.of(queryKeyId),
                        List.of(queryNorm, keyNorm),
                        List.of(Buffer.QK_PROJECTED),
                        List.of(Buffer.ATTENTION_QK_NORMALIZED),
                        attentionQkWidth,
                        attentionQkWidth);
                int kvAppendId = addNode(
                        nodes,
                        Kind.ATTENTION_KV_APPEND,
                        layerIndex,
                        List.of(qkNormRopeId, gateValueId),
                        List.of(),
                        List.of(Buffer.ATTENTION_QK_NORMALIZED, Buffer.VALUE_Z_PROJECTED),
                        List.of(),
                        attentionQkWidth + attentionGateValueWidth,
                        0);
                int attentionId = addNode(
                        nodes,
                        Kind.ATTENTION_CAUSAL,
                        layerIndex,
                        List.of(qkNormRopeId, gateValueId, kvAppendId),
                        List.of(),
                        List.of(Buffer.ATTENTION_QK_NORMALIZED, Buffer.VALUE_Z_PROJECTED),
                        List.of(Buffer.ATTENTION_CONTEXT),
                        attentionQkWidth + attentionGateValueWidth,
                        attentionQueryWidth);
                mixerProjectionId = addNode(
                        nodes,
                        Kind.Q3_LINEAR,
                        layerIndex,
                        List.of(attentionId),
                        List.of(attentionOutput),
                        List.of(Buffer.ATTENTION_CONTEXT),
                        List.of(Buffer.MIXER_DELTA),
                        attentionQueryWidth,
                        hidden);
            } else {
                throw new IllegalArgumentException("unsupported declared Qwen layer type at " + layerIndex);
            }

            int mixerResidualId = addNode(
                    nodes,
                    Kind.RESIDUAL_ADD,
                    layerIndex,
                    List.of(precedingLayer, mixerProjectionId),
                    List.of(),
                    List.of(layerInput, Buffer.MIXER_DELTA),
                    List.of(Buffer.MIXER_HIDDEN),
                    hidden,
                    hidden);
            int postNormId = addNode(
                    nodes,
                    Kind.RMS_NORM_UNIT_OFFSET,
                    layerIndex,
                    List.of(mixerResidualId),
                    List.of(postNorm),
                    List.of(Buffer.MIXER_HIDDEN),
                    List.of(Buffer.POST_MIXER_NORMALIZED),
                    hidden,
                    hidden);
            TensorHandle gateUp = validateQuantized(
                    ffn.gateUp(), Math.multiplyExact(2, intermediate), hidden, WeightFormat.Q3_G64_FP16);
            TensorHandle down = validateQuantized(ffn.down(), hidden, intermediate, WeightFormat.Q3_G64_FP16);
            int gateUpId = addNode(
                    nodes,
                    Kind.Q3_LINEAR,
                    layerIndex,
                    List.of(postNormId),
                    List.of(gateUp),
                    List.of(Buffer.POST_MIXER_NORMALIZED),
                    List.of(Buffer.GATE_UP),
                    hidden,
                    2 * intermediate);
            int swigluId = addNode(
                    nodes,
                    Kind.SWIGLU,
                    layerIndex,
                    List.of(gateUpId),
                    List.of(),
                    List.of(Buffer.GATE_UP),
                    List.of(Buffer.SWIGLU),
                    2 * intermediate,
                    intermediate);
            int downId = addNode(
                    nodes,
                    Kind.Q3_LINEAR,
                    layerIndex,
                    List.of(swigluId),
                    List.of(down),
                    List.of(Buffer.SWIGLU),
                    List.of(Buffer.FFN_DELTA),
                    intermediate,
                    hidden);
            precedingLayer = addNode(
                    nodes,
                    Kind.RESIDUAL_ADD,
                    layerIndex,
                    List.of(mixerResidualId, downId),
                    List.of(),
                    List.of(Buffer.MIXER_HIDDEN, Buffer.FFN_DELTA),
                    List.of(Buffer.FINAL_HIDDEN_STATE),
                    hidden,
                    hidden);
        }

        int finalNormId = addNode(
                nodes,
                Kind.RMS_NORM_UNIT_OFFSET,
                -1,
                List.of(precedingLayer),
                List.of(finalNorm),
                List.of(Buffer.FINAL_HIDDEN_STATE),
                List.of(Buffer.FINAL_NORMALIZED),
                hidden,
                hidden);
        addNode(
                nodes,
                Kind.Q3_LINEAR,
                -1,
                List.of(finalNormId),
                List.of(outputHead),
                List.of(Buffer.FINAL_NORMALIZED),
                List.of(Buffer.LOGITS),
                hidden,
                config.vocabSize());

        return new PlanData(
                List.copyOf(nodes),
                List.of(),
                fullBufferSpecs(
                        hidden,
                        intermediate,
                        config.vocabSize(),
                        Math.max(gdnQueryKeyWidth, attentionQkWidth),
                        Math.max(gdnValueZWidth, attentionGateValueWidth),
                        valueHeads,
                        gdnConvolutionWidth,
                        gdnValueWidth,
                        attentionQkWidth,
                        attentionQueryWidth),
                true,
                true);
    }

    private static List<BufferSpec> fullBufferSpecs(
            int hidden,
            int intermediate,
            int vocabulary,
            int queryKeyWidth,
            int valueZWidth,
            int valueHeads,
            int convolutionWidth,
            int valueWidth,
            int attentionQkWidth,
            int attentionQueryWidth) {
        return List.of(
                spec(Buffer.HIDDEN_STATE, hidden, ElementType.BF16),
                spec(Buffer.MIXER_HIDDEN, hidden, ElementType.BF16),
                spec(Buffer.FINAL_HIDDEN_STATE, hidden, ElementType.BF16),
                spec(Buffer.INPUT_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.QK_PROJECTED, queryKeyWidth, ElementType.BF16),
                spec(Buffer.VALUE_Z_PROJECTED, valueZWidth, ElementType.BF16),
                spec(Buffer.A_PROJECTED, valueHeads, ElementType.FP32),
                spec(Buffer.B_PROJECTED, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_ALPHA, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_BETA, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_CONVOLVED, convolutionWidth, ElementType.BF16),
                spec(Buffer.GDN_RECURRENT, valueWidth, ElementType.BF16),
                spec(Buffer.GDN_NORMALIZED, valueWidth, ElementType.BF16),
                spec(Buffer.ATTENTION_QK_NORMALIZED, attentionQkWidth, ElementType.BF16),
                spec(Buffer.ATTENTION_CONTEXT, attentionQueryWidth, ElementType.BF16),
                spec(Buffer.MIXER_DELTA, hidden, ElementType.BF16),
                spec(Buffer.POST_MIXER_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.GATE_UP, 2 * intermediate, ElementType.BF16),
                spec(Buffer.SWIGLU, intermediate, ElementType.BF16),
                spec(Buffer.FFN_DELTA, hidden, ElementType.BF16),
                spec(Buffer.FINAL_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.LOGITS, vocabulary, ElementType.BF16));
    }

    private static PlanData operatorSlice(
            QwenWeights weights, TensorHandle normWeight, List<TensorHandle> projections) {
        Objects.requireNonNull(weights.config(), "config");
        Objects.requireNonNull(projections, "projections");
        int hiddenSize = weights.config().hiddenSize();
        int vocabularySize = weights.config().vocabSize();
        if (hiddenSize <= 0
                || vocabularySize <= 0
                || (normWeight == null && !projections.isEmpty())
                || (normWeight != null && projections.isEmpty())) {
            throw new IllegalArgumentException("invalid operator slice");
        }
        TensorHandle embedding =
                validateQuantized(weights.tokenEmbedding(), vocabularySize, hiddenSize, WeightFormat.Q3_G64_FP16);
        List<Instruction> nodes = new ArrayList<>();
        nodes.add(node(
                0,
                Kind.EMBEDDING,
                List.of(),
                List.of(embedding),
                List.of(),
                List.of(Buffer.HIDDEN_STATE),
                0,
                hiddenSize));
        if (normWeight != null) {
            TensorHandle norm = validateNorm(normWeight, hiddenSize);
            nodes.add(node(
                    1,
                    Kind.RMS_NORM,
                    List.of(0),
                    List.of(norm),
                    List.of(Buffer.HIDDEN_STATE),
                    List.of(Buffer.INPUT_NORMALIZED),
                    hiddenSize,
                    hiddenSize));
            for (int index = 0; index < projections.size(); index++) {
                TensorHandle projection = Objects.requireNonNull(projections.get(index), "projection");
                long[] shape = projection.shape();
                if (shape == null || shape.length != 2 || shape[0] <= 0 || shape[0] > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("invalid Q3 projection shape");
                }
                nodes.add(new Instruction(
                        nodes.size(),
                        Kind.Q3_LINEAR,
                        List.of(1),
                        List.of(validateQuantized(projection, shape[0], hiddenSize, WeightFormat.Q3_G64_FP16)),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.SLICE_PROJECTION),
                        hiddenSize,
                        Math.toIntExact(shape[0]),
                        index));
            }
        }
        List<Integer> widths = nodes.stream()
                .filter(instruction -> instruction.kind() == Kind.Q3_LINEAR)
                .map(Instruction::outputWidth)
                .toList();
        return new PlanData(List.copyOf(nodes), widths, List.of(), false);
    }

    private static Instruction node(
            int id,
            Kind kind,
            List<Integer> dependencies,
            List<TensorHandle> weights,
            List<Buffer> inputs,
            List<Buffer> outputs,
            int inputWidth,
            int outputWidth) {
        return new Instruction(id, kind, dependencies, weights, inputs, outputs, inputWidth, outputWidth, -1);
    }

    private static int addNode(
            List<Instruction> nodes,
            Kind kind,
            int layerIndex,
            List<Integer> dependencies,
            List<TensorHandle> weights,
            List<Buffer> inputs,
            List<Buffer> outputs,
            int inputWidth,
            int outputWidth) {
        int id = nodes.size();
        nodes.add(new Instruction(
                id, kind, dependencies, weights, inputs, outputs, inputWidth, outputWidth, -1, layerIndex));
        return id;
    }

    private static BufferSpec spec(Buffer buffer, int width, ElementType type) {
        return new BufferSpec(buffer, width, type);
    }

    private static TensorHandle validateNorm(TensorHandle norm, int width) {
        return validateDirect(norm, new long[] {width}, WeightFormat.BF16, (long) width * Short.BYTES);
    }

    private static TensorHandle validateFp32Vector(TensorHandle vector, int width) {
        return validateDirect(vector, new long[] {width}, WeightFormat.FP32, (long) width * Float.BYTES);
    }

    private static TensorHandle validateBf16Matrix(TensorHandle matrix, int rows, int columns) {
        return validateDirect(
                matrix, new long[] {rows, columns}, WeightFormat.BF16, (long) rows * columns * Short.BYTES);
    }

    private static TensorHandle validateDirect(
            TensorHandle handle, long[] expectedShape, WeightFormat format, long expectedByteSize) {
        Objects.requireNonNull(handle, "weight");
        long[] shape = handle.shape();
        if (shape == null
                || !java.util.Arrays.equals(shape, expectedShape)
                || handle.deviceAddress() == 0
                || handle.dataType() != TensorDataType.BF16
                || handle.format() != format
                || handle.layout() != WeightLayout.CONTIGUOUS_LE_V1
                || handle.byteSize() != expectedByteSize) {
            throw new IllegalArgumentException("unsupported direct weight layout or dimensions: " + handle.name());
        }
        return copyHandle(handle);
    }

    private static TensorHandle validateQuantized(TensorHandle handle, long rows, int width, WeightFormat format) {
        Objects.requireNonNull(handle, "weight");
        long[] shape = handle.shape();
        if (shape == null
                || shape.length != 2
                || shape[0] != rows
                || shape[1] != width
                || width % 64 != 0
                || handle.deviceAddress() == 0
                || handle.dataType() != TensorDataType.BF16
                || handle.format() != format
                || handle.layout() != WeightLayout.ROW_SPLIT_K128_V1
                || handle.byteSize()
                        != CompactTensorLayout.expectedByteSize(
                                shape, handle.dataType(), handle.format(), handle.layout())) {
            throw new IllegalArgumentException("unsupported quantized weight layout or dimensions: " + handle.name());
        }
        return copyHandle(handle);
    }

    private static TensorHandle copyHandle(TensorHandle handle) {
        return new TensorHandle(
                handle.name(),
                handle.shape().clone(),
                handle.dataType(),
                handle.format(),
                handle.layout(),
                handle.deviceAddress(),
                handle.byteSize());
    }
}
