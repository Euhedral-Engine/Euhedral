package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.artifact.WeightStaging;
import io.euhedral_execution.inference.core.model.qwen38.loader.AttentionWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Config;
import io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Weights;
import io.euhedral_execution.inference.core.model.qwen38.loader.DenseFfnWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.GdnWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.LayerWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.MtpWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.Weights;
import io.euhedral_execution.inference.core.runtime.graph.StageTopology;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/// Immutable operation instructions and dependency edges for the loaded Qwen text model.
/// Model weights are borrowed; sequence state and quantum workspace have separate owners.
public final class ExecutionPlan {

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
        Q3_GATE_UP_SWIGLU,
        SWIGLU,
        ATTENTION_QK_NORM_ROPE,
        ATTENTION_KV_APPEND,
        ATTENTION_CAUSAL,
        /// MTP stem (docs/MTP_CONTRACT.md §2): packs [RMSNorm₁₊w(embedding); RMSNorm₁₊w(seed hidden)] per row.
        MTP_STEM,
        /// Copies a host-backed weight into its staging slot. Its weight's `hostAddress` is the source
        /// and its `deviceAddress` the slot.
        WEIGHT_TRANSFER,
        /// Copies the target's hidden rows after one tapped layer into the sequence's DFlash2 tap rows (slot
        /// `outputBufferIndex`), in quanta that seed drafting; a no-op otherwise.
        DFLASH_TAP,
        /// DFlash2 drafter operators (docs/DFLASH2.md). A linear with no input buffer reads the sequence's tap rows.
        DFLASH_LINEAR,
        DFLASH_RMS_NORM,
        /// The dynamic convolution's prepare (`outputBufferIndex` 0) or finish (1) kernel.
        DFLASH_CONV,
        DFLASH_CONTEXT_KV,
        DFLASH_BLOCK_QK,
        DFLASH_ATTENTION,
        DFLASH_SWIGLU,
        /// The target's output head over the block's proposal rows (every row but the anchor).
        DFLASH_LM_HEAD,
        DFLASH_TOPK,
        DFLASH_SELECT
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
        FINAL_NORMALIZED,
        LOGITS,
        SLICE_PROJECTION,
        /// MTP stem scratch: one normalized half, then the packed [embedding; hidden] rows.
        MTP_NORMED,
        MTP_PACKED,
        /// DFlash2 drafter buffers.
        DRAFT_FUSED,
        DRAFT_CONTEXT,
        DRAFT_DYNAMIC,
        DRAFT_CONVOLVED,
        DRAFT_QUERY,
        DRAFT_KV,
        DRAFT_QUERY_ROPE,
        DRAFT_KEY_ROPE,
        DRAFT_TOPK_VALUES,
        DRAFT_TOPK_INDICES,
        DRAFT_TOPK_SCRATCH,
        DRAFT_ATTENTION_PARTIAL,
        DRAFT_SELECTOR,
        DRAFT_PROPOSAL,
        DRAFT_SCORES
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
            weights = weights.stream().map(ExecutionPlan::copyHandle).toList();
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
            return this.weights.stream().map(ExecutionPlan::copyHandle).toList();
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

        public WeightFormat weightFormat() {
            return weightFormat(0);
        }

        public WeightFormat weightFormat(int weightIndex) {
            return this.weights.get(weightIndex).format();
        }

        public WeightLayout weightLayout() {
            return weightLayout(0);
        }

        public WeightLayout weightLayout(int weightIndex) {
            return this.weights.get(weightIndex).layout();
        }

        public String weightName(int weightIndex) {
            return this.weights.get(weightIndex).name();
        }
    }

    record PlanData(
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

    /// Prefill specializations derived from the full-model reference topology. `SMALL` covers decode and
    /// quanta below the 64-row region tile; `REGIONS` fuses the FFN gate/up projection with SwiGLU.
    private enum PrefillView {
        SMALL,
        REGIONS
    }

    /// Row threshold for the gate/up and down regions (one 64-row prefill tile).
    private static final int REGION_MIN_ROWS = 64;

    private final Weights weights;
    /// The plan's own topology.
    private final Shape shape;
    private final WeightStaging staging;
    // The views of a full model; null otherwise.
    private final Shape smallPrefill;
    private final Shape decode;
    /// The decode view without the transfers of its first ring slots, for a quantum that finds them loaded by the
    /// prefetch of a DFlash2 block (null when no weight is host-backed).
    private final Shape decodePreloaded;
    private final Shape regionPrefill;
    /// The MTP draft view; null without loaded MTP weights and draft head.
    private final Shape mtpDraft;
    /// The DFlash2 views, null without the loaded drafter: its block (DRAFT) and its context rows (DRAFT_CONTEXT).
    private final Shape dflashBlock;
    private final Shape dflashContext;

    /// Fixed storage lifetime pairs of the region prefill views: each value lives in its owner's storage.
    static final List<Map.Entry<Buffer, Buffer>> REGION_STORAGE = List.of(
            Map.entry(Buffer.FINAL_HIDDEN_STATE, Buffer.HIDDEN_STATE),
            Map.entry(Buffer.POST_MIXER_NORMALIZED, Buffer.INPUT_NORMALIZED),
            Map.entry(Buffer.FFN_DELTA, Buffer.MIXER_DELTA),
            Map.entry(Buffer.SWIGLU, Buffer.VALUE_Z_PROJECTED));

    /// Adds the ordering edges that a single device stream used to provide implicitly. For every
    /// storage (aliased buffers count as one when `reuseStorage`), each reader is ordered after the
    /// last writer, and each writer after the last writer and every reader since it. Edges the
    /// dependencies already imply transitively are not added.
    static int[][] withStorageHazards(List<Instruction> instructions, int[][] dependencies, boolean reuseStorage) {
        int count = instructions.size();
        Map<Buffer, Buffer> owners = new EnumMap<>(Buffer.class);
        if (reuseStorage)
            for (Map.Entry<Buffer, Buffer> pair : REGION_STORAGE) owners.put(pair.getKey(), pair.getValue());
        int storages = Buffer.values().length;
        int[] lastWriter = new int[storages];
        java.util.Arrays.fill(lastWriter, -1);
        List<List<Integer>> readers = new ArrayList<>(storages);
        for (int storage = 0; storage < storages; storage++) readers.add(new ArrayList<>());
        java.util.BitSet[] ancestors = new java.util.BitSet[count];
        int[][] ordered = new int[count][];
        for (int stage = 0; stage < count; stage++) {
            Instruction instruction = instructions.get(stage);
            java.util.LinkedHashSet<Integer> edges = new java.util.LinkedHashSet<>();
            java.util.BitSet reach = new java.util.BitSet(count);
            for (int dependency : dependencies[stage]) {
                edges.add(dependency);
                reach.or(ancestors[dependency]);
                reach.set(dependency);
            }
            java.util.TreeSet<Integer> required = new java.util.TreeSet<>();
            for (Buffer buffer : instruction.inputBuffers()) {
                int storage = owners.getOrDefault(buffer, buffer).ordinal();
                if (lastWriter[storage] >= 0) required.add(lastWriter[storage]);
            }
            for (Buffer buffer : instruction.outputBuffers()) {
                int storage = owners.getOrDefault(buffer, buffer).ordinal();
                if (lastWriter[storage] >= 0) required.add(lastWriter[storage]);
                required.addAll(readers.get(storage));
            }
            // Latest first: an earlier requirement is often already an ancestor of a later one.
            for (int producer : required.descendingSet()) {
                if (producer == stage || reach.get(producer)) continue;
                edges.add(producer);
                reach.or(ancestors[producer]);
                reach.set(producer);
            }
            ancestors[stage] = reach;
            ordered[stage] = edges.stream().mapToInt(Integer::intValue).toArray();
            for (Buffer buffer : instruction.inputBuffers())
                readers.get(owners.getOrDefault(buffer, buffer).ordinal()).add(stage);
            for (Buffer buffer : instruction.outputBuffers()) {
                int storage = owners.getOrDefault(buffer, buffer).ordinal();
                lastWriter[storage] = stage;
                readers.get(storage).clear();
            }
        }
        return ordered;
    }

    /// Builds the production plan. For a complete model, prefill quanta automatically select the
    /// retained region architecture by row count and geometry; decode runs the small topology.
    /// Staged plans for partial weights (embedding-only, layer zero) remain reference-only.
    public ExecutionPlan(Weights weights) {
        this(weights, (WeightStaging) null);
    }

    /// Builds the production plan for weights that may be host-backed: every executed view stages
    /// them through `staging`, which must be present when any weight is host-backed.
    public ExecutionPlan(Weights weights, WeightStaging staging) {
        this(Objects.requireNonNull(weights, "weights"), planFromLoadedWeights(weights), staging);
    }

    /// Unfused reference topology for every execution kind. This is a correctness oracle for tests
    /// and is never selected by the runtime.
    public static ExecutionPlan reference(Weights weights) {
        Objects.requireNonNull(weights, "weights");
        PlanData data = planFromLoadedWeights(weights);
        return new ExecutionPlan(
                weights,
                new PlanData(data.instructions(), data.projectionWidths(), data.bufferSpecs(), data.firstLayer()),
                null);
    }

    /// The staging ring of this plan family, or null when no weight is host-backed.
    public WeightStaging staging() {
        return this.staging;
    }

    /// Selects the view a quantum of `kind` over `rows` rows runs. A plan without views runs its own topology.
    public Shape forExecution(Quantum.ExecutionKind kind, int rows) {
        Objects.requireNonNull(kind, "kind");
        if (rows <= 0) throw new IllegalArgumentException("rows must be positive");
        ExecutionPlan family = this;
        if (family.regionPrefill == null) return family.shape;
        // Decode runs its own instance of the small topology: rounded residual add + RMSNorm and the
        // joint GDN A/B projection + control are single region launches, as in short prefill quanta.
        if (kind == Quantum.ExecutionKind.DRAFT) {
            if (family.mtpDraft != null) return family.mtpDraft;
            if (family.dflashBlock != null) return family.dflashBlock;
            throw new IllegalStateException("the model has no loaded drafter");
        }
        if (kind == Quantum.ExecutionKind.DRAFT_CONTEXT) {
            if (family.dflashContext == null)
                throw new IllegalStateException("the model has no loaded DFlash2 drafter");
            return family.dflashContext;
        }
        if (kind == Quantum.ExecutionKind.DECODE || kind == Quantum.ExecutionKind.VERIFY) return family.decode;
        if (rows < REGION_MIN_ROWS) return family.smallPrefill;
        return family.regionPrefill;
    }

    static final String DRAFT_HEAD = "text/draft_head";

    /// Whether this plan's family can draft with MTP.
    public boolean drafts() {
        return this.mtpDraft != null;
    }

    /// Whether this plan's family can draft with DFlash2.
    public boolean draftsWithDFlash2() {
        return this.dflashBlock != null;
    }

    /// Rows of the draft head: the draft view's logits width.
    public int draftVocabularySize() {
        TensorHandle head = this.weights.runtimeObjects().get(DRAFT_HEAD);
        if (head == null) throw new IllegalStateException("the model has no draft head");
        return Math.toIntExact(head.shape()[0]);
    }

    /// The MTP draft view (docs/MTP_CONTRACT.md §2): embedding of each row's token, the MTP stem with the
    /// context's seed hidden rows, the input projection, then the MTP layer, built by [#fullModel] as a
    /// one-layer model with `mtp/final_norm` and the draft head, at the layer index after the base
    /// layers (its own attention cache).
    private static PlanData mtpDraft(Weights weights) {
        Qwen38Config c = weights.config();
        MtpWeights mtp = weights.mtp();
        TensorHandle head = weights.runtimeObjects().get(DRAFT_HEAD);
        int hidden = c.hiddenSize();
        int mtpLayer = c.numHiddenLayers();
        Qwen38Config one = new Qwen38Config(
                Math.toIntExact(head.shape()[0]),
                hidden,
                1,
                c.numAttentionHeads(),
                c.numKeyValueHeads(),
                c.attentionHeadDim(),
                c.intermediateSize(),
                c.linearNumKeyHeads(),
                c.linearNumValueHeads(),
                c.linearKeyHeadDim(),
                c.linearValueHeadDim(),
                c.linearConvKernelDim(),
                c.rmsNormEpsilon(),
                c.ropeTheta(),
                c.partialRotaryFactor(),
                c.maxPositionEmbeddings(),
                c.hiddenActivation(),
                new LayerType[] {LayerType.FULL_ATTENTION},
                c.numExperts(),
                c.numExpertsPerToken(),
                c.moeIntermediateSize(),
                c.sharedExpertIntermediateSize(),
                c.tieWordEmbeddings(),
                c.attentionOutputGate(),
                0);
        Weights single = new Weights(
                one, weights.tokenEmbedding(), new LayerWeights[] {mtp.layer()}, mtp.finalNorm(), head, null);
        PlanData layer = fullModel(single, validateEmbedding(weights.tokenEmbedding(), c.vocabSize(), hidden));
        TensorHandle projection = validateQuantized(mtp.projection(), hidden, 2 * hidden, WeightFormat.Q3_G64_FP16);
        TensorHandle embeddingNorm = validateNorm(mtp.embeddingNorm(), hidden);
        TensorHandle hiddenNorm = validateNorm(mtp.hiddenNorm(), hidden);
        List<Instruction> source = layer.instructions();
        Instruction embedding = source.getFirst();
        if (embedding.kind() != Kind.EMBEDDING)
            throw new IllegalStateException("one-layer plan must start with the embedding");
        List<Instruction> nodes = new ArrayList<>();
        nodes.add(new Instruction(
                0,
                Kind.EMBEDDING,
                List.of(),
                embedding.weights,
                List.of(),
                List.of(Buffer.HIDDEN_STATE),
                0,
                hidden,
                -1,
                -1));
        nodes.add(new Instruction(
                1,
                Kind.MTP_STEM,
                List.of(0),
                List.of(embeddingNorm, hiddenNorm),
                List.of(Buffer.HIDDEN_STATE),
                List.of(Buffer.MTP_PACKED),
                hidden,
                2 * hidden,
                -1,
                mtpLayer));
        nodes.add(new Instruction(
                2,
                Kind.Q3_LINEAR,
                List.of(1),
                List.of(projection),
                List.of(Buffer.MTP_PACKED),
                List.of(Buffer.HIDDEN_STATE),
                2 * hidden,
                hidden,
                -1,
                mtpLayer));
        for (int index = 1; index < source.size(); index++) {
            Instruction next = source.get(index);
            nodes.add(new Instruction(
                    index + 2,
                    next.kind(),
                    next.dependencies().stream().map(id -> id + 2).toList(),
                    next.weights,
                    next.inputBuffers(),
                    next.outputBuffers(),
                    next.inputWidth(),
                    next.outputWidth(),
                    next.outputBufferIndex(),
                    next.layerIndex() == 0 ? mtpLayer : next.layerIndex()));
        }
        List<BufferSpec> buffers = new ArrayList<>(layer.bufferSpecs());
        buffers.add(new BufferSpec(Buffer.MTP_NORMED, hidden, ElementType.BF16));
        buffers.add(new BufferSpec(Buffer.MTP_PACKED, 2 * hidden, ElementType.BF16));
        return new PlanData(List.copyOf(nodes), layer.projectionWidths(), List.copyOf(buffers), true, false);
    }

    /// Builds an embedding-only plan for callers that intentionally validate only token lookup.
    public static ExecutionPlan embeddingOnly(Weights weights) {
        Objects.requireNonNull(weights, "weights");
        int hiddenSize = weights.config().hiddenSize();
        int vocabularySize = weights.config().vocabSize();
        TensorHandle embedding = validateEmbedding(weights.tokenEmbedding(), vocabularySize, hiddenSize);
        return new ExecutionPlan(weights, embeddingOnly(embedding, hiddenSize));
    }

    /// Builds a dependency-graph prefix ending after the requested real layer, for staged validation.
    public static ExecutionPlan prefix(Weights weights, int layerCount) {
        Objects.requireNonNull(weights, "weights");
        if (layerCount <= 0 || layerCount > weights.config().numHiddenLayers()) {
            throw new IllegalArgumentException("layerCount must select a non-empty model prefix");
        }
        ExecutionPlan fullPlan = new ExecutionPlan(weights);
        int terminalId = -1;
        for (Instruction instruction : fullPlan.instructions()) {
            if (instruction.layerIndex() == layerCount - 1
                    && instruction.kind() == Kind.RESIDUAL_ADD
                    && instruction.outputBuffers().contains(Buffer.FINAL_HIDDEN_STATE)) {
                terminalId = instruction.id();
            }
        }
        if (terminalId < 0) throw new IllegalArgumentException("requested layer prefix has no final residual");
        List<Instruction> instructions = List.copyOf(fullPlan.instructions().subList(0, terminalId + 1));
        boolean[] usedBuffers = new boolean[Buffer.values().length];
        for (Instruction instruction : instructions) {
            for (Buffer buffer : instruction.inputBuffers()) usedBuffers[buffer.ordinal()] = true;
            for (Buffer buffer : instruction.outputBuffers()) usedBuffers[buffer.ordinal()] = true;
        }
        List<BufferSpec> buffers = fullPlan.bufferSpecs().stream()
                .filter(spec -> usedBuffers[spec.buffer().ordinal()])
                .toList();
        return new ExecutionPlan(weights, new PlanData(instructions, List.of(), buffers, true));
    }

    /// A standalone operator slice retained for low-level operation validation.
    public ExecutionPlan(Weights weights, TensorHandle normWeight, List<TensorHandle> projections) {
        this(Objects.requireNonNull(weights, "weights"), operatorSlice(weights, normWeight, projections));
    }

    private ExecutionPlan(Weights weights, PlanData data) {
        this(weights, data, null);
    }

    /// The plan of `data`; a full model also builds its views, which stage host-backed weights through `staging`.
    private ExecutionPlan(Weights weights, PlanData data, WeightStaging staging) {
        this.weights = weights;
        this.staging = staging;
        this.shape = new Shape(Shape.View.OWN, this, data, false, false);
        if (!data.fullModel()) {
            this.smallPrefill = null;
            this.decode = null;
            this.decodePreloaded = null;
            this.regionPrefill = null;
            this.mtpDraft = null;
            this.dflashBlock = null;
            this.dflashContext = null;
            return;
        }
        // Drafting needs the MTP attention split like a base layer's, which only a speculative load does.
        this.mtpDraft = weights.mtp() != null
                        && weights.mtp().layer().mixer()
                                instanceof io.euhedral_execution.inference.core.model.qwen38.loader.AttentionWeights
                        && weights.runtimeObjects().containsKey(DRAFT_HEAD)
                ? new Shape(Shape.View.MTP_DRAFT, this, staged(mtpDraft(weights), staging), false, false)
                : null;
        DFlash2Weights drafter = weights.dflash2();
        // A DFlash2 block leaves the host-backed transfer lane idle for milliseconds before every verification, so it
        // copies the decode view's first ring slots, and a verification that finds them loaded skips their transfers.
        List<TensorHandle> firstUses = staging == null
                ? List.of()
                : firstHostBackedUses(prefillView(data, PrefillView.SMALL), staging.slots());
        this.dflashBlock = drafter == null
                ? null
                : firstUses.isEmpty()
                        ? new Shape(
                                Shape.View.DFLASH_BLOCK,
                                this,
                                staged(dflash2Block(weights, drafter), staging),
                                false,
                                false)
                        : new Shape(
                                Shape.View.DFLASH_BLOCK,
                                this,
                                withPrefetch(staged(dflash2Block(weights, drafter), staging), firstUses),
                                false,
                                true);
        this.dflashContext = drafter == null
                ? null
                : new Shape(Shape.View.DFLASH_CONTEXT, this, staged(dflash2Context(drafter), staging), false, false);
        this.smallPrefill = prefillShape(Shape.View.SMALL_PREFILL, data, PrefillView.SMALL);
        this.decode = prefillShape(Shape.View.DECODE, data, PrefillView.SMALL);
        this.decodePreloaded = this.dflashBlock != null && this.dflashBlock.prefetchesRing()
                ? new Shape(
                        Shape.View.DECODE_PRELOADED,
                        this,
                        staged(prefillView(data, PrefillView.SMALL), staging, true),
                        false,
                        false)
                : null;
        this.regionPrefill = prefillShape(Shape.View.REGION_PREFILL, data, PrefillView.REGIONS);
    }

    private Shape prefillShape(Shape.View view, PlanData data, PrefillView prefill) {
        PlanData selected = prefillView(data, prefill);
        // The named lifetime pairs are qualified only for the region views with a fused FFN, not for
        // a shape that falls back to ordinary FFN.
        boolean regions = prefill == PrefillView.REGIONS;
        boolean hasFusedFfn = selected.instructions().stream().anyMatch(i -> i.kind() == Kind.Q3_GATE_UP_SWIGLU);
        return new Shape(view, this, staged(selected, this.staging), regions && hasFusedFfn, false);
    }

    /// The view to run when the ring holds what [Shape#prefetchesRing] loads: the decode view without the transfers of
    /// its first slots, or `view` when it has no such variant.
    Shape preloadedVariant(Shape view) {
        return view == this.decode && this.decodePreloaded != null ? this.decodePreloaded : view;
    }

    /// The plan's own topology: the reference or unfused form it was built as, which a plan without views runs.
    public Shape shape() {
        return this.shape;
    }

    /// The host-backed weights of `data`'s first `slots` staging uses, in use order: the weights its staged view
    /// copies into slots 0 .. slots - 1 before any slot is reused.
    private static List<TensorHandle> firstHostBackedUses(PlanData data, int slots) {
        List<TensorHandle> uses = new ArrayList<>();
        for (Instruction instruction : data.instructions()) {
            for (TensorHandle weight : instruction.weights) {
                if (weight.hostBacked() && uses.size() < slots) uses.add(weight);
            }
        }
        return List.copyOf(uses);
    }

    /// `data` with leaf transfers that copy `uses` into staging slots 0, 1, ... (the decode view's first slots).
    private PlanData withPrefetch(PlanData data, List<TensorHandle> uses) {
        List<Instruction> result = new ArrayList<>(data.instructions());
        for (int use = 0; use < uses.size(); use++) {
            TensorHandle weight = uses.get(use);
            TensorHandle transfer = new TensorHandle(
                    weight.name(),
                    weight.shape(),
                    weight.dataType(),
                    weight.format(),
                    weight.layout(),
                    this.staging.slotAddress(use),
                    weight.byteSize(),
                    weight.hostAddress());
            result.add(new Instruction(
                    result.size(),
                    Kind.WEIGHT_TRANSFER,
                    List.of(),
                    List.of(transfer),
                    List.of(),
                    List.of(),
                    0,
                    0,
                    -1,
                    -1));
        }
        return new PlanData(
                List.copyOf(result), data.projectionWidths(), data.bufferSpecs(), data.firstLayer(), data.fullModel());
    }

    /// Rewrites a view so that each use of a host-backed weight reads a staging slot. Use `u` takes slot
    /// `u mod slots`; its transfer follows the use that last read the slot (use `u - slots`), or starts
    /// the quantum, and precedes the consumer. Transfers are leaves apart from those two edges, so each
    /// copy starts as soon as its slot is free.
    static PlanData staged(PlanData data, WeightStaging staging) {
        return staged(data, staging, false);
    }

    /// [#staged(PlanData, WeightStaging)], with `preloaded` leaving out the transfers of the first `slots` uses: the
    /// quantum runs only after a prefetch loaded them into slots 0 .. slots - 1 ([#prefetchesRing]).
    static PlanData staged(PlanData data, WeightStaging staging, boolean preloaded) {
        List<Instruction> source = data.instructions();
        boolean hostBacked = source.stream().flatMap(i -> i.weights.stream()).anyMatch(TensorHandle::hostBacked);
        if (!hostBacked) return data;
        if (staging == null) throw new IllegalArgumentException("host-backed weights need a staging ring");
        int slots = staging.slots();
        for (Instruction instruction : source) {
            long uses = instruction.weights.stream()
                    .filter(TensorHandle::hostBacked)
                    .count();
            if (uses > slots) throw new IllegalArgumentException("an instruction stages more weights than slots");
            for (TensorHandle weight : instruction.weights) {
                if (weight.hostBacked() && weight.byteSize() > staging.slotBytes())
                    throw new IllegalArgumentException("host-backed weight exceeds a staging slot: " + weight.name());
            }
        }
        // Each use's weight, in order, and the transfers still to emit once a slot frees.
        List<TensorHandle> useWeights = new ArrayList<>();
        List<Integer> useLayers = new ArrayList<>();
        for (Instruction instruction : source) {
            for (TensorHandle weight : instruction.weights) {
                if (weight.hostBacked()) {
                    useWeights.add(weight);
                    useLayers.add(instruction.layerIndex());
                }
            }
        }
        List<Instruction> result = new ArrayList<>();
        int[] remapped = new int[source.size()];
        int[] transferIds = new int[useWeights.size()];
        java.util.function.IntConsumer emitTransfer = use -> {
            TensorHandle weight = useWeights.get(use);
            TensorHandle transfer = new TensorHandle(
                    weight.name(),
                    weight.shape(),
                    weight.dataType(),
                    weight.format(),
                    weight.layout(),
                    staging.slotAddress(use % slots),
                    weight.byteSize(),
                    weight.hostAddress());
            List<Integer> dependencies = use < slots ? List.of() : List.of(consumerOf(use - slots, source, remapped));
            transferIds[use] = result.size();
            result.add(new Instruction(
                    result.size(),
                    Kind.WEIGHT_TRANSFER,
                    dependencies,
                    List.of(transfer),
                    List.of(),
                    List.of(),
                    0,
                    0,
                    -1,
                    useLayers.get(use)));
        };
        if (!preloaded) for (int use = 0; use < Math.min(slots, useWeights.size()); use++) emitTransfer.accept(use);
        int use = 0;
        for (Instruction instruction : source) {
            List<Integer> dependencies = new ArrayList<>(remapDependencies(instruction.dependencies(), remapped));
            List<TensorHandle> weights = new ArrayList<>();
            int first = use;
            for (TensorHandle weight : instruction.weights) {
                if (!weight.hostBacked()) {
                    weights.add(weight);
                    continue;
                }
                if (!preloaded || use >= slots) dependencies.add(transferIds[use]);
                weights.add(new TensorHandle(
                        weight.name(),
                        weight.shape(),
                        weight.dataType(),
                        weight.format(),
                        weight.layout(),
                        staging.slotAddress(use % slots),
                        weight.byteSize()));
                use++;
            }
            remapped[instruction.id()] = result.size();
            result.add(new Instruction(
                    result.size(),
                    instruction.kind(),
                    dependencies.stream().distinct().sorted().toList(),
                    weights,
                    instruction.inputBuffers(),
                    instruction.outputBuffers(),
                    instruction.inputWidth(),
                    instruction.outputWidth(),
                    instruction.outputBufferIndex(),
                    instruction.layerIndex()));
            for (int staged = first; staged < use; staged++) {
                if (staged + slots < useWeights.size()) emitTransfer.accept(staged + slots);
            }
        }
        return new PlanData(
                List.copyOf(result), data.projectionWidths(), data.bufferSpecs(), data.firstLayer(), data.fullModel());
    }

    /// The remapped id of the instruction that reads use `use`; it is always already emitted.
    private static int consumerOf(int use, List<Instruction> source, int[] remapped) {
        int seen = 0;
        for (Instruction instruction : source) {
            for (TensorHandle weight : instruction.weights) {
                if (weight.hostBacked() && seen++ == use) return remapped[instruction.id()];
            }
        }
        throw new IllegalStateException("no consumer for staged use " + use);
    }

    /// Rewrites the reference layer DAG into the retained prefill regions:
    /// A residual+RMSNorm, D early joint GDN projection/control, and, from 64 rows, B gate/up+SwiGLU.
    /// The attention producers stay four leaf frames (Q projection, QK norm and
    /// RoPE, KV projection, cache append), so the KV branch runs on its own lane beside the Q branch.
    private static PlanData prefillView(PlanData data, PrefillView view) {
        boolean regions = view == PrefillView.REGIONS;
        List<Instruction> source = earlyControlOrder(data.instructions());
        List<Instruction> result = new ArrayList<>();
        int[] remapped = new int[source.size()];
        java.util.Arrays.fill(remapped, -1);
        for (int index = 0; index < source.size(); index++) {
            Instruction first = source.get(index);
            Instruction next = index + 1 < source.size() ? source.get(index + 1) : null;
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
            if (regions
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
            result.add(new Instruction(
                    result.size(),
                    first.kind(),
                    dependencies,
                    first.weights(),
                    first.inputBuffers(),
                    first.outputBuffers(),
                    first.inputWidth(),
                    first.outputWidth(),
                    first.outputBufferIndex(),
                    first.layerIndex()));
        }
        boolean needsGateUp = result.stream().anyMatch(i -> i.outputBuffers().contains(Buffer.GATE_UP));
        boolean needsSwiGlu = result.stream().anyMatch(i -> i.outputBuffers().contains(Buffer.SWIGLU));
        List<BufferSpec> buffers = new ArrayList<>(data.bufferSpecs().stream()
                .filter(spec -> needsSwiGlu || spec.buffer() != Buffer.SWIGLU)
                .filter(spec -> needsGateUp || spec.buffer() != Buffer.GATE_UP)
                .filter(spec -> spec.buffer() != Buffer.A_PROJECTED && spec.buffer() != Buffer.B_PROJECTED)
                .toList());
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

    /// The stage specs of the plan's own topology ([#shape]).
    public List<Instruction> instructions() {
        return this.shape.instructions();
    }

    /// The topology of the plan's own shape ([#shape]).
    public StageTopology stageTopology() {
        return this.shape.topology();
    }

    public StageTopology topology() {
        return this.shape.topology();
    }

    public List<Integer> successors(int instructionId) {
        return this.shape.successors(instructionId);
    }

    List<Integer> projectionWidths() {
        return this.shape.projectionWidths();
    }

    public List<BufferSpec> bufferSpecs() {
        return this.shape.bufferSpecs();
    }

    /// Every shape of this plan: its own and each view, once.
    List<Shape> shapes() {
        List<Shape> all = new ArrayList<>();
        for (Shape shape : new Shape[] {
            this.shape,
            this.smallPrefill,
            this.decode,
            this.decodePreloaded,
            this.regionPrefill,
            this.mtpDraft,
            this.dflashBlock,
            this.dflashContext
        }) {
            if (shape != null && all.stream().noneMatch(existing -> existing == shape)) all.add(shape);
        }
        return all;
    }

    /// The views this plan owns besides its own topology; empty for a plan without views.
    List<Shape> executionVariants() {
        if (this.regionPrefill == null) return List.of();
        return List.of(this.decode, this.smallPrefill, this.regionPrefill);
    }

    boolean reusePrefillStorage() {
        return this.shape.reusePrefillStorage();
    }

    public boolean hasFirstLayer() {
        return this.shape.hasFirstLayer();
    }

    public int bufferWidth(Buffer buffer) {
        return this.shape.bufferWidth(buffer);
    }

    public ElementType bufferElementType(Buffer buffer) {
        return this.shape.bufferElementType(buffer);
    }

    public Weights weights() {
        return this.weights;
    }

    private static PlanData planFromLoadedWeights(Weights weights) {
        Objects.requireNonNull(weights.config(), "config");
        int hiddenSize = weights.config().hiddenSize();
        int vocabularySize = weights.config().vocabSize();
        if (hiddenSize <= 0 || vocabularySize <= 0) {
            throw new IllegalArgumentException("invalid model dimensions");
        }
        TensorHandle embedding = validateEmbedding(weights.tokenEmbedding(), vocabularySize, hiddenSize);
        LayerWeights[] layers = weights.layers();
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

    private static PlanData firstLayer(Weights weights, TensorHandle embedding) {
        Qwen38Config config = weights.config();
        LayerWeights[] layers = weights.layers();
        if (layers[0] == null || layers[0].index() != 0) {
            throw new IllegalArgumentException("loaded weights do not contain actual text layer zero");
        }
        LayerType[] layerTypes = config.layerTypes();
        if (layerTypes == null || layerTypes.length == 0 || layerTypes[0] != LayerType.GATED_DELTA_NET) {
            throw new IllegalArgumentException("layer zero is not the loaded GDN topology");
        }
        if (!(layers[0].mixer() instanceof GdnWeights gdn) || !(layers[0].ffn() instanceof DenseFfnWeights ffn)) {
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

    /// The DFlash2 tap slot of target layer `layer`'s output, or -1.
    private static int tapSlot(DFlash2Weights drafter, int layer) {
        if (drafter == null) return -1;
        int[] layers = drafter.config().targetLayers();
        for (int slot = 0; slot < layers.length; slot++) if (layers[slot] == layer) return slot;
        return -1;
    }

    /// The DFlash2 context view (DRAFT_CONTEXT, docs/DFLASH2.md): the feature fusion of the sequence's tap rows, the
    /// hidden norm, then each draft layer's key/value projection, key norm and RoPE into the drafter's ring.
    private static PlanData dflash2Context(DFlash2Weights drafter) {
        DFlash2Config c = drafter.config();
        int hidden = c.hiddenSize();
        int kv = 2 * c.keyValueWidth();
        List<Instruction> nodes = new ArrayList<>();
        int fused = addNode(
                nodes,
                Kind.DFLASH_LINEAR,
                -1,
                List.of(),
                List.of(drafter.fc()),
                List.of(),
                List.of(Buffer.DRAFT_FUSED),
                c.tapWidth(),
                hidden);
        int context = addNode(
                nodes,
                Kind.DFLASH_RMS_NORM,
                -1,
                List.of(fused),
                List.of(drafter.hiddenNorm()),
                List.of(Buffer.DRAFT_FUSED),
                List.of(Buffer.DRAFT_CONTEXT),
                hidden,
                hidden);
        DFlash2Weights.Layer[] layers = drafter.layers();
        for (int layer = 0; layer < layers.length; layer++) {
            int projected = addNode(
                    nodes,
                    Kind.DFLASH_LINEAR,
                    layer,
                    List.of(context),
                    List.of(layers[layer].keyValue()),
                    List.of(Buffer.DRAFT_CONTEXT),
                    List.of(Buffer.DRAFT_KV),
                    hidden,
                    kv);
            addNode(
                    nodes,
                    Kind.DFLASH_CONTEXT_KV,
                    layer,
                    List.of(projected),
                    List.of(layers[layer].keyNorm()),
                    List.of(Buffer.DRAFT_KV),
                    List.of(),
                    kv,
                    kv);
        }
        List<BufferSpec> buffers = List.of(
                spec(Buffer.DRAFT_FUSED, hidden, ElementType.BF16),
                spec(Buffer.DRAFT_CONTEXT, hidden, ElementType.BF16),
                spec(Buffer.DRAFT_KV, kv, ElementType.BF16));
        return new PlanData(List.copyOf(nodes), List.of(), buffers, true, false);
    }

    /// The DFlash2 block view (DRAFT, docs/DFLASH2.md): the anchor and mask tokens' embedding, the draft layers over
    /// all rows of the block at once, the final norm, the target's output head over the proposal rows, their top
    /// candidates, and the selector's path.
    private static PlanData dflash2Block(Weights weights, DFlash2Weights drafter) {
        DFlash2Config c = drafter.config();
        Qwen38Config target = weights.config();
        int hidden = c.hiddenSize();
        int kv = 2 * c.keyValueWidth();
        TensorHandle embedding = validateEmbedding(weights.tokenEmbedding(), target.vocabSize(), hidden);
        TensorHandle outputHead =
                validateQuantized(weights.lmHead(), target.vocabSize(), hidden, WeightFormat.Q3_G64_FP16);
        List<Instruction> nodes = new ArrayList<>();
        int residual = addNode(
                nodes,
                Kind.EMBEDDING,
                -1,
                List.of(),
                List.of(embedding),
                List.of(),
                List.of(Buffer.HIDDEN_STATE),
                0,
                hidden);
        DFlash2Weights.Layer[] layers = drafter.layers();
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            final int index = layerIndex;
            DFlash2Weights.Layer layer = layers[index];
            int normed = addNode(
                    nodes,
                    Kind.DFLASH_RMS_NORM,
                    index,
                    List.of(residual),
                    List.of(layer.inputNorm()),
                    List.of(Buffer.HIDDEN_STATE),
                    List.of(Buffer.INPUT_NORMALIZED),
                    hidden,
                    hidden);
            int delta = convolved(
                    nodes,
                    index,
                    normed,
                    layer.attentionConvProjection(),
                    layer.attentionConvBase(),
                    c,
                    convolvedInput -> {
                        int query = addNode(
                                nodes,
                                Kind.DFLASH_LINEAR,
                                index,
                                List.of(convolvedInput),
                                List.of(layer.query()),
                                List.of(Buffer.DRAFT_CONVOLVED),
                                List.of(Buffer.DRAFT_QUERY),
                                hidden,
                                c.queryWidth());
                        int keyValue = addNode(
                                nodes,
                                Kind.DFLASH_LINEAR,
                                index,
                                List.of(convolvedInput),
                                List.of(layer.keyValue()),
                                List.of(Buffer.DRAFT_CONVOLVED),
                                List.of(Buffer.DRAFT_KV),
                                hidden,
                                kv);
                        int rotated = addNode(
                                nodes,
                                Kind.DFLASH_BLOCK_QK,
                                index,
                                List.of(query, keyValue),
                                List.of(layer.queryNorm(), layer.keyNorm()),
                                List.of(Buffer.DRAFT_QUERY, Buffer.DRAFT_KV),
                                List.of(Buffer.DRAFT_QUERY_ROPE, Buffer.DRAFT_KEY_ROPE),
                                c.queryWidth(),
                                c.queryWidth());
                        int attended = addNode(
                                nodes,
                                Kind.DFLASH_ATTENTION,
                                index,
                                List.of(rotated),
                                List.of(),
                                List.of(Buffer.DRAFT_QUERY_ROPE, Buffer.DRAFT_KEY_ROPE, Buffer.DRAFT_KV),
                                List.of(Buffer.ATTENTION_CONTEXT, Buffer.DRAFT_ATTENTION_PARTIAL),
                                c.queryWidth(),
                                c.queryWidth());
                        return addNode(
                                nodes,
                                Kind.DFLASH_LINEAR,
                                index,
                                List.of(attended),
                                List.of(layer.output()),
                                List.of(Buffer.ATTENTION_CONTEXT),
                                List.of(Buffer.MIXER_DELTA),
                                c.queryWidth(),
                                hidden);
                    });
            residual = addNode(
                    nodes,
                    Kind.RESIDUAL_ADD,
                    index,
                    List.of(residual, delta),
                    List.of(),
                    List.of(Buffer.HIDDEN_STATE, Buffer.FFN_DELTA),
                    List.of(Buffer.HIDDEN_STATE),
                    hidden,
                    hidden);
            int postNormed = addNode(
                    nodes,
                    Kind.DFLASH_RMS_NORM,
                    index,
                    List.of(residual),
                    List.of(layer.postAttentionNorm()),
                    List.of(Buffer.HIDDEN_STATE),
                    List.of(Buffer.INPUT_NORMALIZED),
                    hidden,
                    hidden);
            int mlpDelta = convolved(
                    nodes, index, postNormed, layer.mlpConvProjection(), layer.mlpConvBase(), c, convolvedInput -> {
                        int gateUp = addNode(
                                nodes,
                                Kind.DFLASH_LINEAR,
                                index,
                                List.of(convolvedInput),
                                List.of(layer.gateUp()),
                                List.of(Buffer.DRAFT_CONVOLVED),
                                List.of(Buffer.GATE_UP),
                                hidden,
                                2 * c.intermediateSize());
                        int activated = addNode(
                                nodes,
                                Kind.DFLASH_SWIGLU,
                                index,
                                List.of(gateUp),
                                List.of(),
                                List.of(Buffer.GATE_UP),
                                List.of(Buffer.SWIGLU),
                                2 * c.intermediateSize(),
                                c.intermediateSize());
                        return addNode(
                                nodes,
                                Kind.DFLASH_LINEAR,
                                index,
                                List.of(activated),
                                List.of(layer.down()),
                                List.of(Buffer.SWIGLU),
                                List.of(Buffer.MIXER_DELTA),
                                c.intermediateSize(),
                                hidden);
                    });
            residual = addNode(
                    nodes,
                    Kind.RESIDUAL_ADD,
                    index,
                    List.of(residual, mlpDelta),
                    List.of(),
                    List.of(Buffer.HIDDEN_STATE, Buffer.FFN_DELTA),
                    List.of(Buffer.HIDDEN_STATE),
                    hidden,
                    hidden);
        }
        int finalNormed = addNode(
                nodes,
                Kind.DFLASH_RMS_NORM,
                -1,
                List.of(residual),
                List.of(drafter.finalNorm()),
                List.of(Buffer.HIDDEN_STATE),
                List.of(Buffer.FINAL_NORMALIZED),
                hidden,
                hidden);
        int logits = addNode(
                nodes,
                Kind.DFLASH_LM_HEAD,
                -1,
                List.of(finalNormed),
                List.of(outputHead),
                List.of(Buffer.FINAL_NORMALIZED),
                List.of(Buffer.LOGITS),
                hidden,
                target.vocabSize());
        int top = addNode(
                nodes,
                Kind.DFLASH_TOPK,
                -1,
                List.of(logits),
                List.of(),
                List.of(Buffer.LOGITS),
                List.of(Buffer.DRAFT_TOPK_VALUES, Buffer.DRAFT_TOPK_INDICES, Buffer.DRAFT_TOPK_SCRATCH),
                target.vocabSize(),
                c.selectorTopK());
        int projected = addNode(
                nodes,
                Kind.DFLASH_LINEAR,
                -1,
                List.of(finalNormed),
                List.of(drafter.selectorProjection()),
                List.of(Buffer.FINAL_NORMALIZED),
                List.of(Buffer.DRAFT_SELECTOR),
                hidden,
                c.selectorRank());
        addNode(
                nodes,
                Kind.DFLASH_SELECT,
                -1,
                List.of(top, projected),
                List.of(drafter.predecessorCodebook(), drafter.successorCodebook()),
                List.of(Buffer.DRAFT_SELECTOR, Buffer.DRAFT_TOPK_VALUES, Buffer.DRAFT_TOPK_INDICES),
                List.of(Buffer.DRAFT_PROPOSAL, Buffer.DRAFT_SCORES),
                c.selectorRank(),
                c.selectorTopK());
        List<BufferSpec> buffers = List.of(
                spec(Buffer.HIDDEN_STATE, hidden, ElementType.BF16),
                spec(Buffer.INPUT_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.DRAFT_DYNAMIC, c.convProjectionWidth(), ElementType.BF16),
                spec(Buffer.DRAFT_CONVOLVED, hidden, ElementType.BF16),
                spec(Buffer.DRAFT_QUERY, c.queryWidth(), ElementType.BF16),
                spec(Buffer.DRAFT_KV, kv, ElementType.BF16),
                spec(Buffer.DRAFT_QUERY_ROPE, c.queryWidth(), ElementType.BF16),
                spec(Buffer.DRAFT_KEY_ROPE, c.keyValueWidth(), ElementType.BF16),
                spec(Buffer.ATTENTION_CONTEXT, c.queryWidth(), ElementType.BF16),
                // Per row: 8 key splits × query heads × (128 partial outputs, maximum, sum).
                spec(Buffer.DRAFT_ATTENTION_PARTIAL, 8 * c.attentionHeads() * 130, ElementType.FP32),
                spec(Buffer.MIXER_DELTA, hidden, ElementType.BF16),
                spec(Buffer.FFN_DELTA, hidden, ElementType.BF16),
                spec(Buffer.GATE_UP, 2 * c.intermediateSize(), ElementType.BF16),
                spec(Buffer.SWIGLU, c.intermediateSize(), ElementType.BF16),
                spec(Buffer.FINAL_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.LOGITS, target.vocabSize(), ElementType.BF16),
                spec(Buffer.DRAFT_TOPK_VALUES, c.selectorTopK(), ElementType.BF16),
                spec(Buffer.DRAFT_TOPK_INDICES, c.selectorTopK(), ElementType.FP32),
                spec(Buffer.DRAFT_TOPK_SCRATCH, 2 * 64 * c.selectorTopK(), ElementType.FP32),
                spec(Buffer.DRAFT_SELECTOR, c.selectorRank(), ElementType.BF16),
                spec(Buffer.DRAFT_PROPOSAL, 1, ElementType.FP32),
                spec(Buffer.DRAFT_SCORES, c.selectorTopK(), ElementType.FP32));
        return new PlanData(List.copyOf(nodes), List.of(), buffers, true, false);
    }

    /// A dynamic convolution around `body`: the kernel projection of the normalized rows, the prepare convolution
    /// into DRAFT_CONVOLVED, `body` (reading it, writing MIXER_DELTA), then the finish convolution into FFN_DELTA.
    /// Returns the finish stage.
    private static int convolved(
            List<Instruction> nodes,
            int layer,
            int normed,
            TensorHandle projection,
            TensorHandle base,
            DFlash2Config c,
            java.util.function.IntUnaryOperator body) {
        int hidden = c.hiddenSize();
        int dynamic = addNode(
                nodes,
                Kind.DFLASH_LINEAR,
                layer,
                List.of(normed),
                List.of(projection),
                List.of(Buffer.INPUT_NORMALIZED),
                List.of(Buffer.DRAFT_DYNAMIC),
                hidden,
                c.convProjectionWidth());
        nodes.add(new Instruction(
                nodes.size(),
                Kind.DFLASH_CONV,
                List.of(normed, dynamic),
                List.of(base),
                List.of(Buffer.INPUT_NORMALIZED, Buffer.DRAFT_DYNAMIC),
                List.of(Buffer.DRAFT_CONVOLVED),
                hidden,
                hidden,
                0,
                layer));
        int prepared = nodes.size() - 1;
        int inner = body.applyAsInt(prepared);
        nodes.add(new Instruction(
                nodes.size(),
                Kind.DFLASH_CONV,
                List.of(inner, dynamic),
                List.of(base),
                List.of(Buffer.MIXER_DELTA, Buffer.DRAFT_DYNAMIC),
                List.of(Buffer.FFN_DELTA),
                hidden,
                hidden,
                1,
                layer));
        return nodes.size() - 1;
    }

    private static PlanData fullModel(Weights weights, TensorHandle embedding) {
        Qwen38Config config = weights.config();
        LayerType[] layerTypes = config.layerTypes();
        LayerWeights[] layers = weights.layers();
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
            LayerWeights layer = layers[layerIndex];
            if (layer == null || layer.index() != layerIndex) {
                throw new IllegalArgumentException("loaded text layers must be indexed contiguously from zero");
            }
            if (!(layer.ffn() instanceof DenseFfnWeights ffn)) {
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
            // The previous layer's output feeds a DFlash2 tap. The tap follows this layer's input norm, which the
            // prefill and decode views fuse with the residual that produced it, and reads only that residual.
            int tap = layerIndex == 0 ? -1 : tapSlot(weights.dflash2(), layerIndex - 1);
            if (tap >= 0)
                nodes.add(new Instruction(
                        nodes.size(),
                        Kind.DFLASH_TAP,
                        List.of(precedingLayer),
                        List.of(),
                        List.of(Buffer.FINAL_HIDDEN_STATE),
                        List.of(),
                        hidden,
                        hidden,
                        tap,
                        layerIndex - 1));

            int mixerProjectionId;
            if (layerTypes[layerIndex] == LayerType.GATED_DELTA_NET) {
                if (!(layer.mixer() instanceof GdnWeights gdn)) {
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
            } else if (layerTypes[layerIndex] == LayerType.FULL_ATTENTION) {
                if (!(layer.mixer() instanceof AttentionWeights attention)) {
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

    private static PlanData operatorSlice(Weights weights, TensorHandle normWeight, List<TensorHandle> projections) {
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
        TensorHandle embedding = validateEmbedding(weights.tokenEmbedding(), vocabularySize, hiddenSize);
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

    /// Projections may be NVFP4 wherever a row-split format is registered; the token embedding, a
    /// gather, has Q3 kernels only.
    private static final WeightFormat EMBEDDING_FORMAT = WeightFormat.Q3_G64_FP16;

    private static TensorHandle validateEmbedding(TensorHandle handle, long rows, int width) {
        if (handle.format() != EMBEDDING_FORMAT) {
            throw new IllegalArgumentException("unsupported token embedding format: " + handle.format());
        }
        return validateQuantized(handle, rows, width, EMBEDDING_FORMAT);
    }

    private static TensorHandle validateQuantized(TensorHandle handle, long rows, int width, WeightFormat format) {
        Objects.requireNonNull(handle, "weight");
        long[] shape = handle.shape();
        if (shape == null
                || shape.length != 2
                || shape[0] != rows
                || shape[1] != width
                || width % 64 != 0
                || (handle.deviceAddress() == 0 && !handle.hostBacked())
                || handle.dataType() != TensorDataType.BF16
                || !(handle.format() == format || handle.format() == WeightFormat.NVFP4)
                || !(handle.layout() == WeightLayout.ROW_SPLIT_K128_V1
                        || (handle.layout() == WeightLayout.ROW_SPLIT_P2E2_V1 && format == WeightFormat.Q3_G64_FP16)
                        || (handle.layout() == WeightLayout.ROW_SPLIT_K128_SD4_V1
                                && handle.format() == WeightFormat.NVFP4))
                || !CompactTensorLayout.acceptsByteSize(
                        shape, handle.dataType(), handle.format(), handle.layout(), handle.byteSize())) {
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
                handle.byteSize(),
                handle.hostAddress(),
                handle.hostMapped());
    }
}
