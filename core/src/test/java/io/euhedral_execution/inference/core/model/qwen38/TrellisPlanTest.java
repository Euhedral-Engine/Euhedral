package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.artifact.WeightStaging;
import io.euhedral_execution.inference.core.artifact.format.LinearFormats;
import io.euhedral_execution.inference.core.artifact.format.LinearKernel;
import io.euhedral_execution.inference.core.artifact.format.LinearLaunch;
import io.euhedral_execution.inference.core.artifact.format.TrellisFormat;
import io.euhedral_execution.inference.core.artifact.format.TrellisSettings;
import io.euhedral_execution.inference.core.gpu.KernelArguments;
import io.euhedral_execution.inference.core.gpu.ScratchUse;
import io.euhedral_execution.inference.core.gpu.TableKernel;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan.Buffer;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan.Kind;
import io.euhedral_execution.inference.core.model.qwen38.loader.AttentionWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.DenseFfnWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.GdnWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.LayerWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.MtpWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.Weights;
import io.euhedral_execution.inference.core.model.qwen38.offload.DiskWindow;
import io.euhedral_execution.inference.core.runtime.graph.StageTopology;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;
import org.junit.jupiter.api.Test;

/// Trellis linears in the plan and the stage chain: the instructions carry the payload and its companions, staging
/// gives only the payload a slot, every view declares the scratch users in order, and each route launches the
/// kernels the native table's contracts name.
class TrellisPlanTest {

    private static final AtomicLong ADDRESSES = new AtomicLong(1L << 30);
    private static final int W = 56;

    // Swift-1.5's geometry.
    private static final int HIDDEN = 5120, INTERMEDIATE = 17408, VOCABULARY = 248320;

    /// A model of `types` layers whose projections are all trellis objects of `W` words per tile, with the
    /// segments of the engine's fused objects; `host` and `disk` name the payloads kept off the device.
    private record Model(Weights weights, Map<String, TensorHandle> objects) {}

    private static Model model(
            LayerType[] types,
            int hidden,
            int intermediate,
            int queryHeads,
            int keyValueHeads,
            int keyHeads,
            int valueHeads,
            Predicate<String> host,
            Predicate<String> disk) {
        Map<String, TensorHandle> objects = new LinkedHashMap<>();
        Builder b = new Builder(objects, host, disk);
        LayerWeights[] layers = new LayerWeights[types.length];
        int queryWidth = queryHeads * 256, attentionQk = (queryHeads + keyValueHeads) * 256;
        int gdnQk = 2 * keyHeads * 128, gdnValue = valueHeads * 128;
        for (int index = 0; index < types.length; index++) {
            String prefix = "text/layers/" + index;
            var mixer = types[index] == LayerType.FULL_ATTENTION
                    ? new AttentionWeights(
                            b.trellis(prefix + "/attention/query_key", attentionQk, hidden, queryWidth, attentionQk),
                            b.trellis(prefix + "/attention/gate_value", attentionQk, hidden, queryWidth, attentionQk),
                            b.direct(prefix + "/attention/query_norm", WeightFormat.BF16, 256),
                            b.direct(prefix + "/attention/key_norm", WeightFormat.BF16, 256),
                            b.trellis(prefix + "/attention/output", hidden, queryWidth, hidden))
                    : new GdnWeights(
                            b.direct(prefix + "/gdn/a_log", WeightFormat.FP32, valueHeads),
                            b.direct(prefix + "/gdn/dt_bias", WeightFormat.FP32, valueHeads),
                            b.direct(prefix + "/gdn/convolution", WeightFormat.BF16, 4, gdnQk + gdnValue),
                            b.direct(prefix + "/gdn/a_projection", WeightFormat.BF16, valueHeads, hidden),
                            b.direct(prefix + "/gdn/b_projection", WeightFormat.BF16, valueHeads, hidden),
                            b.trellis(prefix + "/gdn/query_key", gdnQk, hidden, gdnQk),
                            b.trellis(prefix + "/gdn/value_z", 2 * gdnValue, hidden, gdnValue, 2 * gdnValue),
                            b.direct(prefix + "/gdn/norm", WeightFormat.BF16, 128),
                            b.trellis(prefix + "/gdn/output", hidden, gdnValue, hidden));
            layers[index] = new LayerWeights(
                    index,
                    b.direct(prefix + "/input_norm", WeightFormat.BF16, hidden),
                    b.direct(prefix + "/post_attention_norm", WeightFormat.BF16, hidden),
                    mixer,
                    new DenseFfnWeights(
                            b.trellis(
                                    prefix + "/mlp/gate_up", 2 * intermediate, hidden, intermediate, 2 * intermediate),
                            b.trellis(prefix + "/mlp/down", hidden, intermediate, hidden)));
        }
        Qwen38Config config = new Qwen38Config(
                VOCABULARY,
                hidden,
                types.length,
                queryHeads,
                keyValueHeads,
                256,
                intermediate,
                keyHeads,
                valueHeads,
                128,
                128,
                4,
                1.0e-6,
                10_000_000.0,
                0.25,
                262144,
                "silu",
                types,
                0,
                0,
                0,
                0,
                false,
                true,
                1);
        Weights weights = new Weights(
                config,
                b.q3("text/token_embedding", VOCABULARY, hidden),
                layers,
                b.direct("text/final_norm", WeightFormat.BF16, hidden),
                b.q3("text/output_head", VOCABULARY, hidden),
                null,
                objects);
        return new Model(weights.withSegmentEnds(b.ends), objects);
    }

    /// Two layers (GDN, attention) of the smallest geometry with a trellis MTP layer and a draft head.
    private static Model mtpModel() {
        LayerType[] types = {LayerType.GATED_DELTA_NET, LayerType.FULL_ATTENTION};
        Model base = model(types, 128, 128, 1, 1, 1, 1, n -> false, n -> false);
        Builder b = new Builder(new LinkedHashMap<>(base.objects()), n -> false, n -> false);
        b.ends.putAll(base.weights().segmentEnds());
        int hidden = 128;
        String prefix = "mtp/layer";
        LayerWeights layer = new LayerWeights(
                0,
                b.direct(prefix + "/input_norm", WeightFormat.BF16, hidden),
                b.direct(prefix + "/post_attention_norm", WeightFormat.BF16, hidden),
                new AttentionWeights(
                        b.trellis(prefix + "/attention/query_key", 512, hidden, 256, 512),
                        b.trellis(prefix + "/attention/gate_value", 512, hidden, 256, 512),
                        b.direct(prefix + "/attention/query_norm", WeightFormat.BF16, 256),
                        b.direct(prefix + "/attention/key_norm", WeightFormat.BF16, 256),
                        b.trellis(prefix + "/attention/output", hidden, 256, hidden)),
                new DenseFfnWeights(
                        b.trellis(prefix + "/mlp/gate_up", 256, hidden, 128, 256),
                        b.trellis(prefix + "/mlp/down", hidden, 128, hidden)));
        MtpWeights mtp = new MtpWeights(
                b.direct("mtp/embedding_norm", WeightFormat.BF16, hidden),
                b.direct("mtp/hidden_norm", WeightFormat.BF16, hidden),
                b.trellis("mtp/input_projection", hidden, 2 * hidden, hidden),
                layer,
                b.direct("mtp/final_norm", WeightFormat.BF16, hidden));
        b.q3(ExecutionPlan.DRAFT_HEAD, VOCABULARY, hidden);
        b.direct("text/draft_head_token_ids", WeightFormat.FP32, VOCABULARY);
        Weights w = base.weights();
        Weights weights = new Weights(
                        w.config(), w.tokenEmbedding(), w.layers(), w.finalNorm(), w.lmHead(), mtp, b.objects)
                .withSegmentEnds(b.ends);
        return new Model(weights, b.objects);
    }

    private static Model fullModel(Predicate<String> host, Predicate<String> disk) {
        LayerType[] types = new LayerType[64];
        for (int index = 0; index < types.length; index++)
            types[index] = index % 4 == 3 ? LayerType.FULL_ATTENTION : LayerType.GATED_DELTA_NET;
        return model(types, HIDDEN, INTERMEDIATE, 24, 4, 16, 48, host, disk);
    }

    /// The builder of handles; every handle is also an entry of `objects`, as the loader's runtime objects are.
    private static final class Builder {
        final Map<String, TensorHandle> objects;
        final Predicate<String> host, disk;
        final Map<String, int[]> ends = new LinkedHashMap<>();

        Builder(Map<String, TensorHandle> objects, Predicate<String> host, Predicate<String> disk) {
            this.objects = objects;
            this.host = host;
            this.disk = disk;
        }

        TensorHandle put(TensorHandle handle) {
            objects.put(handle.name(), handle);
            return handle;
        }

        TensorHandle direct(String name, WeightFormat format, int... dimensions) {
            long[] shape = new long[dimensions.length];
            long elements = 1;
            for (int index = 0; index < dimensions.length; index++) {
                shape[index] = dimensions[index];
                elements *= dimensions[index];
            }
            return put(new TensorHandle(
                    name,
                    shape,
                    TensorDataType.BF16,
                    format,
                    WeightLayout.CONTIGUOUS_LE_V1,
                    ADDRESSES.getAndAdd(1 << 12),
                    elements * (format == WeightFormat.FP32 ? 4 : 2)));
        }

        TensorHandle q3(String name, int rows, int columns) {
            long[] shape = {rows, columns};
            return put(new TensorHandle(
                    name,
                    shape,
                    TensorDataType.BF16,
                    WeightFormat.Q3_G64_FP16,
                    WeightLayout.ROW_SPLIT_K128_V1,
                    ADDRESSES.getAndAdd(1 << 12),
                    CompactTensorLayout.expectedByteSize(
                            shape, TensorDataType.BF16, WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_K128_V1)));
        }

        /// A trellis object of `n` rows by `k` columns cut into segments ending at `ends`, with its companions.
        TensorHandle trellis(String name, int n, int k, int... ends) {
            long[] shape = {k / 16, n / 16, W};
            long bytes = (long) (k / 16) * (n / 16) * W * 2;
            TensorHandle payload;
            if (disk.test(name))
                payload = new TensorHandle(
                        name,
                        shape,
                        TensorDataType.BF16,
                        WeightFormat.TRELLIS,
                        WeightLayout.TRELLIS_TILE16_V1,
                        0L,
                        bytes,
                        0L,
                        false,
                        (ADDRESSES.getAndAdd(1 << 12) << 8) + 256);
            else if (host.test(name))
                payload = new TensorHandle(
                        name,
                        shape,
                        TensorDataType.BF16,
                        WeightFormat.TRELLIS,
                        WeightLayout.TRELLIS_TILE16_V1,
                        0L,
                        bytes,
                        ADDRESSES.getAndAdd(1 << 12));
            else
                payload = new TensorHandle(
                        name,
                        shape,
                        TensorDataType.BF16,
                        WeightFormat.TRELLIS,
                        WeightLayout.TRELLIS_TILE16_V1,
                        ADDRESSES.getAndAdd(1 << 12),
                        bytes);
            put(payload);
            this.ends.put(name, ends);
            int s = ends.length;
            put(new TensorHandle(
                    name + ".suh",
                    new long[] {s, k},
                    TensorDataType.FP16,
                    WeightFormat.FP16,
                    WeightLayout.CONTIGUOUS_LE_V1,
                    ADDRESSES.getAndAdd(1 << 12),
                    (long) s * k * 2));
            put(new TensorHandle(
                    name + ".svh",
                    new long[] {n},
                    TensorDataType.FP16,
                    WeightFormat.FP16,
                    WeightLayout.CONTIGUOUS_LE_V1,
                    ADDRESSES.getAndAdd(1 << 12),
                    (long) n * 2));
            put(new TensorHandle(
                    name + ".segments",
                    new long[] {s},
                    TensorDataType.INT32,
                    WeightFormat.I32,
                    WeightLayout.CONTIGUOUS_LE_V1,
                    ADDRESSES.getAndAdd(1 << 12),
                    (long) s * 4));
            return payload;
        }
    }

    private static List<ExecutionPlan.Instruction> trellisLinears(Shape shape) {
        return shape.instructions().stream()
                .filter(i -> i.kind() == Kind.TRELLIS_LINEAR)
                .toList();
    }

    @Test
    void aFullTrellisPlanBuildsWithCompanionsOnEveryLinear() {
        Model model = fullModel(name -> false, name -> false);
        ExecutionPlan plan = new ExecutionPlan(model.weights());
        for (Shape shape : plan.shapes()) {
            for (ExecutionPlan.Instruction instruction : shape.instructions()) {
                assertFalse(
                        instruction.kind() == Kind.Q3_GATE_UP_SWIGLU, "a trellis gate_up is never fused with SwiGLU");
                if (instruction.kind() != Kind.TRELLIS_LINEAR) continue;
                List<TensorHandle> weights = instruction.weights();
                String name = weights.getFirst().name();
                assertEquals(
                        List.of(name, name + ".suh", name + ".svh", name + ".segments"),
                        weights.stream().map(TensorHandle::name).toList());
                for (TensorHandle weight : weights)
                    assertEquals(model.objects().get(weight.name()).deviceAddress(), weight.deviceAddress());
                assertEquals(WeightFormat.TRELLIS, weights.getFirst().format());
            }
        }
        // Five trellis linears per layer; the output head is Q3.
        assertEquals(
                64 * 5,
                trellisLinears(plan.forExecution(Quantum.ExecutionKind.DECODE, 1))
                        .size());
        assertEquals(
                64 * 5,
                trellisLinears(plan.forExecution(Quantum.ExecutionKind.PREFILL, 512))
                        .size());
        assertTrue(plan.forExecution(Quantum.ExecutionKind.DECODE, 1).instructions().stream()
                .anyMatch(i -> i.kind() == Kind.Q3_LINEAR && i.outputBuffers().contains(Buffer.LOGITS)));
    }

    /// The staging ring of `gate_up` and `down` payloads kept on the host and the GDN outputs on disk.
    private static ExecutionPlan stagedPlan(Model[] holder) {
        Model model = fullModel(
                name -> name.endsWith("/mlp/gate_up") || name.endsWith("/mlp/down"),
                name -> name.endsWith("/gdn/output"));
        holder[0] = model;
        long largest = model.objects().values().stream()
                .filter(h -> h.hostBacked() || h.diskBacked())
                .mapToLong(TensorHandle::byteSize)
                .max()
                .orElseThrow();
        var staging = new WeightStaging(1L << 40, WeightStaging.slotBytesFor(largest), WeightStaging.SLOTS);
        long slot = DiskWindow.slotBytesFor(largest);
        var window = new DiskWindow(
                MemorySegment.ofAddress(1L << 42).reinterpret(slot * DiskWindow.SLOTS),
                MemorySegment.ofAddress(1L << 43).reinterpret(4096),
                slot);
        return new ExecutionPlan(model.weights(), staging, window);
    }

    @Test
    void stagedPlansGiveOnlyPayloadsASlot() {
        Model[] holder = new Model[1];
        ExecutionPlan plan = stagedPlan(holder);
        WeightStaging staging = plan.staging();
        long end = staging.slotAddress(0) + (long) staging.slots() * staging.slotBytes();
        int staged = 0;
        for (Shape shape : List.of(
                plan.forExecution(Quantum.ExecutionKind.DECODE, 1),
                plan.forExecution(Quantum.ExecutionKind.VERIFY, 16),
                plan.forExecution(Quantum.ExecutionKind.PREFILL, 512))) {
            for (ExecutionPlan.Instruction instruction : trellisLinears(shape)) {
                List<TensorHandle> weights = instruction.weights();
                TensorHandle payload =
                        holder[0].objects().get(weights.getFirst().name());
                boolean off = payload.hostBacked() || payload.diskBacked();
                long address = weights.getFirst().deviceAddress();
                assertEquals(
                        off,
                        address >= staging.slotAddress(0) && address < end,
                        weights.getFirst().name());
                if (off) staged++;
                for (int index = 1; index < 4; index++) {
                    TensorHandle companion = weights.get(index);
                    assertEquals(
                            holder[0].objects().get(companion.name()).deviceAddress(),
                            companion.deviceAddress(),
                            "a companion keeps its device address: " + companion.name());
                    assertFalse(companion.hostBacked() || companion.diskBacked());
                }
            }
            // Transfers carry payloads only.
            for (ExecutionPlan.Instruction instruction : shape.instructions())
                if (instruction.kind() == Kind.WEIGHT_TRANSFER || instruction.kind() == Kind.DISK_TRANSFER) {
                    assertEquals(1, instruction.weights().size());
                    assertEquals(
                            WeightFormat.TRELLIS,
                            instruction.weights().getFirst().format());
                }
        }
        assertEquals(3 * (64 * 2 + 48), staged, "gate_up and down of every layer, the output of the GDN layers");
    }

    private static List<Integer> declared(Shape shape) {
        List<Integer> stages = new ArrayList<>();
        for (int stage = 0; stage < shape.topology().size(); stage++)
            if (shape.scratchUse(stage) != null) stages.add(stage);
        return stages;
    }

    private static BitSet[] descendants(StageTopology topology) {
        BitSet[] below = new BitSet[topology.size()];
        for (int stage = topology.size() - 1; stage >= 0; stage--) {
            BitSet set = new BitSet();
            for (int next : topology.submittedSuccessors(stage)) {
                set.set(next);
                set.or(below[next]);
            }
            below[stage] = set;
        }
        return below;
    }

    @Test
    void everyViewDeclaresTheTrellisScratchAndOrdersItsUsers() {
        Model[] holder = new Model[1];
        for (ExecutionPlan plan :
                List.of(new ExecutionPlan(fullModel(n -> false, n -> false).weights()), stagedPlan(holder))) {
            for (Shape shape : plan.shapes()) {
                List<Integer> users = declared(shape);
                for (int stage = 0; stage < shape.topology().size(); stage++) {
                    if (shape.instructions().get(stage).kind() == Kind.TRELLIS_LINEAR)
                        assertEquals(ScratchUse.TRELLIS_LINEAR, shape.scratchUse(stage), shape.view() + " " + stage);
                }
                BitSet[] below = descendants(shape.topology());
                // The users of one scratch buffer are totally ordered.
                for (int i = 0; i < users.size(); i++)
                    for (int j = i + 1; j < users.size(); j++)
                        if (shape.scratchRegion(users.get(i)) == shape.scratchRegion(users.get(j)))
                            assertTrue(
                                    below[users.get(i)].get(users.get(j)),
                                    shape.view() + ": scratch users " + users.get(i) + " and " + users.get(j)
                                            + " may overlap");
                boolean decodeType = shape.view() == Shape.View.DECODE
                        || shape.view() == Shape.View.VERIFY
                        || shape.view() == Shape.View.DECODE_PRELOADED
                        || shape.view() == Shape.View.VERIFY_PRELOADED;
                int branches = 0;
                for (int stage : users) {
                    ExecutionPlan.Instruction instruction = shape.instructions().get(stage);
                    boolean second = instruction.kind() == Kind.TRELLIS_LINEAR
                            && instruction.outputBuffers().contains(Buffer.VALUE_Z_PROJECTED);
                    assertEquals(decodeType && second ? 1 : 0, shape.scratchRegion(stage), shape.view() + " " + stage);
                    // In a decode-type view the query/key and value/z projections of a layer may run together.
                    if (second
                            && decodeType
                            && instruction.weights().getFirst().name().contains("/gdn/")) {
                        int queryKey = stage - 1;
                        assertEquals(
                                List.of(Buffer.QK_PROJECTED),
                                shape.instructions().get(queryKey).outputBuffers());
                        assertFalse(below[queryKey].get(stage), shape.view() + " layer branches are serialized");
                        assertFalse(below[stage].get(queryKey));
                        branches++;
                    } else if (second && !decodeType) {
                        assertEquals(0, shape.scratchRegion(stage));
                    }
                }
                if (decodeType) assertEquals(48, branches, shape.view().toString());
            }
            for (Quantum.ExecutionKind kind : List.of(
                    Quantum.ExecutionKind.DECODE, Quantum.ExecutionKind.VERIFY, Quantum.ExecutionKind.PREFILL)) {
                Shape shape = plan.forExecution(kind, kind == Quantum.ExecutionKind.DECODE ? 1 : 512);
                assertEquals(
                        64 * 5,
                        declared(shape).stream()
                                .filter(stage -> shape.instructions().get(stage).kind() == Kind.TRELLIS_LINEAR)
                                .count(),
                        kind.toString());
            }
        }
    }

    private static long align(long value) {
        return (value + 255) / 256 * 256;
    }

    @Test
    void theSharedWorkspaceScratchIsTheLargestOfBothRoutesAtTheWorkspaceRows() {
        Model model = fullModel(n -> false, n -> false);
        ExecutionPlan plan = new ExecutionPlan(model.weights());
        TrellisSettings settings = new ExecutionFixtures.RecordingGpu().trellisSettings();
        int rows = 512;
        long expected = 0;
        for (Shape shape : plan.shapes()) {
            for (ExecutionPlan.Instruction instruction : trellisLinears(shape)) {
                long k = instruction.inputWidth(), n = instruction.outputWidth();
                long s = instruction.weights().get(1).shape()[0];
                long activations = align(s * rows * k * 2);
                int splits = TrellisFormat.splits((int) k, (int) n, settings.targetCtas());
                // Launches from the whole-K rows keep no split partials.
                long splitRows = Math.min(rows, settings.wholeKRows() - 1);
                long mma = activations + (splits > 1 ? splits * splitRows * n * 4 : 0);
                long prefill = activations + align(n * k * 2) + align(rows * n * 2);
                expected = Math.max(expected, rows > settings.mmaMaxRows() ? Math.max(mma, prefill) : mma);
            }
        }
        assertTrue(expected > 0);
        // The second buffer holds the tensor-core route of the value/z projections alone.
        long decodeExpected = 0;
        for (Shape shape : plan.shapes())
            for (ExecutionPlan.Instruction instruction : trellisLinears(shape)) {
                if (shape.scratchRegion(instruction.id()) != 1) continue;
                long k = instruction.inputWidth(), n = instruction.outputWidth();
                long s = instruction.weights().get(1).shape()[0];
                int splits = TrellisFormat.splits((int) k, (int) n, settings.targetCtas());
                long splitRows = Math.min(rows, settings.wholeKRows() - 1);
                decodeExpected = Math.max(
                        decodeExpected, align(s * rows * k * 2) + (splits > 1 ? splits * splitRows * n * 4 : 0));
            }
        assertTrue(decodeExpected > 0 && decodeExpected < expected);
        try (var workspace = new SharedWorkspace(new ExecutionFixtures.RecordingGpu(), plan, rows)) {
            assertEquals(expected, workspace.scratchBytes());
            assertEquals(expected, workspace.scratchBytes(0));
            assertEquals(decodeExpected, workspace.scratchBytes(1));
        }
    }

    @Test
    void theMtpLayerAndTheDraftRunTrellisOnTheTensorCoreRoute() {
        Model model = mtpModel();
        ExecutionPlan plan = new ExecutionPlan(model.weights());
        Shape draft = plan.forExecution(Quantum.ExecutionKind.DRAFT, 4);
        assertEquals(Shape.View.MTP_DRAFT, draft.view());
        List<ExecutionPlan.Instruction> linears = trellisLinears(draft);
        // Projection, then the layer's query/key, gate/value, output, gate/up and down.
        assertEquals(6, linears.size());
        for (ExecutionPlan.Instruction instruction : linears) {
            String name = instruction.weights().getFirst().name();
            assertTrue(name.startsWith("mtp/"), name);
            assertEquals(
                    List.of(name, name + ".suh", name + ".svh", name + ".segments"),
                    instruction.weights().stream().map(TensorHandle::name).toList());
            assertArrayEquals(model.weights().segmentEnds().get(name), instruction.segmentEnds(), name);
        }
        assertTrue(
                linears.stream().anyMatch(i -> i.weights().getFirst().name().equals("mtp/layer/attention/query_key")));
        assertTrue(
                linears.stream().anyMatch(i -> i.weights().getFirst().name().equals("mtp/layer/attention/gate_value")));
        // A draft of four rows is below the limit; a catch-up above it would not be, but is not a decode-type view.
        LaunchLog gpu = new LaunchLog();
        ExecutionPlan.Instruction projection = linears.getFirst();
        FORMAT.linear(
                gpu,
                new LinearLaunch(
                        projection.kind(),
                        projection.weights(),
                        0x1000,
                        0x2000,
                        4,
                        projection.inputWidth(),
                        projection.outputWidth(),
                        1L << 36,
                        1L << 30,
                        false,
                        false,
                        projection.segmentEnds()));
        assertEquals(
                List.of(LinearKernel.TRELLIS_PREPASS.symbol(), LinearKernel.TRELLIS_MMA.symbol()),
                gpu.names().subList(0, 2));
        assertFalse(gpu.names().contains(LinearKernel.TRELLIS_DECODE_BF16.symbol()));
    }

    @Test
    void anObjectWithoutHostSegmentEndsIsRefusedAtPlanTime() {
        Model model = fullModel(n -> false, n -> false);
        Weights w = model.weights();
        Weights bare = w.withSegmentEnds(Map.of());
        assertThrows(IllegalArgumentException.class, () -> new ExecutionPlan(bare));
    }

    @Test
    void prefillOrdersTheGdnProjectionsBehindTheControlRegion() {
        ExecutionPlan plan = new ExecutionPlan(fullModel(n -> false, n -> false).weights());
        for (Shape shape : List.of(
                plan.forExecution(Quantum.ExecutionKind.PREFILL, 16),
                plan.forExecution(Quantum.ExecutionKind.PREFILL, 512))) {
            int checked = 0;
            for (ExecutionPlan.Instruction instruction : shape.instructions()) {
                if (instruction.kind() != Kind.TRELLIS_LINEAR
                        || !instruction.outputBuffers().equals(List.of(Buffer.QK_PROJECTED))
                        || !instruction.weights().getFirst().name().contains("/gdn/")) continue;
                int layer = instruction.layerIndex();
                int control = shape.instructions().stream()
                        .filter(i -> i.layerIndex() == layer
                                && (i.kind() == Kind.GDN_CONTROL || i.kind() == Kind.GDN_PROJECT_CONTROL))
                        .mapToInt(ExecutionPlan.Instruction::id)
                        .min()
                        .orElseThrow();
                assertTrue(control < instruction.id(), shape.view() + " layer " + layer);
                checked++;
            }
            assertEquals(48, checked);
        }
    }

    // routes

    private record Launch(TableKernel kernel, int[] grid, int[] block, int shared, long[] words, byte[] sizes) {
        String name() {
            return kernel.symbol();
        }
    }

    /// Logs the table launches and answers a host read of a segments companion with its ends.
    private static final class LaunchLog extends ExecutionFixtures.RecordingGpu {
        final List<Launch> launches = new ArrayList<>();
        boolean exact;

        @Override
        public boolean exactNumerics() {
            return this.exact;
        }

        @Override
        public void launchTableKernel(
                TableKernel kernel,
                int gridX,
                int gridY,
                int gridZ,
                int blockX,
                int blockY,
                int blockZ,
                int sharedBytes,
                KernelArguments arguments) {
            this.launches.add(new Launch(
                    kernel,
                    new int[] {gridX, gridY, gridZ},
                    new int[] {blockX, blockY, blockZ},
                    sharedBytes,
                    Arrays.copyOf(arguments.words(), arguments.count()),
                    Arrays.copyOf(arguments.sizes(), arguments.count())));
        }

        /// A launch never reads the device on the host.
        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            throw new IllegalStateException("a launch read the device on the host");
        }

        List<String> names() {
            return this.launches.stream().map(Launch::name).toList();
        }
    }

    private static final Map<Long, int[]> ENDS = new java.util.concurrent.ConcurrentHashMap<>();

    /// An object of `n` by `k` with `ends`, its handles at addresses recorded in `gpu`.
    private static List<TensorHandle> object(LaunchLog gpu, int n, int k, int... ends) {
        Builder b = new Builder(new LinkedHashMap<>(), name -> false, name -> false);
        TensorHandle payload = b.trellis("object", n, k, ends);
        List<TensorHandle> weights = new ArrayList<>(List.of(payload));
        for (String companion : List.of(".suh", ".svh", ".segments")) weights.add(b.objects.get("object" + companion));
        ENDS.put(weights.get(3).deviceAddress(), ends);
        return weights;
    }

    private static LinearLaunch launch(
            List<TensorHandle> weights, int rows, int k, int n, long scratch, long scratchBytes, boolean verify) {
        return new LinearLaunch(
                Kind.TRELLIS_LINEAR,
                weights,
                0x1000,
                0x2000,
                rows,
                k,
                n,
                scratch,
                scratchBytes,
                verify,
                false,
                ENDS.get(weights.get(3).deviceAddress()));
    }

    private static final TrellisFormat FORMAT = new TrellisFormat(new TrellisSettings(32, 140, 128, 1024));

    private static void assertWords(Launch launch, long[] words, int... sizes) {
        assertArrayEquals(words, launch.words(), launch.name());
        assertEquals(sizes.length, launch.sizes().length, launch.name());
        for (int index = 0; index < sizes.length; index++)
            assertEquals(sizes[index], launch.sizes()[index], launch.name() + " argument " + index);
    }

    @Test
    void aDecodeRowRunsPrepassAndTheTensorCoreGemm() {
        LaunchLog gpu = new LaunchLog();
        // gate_up: k 5120, n 34816, two segments; one K split.
        var weights = object(gpu, 34816, 5120, 12288, 34816);
        long scratch = 1L << 36;
        FORMAT.linear(gpu, launch(weights, 1, 5120, 34816, scratch, 1L << 30, false));
        assertEquals(List.of(LinearKernel.TRELLIS_PREPASS.symbol(), LinearKernel.TRELLIS_MMA.symbol()), gpu.names());
        Launch prepass = gpu.launches.get(0), mma = gpu.launches.get(1);
        assertArrayEquals(new int[] {10, 1, 2}, prepass.grid());
        assertEquals(128, prepass.block()[0]);
        assertWords(
                prepass,
                new long[] {0x1000, weights.get(1).deviceAddress(), scratch, 1, 5120, 2, 0},
                8,
                8,
                8,
                4,
                4,
                4,
                4);
        assertArrayEquals(new int[] {272, 1, 1}, mma.grid());
        assertEquals(256, mma.block()[0]);
        long partials = scratch + align(2L * 1 * 5120 * 2);
        assertWords(
                mma,
                new long[] {
                    scratch,
                    weights.get(0).deviceAddress(),
                    weights.get(3).deviceAddress(),
                    weights.get(2).deviceAddress(),
                    partials,
                    0x2000,
                    1,
                    5120,
                    34816,
                    W,
                    1,
                    2
                },
                8,
                8,
                8,
                8,
                8,
                8,
                4,
                4,
                4,
                4,
                4,
                4);
    }

    @Test
    void aSplitGemmEndsWithTheEpilogue() {
        LaunchLog gpu = new LaunchLog();
        // down: k 17408, n 5120: ceil(140 / 40) = 4 splits.
        var weights = object(gpu, 5120, 17408, 5120);
        assertEquals(4, TrellisFormat.splits(17408, 5120, 140));
        long scratch = 1L << 36;
        FORMAT.linear(gpu, launch(weights, 1, 17408, 5120, scratch, 1L << 30, false));
        assertEquals(
                List.of(
                        LinearKernel.TRELLIS_PREPASS.symbol(),
                        LinearKernel.TRELLIS_MMA.symbol(),
                        LinearKernel.TRELLIS_EPILOGUE.symbol()),
                gpu.names());
        long partials = scratch + align(17408L * 2);
        Launch mma = gpu.launches.get(1), epilogue = gpu.launches.get(2);
        assertArrayEquals(new int[] {40, 4, 1}, mma.grid());
        assertEquals(4, mma.words()[10]);
        assertEquals(partials, mma.words()[4]);
        assertArrayEquals(new int[] {10, 1, 1}, epilogue.grid());
        assertEquals(128, epilogue.block()[0]);
        assertWords(
                epilogue, new long[] {partials, 4, weights.get(2).deviceAddress(), 0x2000, 1, 5120}, 8, 4, 8, 8, 4, 4);
    }

    @Test
    void aVerificationNeverLeavesTheTensorCoreRoute() {
        LaunchLog gpu = new LaunchLog();
        var weights = object(gpu, 5120, 17408, 5120);
        // 40 rows is above the limit of 32, but a verification stays row-exact.
        FORMAT.linear(gpu, launch(weights, 40, 17408, 5120, 1L << 36, 1L << 30, true));
        assertEquals(
                List.of(
                        LinearKernel.TRELLIS_PREPASS.symbol(),
                        LinearKernel.TRELLIS_MMA.symbol(),
                        LinearKernel.TRELLIS_EPILOGUE.symbol()),
                gpu.names());
        assertArrayEquals(new int[] {34, 40, 1}, gpu.launches.get(0).grid());
        assertEquals(0, gpu.launches.get(0).words()[6], "FP16 activations");
        assertTrue(FORMAT.rowExact(gpu, Kind.TRELLIS_LINEAR));
    }

    @Test
    void aLaunchFromTheWideRowsTakesBlocksOfSixtyFourRows() {
        LaunchLog gpu = new LaunchLog();
        var weights = object(gpu, 5120, 17408, 5120);
        TrellisFormat wide = new TrellisFormat(new TrellisSettings(32, 140, 128, 128));
        wide.linear(gpu, launch(weights, 200, 17408, 5120, 1L << 36, 1L << 30, true));
        assertEquals(
                List.of(LinearKernel.TRELLIS_PREPASS.symbol(), LinearKernel.TRELLIS_MMA_ROWS64.symbol()), gpu.names());
        assertArrayEquals(new int[] {40, 4, 1}, gpu.launches.get(1).grid());
    }

    @Test
    void aLaunchFromTheWholeKRowsRunsEverySplitInOneBlockWithoutPartials() {
        LaunchLog gpu = new LaunchLog();
        // down: k 17408, n 5120, 4 splits; a verification of 128 rows, the whole-K rows.
        var weights = object(gpu, 5120, 17408, 5120);
        long scratch = 1L << 36;
        FORMAT.linear(gpu, launch(weights, 128, 17408, 5120, scratch, 1L << 30, true));
        assertEquals(
                List.of(LinearKernel.TRELLIS_PREPASS.symbol(), LinearKernel.TRELLIS_MMA_ROWS.symbol()), gpu.names());
        Launch mma = gpu.launches.get(1);
        assertArrayEquals(new int[] {40, 4, 1}, mma.grid());
        assertEquals(256, mma.block()[0]);
        assertWords(
                mma,
                new long[] {
                    scratch,
                    weights.get(0).deviceAddress(),
                    weights.get(3).deviceAddress(),
                    weights.get(2).deviceAddress(),
                    0x2000,
                    128,
                    17408,
                    5120,
                    W,
                    4,
                    1
                },
                8,
                8,
                8,
                8,
                8,
                4,
                4,
                4,
                4,
                4,
                4);
    }

    @Test
    void rowsAboveTheLimitDecodeToBf16AndMultiplyPerSegment() {
        LaunchLog gpu = new LaunchLog();
        var weights = object(gpu, 34816, 5120, 12288, 34816);
        int rows = 33;
        long scratch = 1L << 36;
        long activations = align(2L * rows * 5120 * 2), decodedBytes = align(34816L * 5120 * 2);
        FORMAT.linear(gpu, launch(weights, rows, 5120, 34816, scratch, 1L << 33, false));
        assertEquals(
                List.of(
                        LinearKernel.TRELLIS_PREPASS.symbol(),
                        LinearKernel.TRELLIS_DECODE_BF16.symbol(),
                        "euhedral_q4_linear_tc_rows_bf16",
                        "euhedral_q4_linear_tc_rows_bf16",
                        LinearKernel.TRELLIS_EPILOGUE_BF16.symbol()),
                gpu.names());
        Launch prepass = gpu.launches.get(0), decode = gpu.launches.get(1);
        assertEquals(1, prepass.words()[6], "BF16 activations");
        assertWords(
                decode,
                new long[] {weights.get(0).deviceAddress(), scratch + activations, 5120, 34816, W},
                8,
                8,
                4,
                4,
                4);
        assertArrayEquals(new int[] {320 * 2176, 1, 1}, decode.grid());
        assertEquals(256, decode.block()[0]);
        long decoded = scratch + activations, rotated = decoded + decodedBytes;
        Launch first = gpu.launches.get(2), second = gpu.launches.get(3);
        assertArrayEquals(new long[] {scratch, decoded, rotated, rows, 5120, 12288}, first.words());
        assertArrayEquals(
                new long[] {
                    scratch + (long) rows * 5120 * 2,
                    decoded + 12288L * 5120 * 2,
                    rotated + (long) rows * 12288 * 2,
                    rows,
                    5120,
                    22528
                },
                second.words());
        Launch epilogue = gpu.launches.get(4);
        assertArrayEquals(new int[] {272, rows, 1}, epilogue.grid());
        assertWords(
                epilogue,
                new long[] {
                    rotated, weights.get(3).deviceAddress(), 2, weights.get(2).deviceAddress(), 0x2000, rows, 34816
                },
                8,
                8,
                4,
                8,
                8,
                4,
                4);
    }

    @Test
    void aNonVerificationAtTheLimitStaysOnTheTensorCoreRoute() {
        LaunchLog gpu = new LaunchLog();
        var weights = object(gpu, 34816, 5120, 34816);
        FORMAT.linear(gpu, launch(weights, 32, 5120, 34816, 1L << 36, 1L << 33, false));
        assertEquals(2, gpu.launches.size());
        assertEquals(LinearKernel.TRELLIS_MMA.symbol(), gpu.names().get(1));
    }

    @Test
    void exactNumericsRunOnlyTheReference() {
        LaunchLog gpu = new LaunchLog();
        gpu.exact = true;
        var weights = object(gpu, 34816, 5120, 12288, 34816);
        // The reference needs no scratch, whatever the rows.
        FORMAT.linear(gpu, launch(weights, 100, 5120, 34816, 0, 0, false));
        assertEquals(List.of(LinearKernel.TRELLIS_REFERENCE.symbol()), gpu.names());
        Launch reference = gpu.launches.getFirst();
        assertArrayEquals(new int[] {100 * 34816, 1, 1}, reference.grid());
        assertEquals(128, reference.block()[0]);
        assertEquals(4 * 5120, reference.shared());
        assertWords(
                reference,
                new long[] {
                    0x1000,
                    weights.get(0).deviceAddress(),
                    weights.get(1).deviceAddress(),
                    weights.get(2).deviceAddress(),
                    weights.get(3).deviceAddress(),
                    2,
                    0x2000,
                    100,
                    5120,
                    34816,
                    W
                },
                8,
                8,
                8,
                8,
                8,
                4,
                8,
                4,
                4,
                4,
                4);
    }

    @Test
    void aLaunchWithoutEnoughScratchIsRefused() {
        LaunchLog gpu = new LaunchLog();
        var weights = object(gpu, 5120, 17408, 5120);
        assertThrows(
                IllegalStateException.class, () -> FORMAT.linear(gpu, launch(weights, 1, 17408, 5120, 0, 0, false)));
        assertThrows(
                IllegalStateException.class,
                () -> FORMAT.linear(gpu, launch(weights, 1, 17408, 5120, 1L << 36, 1024, false)));
        assertTrue(gpu.launches.isEmpty());
    }

    @Test
    void scratchBytesFollowTheRoutes() {
        LaunchLog gpu = new LaunchLog();
        var weights = object(gpu, 34816, 5120, 12288, 34816);
        long activations = align(2L * 32 * 5120 * 2);
        // At most the limit: the tensor-core route alone (one split here, no partials).
        assertEquals(activations, FORMAT.scratchBytes(gpu, ScratchUse.TRELLIS_LINEAR, 32, 5120, 34816, weights));
        long prefill = align(2L * 64 * 5120 * 2) + align(34816L * 5120 * 2) + align(64L * 34816 * 2);
        assertEquals(prefill, FORMAT.scratchBytes(gpu, ScratchUse.TRELLIS_LINEAR, 64, 5120, 34816, weights));
    }

    @Test
    void theFormatAnswersTheExecutionRules() {
        TrellisFormat registered =
                (TrellisFormat) LinearFormats.find(WeightFormat.TRELLIS, WeightLayout.TRELLIS_TILE16_V1)
                        .orElseThrow();
        for (Kind role : List.of(Kind.Q3_LINEAR, Kind.Q4_LINEAR, Kind.Q5_LINEAR))
            assertEquals(Kind.TRELLIS_LINEAR, registered.kind(role));
        assertEquals(ScratchUse.TRELLIS_LINEAR, registered.scratchUse(Kind.TRELLIS_LINEAR, false));
        assertEquals(ScratchUse.TRELLIS_LINEAR, registered.scratchUse(Kind.TRELLIS_LINEAR, true));
        assertFalse(registered.gateUpRegion());
        assertNotEquals(0, registered.settings(new LaunchLog()).mmaMaxRows());
        var weights = object(new LaunchLog(), 5120, 17408, 5120);
        assertEquals(
                weights.getFirst(),
                registered.validateHandle(weights.getFirst(), 5120, 17408, WeightFormat.Q3_G64_FP16));
        assertThrows(
                IllegalArgumentException.class,
                () -> registered.validateHandle(weights.getFirst(), 5120, 5120, WeightFormat.Q3_G64_FP16));
    }
}
