package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan.Buffer;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan.Kind;
import io.euhedral_execution.inference.core.model.qwen38.Quantum.ExecutionKind;
import java.util.List;
import org.junit.jupiter.api.Test;

/// Production prefill routing: the fused gate/up region from 64 rows, residual-norm and early GDN control
/// regions below 64 rows, and the small topology in a separate view for decode.
class PrefillRouteTest {
    private static final int[] QUALIFIED_ROWS = {64, 256, 512, 1024};

    @Test
    void decodeRunsTheSmallTopologyInItsOwnView() {
        var weights = ExecutionFixtures.statefulCompactWeights();
        var plan = new ExecutionPlan(weights);
        var reference = ExecutionPlan.reference(weights);
        var decode = plan.forExecution(ExecutionKind.DECODE, 1);
        var small = plan.forExecution(ExecutionKind.PREFILL, 1);
        for (int rows : new int[] {1, 64, 256}) assertSame(decode, plan.forExecution(ExecutionKind.DECODE, rows));
        // Same fused topology as short prefill, but its own plan: graphs and logits stay per view.
        assertNotSame(small, decode);
        assertNotSame(plan, decode);
        assertEquals(small.bufferSpecs(), decode.bufferSpecs());
        assertTrue(has(decode, Kind.RESIDUAL_RMS_NORM));
        assertTrue(has(decode, Kind.GDN_PROJECT_CONTROL));
        assertFalse(decode.reusePrefillStorage());
        assertEquals(kinds(reference), kinds(plan));
        assertEquals(reference.bufferSpecs(), plan.bufferSpecs());
        assertFalse(plan.reusePrefillStorage());
    }

    @Test
    void topologyOrdersEveryPairOfStagesThatShareStorage() {
        var weights = ExecutionFixtures.statefulCompactWeights(8, 5120, 17408);
        var plan = new ExecutionPlan(weights);
        for (var view : List.of(
                plan,
                plan.forExecution(ExecutionKind.DECODE, 1),
                plan.forExecution(ExecutionKind.PREFILL, 1),
                plan.forExecution(ExecutionKind.PREFILL, 512),
                plan.forExecution(ExecutionKind.PREFILL, 1024))) {
            var topology = view.stageTopology();
            int count = topology.size();
            var reach = new java.util.BitSet[count];
            for (int stage = 0; stage < count; stage++) reach[stage] = new java.util.BitSet(count);
            for (int stage = 0; stage < count; stage++) {
                for (int successor : topology.submittedSuccessors(stage)) {
                    reach[successor].or(reach[stage]);
                    reach[successor].set(stage);
                }
            }
            var owners = new java.util.EnumMap<Buffer, Buffer>(Buffer.class);
            if (view.reusePrefillStorage())
                for (var pair : ExecutionPlan.REGION_STORAGE) owners.put(pair.getKey(), pair.getValue());
            var instructions = view.instructions();
            for (var later : instructions) {
                for (var earlier : instructions.subList(0, later.id())) {
                    boolean conflict = false;
                    for (Buffer written : later.outputBuffers())
                        for (Buffer touched : concat(earlier.inputBuffers(), earlier.outputBuffers()))
                            conflict |= owners.getOrDefault(written, written) == owners.getOrDefault(touched, touched);
                    for (Buffer read : later.inputBuffers())
                        for (Buffer written : earlier.outputBuffers())
                            conflict |= owners.getOrDefault(read, read) == owners.getOrDefault(written, written);
                    if (conflict) assertTrue(reach[later.id()].get(earlier.id()), earlier + " -> " + later);
                }
            }
        }
    }

    private static List<Buffer> concat(List<Buffer> first, List<Buffer> second) {
        var all = new java.util.ArrayList<>(first);
        all.addAll(second);
        return all;
    }

    @Test
    void unsupportedAttentionHeadDimensionIsRejectedWhenThePlanIsBuilt() {
        var weights = ExecutionFixtures.statefulCompactWeightsWithHeadDim(128);
        assertEquals(128, weights.config().attentionHeadDim());
        assertThrows(IllegalArgumentException.class, () -> new ExecutionPlan(weights));
    }

    @Test
    void rotaryWidthIsQualifiedUsingTheSameRoundingAsTheFrame() {
        var weights = ExecutionFixtures.statefulCompactWeightsWithRotaryFactor(0.50234375);
        assertThrows(IllegalArgumentException.class, () -> new ExecutionPlan(weights));
    }

    @Test
    void referencePlanIsNeverSpecialized() {
        var reference = ExecutionPlan.reference(ExecutionFixtures.statefulCompactWeights(8, 5120, 17408));
        for (int rows : new int[] {1, 63, 64, 256, 512, 1024})
            assertSame(reference, reference.forExecution(ExecutionKind.PREFILL, rows));
        assertTrue(reference.executionVariants().isEmpty());
    }

    @Test
    void smallPrefillUsesOnlyResidualNormAndEarlyControlRegions() {
        var weights = ExecutionFixtures.statefulCompactWeights();
        var plan = new ExecutionPlan(weights);
        var small = plan.forExecution(ExecutionKind.PREFILL, 1);
        assertSame(small, plan.forExecution(ExecutionKind.PREFILL, 63));
        assertTrue(has(small, Kind.RESIDUAL_RMS_NORM));
        assertTrue(has(small, Kind.GDN_PROJECT_CONTROL));
        assertTrue(has(small, Kind.ATTENTION_KV_APPEND));
        assertFalse(has(small, Kind.Q3_GATE_UP_SWIGLU));
        assertFalse(small.reusePrefillStorage());
        assertTopology(small);
    }

    @Test
    void combinedArchitectureIsSelectedAutomaticallyFromSixtyFourRows() {
        var weights = ExecutionFixtures.statefulCompactWeights();
        var plan = new ExecutionPlan(weights);
        var combined = plan.forExecution(ExecutionKind.PREFILL, 64);
        for (int rows : QUALIFIED_ROWS) assertSame(combined, plan.forExecution(ExecutionKind.PREFILL, rows));
        assertCombined(weights.config().numHiddenLayers(), combined);
    }

    @Test
    void residualNormRegionsKeepTheFinalModelBoundary() {
        var weights = ExecutionFixtures.statefulCompactWeights();
        var combined = new ExecutionPlan(weights).forExecution(ExecutionKind.PREFILL, 256);
        int layers = weights.config().numHiddenLayers();
        assertEquals(layers * 2L - 1, count(combined, Kind.RESIDUAL_RMS_NORM));
        for (var region : combined.instructions().stream()
                .filter(i -> i.kind() == Kind.RESIDUAL_RMS_NORM)
                .toList()) {
            assertEquals(2, region.inputBuffers().size());
            assertEquals(2, region.outputBuffers().size());
            assertEquals(1, region.weights().size());
        }
        assertTrue(combined.instructions().stream()
                .anyMatch(i -> i.kind() == Kind.RESIDUAL_ADD && i.layerIndex() == layers - 1));
    }

    @Test
    void controlRegionIsSubmittedBeforeHeavyMixerProjections() {
        var combined =
                new ExecutionPlan(ExecutionFixtures.statefulCompactWeights()).forExecution(ExecutionKind.PREFILL, 256);
        int control = firstId(combined, Kind.GDN_PROJECT_CONTROL);
        int heavy = firstId(combined, Kind.Q4_LINEAR);
        assertTrue(control < heavy);
    }

    @Test
    void contextRequalifiesAnyViewThroughItsOwner() {
        var owner = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(8, 5120, 17408));
        var view = owner.forExecution(ExecutionKind.PREFILL, 256);
        for (int rows : new int[] {1, 63, 64, 256, 512}) {
            var sequence = new Sequence(rows);
            try {
                var context = new Quantum(view, sequence, ExecutionKind.PREFILL, 0, new int[rows]);
                assertSame(owner.forExecution(ExecutionKind.PREFILL, rows), context.plan());
                assertSame(
                        owner.forExecution(ExecutionKind.DECODE, 1),
                        context.plan().forExecution(ExecutionKind.DECODE, 1));
                assertSame(owner, context.plan().executionOwner());
            } finally {
                sequence.complete();
            }
        }
    }

    @Test
    void stagedPlansRemainReferenceOnly() {
        var weights = ExecutionFixtures.statefulCompactWeights();
        var prefix = ExecutionPlan.prefix(weights, 1);
        var embedding = ExecutionPlan.embeddingOnly(weights);
        for (var plan : List.of(prefix, embedding)) {
            assertSame(plan, plan.forExecution(ExecutionKind.PREFILL, 256));
            assertTrue(plan.executionVariants().isEmpty());
        }
    }

    @Test
    void combinedWorkspaceReusesDisjointLifetimesAndFreesEachOwnerOnce() {
        var plan =
                new ExecutionPlan(ExecutionFixtures.statefulCompactWeights()).forExecution(ExecutionKind.PREFILL, 256);
        var gpu = new ExecutionFixtures.RecordingGpu();
        try (var workspace = new Workspace(gpu, 256, plan, LogitsRequirement.NONE)) {
            workspace.allocateBuffers();
            assertShared(workspace, Buffer.HIDDEN_STATE, Buffer.FINAL_HIDDEN_STATE);
            assertShared(workspace, Buffer.INPUT_NORMALIZED, Buffer.POST_MIXER_NORMALIZED);
            assertShared(workspace, Buffer.MIXER_DELTA, Buffer.FFN_DELTA);
            assertShared(workspace, Buffer.VALUE_Z_PROJECTED, Buffer.SWIGLU);
            assertThrows(IllegalStateException.class, () -> workspace.detachAddress(Buffer.SWIGLU));
            assertThrows(IllegalStateException.class, () -> workspace.address(Buffer.GATE_UP));
            assertThrows(IllegalStateException.class, () -> workspace.address(Buffer.A_PROJECTED));
        }
        assertEquals(gpu.allocations.size(), gpu.frees.size());
        assertEquals(gpu.frees.size(), new java.util.HashSet<>(gpu.frees).size());
    }

    @Test
    void unsupportedQ45InputWidthIsRejectedBeforeRuntimeSubmission() {
        var weights = ExecutionFixtures.statefulCompactWeights(8, 192, 128);
        assertThrows(IllegalArgumentException.class, () -> new ExecutionPlan(weights));
    }

    @Test
    void oversizedGdnConvolutionKernelIsRejectedBeforeRuntimeSubmission() {
        var weights = ExecutionFixtures.statefulCompactWeightsWithKernelWidth(33);
        assertThrows(IllegalArgumentException.class, () -> new ExecutionPlan(weights));
    }

    @Test
    void epsilonMustRemainPositiveAndFiniteAfterConversionToNativeFloat() {
        for (double epsilon : new double[] {Double.MIN_NORMAL, Double.MAX_VALUE}) {
            var weights = ExecutionFixtures.statefulCompactWeightsWithEpsilon(epsilon);
            assertThrows(IllegalArgumentException.class, () -> new ExecutionPlan(weights));
        }
    }

    private static void assertCombined(int layers, ExecutionPlan plan) {
        assertEquals(layers, count(plan, Kind.Q3_GATE_UP_SWIGLU));
        assertFalse(has(plan, Kind.SWIGLU));
        // The attention producers stay leaf frames: Q4 -> QK norm/RoPE and Q5 -> cache append.
        assertTrue(has(plan, Kind.ATTENTION_KV_APPEND));
        assertTrue(has(plan, Kind.ATTENTION_QK_NORM_ROPE));
        assertTrue(hasBuffer(plan, Buffer.ATTENTION_QK_NORMALIZED));
        assertTrue(has(plan, Kind.GDN_PROJECT_CONTROL));
        assertFalse(has(plan, Kind.BF16_LINEAR));
        assertFalse(has(plan, Kind.GDN_CONTROL));
        assertTrue(has(plan, Kind.RESIDUAL_RMS_NORM));
        for (var buffer : List.of(Buffer.GATE_UP, Buffer.A_PROJECTED, Buffer.B_PROJECTED)) {
            assertFalse(hasBuffer(plan, buffer), buffer.name());
            assertFalse(plan.instructions().stream()
                    .anyMatch(i -> i.inputBuffers().contains(buffer)
                            || i.outputBuffers().contains(buffer)));
        }
        assertTrue(plan.reusePrefillStorage());
        assertTopology(plan);
    }

    private static void assertShared(Workspace workspace, Buffer owner, Buffer view) {
        assertEquals(workspace.address(owner), workspace.address(view), owner + "/" + view);
    }

    private static List<Kind> kinds(ExecutionPlan plan) {
        return plan.instructions().stream().map(ExecutionPlan.Instruction::kind).toList();
    }

    private static boolean has(ExecutionPlan plan, Kind kind) {
        return count(plan, kind) > 0;
    }

    private static long count(ExecutionPlan plan, Kind kind) {
        return plan.instructions().stream().filter(i -> i.kind() == kind).count();
    }

    private static boolean hasBuffer(ExecutionPlan plan, Buffer buffer) {
        return plan.bufferSpecs().stream().anyMatch(s -> s.buffer() == buffer);
    }

    private static int firstId(ExecutionPlan plan, Kind kind) {
        return plan.instructions().stream()
                .filter(i -> i.kind() == kind)
                .findFirst()
                .orElseThrow()
                .id();
    }

    private static void assertTopology(ExecutionPlan plan) {
        for (int index = 0; index < plan.instructions().size(); index++) {
            var instruction = plan.instructions().get(index);
            assertEquals(index, instruction.id());
            for (int dependency : instruction.dependencies()) {
                assertTrue(dependency < index);
                assertTrue(plan.successors(dependency).contains(index));
            }
        }
    }
}
