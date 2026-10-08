package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.model.qwen38.loader.LayerWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.Weights;
import io.euhedral_execution.inference.core.runtime.graph.GraphShape;
import org.junit.jupiter.api.Test;

/// A prompt's chunks hand sequence state to one another: each layer's GDN state and KV rows are carried state,
/// and the prompt reserves its KV rows once, at admission.
class PromptStateTest {

    private static final ExecutionPlan PLAN = new ExecutionPlan(EngineExecutionFixture.weights());

    @Test
    void gdnAndKvStagesCarryTheirLayersState() {
        Shape view = PLAN.forExecution(Quantum.ExecutionKind.PREFILL, 64);
        int layers = PLAN.weights().config().numHiddenLayers();
        assertEquals(Shape.seedRowsKey(layers) + 3, view.carriedStateCount());
        boolean gdn = false, kv = false;
        for (int stage = 0; stage < view.topology().size(); stage++) {
            ExecutionPlan.Instruction instruction = view.instructions().get(stage);
            int layer = instruction.layerIndex();
            switch (instruction.kind()) {
                case GDN_CONVOLUTION, GDN_RECURRENCE -> {
                    assertArrayEquals(
                            new int[] {2 * layer},
                            view.carriedState(stage),
                            instruction.kind().name());
                    gdn = true;
                }
                case ATTENTION_KV_APPEND, ATTENTION_CAUSAL -> {
                    assertArrayEquals(
                            new int[] {2 * layer + 1},
                            view.carriedState(stage),
                            instruction.kind().name());
                    kv = true;
                }
                default ->
                    assertArrayEquals(
                            GraphShape.NO_BUFFERS,
                            view.carriedState(stage),
                            instruction.kind().name());
            }
        }
        assertTrue(gdn && kv, "the fixture has GDN and attention layers");
    }

    @Test
    void aDraftingPlansSeedRowsAndMtpCacheAreCarriedState() {
        var plan = new ExecutionPlan(ExecutionFixtures.mtpCompactWeights(8));
        int layers = plan.weights().config().numHiddenLayers();
        int seed = Shape.seedRowsKey(layers);
        Shape prefill = plan.forExecution(Quantum.ExecutionKind.PREFILL, 64);
        assertTrue(carries(prefill, ExecutionPlan.Kind.RMS_NORM, seed), "the base final norm writes the seed rows");
        Shape draft = plan.forExecution(Quantum.ExecutionKind.DRAFT, 1);
        assertTrue(carries(draft, ExecutionPlan.Kind.MTP_STEM, seed), "the MTP stem reads them");
        assertTrue(carries(draft, ExecutionPlan.Kind.RMS_NORM, seed), "the MTP final norm writes the recursion row");
        assertTrue(
                carries(draft, ExecutionPlan.Kind.ATTENTION_KV_APPEND, 2 * layers + 1),
                "the MTP layer appends to its own cache");
    }

    private static boolean carries(Shape view, ExecutionPlan.Kind kind, int key) {
        for (int stage = 0; stage < view.topology().size(); stage++)
            if (view.instructions().get(stage).kind() == kind
                    || kind == ExecutionPlan.Kind.RMS_NORM
                            && view.instructions().get(stage).kind() == ExecutionPlan.Kind.RMS_NORM_UNIT_OFFSET)
                for (int carried : view.carriedState(stage)) if (carried == key) return true;
        return false;
    }

    @Test
    void aFirstLayerPlansGdnStagesCarryLayerZerosState() {
        var full = EngineExecutionFixture.weights();
        var firstLayer = new Weights(
                full.config(),
                full.tokenEmbedding(),
                new LayerWeights[] {full.layers()[0]},
                full.finalNorm(),
                full.lmHead(),
                null);
        Shape view = new ExecutionPlan(firstLayer).forExecution(Quantum.ExecutionKind.PREFILL, 64);
        boolean gdn = false;
        for (int stage = 0; stage < view.topology().size(); stage++) {
            ExecutionPlan.Instruction instruction = view.instructions().get(stage);
            switch (instruction.kind()) {
                case GDN_CONVOLUTION, GDN_RECURRENCE -> {
                    assertArrayEquals(
                            new int[] {0},
                            view.carriedState(stage),
                            instruction.kind().name());
                    gdn = true;
                }
                default -> {}
            }
        }
        assertTrue(gdn, "the first layer is a GDN layer");
    }

    @Test
    void aPromptQuantumReservesEveryRowAtAdmission() {
        var gpu = new MemoryGpu();
        var sequence = new Sequence(1);
        var quantum = Quantum.prompt(PLAN, sequence, 0, new int[600], 256, LogitsRequirement.NONE, null);
        quantum.begin(gpu, () -> {});
        var attention = (AttentionStates) sequence.kvCacheState();
        for (int layer = 0; layer < PLAN.weights().config().numHiddenLayers(); layer++) {
            if (PLAN.weights().config().layerTypes()[layer] != LayerType.FULL_ATTENTION) continue;
            assertTrue(attention.forLayer(layer).reserves(0, 600), "layer " + layer + " reserved the prompt");
            assertEquals(0, attention.forLayer(layer).submittedLength(), "nothing is submitted at admission");
        }
    }
}
