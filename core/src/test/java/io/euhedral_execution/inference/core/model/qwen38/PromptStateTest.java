package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.generation.LogitsRequirement;
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
        assertEquals(2 * (layers + 1), view.carriedStateCount());
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
