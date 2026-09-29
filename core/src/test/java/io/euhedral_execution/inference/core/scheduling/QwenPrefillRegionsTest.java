package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.junit.jupiter.api.Test;

class QwenPrefillRegionsTest {
    @Test
    void prefillCollapsesResidualNormBoundariesWithoutChangingDecode() {
        var weights = QwenExecutionFixtures.statefulCompactWeights();
        var reference = new QwenExecutionPlan(weights);
        var owner = new QwenExecutionPlan(weights, QwenExecutionPlan.PrefillRegions.RESIDUAL_NORM);
        var prefill = owner.forExecution(QwenExecutionContext.ExecutionKind.PREFILL);
        assertSame(owner, owner.forExecution(QwenExecutionContext.ExecutionKind.DECODE));
        assertEquals(
                reference.instructions().stream()
                        .map(QwenExecutionPlan.Instruction::kind)
                        .toList(),
                owner.instructions().stream()
                        .map(QwenExecutionPlan.Instruction::kind)
                        .toList());
        long expected = weights.config().numHiddenLayers() * 2L - 1;
        var regions = prefill.instructions().stream()
                .filter(i -> i.kind() == QwenExecutionPlan.Kind.RESIDUAL_RMS_NORM)
                .toList();
        assertEquals(expected, regions.size());
        assertEquals(
                reference.instructions().size() - expected,
                prefill.instructions().size());
        for (var region : regions) {
            assertEquals(2, region.inputBuffers().size());
            assertEquals(2, region.outputBuffers().size());
            assertEquals(1, region.weights().size());
        }
        assertEquals(reference.bufferSpecs(), prefill.bufferSpecs());
        assertTrue(prefill.instructions().stream()
                .anyMatch(i -> i.kind() == QwenExecutionPlan.Kind.RESIDUAL_ADD
                        && i.layerIndex() == weights.config().numHiddenLayers() - 1));
        assertTopology(prefill);
    }

    @Test
    void controlRegionRemovesProjectedGraphBuffers() {
        var weights = QwenExecutionFixtures.statefulCompactWeights();
        var plan = new QwenExecutionPlan(weights, QwenExecutionPlan.PrefillRegions.GDN_CONTROL)
                .forExecution(QwenExecutionContext.ExecutionKind.PREFILL);
        assertTrue(plan.instructions().stream().anyMatch(i -> i.kind() == QwenExecutionPlan.Kind.GDN_PROJECT_CONTROL));
        assertFalse(plan.instructions().stream()
                .anyMatch(i -> i.kind() == QwenExecutionPlan.Kind.BF16_LINEAR
                        || i.kind() == QwenExecutionPlan.Kind.GDN_CONTROL));
        for (var buffer : List.of(QwenExecutionPlan.Buffer.A_PROJECTED, QwenExecutionPlan.Buffer.B_PROJECTED)) {
            assertFalse(plan.bufferSpecs().stream().anyMatch(s -> s.buffer() == buffer));
            assertFalse(plan.instructions().stream()
                    .anyMatch(i -> i.inputBuffers().contains(buffer)
                            || i.outputBuffers().contains(buffer)));
        }
        assertTopology(plan);
    }

    @Test
    void quantumSelectsPrefillOnlyAndKeepsDecodeTopology() {
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.statefulCompactWeights(), QwenExecutionPlan.PrefillRegions.RESIDUAL_NORM_CONTROL);
        var sequence = new QwenSequenceState(123);
        var prefill = new QwenExecutionContext(
                plan, sequence, QwenExecutionContext.ExecutionKind.PREFILL, 0, new int[] {1, 2});
        var decode =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        assertSame(plan.forExecution(QwenExecutionContext.ExecutionKind.PREFILL), prefill.plan());
        assertSame(plan, decode.plan());
    }

    @Test
    void contextRequalifiesDerivedViewsUsingTheirOriginalPlanOwner() {
        var owner = new QwenExecutionPlan(
                QwenExecutionFixtures.statefulCompactWeights(), QwenExecutionPlan.PrefillRegions.STREAMED_KV);
        var view = owner.forExecution(QwenExecutionContext.ExecutionKind.PREFILL);
        for (int rows : new int[] {1, 63, 64, 256, 512}) {
            var context = new QwenExecutionContext(
                    view, new QwenSequenceState(rows), QwenExecutionContext.ExecutionKind.PREFILL, 0, new int[rows]);
            assertSame(owner.forExecution(QwenExecutionContext.ExecutionKind.PREFILL, rows), context.plan());
            assertSame(owner, context.plan().forExecution(QwenExecutionContext.ExecutionKind.DECODE, 1));
        }
    }

    @Test
    void gateUpRegionEliminatesWideBoundaryOnlyForQualifiedPrefillRows() {
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.statefulCompactWeights(), QwenExecutionPlan.PrefillRegions.GATE_UP_SWIGLU);
        var prefill = plan.forExecution(QwenExecutionContext.ExecutionKind.PREFILL, 256);
        assertTrue(prefill.instructions().stream().anyMatch(i -> i.kind() == QwenExecutionPlan.Kind.Q3_GATE_UP_SWIGLU));
        assertFalse(prefill.bufferSpecs().stream().anyMatch(s -> s.buffer() == QwenExecutionPlan.Buffer.GATE_UP));
        assertFalse(prefill.instructions().stream().anyMatch(i -> i.kind() == QwenExecutionPlan.Kind.SWIGLU));
        assertSame(plan, plan.forExecution(QwenExecutionContext.ExecutionKind.PREFILL, 1));
        assertSame(plan, plan.forExecution(QwenExecutionContext.ExecutionKind.PREFILL, 32));
        assertSame(plan, plan.forExecution(QwenExecutionContext.ExecutionKind.DECODE, 256));
        assertTopology(prefill);
    }

    @Test
    void streamedFfnFallsBackOutsideItsExactQualifiedGeometry() {
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.statefulCompactWeights(), QwenExecutionPlan.PrefillRegions.STREAMED_FFN);
        var selected = plan.forExecution(QwenExecutionContext.ExecutionKind.PREFILL, 256);
        assertFalse(selected.instructions().stream().anyMatch(i -> i.kind() == QwenExecutionPlan.Kind.FFN_STREAMED));
        assertTrue(
                selected.instructions().stream().anyMatch(i -> i.kind() == QwenExecutionPlan.Kind.Q3_GATE_UP_SWIGLU));
    }

    @Test
    void attentionProducerRegionRemovesAppendAndNormalizedBufferButKeepsDecode() {
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.statefulCompactWeights(), QwenExecutionPlan.PrefillRegions.ATTENTION_KV);
        var prefill = plan.forExecution(QwenExecutionContext.ExecutionKind.PREFILL, 256);
        assertTrue(
                prefill.instructions().stream().anyMatch(i -> i.kind() == QwenExecutionPlan.Kind.ATTENTION_PRODUCERS));
        assertFalse(
                prefill.instructions().stream().anyMatch(i -> i.kind() == QwenExecutionPlan.Kind.ATTENTION_KV_APPEND));
        assertFalse(prefill.bufferSpecs().stream()
                .anyMatch(s -> s.buffer() == QwenExecutionPlan.Buffer.ATTENTION_QK_NORMALIZED));
        assertSame(plan, plan.forExecution(QwenExecutionContext.ExecutionKind.PREFILL, 63));
        assertSame(plan, plan.forExecution(QwenExecutionContext.ExecutionKind.DECODE, 256));
        assertTopology(prefill);
    }

    @Test
    void combinedRegionWorkspaceReusesDisjointLifetimesAndFreesEachOwnerOnce() {
        var plan = new QwenExecutionPlan(
                        QwenExecutionFixtures.statefulCompactWeights(), QwenExecutionPlan.PrefillRegions.COMBINED_KV)
                .forExecution(QwenExecutionContext.ExecutionKind.PREFILL, 256);
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        try (var workspace = new QwenExecutionWorkspace(gpu, 256, plan, QwenLogitsRequirement.NONE)) {
            workspace.allocateBuffers();
            assertEquals(
                    workspace.address(QwenExecutionPlan.Buffer.HIDDEN_STATE),
                    workspace.address(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE));
            assertEquals(
                    workspace.address(QwenExecutionPlan.Buffer.INPUT_NORMALIZED),
                    workspace.address(QwenExecutionPlan.Buffer.POST_MIXER_NORMALIZED));
            assertEquals(
                    workspace.address(QwenExecutionPlan.Buffer.MIXER_DELTA),
                    workspace.address(QwenExecutionPlan.Buffer.FFN_DELTA));
            assertEquals(
                    workspace.address(QwenExecutionPlan.Buffer.VALUE_Z_PROJECTED),
                    workspace.address(QwenExecutionPlan.Buffer.SWIGLU));
            assertThrows(IllegalStateException.class, () -> workspace.detachAddress(QwenExecutionPlan.Buffer.SWIGLU));
        }
        assertEquals(gpu.allocations.size(), gpu.frees.size());
        assertEquals(gpu.frees.size(), new java.util.HashSet<>(gpu.frees).size());
    }

    @Test
    void controlBranchIsReadyBeforeHeavyMixerProjectionsAreSubmitted() {
        var plan = new QwenExecutionPlan(
                        QwenExecutionFixtures.statefulCompactWeights(), QwenExecutionPlan.PrefillRegions.GDN_CONTROL)
                .forExecution(QwenExecutionContext.ExecutionKind.PREFILL, 256);
        int control = plan.instructions().stream()
                .filter(i -> i.kind() == QwenExecutionPlan.Kind.GDN_PROJECT_CONTROL)
                .findFirst()
                .orElseThrow()
                .id();
        int heavy = plan.instructions().stream()
                .filter(i -> i.kind() == QwenExecutionPlan.Kind.Q4_LINEAR)
                .findFirst()
                .orElseThrow()
                .id();
        assertTrue(control < heavy);
    }

    private static void assertTopology(QwenExecutionPlan plan) {
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
