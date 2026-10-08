package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.ScratchUse;
import io.euhedral_execution.inference.core.runtime.graph.StageTopology;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import org.junit.jupiter.api.Test;

/// The expansion scratch is one workspace buffer: the stages that will use it are declared, take it in an order the
/// shape's edges give, and nothing else changes.
class ScratchDeclarationTest {

    private static final ExecutionPlan PLAN = new ExecutionPlan(EngineExecutionFixture.weights());

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
    void theRegionViewDeclaresItsGateUpRegionsAndQuantizedLinears() {
        Shape region = PLAN.forExecution(Quantum.ExecutionKind.PREFILL, 64);
        boolean gateUp = false, linear = false;
        for (int stage : declared(region)) {
            ExecutionPlan.Kind kind = region.instructions().get(stage).kind();
            gateUp |= kind == ExecutionPlan.Kind.Q3_GATE_UP_SWIGLU;
            linear |= kind == ExecutionPlan.Kind.Q4_LINEAR || kind == ExecutionPlan.Kind.Q5_LINEAR;
            assertTrue(kind != ExecutionPlan.Kind.BF16_LINEAR, "a BF16 linear takes no scratch");
        }
        assertTrue(gateUp && linear, "region view declares " + declared(region));
        assertTrue(region.workspaceBuffers(declared(region).getFirst()).length > 0);
        int scratch = SharedWorkspace.scratchBuffer(PLAN);
        for (int stage : declared(region))
            assertTrue(java.util.Arrays.stream(region.workspaceBuffers(stage)).anyMatch(b -> b == scratch));
    }

    @Test
    void theDecodeViewDeclaresNoQuantizedLinear() {
        Shape decode = PLAN.forExecution(Quantum.ExecutionKind.DECODE, 1);
        for (int stage : declared(decode)) {
            ExecutionPlan.Kind kind = decode.instructions().get(stage).kind();
            assertFalse(
                    kind == ExecutionPlan.Kind.Q4_LINEAR
                            || kind == ExecutionPlan.Kind.Q5_LINEAR
                            || kind == ExecutionPlan.Kind.Q3_LINEAR,
                    "decode rows use no scratch: " + kind);
        }
    }

    @Test
    void aVerificationOfMoreThanEightRowsDeclaresItsQuantizedLinears() {
        Shape verify = PLAN.forExecution(Quantum.ExecutionKind.VERIFY, 16);
        boolean linear = false;
        for (int stage : declared(verify)) {
            ExecutionPlan.Kind kind = verify.instructions().get(stage).kind();
            linear |= kind == ExecutionPlan.Kind.Q4_LINEAR || kind == ExecutionPlan.Kind.Q5_LINEAR;
        }
        assertTrue(linear, "a 16-row verification expands its linears into the scratch: " + declared(verify));
        assertEquals(
                PLAN.forExecution(Quantum.ExecutionKind.DECODE, 1).instructions().stream()
                        .map(ExecutionPlan.Instruction::kind)
                        .toList(),
                verify.instructions().stream()
                        .map(ExecutionPlan.Instruction::kind)
                        .toList(),
                "the same work as a decode token");
        assertEquals(
                PLAN.forExecution(Quantum.ExecutionKind.DECODE, 1),
                PLAN.forExecution(Quantum.ExecutionKind.VERIFY, 4),
                "a verification of up to eight rows runs the decode view");
    }

    @Test
    void declaredStagesAreTotallyOrdered() {
        for (Shape shape : PLAN.shapes()) {
            List<Integer> users = declared(shape);
            BitSet[] below = descendants(shape.topology());
            for (int i = 0; i < users.size(); i++)
                for (int j = i + 1; j < users.size(); j++)
                    assertTrue(
                            below[users.get(i)].get(users.get(j)),
                            shape.view() + ": scratch users " + users.get(i) + " and " + users.get(j) + " may overlap");
        }
    }

    @Test
    void aGateUpRegionUsesTheGateUpRoute() {
        Shape region = PLAN.forExecution(Quantum.ExecutionKind.PREFILL, 64);
        for (int stage : declared(region))
            if (region.instructions().get(stage).kind() == ExecutionPlan.Kind.Q3_GATE_UP_SWIGLU)
                assertTrue(region.scratchUse(stage) == ScratchUse.Q3_GATE_UP
                        || region.scratchUse(stage) == ScratchUse.NVFP4_GATE_UP);
        assertEquals(PLAN.shapes().size(), PLAN.shapes().stream().distinct().count());
    }

    @Test
    @org.junit.jupiter.api.Timeout(20)
    void declaredStagesSubmitWithTheWorkspaceScratchBound() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var plan = new ExecutionPlan(weights);
        List<Long> bound = java.util.Collections.synchronizedList(new ArrayList<>());
        var gpu = new EngineExecutionFixture.SamplingGpu(weights.config().vocabSize()) {
            @Override
            public long scratchBytes(
                    ScratchUse use,
                    int rows,
                    int inFeatures,
                    int outFeatures,
                    io.euhedral_execution.inference.core.artifact.WeightLayout layout) {
                return 4096;
            }

            @Override
            public void withScratch(long address, long bytes, Runnable submit) {
                bound.add(address);
                assertEquals(4096, bytes);
                submit.run();
            }
        };
        var runtime = new Execution(ExecutionFixtures.inlineLattice(), plan, gpu);
        try {
            var quantum = new Quantum(
                    plan,
                    new Sequence(51),
                    Quantum.ExecutionKind.PREFILL,
                    0,
                    new int[64],
                    io.euhedral_execution.inference.core.generation.LogitsRequirement.NONE);
            assertEquals(
                    Quantum.Status.SUCCESS,
                    runtime.submit(quantum)
                            .get(10, java.util.concurrent.TimeUnit.SECONDS)
                            .status());
            assertEquals(declared(quantum.shape()).size(), bound.size(), "every declared stage, once");
            assertEquals(1, new java.util.HashSet<>(bound).size(), "one scratch buffer");
            assertTrue(bound.getFirst() != 0);
        } finally {
            runtime.close();
        }
    }
}
