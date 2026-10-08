package io.euhedral_execution.inference.core.model.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// One workspace per plan: every row capacity leases the same one, sized for the plan's chunk, and the stages that
/// take the expansion scratch find it bound there instead of allocating their own. Opt-in with the artifact present.
@ModelGroup.FlashNext
class SingleWorkspaceCudaIntegrationTest {

    private static final int CONTEXT = 256;

    @Test
    @Timeout(value = 2, unit = TimeUnit.MINUTES)
    void everyRowCapacityLeasesTheOneWorkspaceAndTheScratchIsBound() throws Exception {
        var loaded = SharedFlashNext.model(SharedFlashNext.ROOMY);
        var gpu = loaded.gpu();
        ExecutionPlan plan = loaded.plan();
        assertEquals(1, plan.workspaceCount(), "the plan allocates one workspace");
        Workspace one = plan.leaseStorage(1).storage();
        assertSame(one, plan.leaseStorage(16).storage());
        assertSame(one, plan.leaseStorage(plan.maxRows()).storage());
        assertEquals(plan.maxRows(), one.rows(), "sized for the plan's chunk");
        assertTrue(one.scratchBytes() > 0, "the native NVFP4 route's scratch is part of the workspace");
        int[] prompt = PerformanceCudaIntegrationTest.corpus(96);
        try (Sequence sequence = plan.newSequence()) {
            long before = gpu.scratchFallbacks();
            Blocking.step(plan, sequence, prompt, 0, 64, null);
            Blocking.step(plan, sequence, prompt, 64, 12, null);
            Blocking.step(plan, sequence, prompt, 76, 1, null);
            assertEquals(before, gpu.scratchFallbacks(), "every scratch use found the workspace's region");
        }
    }
}
