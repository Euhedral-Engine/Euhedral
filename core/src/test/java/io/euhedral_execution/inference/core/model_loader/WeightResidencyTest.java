package io.euhedral_execution.inference.core.model_loader;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class WeightResidencyTest {
    @Test
    void executedResidencyLeavesUnusedRoutesInTheArtifact() {
        for (String name : new String[] {"vision/patch_embedding", "mtp/input_projection", "text/draft_head"}) {
            assertTrue(WeightResidency.ALL.uploads(name), name);
            assertFalse(WeightResidency.EXECUTED.uploads(name), name);
        }
        assertTrue(WeightResidency.EXECUTED.uploads("text/layers/0/mlp/gate_up"));
        assertTrue(WeightResidency.EXECUTED.uploads("text/output_head"));
    }
}
