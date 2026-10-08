package io.euhedral_execution.inference.core.model.qwen38;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

/// Every view of the fixture plans keeps the topology and specs recorded before 3.8's views moved into shapes
/// (`-Peuhedral.shapes.record=true` records instead of comparing).
class ShapeIdentityTest {

    private static final Path GOLDENS = Path.of("src/test/resources/shapes");

    @Test
    void theFullFixtureModelsViewsKeepTheirShape() throws Exception {
        ShapeDescription.check(
                GOLDENS,
                "fixture-full",
                ShapeDescription.views(new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(64))));
    }

    @Test
    void theFullFixtureModelsPromptGraphKeepsItsShape() throws Exception {
        ShapeDescription.check(
                GOLDENS,
                "fixture-full-prompt",
                ShapeDescription.prompt(new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(64)), 64, 5, 4));
    }

    @Test
    void theReferencePlanKeepsItsShape() throws Exception {
        ShapeDescription.check(
                GOLDENS,
                "fixture-reference",
                ShapeDescription.views(ExecutionPlan.reference(ExecutionFixtures.statefulCompactWeights(64))));
    }

    @Test
    void thePrefixAndEmbeddingPlansKeepTheirShape() throws Exception {
        var weights = ExecutionFixtures.statefulCompactWeights(64);
        ShapeDescription.check(GOLDENS, "fixture-prefix-1", ShapeDescription.views(ExecutionPlan.prefix(weights, 1)));
        ShapeDescription.check(
                GOLDENS, "fixture-embedding", ShapeDescription.views(ExecutionPlan.embeddingOnly(weights)));
    }

    @Test
    void theOperatorSliceKeepsItsShape() throws Exception {
        var plan = new ExecutionPlan(
                ExecutionFixtures.weights(),
                ExecutionFixtures.norm(),
                List.of(ExecutionFixtures.q3("projection", 64, 201)));
        ShapeDescription.check(GOLDENS, "fixture-operator-slice", ShapeDescription.views(plan));
    }
}
