package io.euhedral_execution.inference.core.runtime.graph;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class ShapeBuilderTest {
    @Test
    void stagesAndEdgesBecomeTheTopology() {
        var builder = new ShapeBuilder<String>();
        int a = builder.stage("a"), b = builder.stage("b"), c = builder.stage("c");
        builder.submitted(a, b);
        builder.retired(a, c);
        builder.submitted(b, c);
        StageTopology topology = builder.build();
        assertEquals(List.of("a", "b", "c"), builder.specs());
        assertArrayEquals(new int[] {a}, topology.roots());
        assertArrayEquals(new int[] {b}, topology.submittedSuccessors(a));
        assertArrayEquals(new int[] {c}, topology.retiredSuccessors(a));
        assertEquals(2, topology.inDegree(c));
    }

    @Test
    void edgesMustPointForwardAndOnce() {
        var builder = new ShapeBuilder<String>();
        int a = builder.stage("a"), b = builder.stage("b");
        assertThrows(IllegalArgumentException.class, () -> builder.submitted(b, a));
        builder.submitted(a, b);
        builder.retired(a, b);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void anAllSubmittedShapeEqualsTheDependencyForm() {
        var builder = new ShapeBuilder<Integer>();
        for (int i = 0; i < 4; i++) builder.stage(i);
        builder.submitted(0, 2);
        builder.submitted(1, 2);
        builder.submitted(2, 3);
        assertEquals(
                StageTopology.submitted(new int[][] {{}, {}, {0, 1}, {2}}).toString(),
                builder.build().toString());
    }

    @Test
    void aChainOrdersOnlyTheStagesNotAlreadyOrdered() {
        var builder = new ShapeBuilder<String>();
        int a = builder.stage("a"), b = builder.stage("b"), c = builder.stage("c"), d = builder.stage("d");
        builder.submitted(a, b);
        builder.submitted(b, c);
        builder.chain(new int[] {a, c, d});
        StageTopology topology = builder.build();
        assertArrayEquals(new int[] {b}, topology.submittedSuccessors(a), "a already reaches c");
        assertArrayEquals(new int[] {d}, topology.submittedSuccessors(c), "c did not reach d");
        assertEquals(1, topology.inDegree(d));
    }
}
