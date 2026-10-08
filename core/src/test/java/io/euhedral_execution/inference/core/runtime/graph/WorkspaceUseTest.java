package io.euhedral_execution.inference.core.runtime.graph;

import static io.euhedral_execution.inference.core.runtime.graph.StageGraphFixtures.dependencies;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

class WorkspaceUseTest {

    @Test
    void entriesAndExitsAreTheFirstAndLastAccessorsOfEachBuffer() {
        // 0 writes A; 1 and 2 read it; 3 writes it after both; 4 touches nothing.
        var topology = StageTopology.submitted(
                dependencies(new int[0], new int[] {0}, new int[] {0}, new int[] {1, 2}, new int[] {3}));
        var use = WorkspaceUse.of(TestShapes.of(topology, new int[][] {{0}, {0}, {0}, {0}, {}}, 1));
        assertArrayEquals(new int[] {0}, use.entries(0));
        assertArrayEquals(new int[] {3}, use.exits(0));
    }

    @Test
    void parallelReadersAreAllExits() {
        var topology = StageTopology.submitted(dependencies(new int[0], new int[] {0}, new int[] {0}));
        var use = WorkspaceUse.of(TestShapes.of(topology, new int[][] {{0}, {0}, {0}}, 1));
        assertArrayEquals(new int[] {0}, use.entries(0));
        assertArrayEquals(new int[] {1, 2}, use.exits(0));
    }

    @Test
    void aBufferNoStageTouchesHasNeitherEntriesNorExits() {
        var topology = StageTopology.submitted(dependencies(new int[0]));
        var use = WorkspaceUse.of(TestShapes.of(topology, new int[][] {{0}}, 2));
        assertArrayEquals(new int[0], use.entries(1));
        assertArrayEquals(new int[0], use.exits(1));
    }
}
