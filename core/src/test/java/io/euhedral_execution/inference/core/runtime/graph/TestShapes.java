package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;

/// Graph shapes for tests: a topology of [StageGraphFixtures.TestStage]s, each touching the workspace buffers
/// listed for it.
final class TestShapes {

    private TestShapes() {}

    static GraphShape of(StageTopology topology, int[][] buffers, int count) {
        return of(topology, buffers, count, new int[topology.size()][0], 0);
    }

    /// As above, each stage also carrying the sequence-state keys listed for it.
    static GraphShape of(StageTopology topology, int[][] buffers, int count, int[][] carried, int carriedCount) {
        return new GraphShape() {
            @Override
            public int[] carriedState(int stage) {
                return carried[stage];
            }

            @Override
            public int carriedStateCount() {
                return carriedCount;
            }

            @Override
            public StageTopology topology() {
                return topology;
            }

            @Override
            public StageFrame createStage(StageGraph graph, int stage, ExecutionGpu gpu) {
                return new StageGraphFixtures.TestStage(graph, stage);
            }

            @Override
            public GraphStorage newStorage(ExecutionGpu gpu) {
                throw new UnsupportedOperationException("test shapes own no storage");
            }

            @Override
            public int[] workspaceBuffers(int stage) {
                return buffers[stage];
            }

            @Override
            public int workspaceBufferCount() {
                return count;
            }
        };
    }
}
