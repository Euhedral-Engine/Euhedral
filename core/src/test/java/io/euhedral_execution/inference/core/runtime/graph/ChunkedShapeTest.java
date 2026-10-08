package io.euhedral_execution.inference.core.runtime.graph;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// A chunked shape is C copies of a chunk template in one graph: chunk c's first users of a workspace buffer or of
/// a piece of carried sequence state follow chunk c-1's last users of it, and nothing else orders the chunks.
@Timeout(30)
class ChunkedShapeTest {

    private static final ChunkedShape.Chunks TEST_CHUNKS = new ChunkedShape.Chunks() {
        @Override
        public StageFrame create(StageGraph graph, int stage, int chunk, int templateStage, ExecutionGpu gpu) {
            return new StageGraphFixtures.TestStage(graph, stage);
        }

        @Override
        public GraphStorage newStorage(ExecutionGpu gpu) {
            throw new UnsupportedOperationException();
        }
    };

    private static boolean edge(StageTopology topology, int from, int to) {
        return Arrays.stream(topology.submittedSuccessors(from)).anyMatch(next -> next == to);
    }

    /// Stage 0 writes buffer 0, stage 1 touches nothing, stage 2 reads buffer 0.
    private static GraphShape bufferTemplate() {
        int[][] dependencies = {{}, {0}, {1}};
        return TestShapes.of(StageTopology.submitted(dependencies), new int[][] {{0}, {}, {0}}, 1);
    }

    @Test
    void aChunksFirstWriterFollowsThePreviousChunksLastReader() {
        GraphShape template = bufferTemplate();
        var shape = new ChunkedShape(template, template, 3, TEST_CHUNKS);
        StageTopology topology = shape.topology();
        assertEquals(9, topology.size());
        for (int chunk = 1; chunk < 3; chunk++)
            assertTrue(edge(topology, shape.stage(chunk - 1, 2), shape.stage(chunk, 0)), "chunk " + chunk);
        assertEquals(2, shape.chunkOf(shape.stage(2, 1)));
        assertEquals(1, shape.templateStageOf(shape.stage(2, 1)));
        assertArrayEquals(new int[] {0}, shape.workspaceBuffers(shape.stage(1, 2)));
    }

    @Test
    void carriedStateChainsLayerByLayer() {
        // Two layer stages, each carrying its own key; no workspace buffers.
        int[][] dependencies = {{}, {0}};
        GraphShape template = TestShapes.of(
                StageTopology.submitted(dependencies), new int[][] {{}, {}}, 0, new int[][] {{0}, {1}}, 2);
        var shape = new ChunkedShape(template, template, 2, TEST_CHUNKS);
        StageTopology topology = shape.topology();
        assertTrue(edge(topology, shape.stage(0, 0), shape.stage(1, 0)), "layer 0 to layer 0");
        assertTrue(edge(topology, shape.stage(0, 1), shape.stage(1, 1)), "layer 1 to layer 1");
        assertFalse(edge(topology, shape.stage(0, 1), shape.stage(1, 0)), "layer 0 does not wait for layer 1");
        assertArrayEquals(new int[] {1}, shape.carriedState(shape.stage(1, 1)));
        assertEquals(2, shape.carriedStateCount());
    }

    @Test
    void stagesWithNothingSharedRunAhead() {
        int[][] dependencies = {{}, {}, {0, 1}};
        GraphShape template = TestShapes.of(StageTopology.submitted(dependencies), new int[][] {{0}, {}, {0}}, 1);
        var shape = new ChunkedShape(template, template, 3, TEST_CHUNKS);
        for (int chunk = 0; chunk < 3; chunk++)
            assertEquals(0, shape.topology().inDegree(shape.stage(chunk, 1)), "an input stage of chunk " + chunk);
    }

    @Test
    void theLastChunkMayUseItsOwnTemplate() {
        GraphShape full = bufferTemplate();
        // The last chunk touches buffer 0 first at its stage 1.
        GraphShape last = TestShapes.of(StageTopology.submitted(new int[][] {{}, {0}}), new int[][] {{}, {0}}, 1);
        var shape = new ChunkedShape(full, last, 3, TEST_CHUNKS);
        assertEquals(8, shape.topology().size());
        assertTrue(edge(shape.topology(), shape.stage(1, 2), shape.stage(2, 1)));
        assertEquals(0, shape.topology().inDegree(shape.stage(2, 0)));
    }

    @Test
    void retiredEdgesOfTheTemplateStayRetired() {
        StageTopology retired = StageTopology.of(
                new int[][] {{}, {0}}, new StageTopology.Boundary[][] {{}, {StageTopology.Boundary.RETIRED}});
        GraphShape template = TestShapes.of(retired, new int[][] {{}, {}}, 0);
        var shape = new ChunkedShape(template, template, 2, TEST_CHUNKS);
        assertArrayEquals(new int[] {shape.stage(1, 1)}, shape.topology().retiredSuccessors(shape.stage(1, 0)));
    }

    @Test
    void aTenThousandChunkShapeBuildsInLinearTime() {
        GraphShape template = bufferTemplate();
        long started = System.nanoTime();
        var shape = new ChunkedShape(template, template, 10_000, TEST_CHUNKS);
        WorkspaceUse.of(shape);
        assertTrue(System.nanoTime() - started < 3_000_000_000L);
        assertEquals(30_000, shape.topology().size());
    }
}
