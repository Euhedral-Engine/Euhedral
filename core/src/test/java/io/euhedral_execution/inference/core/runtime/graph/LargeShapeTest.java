package io.euhedral_execution.inference.core.runtime.graph;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// A prompt graph has tens of thousands of stages: the analyses of a shape stay linear in its size.
@Timeout(60)
class LargeShapeTest {

    private static long allocatedBytes() {
        return ((com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean()).getCurrentThreadAllocatedBytes();
    }

    @Test
    void workspaceUseOfAHundredThousandStagesStaysLinear() {
        int stages = 100_000;
        int[][] dependencies = new int[stages][];
        int[][] buffers = new int[stages][];
        dependencies[0] = new int[0];
        for (int stage = 1; stage < stages; stage++) dependencies[stage] = new int[] {stage - 1};
        for (int stage = 0; stage < stages; stage++) buffers[stage] = new int[] {stage % 40};
        GraphShape shape = TestShapes.of(StageTopology.submitted(dependencies), buffers, 40);
        long before = allocatedBytes();
        long started = System.nanoTime();
        WorkspaceUse use = WorkspaceUse.of(shape);
        long nanos = System.nanoTime() - started;
        long bytes = allocatedBytes() - before;
        assertTrue(nanos < 2_000_000_000L, "took " + nanos / 1_000_000 + " ms");
        assertTrue(bytes < (64L << 20), "allocated " + (bytes >> 20) + " MiB");
        assertArrayEquals(new int[] {0}, use.entries(0));
        assertArrayEquals(new int[] {stages - 40}, use.exits(0));
    }

    @Test
    void entriesAndExitsMatchABruteForceAnswerOnRandomGraphs() {
        Random random = new Random(42);
        for (int round = 0; round < 20; round++) {
            int stages = 300;
            int count = 6;
            int[][] dependencies = new int[stages][];
            int[][] buffers = new int[stages][];
            for (int stage = 0; stage < stages; stage++) {
                List<Integer> producers = new ArrayList<>();
                for (int earlier = Math.max(0, stage - 12); earlier < stage; earlier++)
                    if (random.nextInt(4) == 0) producers.add(earlier);
                dependencies[stage] =
                        producers.stream().mapToInt(Integer::intValue).toArray();
                List<Integer> touched = new ArrayList<>();
                for (int buffer = 0; buffer < count; buffer++) if (random.nextInt(5) == 0) touched.add(buffer);
                buffers[stage] = touched.stream().mapToInt(Integer::intValue).toArray();
            }
            StageTopology topology = StageTopology.submitted(dependencies);
            WorkspaceUse use = WorkspaceUse.of(TestShapes.of(topology, buffers, count));
            BitSet[] below = new BitSet[stages];
            for (int stage = stages - 1; stage >= 0; stage--) {
                below[stage] = new BitSet(stages);
                for (int next : topology.submittedSuccessors(stage)) {
                    below[stage].set(next);
                    below[stage].or(below[next]);
                }
            }
            for (int buffer = 0; buffer < count; buffer++) {
                BitSet all = new BitSet(stages);
                for (int stage = 0; stage < stages; stage++)
                    for (int b : buffers[stage]) if (b == buffer) all.set(stage);
                BitSet reached = new BitSet(stages);
                for (int stage = all.nextSetBit(0); stage >= 0; stage = all.nextSetBit(stage + 1))
                    reached.or(below[stage]);
                List<Integer> entries = new ArrayList<>(), exits = new ArrayList<>();
                for (int stage = all.nextSetBit(0); stage >= 0; stage = all.nextSetBit(stage + 1)) {
                    if (!reached.get(stage)) entries.add(stage);
                    if (!below[stage].intersects(all)) exits.add(stage);
                }
                assertArrayEquals(entries.stream().mapToInt(Integer::intValue).toArray(), use.entries(buffer));
                assertArrayEquals(exits.stream().mapToInt(Integer::intValue).toArray(), use.exits(buffer));
            }
        }
    }

    @Test
    void chainingUsersAcrossALongShapeAllocatesLittle() {
        var builder = new ShapeBuilder<Integer>();
        int stages = 40_000;
        for (int stage = 0; stage < stages; stage++) {
            builder.stage(stage);
            if (stage > 0) builder.submitted(stage - 1, stage);
        }
        int[] users = new int[stages / 10];
        for (int i = 0; i < users.length; i++) users[i] = i * 10;
        long before = allocatedBytes();
        builder.chain(users);
        long bytes = allocatedBytes() - before;
        assertTrue(bytes < (8L << 20), "allocated " + (bytes >> 20) + " MiB");
    }
}
