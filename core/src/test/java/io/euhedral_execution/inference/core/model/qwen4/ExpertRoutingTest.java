package io.euhedral_execution.inference.core.model.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The block plan on the CPU: grouping by expert, each expert's contiguous work items, and the descriptor it writes.
class ExpertRoutingTest {

    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static void route(
            SplittableRandom random, int rows, int topK, int experts, boolean skewed, int[] ids, short[] weights) {
        for (int r = 0; r < rows; r++) {
            for (int k = 0; k < topK; k++) {
                int id;
                boolean fresh;
                do {
                    id = skewed
                            ? (int) Math.min(experts - 1, Math.abs(random.nextGaussian()) * experts / 6)
                            : random.nextInt(experts);
                    fresh = true;
                    for (int j = 0; j < k; j++) if (ids[r * topK + j] == id) fresh = false;
                } while (!fresh);
                ids[r * topK + k] = id;
                weights[r * topK + k] = (short) (0x3c00 + random.nextInt(0x400));
            }
        }
    }

    /// Replays the descriptor on the CPU: every expert's items cover exactly its pairs and name its slot, every pair
    /// is a router entry, and every row's list adds its pairs in ascending expert id.
    private static void check(
            ExpertRouting plan, int rows, int topK, int experts, int[] ids, short[] weights, long[] slotAddress) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment descriptor = arena.allocate(plan.descriptorBytes(), 16);
            plan.fill(descriptor);
            int active = plan.activeExperts();
            for (int i = 0; i < active; i++) plan.setSlot(descriptor, i, slotAddress[plan.activeExpert(i)]);
            int pairs = plan.pairCount();
            int[] pairExpert = new int[pairs];
            java.util.Arrays.fill(pairExpert, -1);
            int previous = -1;
            for (int i = 0; i < active; i++) {
                int expert = plan.activeExpert(i);
                assertTrue(expert > previous, "active experts ascend");
                previous = expert;
                assertEquals(slotAddress[expert], descriptor.get(LONG, plan.slotsOffset() + 8L * i));
                int covered = 0;
                for (int item = plan.itemStart(i); item < plan.itemStart(i) + plan.itemCount(i); item++) {
                    long at = plan.itemsOffset() + 16L * item;
                    int slot = descriptor.get(INT, at),
                            begin = descriptor.get(INT, at + 4),
                            count = descriptor.get(INT, at + 8);
                    assertEquals(i, slot, "an expert's items name its own slot");
                    assertTrue(count >= 1 && count <= ExpertRouting.ITEM_PAIRS, "item count " + count);
                    for (int c = 0; c < count; c++) {
                        assertEquals(-1, pairExpert[begin + c], "a pair is in one item");
                        pairExpert[begin + c] = expert;
                        covered++;
                    }
                }
                assertEquals(plan.expertPairCount(expert), covered);
            }
            for (int p = 0; p < pairs; p++) {
                int row = descriptor.get(INT, plan.pairsOffset() + 8L * p);
                int weight = descriptor.get(INT, plan.pairsOffset() + 8L * p + 4);
                boolean found = false;
                for (int k = 0; k < topK; k++)
                    if (ids[row * topK + k] == pairExpert[p] && (weights[row * topK + k] & 0xffff) == weight)
                        found = true;
                assertTrue(found, "pair " + p + " is a router entry");
            }
            boolean[] listed = new boolean[pairs];
            for (int row = 0; row < rows; row++) {
                int from = descriptor.get(INT, plan.rowOffsetsOffset() + 4L * row);
                int to = descriptor.get(INT, plan.rowOffsetsOffset() + 4L * (row + 1));
                int last = -1;
                int expected = 0;
                for (int k = 0; k < topK; k++) if (ids[row * topK + k] < experts) expected++;
                assertEquals(expected, to - from, "row " + row + " lists each of its pairs");
                for (int i = from; i < to; i++) {
                    int p = descriptor.get(INT, plan.rowPairsOffset() + 4L * i);
                    assertEquals(row, descriptor.get(INT, plan.pairsOffset() + 8L * p));
                    assertTrue(!listed[p]);
                    listed[p] = true;
                    assertTrue(pairExpert[p] >= last, "a row's additions ascend in expert id");
                    last = pairExpert[p];
                }
            }
            for (boolean l : listed) assertTrue(l);
        }
    }

    @Test
    void everyPairIsInItsExpertsItemsAndEveryRowAddsInAscendingExpertOrder() {
        SplittableRandom random = new SplittableRandom(5);
        int experts = 512, topK = 10;
        long[] slots = new long[experts];
        for (int e = 0; e < experts; e++) slots[e] = 0x7000_0000_0000L + (long) e * 2_768_896;
        for (int rows : new int[] {1, 2, 8, 64, 100, 512}) {
            for (boolean skewed : new boolean[] {false, true}) {
                int[] ids = new int[rows * topK];
                short[] weights = new short[rows * topK];
                route(random, rows, topK, experts, skewed, ids, weights);
                ExpertRouting plan = new ExpertRouting(experts, topK, 512);
                plan.plan(rows, ids, weights);
                check(plan, rows, topK, experts, ids, weights, slots);
            }
        }
    }

    @Test
    void paddingIdsAreSkippedAndRepeatedExpertsKeepTheirPairs() {
        int experts = 16, topK = 4, rows = 3;
        int[] ids = {1, 16, 1, 5, 16, 16, 16, 16, 2, 3, 4, 5};
        short[] weights = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        ExpertRouting plan = new ExpertRouting(experts, topK, 4);
        plan.plan(rows, ids, weights);
        assertEquals(5, plan.activeExperts());
        assertEquals(2, plan.expertPairCount(1));
        assertEquals(0, plan.expertPairCount(16 - 1));
        long[] slots = new long[experts];
        for (int e = 0; e < experts; e++) slots[e] = 4096L * (e + 1);
        check(plan, rows, topK, experts, ids, weights, slots);
    }

    @Test
    void anExpertWithManyRowsHasSeveralItemsAndBadIdsAreRejected() {
        ExpertRouting plan = new ExpertRouting(8, 2, 40);
        int[] ids = new int[80];
        for (int r = 0; r < 40; r++) {
            ids[2 * r] = 3;
            ids[2 * r + 1] = 8;
        }
        plan.plan(40, ids, new short[80]);
        assertEquals(1, plan.activeExperts());
        assertEquals(3, plan.itemCount(0), "40 pairs are three items of at most 16");
        assertThrows(IllegalArgumentException.class, () -> plan.plan(2, new int[] {0, 9, 1, 2}, new short[4]));
        assertThrows(IllegalArgumentException.class, () -> plan.plan(41, new int[82], new short[82]));
        plan.plan(2, new int[] {8, 8, 8, 8}, new short[4]);
        assertEquals(0, plan.activeExperts());
    }

    @Test
    void scratchAndDescriptorSizesAreBoundedByTheCapacities() {
        ExpertRouting plan = new ExpertRouting(512, 10, 512);
        assertEquals(5120, plan.maxPairs());
        assertEquals(512, plan.maxActive());
        assertEquals(10, new ExpertRouting(512, 10, 1).maxActive(), "a decode token names ten experts at most");
        assertEquals(1024 * (640 + 2560) * 2L, ExpertRouting.scratchBytes(1024));
        assertTrue(plan.descriptorBytes() % 16 == 0);
        assertTrue(plan.rowPairsOffset() + 4 * 5120 <= plan.descriptorBytes());
        assertTrue(plan.itemsOffset() >= 8 * 512);
    }

    @Test
    void geometryChecksTheRecordLayoutTheKernelsAssume() {
        ExpertOps.Geometry geometry = ExpertOps.Geometry.flashNext();
        assertEquals(2560, geometry.hidden());
        assertEquals(640, geometry.inter());
        assertEquals(1_843_456, geometry.downOffset());
        assertThrows(IllegalArgumentException.class, () -> new ExpertOps.Geometry(2560, 600, 0, 1_843_456));
        assertThrows(IllegalArgumentException.class, () -> new ExpertOps.Geometry(2560, 640, 0, 1_843_457));
        assertThrows(IllegalArgumentException.class, () -> new ExpertOps.Geometry(2500, 640, 0, 1_843_456));
        assertEquals(16 * (640 + 2560) * 2L, ExpertOps.Scratch.bytes(16, 640, 2560));
        assertEquals(1280 * 16L, ExpertOps.Scratch.weightedOffset(16, 640));
    }
}
