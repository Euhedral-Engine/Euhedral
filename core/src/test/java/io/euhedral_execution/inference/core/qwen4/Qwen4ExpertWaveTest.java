package io.euhedral_execution.inference.core.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The wave builder on the CPU: grouping, ordering, wave limits and the descriptor it writes.
class Qwen4ExpertWaveTest {

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

    /// Replays every wave's descriptor on the CPU: the (row, expert, weight) triples and, per row, the expert ids in
    /// the order the combine kernel would add them.
    private static void check(
            Qwen4ExpertWave plan, int rows, int topK, int experts, int[] ids, short[] weights, long[] slotAddress) {
        int[] expectedPairsPerRow = new int[rows];
        boolean[][] seen = new boolean[rows][experts + 1];
        int[] lastExpert = new int[rows];
        java.util.Arrays.fill(lastExpert, -1);
        int totalPairs = 0;
        int previousWaveLast = -1;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment descriptor = arena.allocate(plan.descriptorBytes(), 16);
            for (int wave = 0; wave < plan.waveCount(); wave++) {
                int waveExperts = plan.waveExpertCount(wave);
                assertTrue(waveExperts >= 1 && waveExperts <= plan.maxWaveExperts());
                assertTrue(plan.wavePairCount(wave) <= plan.maxWavePairs());
                plan.fill(wave, descriptor);
                for (int i = 0; i < waveExperts; i++)
                    plan.setSlot(descriptor, i, slotAddress[plan.waveExpert(wave, i)]);
                int pairs = plan.wavePairCount(wave);
                int items = plan.waveItemCount(wave);
                int covered = 0;
                int lastItemExpert = -1;
                int[] pairExpert = new int[pairs];
                java.util.Arrays.fill(pairExpert, -1);
                for (int item = 0; item < items; item++) {
                    long at = plan.itemsOffset() + 16L * item;
                    int slot = descriptor.get(INT, at),
                            begin = descriptor.get(INT, at + 4),
                            count = descriptor.get(INT, at + 8);
                    assertTrue(count >= 1 && count <= 8, "item count " + count);
                    assertTrue(slot >= lastItemExpert, "items ascend");
                    lastItemExpert = slot;
                    int expert = plan.waveExpert(wave, slot);
                    assertEquals(slotAddress[expert], descriptor.get(LONG, plan.slotsOffset() + 8L * slot));
                    for (int c = 0; c < count; c++) {
                        assertEquals(-1, pairExpert[begin + c], "a pair is in one item");
                        pairExpert[begin + c] = expert;
                        covered++;
                    }
                }
                assertEquals(pairs, covered);
                for (int p = 0; p < pairs; p++) {
                    int row = descriptor.get(INT, plan.pairsOffset() + 8L * p);
                    int weight = descriptor.get(INT, plan.pairsOffset() + 8L * p + 4);
                    int expert = pairExpert[p];
                    assertTrue(expert > previousWaveLast, "experts ascend across waves");
                    boolean found = false;
                    for (int k = 0; k < topK; k++) {
                        if (ids[row * topK + k] == expert && (weights[row * topK + k] & 0xffff) == weight) found = true;
                    }
                    assertTrue(found, "pair " + p + " of wave " + wave + " is a router entry");
                }
                int lastOfWave = pairExpert[pairs - 1];
                // row lists: every pair once, ascending expert inside a row
                boolean[] listed = new boolean[pairs];
                for (int row = 0; row < rows; row++) {
                    int from = descriptor.get(INT, plan.rowOffsetsOffset() + 4L * row);
                    int to = descriptor.get(INT, plan.rowOffsetsOffset() + 4L * (row + 1));
                    for (int i = from; i < to; i++) {
                        int p = descriptor.get(INT, plan.rowPairsOffset() + 4L * i);
                        assertEquals(
                                row,
                                descriptor.get(INT, plan.pairsOffset() + 8L * p),
                                "row list entry belongs to its row");
                        assertTrue(!listed[p]);
                        listed[p] = true;
                        int expert = pairExpert[p];
                        assertTrue(expert >= lastExpert[row], "a row's additions ascend in expert id");
                        lastExpert[row] = expert;
                        expectedPairsPerRow[row]++;
                        seen[row][expert] = true;
                    }
                }
                for (boolean l : listed) assertTrue(l);
                totalPairs += pairs;
                previousWaveLast = lastOfWave;
            }
        }
        int routed = 0;
        for (int i = 0; i < rows * topK; i++) if (ids[i] < experts) routed++;
        assertEquals(routed, totalPairs);
        for (int row = 0; row < rows; row++) {
            int expected = 0;
            for (int k = 0; k < topK; k++) if (ids[row * topK + k] < experts) expected++;
            assertEquals(expected, expectedPairsPerRow[row]);
        }
    }

    @Test
    void waveSplitsKeepEveryPairOnceAndAscendPerRow() {
        SplittableRandom random = new SplittableRandom(5);
        int experts = 512, topK = 10;
        long[] slots = new long[experts];
        for (int e = 0; e < experts; e++) slots[e] = 0x7000_0000_0000L + (long) e * 2_768_896;
        int[][] limits = {{64, 5120}, {20, 600}, {1, 512}, {7, 512}, {512, 512}, {64, 1024}};
        int[] rowCounts = {1, 2, 8, 64, 100, 512};
        for (int[] limit : limits) {
            for (int rows : rowCounts) {
                for (boolean skewed : new boolean[] {false, true}) {
                    int[] ids = new int[rows * topK];
                    short[] weights = new short[rows * topK];
                    route(random, rows, topK, experts, skewed, ids, weights);
                    Qwen4ExpertWave plan = new Qwen4ExpertWave(experts, topK, 512, limit[0], limit[1]);
                    plan.plan(rows, ids, weights);
                    check(plan, rows, topK, experts, ids, weights, slots);
                }
            }
        }
    }

    @Test
    void paddingIdsAreSkippedAndRepeatedExpertsKeepTheirPairs() {
        int experts = 16, topK = 4, rows = 3;
        int[] ids = {1, 16, 1, 5, 16, 16, 16, 16, 2, 3, 4, 5};
        short[] weights = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        Qwen4ExpertWave plan = new Qwen4ExpertWave(experts, topK, 4, 3, 16);
        plan.plan(rows, ids, weights);
        assertEquals(5, plan.activeExperts());
        assertEquals(2, plan.expertPairCount(1));
        assertEquals(0, plan.expertPairCount(16 - 1));
        assertEquals(2, plan.waveCount());
        assertEquals(3, plan.waveExpertCount(0));
        long[] slots = new long[experts];
        for (int e = 0; e < experts; e++) slots[e] = 4096L * (e + 1);
        check(plan, rows, topK, experts, ids, weights, slots);
    }

    @Test
    void anExpertThatDoesNotFitAWaveAndBadIdsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Qwen4ExpertWave(8, 2, 4, 2, 3));
        Qwen4ExpertWave plan = new Qwen4ExpertWave(8, 2, 4, 2, 4);
        assertThrows(IllegalArgumentException.class, () -> plan.plan(2, new int[] {0, 9, 1, 2}, new short[4]));
        assertThrows(IllegalArgumentException.class, () -> plan.plan(5, new int[10], new short[10]));
        plan.plan(2, new int[] {0, 8, 8, 8}, new short[4]);
        assertEquals(1, plan.waveCount());
        assertEquals(1, plan.wavePairCount(0));
        plan.plan(2, new int[] {8, 8, 8, 8}, new short[4]);
        assertEquals(0, plan.waveCount());
    }

    @Test
    void scratchAndDescriptorSizesAreBoundedByTheCapacities() {
        Qwen4ExpertWave plan = new Qwen4ExpertWave(512, 10, 512, 64, 1024);
        assertEquals(1024 * (640 + 2560) * 2L, Qwen4ExpertWave.scratchBytes(1024));
        assertTrue(plan.descriptorBytes() % 16 == 0);
        assertTrue(plan.rowPairsOffset() + 4 * 1024 <= plan.descriptorBytes());
        assertTrue(plan.itemsOffset() >= 8 * 64);
    }
}
