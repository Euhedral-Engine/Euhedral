package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.Test;

class KnapsackTest {

    private static double bestByBruteForce(double[][] score, long[][] cost, long budget) {
        int n = score.length;
        double best = Double.NEGATIVE_INFINITY;
        int[] pick = new int[n];
        long total = 1;
        for (double[] row : score) total *= row.length;
        for (long code = 0; code < total; code++) {
            long rest = code;
            long bytes = 0;
            double value = 0;
            for (int i = 0; i < n; i++) {
                pick[i] = (int) (rest % score[i].length);
                rest /= score[i].length;
                bytes += cost[i][pick[i]];
                value += score[i][pick[i]];
            }
            if (bytes <= budget && value > best) best = value;
        }
        return best;
    }

    @Test
    void matchesBruteForceOnRandomSmallInstances() {
        Random random = new Random(5);
        for (int trial = 0; trial < 200; trial++) {
            int n = 2 + random.nextInt(5), k = 2 + random.nextInt(3);
            double[][] score = new double[n][k];
            long[][] cost = new long[n][k];
            long minimum = 0, maximum = 0;
            for (int i = 0; i < n; i++) {
                long low = Long.MAX_VALUE, high = 0;
                for (int j = 0; j < k; j++) {
                    score[i][j] = random.nextGaussian();
                    cost[i][j] = 1 + random.nextInt(20);
                    low = Math.min(low, cost[i][j]);
                    high = Math.max(high, cost[i][j]);
                }
                minimum += low;
                maximum += high;
            }
            long budget = minimum + random.nextLong(maximum - minimum + 1);
            int[] choice = Knapsack.solve(score, cost, budget, 1);
            long bytes = 0;
            double value = 0;
            for (int i = 0; i < n; i++) {
                bytes += cost[i][choice[i]];
                value += score[i][choice[i]];
            }
            assertTrue(bytes <= budget, "trial " + trial);
            assertEquals(bestByBruteForce(score, cost, budget), value, 1e-12, "trial " + trial);
        }
    }

    @Test
    void roundsCostsToTheUnit() {
        // costs 100 and 300 with unit 100 are 1 and 3 units; a budget of 4 units takes the better of the two groups.
        double[][] score = {{0, 1}, {0, 2}};
        long[][] cost = {{100, 300}, {100, 300}};
        assertArrayEquals(new int[] {0, 1}, Knapsack.solve(score, cost, 400, 100));
    }

    @Test
    void refusesABudgetBelowTheCheapestChoices() {
        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> Knapsack.solve(new double[][] {{0, 1}}, new long[][] {{5, 9}}, 4, 1));
        assertTrue(failure.getMessage().contains("budget"), failure.getMessage());
    }
}
