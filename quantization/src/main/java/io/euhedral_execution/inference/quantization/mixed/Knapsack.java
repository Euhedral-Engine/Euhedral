package io.euhedral_execution.inference.quantization.mixed;

import java.util.Arrays;

/// The multiple-choice knapsack: one option per group, the most score under a byte budget.
public final class Knapsack {

    private Knapsack() {}

    /// The option of each group that maximizes the sum of `score` with the sum of `cost` at most `budget`, by
    /// dynamic programming over the budget in steps of `unit` bytes: a cost counts as its nearest whole number
    /// of units, the
    /// budget as its whole units. Time and memory `O(groups * options * budget / unit)`.
    ///
    /// @throws IllegalStateException when even the cheapest options exceed the budget
    public static int[] solve(double[][] score, long[][] cost, long budget, long unit) {
        if (unit < 1) throw new IllegalArgumentException("the unit must be at least one byte");
        if (score.length != cost.length) throw new IllegalArgumentException("score and cost differ in groups");
        int groups = score.length;
        int capacity = Math.toIntExact(budget / unit);
        int[][] units = new int[groups][];
        for (int i = 0; i < groups; i++) {
            if (score[i].length != cost[i].length)
                throw new IllegalArgumentException("score and cost differ in options at group " + i);
            units[i] = new int[cost[i].length];
            for (int k = 0; k < cost[i].length; k++)
                units[i][k] = Math.toIntExact(Math.round((double) cost[i][k] / unit));
        }
        byte[][] choice = new byte[groups][capacity + 1];
        double[] best = new double[capacity + 1];
        double[] next = new double[capacity + 1];
        for (int i = 0; i < groups; i++) {
            Arrays.fill(next, Double.NEGATIVE_INFINITY);
            byte[] picked = choice[i];
            for (int k = 0; k < units[i].length; k++) {
                int weight = units[i][k];
                double value = score[i][k];
                for (int b = weight; b <= capacity; b++) {
                    double candidate = best[b - weight] + value;
                    if (candidate > next[b]) {
                        next[b] = candidate;
                        picked[b] = (byte) k;
                    }
                }
            }
            double[] swap = best;
            best = next;
            next = swap;
        }
        if (best[capacity] == Double.NEGATIVE_INFINITY)
            throw new IllegalStateException("the budget of " + budget + " bytes is below the cheapest choices");
        int[] result = new int[groups];
        int remaining = capacity;
        for (int i = groups - 1; i >= 0; i--) {
            result[i] = choice[i][remaining];
            remaining -= units[i][result[i]];
        }
        return result;
    }
}
