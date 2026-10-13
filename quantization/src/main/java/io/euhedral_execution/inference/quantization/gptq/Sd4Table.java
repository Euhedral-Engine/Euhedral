package io.euhedral_execution.inference.quantization.gptq;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/// The SD4 scale table of a tensor from its per-nearest-code costs (tools/euhedral_artifacts/nvfp4.py:
/// sd4_cost_bounds and sd4_table), on the host in float64.
public final class Sd4Table {

    public static final int CODES = 127;
    public static final int ENTRIES = 16;
    public static final int ABOVE = 40;

    private Sd4Table() {}

    /// costs[nearest][code] (measured from `below` under to ABOVE over each nearest code), energy and counts per
    /// nearest code -> the 16 ascending E4M3 codes minimizing the summed least cost of every nearest code.
    public static byte[] choose(double[][] costs, double[] energy, long[] counts, int below) {
        double[][] bounded = bounds(costs, energy, counts, below);
        int[] table = table(bounded);
        byte[] bytes = new byte[ENTRIES];
        for (int i = 0; i < ENTRIES; i++) bytes[i] = (byte) table[i];
        return bytes;
    }

    /// sd4_cost_bounds: too far above costs the blocks' energy, too far below is not allowed, nearest codes without
    /// blocks cost nothing.
    static double[][] bounds(double[][] costs, double[] energy, long[] counts, int below) {
        double[][] out = new double[CODES][CODES];
        for (int o = 0; o < CODES; o++) {
            for (int c = 0; c < CODES; c++) {
                double cost = costs[o][c];
                if (c > o + ABOVE) cost = energy[o];
                if (c < o - below) cost = Double.POSITIVE_INFINITY;
                if (counts[o] == 0) cost = 0.0;
                out[o][c] = cost;
            }
        }
        return out;
    }

    /// sd4_table: dynamic program over the ascending entries; ties go to the earliest choice.
    static int[] table(double[][] costs) {
        int n = CODES;
        double[] below = new double[n], above = new double[n];
        for (int b = 0; b < n; b++) {
            for (int o = 0; o < b; o++) below[b] += costs[o][b];
            for (int o = b; o < n; o++) above[b] += costs[o][b];
        }
        double[][] segment = new double[n][n];
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < n; b++) {
                if (a >= b) {
                    segment[a][b] = Double.POSITIVE_INFINITY;
                    continue;
                }
                double sum = 0;
                for (int o = a; o < b; o++) sum += Math.min(costs[o][a], costs[o][b]);
                segment[a][b] = sum;
            }
        }
        double[] best = below.clone();
        List<int[]> previous = new ArrayList<>();
        for (int step = 0; step < ENTRIES - 1; step++) {
            int[] argmin = new int[n];
            double[] next = new double[n];
            for (int b = 0; b < n; b++) {
                int choice = 0;
                double least = best[0] + segment[0][b];
                for (int a = 1; a < n; a++) {
                    double total = best[a] + segment[a][b];
                    if (total < least) {
                        least = total;
                        choice = a;
                    }
                }
                argmin[b] = choice;
                next[b] = least;
            }
            previous.add(argmin);
            best = next;
        }
        int last = 0;
        for (int b = 1; b < n; b++) if (best[b] + above[b] < best[last] + above[last]) last = b;
        int[] table = new int[ENTRIES];
        table[0] = last;
        for (int i = 0; i < previous.size(); i++) table[i + 1] = previous.get(previous.size() - 1 - i)[table[i]];
        Arrays.sort(table);
        return table;
    }
}
