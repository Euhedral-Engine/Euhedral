package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

class RcoTest {

    /// f(p) = sum over objects of (expected value - target)^2, separable and quadratic in the probabilities.
    private record Quadratic(double[][] value, double[] target) implements Rco.Objective {
        @Override
        public double value(double[][] p) {
            double sum = 0;
            for (int i = 0; i < p.length; i++) {
                double expected = 0;
                for (int k = 0; k < p[i].length; k++) expected += p[i][k] * value[i][k];
                sum += (expected - target[i]) * (expected - target[i]);
            }
            return sum;
        }

        @Override
        public double[][] gradient(double[][] p) {
            double[][] gradient = new double[p.length][];
            for (int i = 0; i < p.length; i++) {
                double expected = 0;
                for (int k = 0; k < p[i].length; k++) expected += p[i][k] * value[i][k];
                gradient[i] = new double[p[i].length];
                for (int k = 0; k < p[i].length; k++) gradient[i][k] = 2 * (expected - target[i]) * value[i][k];
            }
            return gradient;
        }
    }

    private static double[][] oneHot(int[] choice, int options) {
        double[][] p = new double[choice.length][options];
        for (int i = 0; i < choice.length; i++) p[i][choice[i]] = 1;
        return p;
    }

    private record Instance(Quadratic objective, long[][] cost, long budget) {}

    private static Instance instance(int n, int k, long seed) {
        Random random = new Random(seed);
        double[][] value = new double[n][k];
        double[] target = new double[n];
        long[][] cost = new long[n][k];
        long minimum = 0, maximum = 0;
        for (int i = 0; i < n; i++) {
            long low = Long.MAX_VALUE, high = 0;
            for (int j = 0; j < k; j++) {
                value[i][j] = random.nextDouble();
                cost[i][j] = 10 + 10L * j + random.nextInt(5);
                low = Math.min(low, cost[i][j]);
                high = Math.max(high, cost[i][j]);
            }
            target[i] = random.nextDouble();
            minimum += low;
            maximum += high;
        }
        return new Instance(new Quadratic(value, target), cost, (minimum + maximum) / 2);
    }

    private static double bruteForce(Instance instance) {
        int n = instance.cost().length, k = instance.cost()[0].length;
        double best = Double.POSITIVE_INFINITY;
        int[] pick = new int[n];
        long total = 1;
        for (int i = 0; i < n; i++) total *= k;
        for (long code = 0; code < total; code++) {
            long rest = code, bytes = 0;
            for (int i = 0; i < n; i++) {
                pick[i] = (int) (rest % k);
                rest /= k;
                bytes += instance.cost()[i][pick[i]];
            }
            if (bytes > instance.budget()) continue;
            best = Math.min(best, instance.objective().value(oneHot(pick, k)));
        }
        return best;
    }

    @Test
    void expectedCostStaysOnTheManifoldEveryStep() {
        Instance instance = instance(8, 3, 1);
        Rco.Result result = Rco.run(
                instance.objective(), instance.cost(), instance.budget(), 1, new Rco.Settings(60, 0.1, 4, 1), 7, 0);
        assertTrue(result.maxManifoldError() <= 1e-6, "manifold error " + result.maxManifoldError());
        double expected = 0;
        for (int i = 0; i < result.p().length; i++)
            for (int k = 0; k < result.p()[i].length; k++) expected += result.p()[i][k] * instance.cost()[i][k];
        assertEquals(instance.budget(), expected, 1e-6 * instance.budget());
    }

    @Test
    void reachesTheBruteForceOptimumOfASeparableQuadratic() {
        int reached = 0;
        for (long seed = 1; seed <= 6; seed++) {
            Instance instance = instance(7, 3, seed);
            List<Rco.Result> results = Rco.solve(
                    instance.objective(),
                    instance.cost(),
                    instance.budget(),
                    1,
                    new Rco.Settings(200, 0.1, 8, 4),
                    seed);
            double best = results.stream().mapToDouble(Rco.Result::value).min().orElseThrow();
            double optimum = bruteForce(instance);
            assertTrue(best >= optimum - 1e-12, "better than the optimum?");
            for (Rco.Result result : results) {
                long bytes = 0;
                for (int i = 0; i < result.choice().length; i++) bytes += instance.cost()[i][result.choice()[i]];
                assertTrue(bytes <= instance.budget());
            }
            if (best <= optimum + 1e-9) reached++;
        }
        assertEquals(6, reached, "instances solved to the brute-force optimum");
    }

    @Test
    void theSameSeedGivesTheSameAssignment() {
        Instance instance = instance(6, 3, 3);
        Rco.Settings settings = new Rco.Settings(40, 0.1, 4, 2);
        assertEquals(
                Arrays.toString(Rco.solve(instance.objective(), instance.cost(), instance.budget(), 1, settings, 5)
                        .get(0)
                        .choice()),
                Arrays.toString(Rco.solve(instance.objective(), instance.cost(), instance.budget(), 1, settings, 5)
                        .get(0)
                        .choice()));
    }
}
