package io.euhedral_execution.inference.quantization.mixed;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/// Riemannian Constrained Optimization, Helcig and Alistarh, arXiv:2605.00649 (github.com/IST-DASLab/RCO),
/// applied to choosing one option per object under a byte budget.
///
/// Every object `i` has logits `alpha[i][k]` over its options and probabilities `p = softmax(alpha)`. The
/// expected bytes `C(alpha) = sum_i sum_k p[i][k] c[i][k]` are held on the budget by working on the manifold
/// `C = budget` (Algorithm 1 of the paper):
///
/// 1. the normal of the manifold is `(grad C)[i][k] = p[i][k] * (c[i][k] - E_{p_i}[c_i])`;
/// 2. a gradient of the objective is estimated by Gumbel straight-through sampling: for each of `samples`
/// draws of    Gumbel noise `G`, the forward pass is the knapsack optimum (exact within the byte budget) of
/// the scores    `(alpha + G) / tau`, the objective's gradient is taken at that hard assignment, and the
/// backward pass goes    through `softmax((alpha + G) / tau)`; `tau` falls exponentially from 1.0 to 0.01 over
/// the steps;
/// 3. the gradient is projected onto the tangent space (the plane orthogonal to the normal);
/// 4. Adam turns the projected gradient into a direction, which is projected onto the tangent space again;
/// 5. `alpha` takes a step of `lr` against the direction;
/// 6. the retraction puts `alpha` back on the manifold by adding `t` times the cost vector, `t` found by
/// bisection on    `[-50, 50]` to 1e-8 (the expected bytes grow with `t`);
/// 7. Adam's first moment is transported to the new tangent space by projecting it there.
///
/// The final assignment is the knapsack optimum of `log p`.
public final class Rco {

    /// The default of `--rco-steps`.
    public static final int DEFAULT_STEPS = 200;

    private static final double BETA1 = 0.9, BETA2 = 0.999, EPSILON = 1e-8;
    private static final double TAU_START = 1.0, TAU_END = 0.01;
    private static final double RETRACTION_RANGE = 50, RETRACTION_TOLERANCE = 1e-8;
    private static final double INITIAL_SPREAD = 0.5;

    private Rco() {}

    /// What is minimized: a function of the soft assignment `p[object][option]` and its gradient.
    public interface Objective {
        double value(double[][] p);

        double[][] gradient(double[][] p);
    }

    /// `steps` of Adam with learning rate `lr`, `samples` Gumbel draws per step, and `restarts` runs from
    /// different starting logits.
    public record Settings(int steps, double lr, int samples, int restarts) {
        public Settings {
            if (steps < 1 || samples < 1 || restarts < 1 || !(lr > 0))
                throw new IllegalArgumentException("steps, samples and restarts must be positive, and lr too");
        }

        public static Settings defaults() {
            return new Settings(DEFAULT_STEPS, 0.1, 8, 4);
        }
    }

    /// The assignment of a run, the probabilities it ended on, the objective at the assignment, and the largest
    /// relative distance of the expected bytes from the manifold after any step.
    public record Result(int[] choice, double[][] p, double value, double maxManifoldError) {}

    /// Every restart of `settings`, best first.
    public static List<Result> solve(
            Objective objective, long[][] cost, long budget, long unit, Settings settings, long seed) {
        List<Result> results = new ArrayList<>();
        for (int restart = 0; restart < settings.restarts(); restart++)
            results.add(run(objective, cost, budget, unit, settings, seed, restart));
        results.sort(Comparator.comparingDouble(Result::value));
        return results;
    }

    /// One run, restart number `restart`. `unit` is the knapsack's step in bytes.
    ///
    /// @throws IllegalStateException when the budget is below the cheapest options
    public static Result run(
            Objective objective, long[][] cost, long budget, long unit, Settings settings, long seed, int restart) {
        int groups = cost.length;
        Random random = new Random(seed * 1_000_003L + restart);
        double[][] c = new double[groups][];
        double minimum = 0, maximum = 0, ranges = 0;
        double[] low = new double[groups];
        for (int i = 0; i < groups; i++) {
            c[i] = new double[cost[i].length];
            double lo = Double.MAX_VALUE, hi = 0;
            for (int k = 0; k < c[i].length; k++) {
                c[i][k] = cost[i][k];
                lo = Math.min(lo, c[i][k]);
                hi = Math.max(hi, c[i][k]);
            }
            low[i] = lo;
            minimum += lo;
            maximum += hi;
            ranges += hi - lo;
        }
        if (budget < minimum)
            throw new IllegalStateException("the budget of " + budget + " bytes is below the cheapest choices");
        // The manifold must lie strictly between the cheapest and the dearest expectation to be reachable.
        double margin = 1e-4 * (maximum - minimum);
        double target = Math.min(Math.max(budget, minimum + margin), maximum - margin);
        double meanRange = Math.max(ranges / groups, 1);
        double[][] direction = new double[groups][];
        for (int i = 0; i < groups; i++) {
            direction[i] = new double[c[i].length];
            for (int k = 0; k < c[i].length; k++) direction[i][k] = (c[i][k] - low[i]) / meanRange;
        }

        double[][] alpha = new double[groups][];
        for (int i = 0; i < groups; i++) {
            alpha[i] = new double[c[i].length];
            for (int k = 0; k < alpha[i].length; k++) alpha[i][k] = INITIAL_SPREAD * random.nextGaussian();
        }
        retract(alpha, c, direction, target);
        double[][] m = new double[groups][], v = new double[groups][];
        for (int i = 0; i < groups; i++) {
            m[i] = new double[c[i].length];
            v[i] = new double[c[i].length];
        }
        double worst = manifoldError(alpha, c, target);
        double bestValue = Double.POSITIVE_INFINITY;
        int[] bestChoice = null;

        double[][] gradient = new double[groups][];
        for (int i = 0; i < groups; i++) gradient[i] = new double[c[i].length];
        double[][] score = new double[groups][];
        for (int i = 0; i < groups; i++) score[i] = new double[c[i].length];
        double[][] hard = new double[groups][];
        for (int i = 0; i < groups; i++) hard[i] = new double[c[i].length];

        for (int step = 1; step <= settings.steps(); step++) {
            double tau = settings.steps() == 1
                    ? TAU_START
                    : TAU_START * Math.pow(TAU_END / TAU_START, (step - 1.0) / (settings.steps() - 1));
            for (double[] row : gradient) Arrays.fill(row, 0);
            for (int sample = 0; sample < settings.samples(); sample++) {
                for (int i = 0; i < groups; i++)
                    for (int k = 0; k < c[i].length; k++)
                        score[i][k] = (alpha[i][k] - Math.log(-Math.log(random.nextDouble() + 1e-300) + 1e-300)) / tau;
                int[] choice = Knapsack.solve(score, cost, budget, unit);
                for (int i = 0; i < groups; i++) {
                    Arrays.fill(hard[i], 0);
                    hard[i][choice[i]] = 1;
                }
                double sampled = objective.value(hard);
                if (sampled < bestValue) {
                    bestValue = sampled;
                    bestChoice = choice;
                }
                double[][] toward = objective.gradient(hard);
                for (int i = 0; i < groups; i++) {
                    double[] y = softmax(score[i]);
                    double dot = 0;
                    for (int k = 0; k < y.length; k++) dot += y[k] * toward[i][k];
                    for (int k = 0; k < y.length; k++)
                        gradient[i][k] += y[k] * (toward[i][k] - dot) / tau / settings.samples();
                }
            }
            double[][] normal = normal(alpha, c);
            project(gradient, normal);
            double correction1 = 1 - Math.pow(BETA1, step), correction2 = 1 - Math.pow(BETA2, step);
            double[][] move = new double[groups][];
            for (int i = 0; i < groups; i++) {
                move[i] = new double[c[i].length];
                for (int k = 0; k < c[i].length; k++) {
                    m[i][k] = BETA1 * m[i][k] + (1 - BETA1) * gradient[i][k];
                    v[i][k] = BETA2 * v[i][k] + (1 - BETA2) * gradient[i][k] * gradient[i][k];
                    move[i][k] = m[i][k] / correction1 / (Math.sqrt(v[i][k] / correction2) + EPSILON);
                }
            }
            project(move, normal);
            for (int i = 0; i < groups; i++)
                for (int k = 0; k < c[i].length; k++) alpha[i][k] -= settings.lr() * move[i][k];
            retract(alpha, c, direction, target);
            project(m, normal(alpha, c));
            worst = Math.max(worst, manifoldError(alpha, c, target));
        }

        double[][] p = new double[groups][];
        double[][] logP = new double[groups][];
        for (int i = 0; i < groups; i++) {
            p[i] = softmax(alpha[i]);
            logP[i] = new double[p[i].length];
            for (int k = 0; k < p[i].length; k++) logP[i][k] = Math.log(Math.max(p[i][k], 1e-300));
        }
        int[] choice = Knapsack.solve(logP, cost, budget, unit);
        double[][] last = new double[groups][];
        for (int i = 0; i < groups; i++) {
            last[i] = new double[c[i].length];
            last[i][choice[i]] = 1;
        }
        double value = objective.value(last);
        // The best assignment any sample produced is kept as well: the run has evaluated the objective at it.
        if (bestValue < value) return new Result(bestChoice, p, bestValue, worst);
        return new Result(choice, p, value, worst);
    }

    private static double[] softmax(double[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (double x : logits) max = Math.max(max, x);
        double[] p = new double[logits.length];
        double sum = 0;
        for (int k = 0; k < p.length; k++) sum += p[k] = Math.exp(logits[k] - max);
        for (int k = 0; k < p.length; k++) p[k] /= sum;
        return p;
    }

    private static double expectedCost(double[][] alpha, double[][] c) {
        double total = 0;
        for (int i = 0; i < alpha.length; i++) {
            double[] p = softmax(alpha[i]);
            for (int k = 0; k < p.length; k++) total += p[k] * c[i][k];
        }
        return total;
    }

    private static double manifoldError(double[][] alpha, double[][] c, double target) {
        return Math.abs(expectedCost(alpha, c) - target) / target;
    }

    /// The normal of the manifold, `p (c - E_p[c])`.
    private static double[][] normal(double[][] alpha, double[][] c) {
        double[][] normal = new double[alpha.length][];
        for (int i = 0; i < alpha.length; i++) {
            double[] p = softmax(alpha[i]);
            double mean = 0;
            for (int k = 0; k < p.length; k++) mean += p[k] * c[i][k];
            normal[i] = new double[p.length];
            for (int k = 0; k < p.length; k++) normal[i][k] = p[k] * (c[i][k] - mean);
        }
        return normal;
    }

    /// Removes from `vector` its component along `normal`.
    private static void project(double[][] vector, double[][] normal) {
        double dot = 0, squares = 0;
        for (int i = 0; i < vector.length; i++)
            for (int k = 0; k < vector[i].length; k++) {
                dot += vector[i][k] * normal[i][k];
                squares += normal[i][k] * normal[i][k];
            }
        if (squares < 1e-300) return;
        double along = dot / squares;
        for (int i = 0; i < vector.length; i++)
            for (int k = 0; k < vector[i].length; k++) vector[i][k] -= along * normal[i][k];
    }

    /// Adds `t * direction` to `alpha` with `t` in `[-50, 50]` such that the expected bytes equal `target`.
    private static void retract(double[][] alpha, double[][] c, double[][] direction, double target) {
        double low = -RETRACTION_RANGE, high = RETRACTION_RANGE;
        double[][] trial = new double[alpha.length][];
        for (int i = 0; i < alpha.length; i++) trial[i] = new double[alpha[i].length];
        while (high - low > RETRACTION_TOLERANCE * 1e-4) {
            double t = 0.5 * (low + high);
            for (int i = 0; i < alpha.length; i++)
                for (int k = 0; k < alpha[i].length; k++) trial[i][k] = alpha[i][k] + t * direction[i][k];
            if (expectedCost(trial, c) < target) low = t;
            else high = t;
        }
        double t = 0.5 * (low + high);
        for (int i = 0; i < alpha.length; i++)
            for (int k = 0; k < alpha[i].length; k++) alpha[i][k] += t * direction[i][k];
    }
}
