package io.euhedral_execution.inference.quantization.mixed;

import io.euhedral_execution.inference.quantization.mixed.SearchState.Measured;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/// Predicts the KL of an allocation from the per-object priors and the candidates measured so far.
///
/// Each object `i` that takes option `o` contributes its prior error `e[i][o]` (divided by the mean prior
/// error, so that features are of order one). With `E_i` that contribution, or its expectation under
/// probabilities `p[i]` of a soft assignment, the model is
///
/// ```
/// x  = sum_{f,b} w[f,b] * phi[f,b]  +  sum_{f<g} v[f,g] * psi[f,g]  +  sum_f u[f] * chi[f]
/// kl = s * x^gamma
/// ```
///
/// with
/// - `phi[f,b]`, the sum of `E_i` over the objects of family `f` in band `b` (a family is the name after the
/// layer   index, `mlp/down`; the band of layer `l` of `L` is `l * bands / L`);
/// - `psi[f,g]`, the sum over layers of (the layer's `E` summed over family `f`) times (the same over family
/// `g`):   the pairs of families within a layer;
/// - `chi[f]`, the sum over bands `b` of `phi[f,b] * phi[f,b+1]`: adjacent bands of one family.
///
/// `gamma` is the monotone bend, `s` its scale. The weights are fitted by ridge regression of `kl^(1/gamma)`
/// on the features, each column scaled to unit RMS, every weight non-negative (an error anywhere can only add KL),
/// shrinking the linear weights towards one common weight and the pair weights towards zero; `gamma` is the value on a
/// grid whose fit has the least RMS error of `log
/// kl`, and `s` the geometric-mean correction of that fit. The prediction is differentiable in `p`:
/// [#gradient] is exact.
public final class FidelityModel {

    /// The default of `--bands`.
    public static final int DEFAULT_BANDS = 8;

    private static final double FLOOR = 1e-12;
    private static final double MINIMUM_KL = 1e-9;
    /// The coordinate-descent sweeps of [#ridgeSolve] at most, and the relative change that ends it sooner.
    private static final int SOLVE_SWEEPS = 2000;
    private static final double SOLVE_TOLERANCE = 1e-10;
    private static final double[] GAMMAS = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0};

    private final Layout layout;
    private final List<Option> options;
    /// The normalized prior errors, `error[object][option]`.
    private final double[][] error;
    private final double meanError;
    private final double[] weights;
    private final double gamma;
    private final double scale;

    private FidelityModel(
            Layout layout,
            List<Option> options,
            double[][] error,
            double meanError,
            double[] weights,
            double gamma,
            double scale) {
        this.layout = layout;
        this.options = options;
        this.error = error;
        this.meanError = meanError;
        this.weights = weights;
        this.gamma = gamma;
        this.scale = scale;
    }

    /// The objects of the model and how they group.
    private static final class Layout {
        final List<String> objects;
        final int[] family, band, layer;
        final int families, bands, layers;
        final List<String> familyNames;
        /// The index of the first pair weight and of the first adjacency weight.
        final int pairBase, adjacentBase, columns;

        Layout(List<String> objects, int bands) {
            this.objects = objects;
            this.bands = bands;
            int n = objects.size();
            family = new int[n];
            band = new int[n];
            layer = new int[n];
            Map<String, Integer> familyIndex = new LinkedHashMap<>();
            int maxLayer = -1;
            for (int i = 0; i < n; i++) {
                String[] parts = objects.get(i).split("/", 4);
                if (parts.length < 4 || !parts[0].equals("text") || !parts[1].equals("layers"))
                    throw new IllegalArgumentException(objects.get(i) + " is not an object of a decoder layer");
                layer[i] = Integer.parseInt(parts[2]);
                maxLayer = Math.max(maxLayer, layer[i]);
                family[i] = familyIndex.computeIfAbsent(parts[3], key -> familyIndex.size());
            }
            layers = maxLayer + 1;
            families = familyIndex.size();
            familyNames = List.copyOf(familyIndex.keySet());
            for (int i = 0; i < n; i++) band[i] = (int) ((long) layer[i] * bands / layers);
            pairBase = families * bands;
            adjacentBase = pairBase + families * families;
            columns = adjacentBase + families;
        }

        int pair(int f, int g) {
            return pairBase + Math.min(f, g) * families + Math.max(f, g);
        }
    }

    /// The families, bands and weights needed to evaluate features of expectations `expected` (per object).
    private static final class Features {
        final double[] phi; // [family * bands + band]
        final double[][] layerFamily; // [layer][family]

        Features(Layout layout, double[] expected) {
            phi = new double[layout.families * layout.bands];
            layerFamily = new double[layout.layers][layout.families];
            for (int i = 0; i < expected.length; i++) {
                phi[layout.family[i] * layout.bands + layout.band[i]] += expected[i];
                layerFamily[layout.layer[i]][layout.family[i]] += expected[i];
            }
        }
    }

    /// The feature vector, columns as the layout numbers them.
    private static double[] features(Layout layout, double[] expected) {
        Features parts = new Features(layout, expected);
        double[] x = new double[layout.columns];
        System.arraycopy(parts.phi, 0, x, 0, parts.phi.length);
        for (double[] row : parts.layerFamily)
            for (int f = 0; f < layout.families; f++)
                for (int g = f + 1; g < layout.families; g++) x[layout.pair(f, g)] += row[f] * row[g];
        for (int f = 0; f < layout.families; f++)
            for (int b = 0; b + 1 < layout.bands; b++)
                x[layout.adjacentBase + f] += parts.phi[f * layout.bands + b] * parts.phi[f * layout.bands + b + 1];
        return x;
    }

    /// As [#fit(List, double, Priors, int, long, double)] with the default bands and no perturbation.
    public static FidelityModel fit(List<Measured> data, double ridge, Priors priors) {
        return fit(data, ridge, priors, DEFAULT_BANDS);
    }

    public static FidelityModel fit(List<Measured> data, double ridge, Priors priors, int bands) {
        return fit(data, ridge, priors, bands, 0, 0);
    }

    /// The model fitted to `data` with ridge `ridge`, over `bands` bands per family. A `perturb` above zero
    /// draws, from `seed`, the point the ridge shrinks towards: each linear weight is scaled by `exp(perturb *
    /// N(0,1))` and each pair weight is given a random contribution of order `perturb`, so that models fitted
    /// from different seeds differ where the data does not decide.
    public static FidelityModel fit(
            List<Measured> data, double ridge, Priors priors, int bands, long seed, double perturb) {
        List<String> names = new ArrayList<>(priors.objects().keySet());
        names.sort(null);
        Layout layout = new Layout(sortedByLayer(names), bands);
        List<Option> options = priors.options();
        double[][] error = new double[names.size()][options.size()];
        double sum = 0;
        int count = 0;
        for (int i = 0; i < names.size(); i++)
            for (int k = 0; k < options.size(); k++) {
                error[i][k] = priors.error(layout.objects.get(i), options.get(k));
                sum += error[i][k];
                count++;
            }
        double mean = sum / count;
        if (!(mean > 0)) throw new IllegalArgumentException("the priors' mean error is " + mean + ", not positive");
        for (double[] row : error) for (int k = 0; k < row.length; k++) row[k] /= mean;
        FidelityModel shell = new FidelityModel(layout, options, error, mean, new double[layout.columns], 1, 1);
        if (data.isEmpty()) {
            double[] weights = new double[layout.columns];
            for (int j = 0; j < layout.pairBase; j++) weights[j] = 1;
            return new FidelityModel(layout, options, error, mean, weights, 1, 1);
        }

        int rows = data.size();
        double[][] x = new double[rows][];
        double[] kl = new double[rows];
        for (int r = 0; r < rows; r++) {
            x[r] = features(layout, shell.expectedOf(data.get(r).a()));
            kl[r] = Math.max(data.get(r).kl(), MINIMUM_KL);
        }
        double[] rms = new double[layout.columns];
        for (double[] row : x) for (int j = 0; j < rms.length; j++) rms[j] += row[j] * row[j];
        for (int j = 0; j < rms.length; j++) {
            rms[j] = Math.sqrt(rms[j] / rows);
            if (!(rms[j] > 1e-12)) rms[j] = 1;
        }
        double[][] scaled = new double[rows][layout.columns];
        for (int r = 0; r < rows; r++) for (int j = 0; j < rms.length; j++) scaled[r][j] = x[r][j] / rms[j];

        Random random = new Random(seed);
        double[] noise = new double[layout.columns];
        for (int j = 0; j < noise.length; j++) noise[j] = perturb > 0 ? random.nextGaussian() : 0;

        FidelityModel best = null;
        double bestScore = Double.POSITIVE_INFINITY;
        for (double gamma : GAMMAS) {
            double[] target = new double[rows];
            double targetSquares = 0;
            for (int r = 0; r < rows; r++) {
                target[r] = Math.pow(kl[r], 1 / gamma);
                targetSquares += target[r] * target[r];
            }
            double targetRms = Math.sqrt(targetSquares / rows);
            // The common linear weight: least squares of the target on the sum of the linear columns.
            double dot = 0, squares = 0;
            for (int r = 0; r < rows; r++) {
                double total = 0;
                for (int j = 0; j < layout.pairBase; j++) total += scaled[r][j];
                dot += total * target[r];
                squares += total * total;
            }
            double common = squares > 0 ? dot / squares : 0;
            double[] prior = new double[layout.columns];
            for (int j = 0; j < layout.columns; j++)
                prior[j] = j < layout.pairBase
                        ? common * Math.exp(perturb * noise[j])
                        : perturb * targetRms / Math.sqrt(layout.columns) * Math.abs(noise[j]);
            double[] theta = ridgeSolve(scaled, target, ridge, prior);
            double[] weights = new double[layout.columns];
            for (int j = 0; j < weights.length; j++) weights[j] = theta[j] / rms[j];
            double logs = 0;
            double[] residual = new double[rows];
            for (int r = 0; r < rows; r++) {
                double value = 0;
                for (int j = 0; j < weights.length; j++) value += weights[j] * x[r][j];
                residual[r] = Math.log(kl[r]) - gamma * Math.log(Math.max(value, FLOOR));
                logs += residual[r];
            }
            double logScale = logs / rows;
            double squaresOfError = 0;
            for (double r : residual) squaresOfError += (r - logScale) * (r - logScale);
            double score = Math.sqrt(squaresOfError / rows);
            if (score < bestScore) {
                bestScore = score;
                best = new FidelityModel(layout, options, error, mean, weights, gamma, Math.exp(logScale));
            }
        }
        return best;
    }

    static List<String> sortedByLayer(List<String> names) {
        List<String> sorted = new ArrayList<>(names);
        sorted.sort((a, b) -> {
            int layerA = Integer.parseInt(a.split("/", 4)[2]), layerB = Integer.parseInt(b.split("/", 4)[2]);
            return layerA != layerB ? Integer.compare(layerA, layerB) : a.compareTo(b);
        });
        return sorted;
    }

    /// Minimizes `|X theta - t|^2 + ridge' |theta - prior|^2` over `theta >= 0` (`ridge'` is `ridge` times the
    /// rows), by coordinate descent on the normal equations, each coordinate clipped at zero. The features are sums
    /// of non-negative prior errors, so non-negative weights keep every prediction's linear part non-negative: an
    /// unconstrained fit could give an allocation a part at or below zero, a predicted KL of nothing with no
    /// gradient, which RCO then seeks out.
    private static double[] ridgeSolve(double[][] x, double[] t, double ridge, double[] prior) {
        int n = prior.length, rows = x.length;
        double[][] a = new double[n][n];
        double[] b = new double[n];
        for (int r = 0; r < rows; r++) {
            double[] row = x[r];
            for (int i = 0; i < n; i++) {
                if (row[i] == 0) continue;
                b[i] += row[i] * t[r];
                for (int j = 0; j < n; j++) a[i][j] += row[i] * row[j];
            }
        }
        double lambda = Math.max(ridge, 1e-12) * Math.max(1, rows);
        for (int i = 0; i < n; i++) {
            a[i][i] += lambda;
            b[i] += lambda * prior[i];
        }
        double[] theta = new double[n];
        for (int i = 0; i < n; i++) theta[i] = Math.max(0, prior[i]);
        for (int sweep = 0; sweep < SOLVE_SWEEPS; sweep++) {
            double change = 0, size = 0;
            for (int i = 0; i < n; i++) {
                double rest = b[i];
                for (int j = 0; j < n; j++) if (j != i) rest -= a[i][j] * theta[j];
                double next = Math.max(0, rest / a[i][i]);
                change = Math.max(change, Math.abs(next - theta[i]));
                size = Math.max(size, Math.abs(next));
                theta[i] = next;
            }
            if (change <= SOLVE_TOLERANCE * Math.max(size, 1e-300)) break;
        }
        return theta;
    }

    /// The objects the model predicts over, in the order of the rows of `p` in [#predictSoft].
    public List<String> objects() {
        return layout.objects;
    }

    /// The families (the names after the layer index), in the order of the weights' first index.
    public List<String> families() {
        return layout.familyNames;
    }

    /// The options, in the order of the columns of `p`.
    public List<Option> options() {
        return options;
    }

    public double gamma() {
        return gamma;
    }

    public double scale() {
        return scale;
    }

    /// The weight of the family (by the order its first object appears) in the band, in the model's own units.
    public double weight(int family, int band) {
        return weights[family * layout.bands + band];
    }

    /// What the family-band weight multiplies a prior error by in the linear law `kl = c * (sum w * e)^gamma`
    /// (the law the model reduces to without pair terms), with `c` taken as one.
    public double effectiveWeight(int family, int band) {
        return Math.pow(scale, 1 / gamma) * weight(family, band) / meanError;
    }

    private double[] expectedOf(Allocation allocation) {
        double[] expected = new double[layout.objects.size()];
        for (int i = 0; i < expected.length; i++) {
            Option option = allocation.choices().get(layout.objects.get(i));
            if (option == null)
                throw new IllegalArgumentException("the allocation has no option for " + layout.objects.get(i));
            int k = options.indexOf(option);
            if (k < 0) throw new IllegalArgumentException(option.key() + " is not an option of the model");
            expected[i] = error[i][k];
        }
        return expected;
    }

    private double[] expectedOf(double[][] p) {
        double[] expected = new double[layout.objects.size()];
        for (int i = 0; i < expected.length; i++)
            for (int k = 0; k < options.size(); k++) expected[i] += p[i][k] * error[i][k];
        return expected;
    }

    private double linear(double[] expected) {
        double[] x = features(layout, expected);
        double value = 0;
        for (int j = 0; j < x.length; j++) value += weights[j] * x[j];
        return value;
    }

    /// The predicted KL of `allocation`, which names an option for every object of the model.
    public double predict(Allocation allocation) {
        return curve(linear(expectedOf(allocation)));
    }

    /// The predicted KL of the soft assignment `p[object][option]`, each row a probability vector.
    public double predictSoft(double[][] p) {
        return curve(linear(expectedOf(p)));
    }

    private double curve(double x) {
        return scale * Math.pow(Math.max(x, FLOOR), gamma);
    }

    /// The derivative of [#predictSoft] with respect to every `p[object][option]`.
    public double[][] gradient(double[][] p) {
        double[] expected = expectedOf(p);
        Features parts = new Features(layout, expected);
        double x = linear(expected);
        double outer = x > FLOOR ? scale * gamma * Math.pow(x, gamma - 1) : 0;
        double[][] gradient = new double[expected.length][options.size()];
        for (int i = 0; i < expected.length; i++) {
            int f = layout.family[i], b = layout.band[i];
            double slope = weights[f * layout.bands + b];
            for (int g = 0; g < layout.families; g++)
                if (g != f) slope += weights[layout.pair(f, g)] * parts.layerFamily[layout.layer[i]][g];
            double adjacent = 0;
            if (b > 0) adjacent += parts.phi[f * layout.bands + b - 1];
            if (b + 1 < layout.bands) adjacent += parts.phi[f * layout.bands + b + 1];
            slope += weights[layout.adjacentBase + f] * adjacent;
            for (int k = 0; k < options.size(); k++) gradient[i][k] = outer * slope * error[i][k];
        }
        return gradient;
    }
}
