package io.euhedral_execution.inference.quantization.mixed;

import io.euhedral_execution.inference.quantization.mixed.SearchState.Measured;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

/// Synthetic objects, priors and a known fidelity law for the search tests.
final class Synthetic {

    static final List<String> FAMILIES = List.of("gdn/query_key", "attention/output", "mlp/gate_up", "mlp/down");
    static final List<Option> OPTIONS =
            List.of(Option.NVFP4, Option.TRELLIS_2_5, Option.TRELLIS_3_0, Option.TRELLIS_4_0);

    final int layers;
    final int bands;
    final List<String> objects = new ArrayList<>();
    final Priors priors;
    /// The true weight of each (family, band), and the bend the true law applies.
    final double[][] weight;
    final double gamma;
    final double scale;

    Synthetic(int layers, int bands, double gamma, double scale, long seed) {
        this.layers = layers;
        this.bands = bands;
        this.gamma = gamma;
        this.scale = scale;
        Random random = new Random(seed);
        Map<String, Map<Option, Double>> errors = new TreeMap<>();
        for (int layer = 0; layer < layers; layer++)
            for (String family : FAMILIES) {
                String name = "text/layers/" + layer + "/" + family;
                objects.add(name);
                Map<Option, Double> byOption = new TreeMap<>();
                double base = 0.001 * (1 + random.nextDouble());
                byOption.put(Option.NVFP4, base * 3.0);
                byOption.put(Option.TRELLIS_2_5, base * 4.0);
                byOption.put(Option.TRELLIS_3_0, base * 2.0);
                byOption.put(Option.TRELLIS_4_0, base * 0.6);
                errors.put(name, byOption);
            }
        priors = new Priors(errors);
        weight = new double[FAMILIES.size()][bands];
        for (double[] row : weight) for (int b = 0; b < bands; b++) row[b] = 0.5 + 1.5 * random.nextDouble();
    }

    int family(String object) {
        return FAMILIES.indexOf(object.substring(object.indexOf('/', "text/layers/".length()) + 1));
    }

    int band(String object) {
        String[] parts = object.split("/");
        return Integer.parseInt(parts[2]) * bands / layers;
    }

    /// The true KL of `allocation`: scale * (sum of weight * prior error)^gamma.
    double truth(Allocation allocation) {
        double x = 0;
        for (String object : objects)
            x += weight[family(object)][band(object)]
                    * priors.error(object, allocation.choices().get(object));
        return scale * Math.pow(x, gamma);
    }

    Allocation random(Random random) {
        return random(random, OPTIONS);
    }

    Allocation random(Random random, List<Option> options) {
        Map<String, Option> choices = new TreeMap<>();
        for (String object : objects) choices.put(object, options.get(random.nextInt(options.size())));
        return new Allocation(choices);
    }

    Measured measured(Allocation allocation, int round) {
        return new Measured(allocation, truth(allocation), 0.0, 0, 0, 0, true, round, "synthetic");
    }
}
