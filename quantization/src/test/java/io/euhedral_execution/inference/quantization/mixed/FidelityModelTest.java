package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.quantization.mixed.SearchState.Measured;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class FidelityModelTest {

    private static List<Measured> data(Synthetic world, int count, long seed) {
        Random random = new Random(seed);
        List<Measured> data = new ArrayList<>();
        for (int i = 0; i < count; i++) data.add(world.measured(world.random(random), 1));
        return data;
    }

    @Test
    void recoversTheLawOfSyntheticData() {
        Synthetic world = new Synthetic(16, 4, 1.2, 3.0, 7);
        FidelityModel model = FidelityModel.fit(data(world, 400, 1), 1e-9, world.priors, 4);
        assertEquals(1.2, model.gamma(), 0.11);
        Random random = new Random(99);
        for (int i = 0; i < 40; i++) {
            Allocation allocation = world.random(random);
            double truth = world.truth(allocation);
            assertEquals(truth, model.predict(allocation), 0.03 * truth, "prediction " + i);
        }
    }

    @Test
    void recoversTheWeightsOfAPurelyLinearLaw() {
        Synthetic world = new Synthetic(16, 4, 1.0, 1.0, 11);
        FidelityModel model = FidelityModel.fit(data(world, 400, 2), 1e-9, world.priors, 4);
        assertEquals(1.0, model.gamma(), 0.11);
        for (int family = 0; family < Synthetic.FAMILIES.size(); family++)
            for (int band = 0; band < 4; band++) {
                double estimated =
                        model.effectiveWeight(model.families().indexOf(Synthetic.FAMILIES.get(family)), band);
                assertEquals(
                        world.weight[family][band], estimated, 0.1 * world.weight[family][band], family + "/" + band);
            }
    }

    @Test
    void theSoftGradientMatchesFiniteDifferences() {
        Synthetic world = new Synthetic(8, 4, 1.3, 2.0, 5);
        // A perturbed prior gives the pair and adjacent-band weights values, so every term is differentiated.
        List<Measured> data = new ArrayList<>(data(world, 120, 3));
        FidelityModel model = FidelityModel.fit(data, 1e-3, world.priors, 4, 11, 0.5);
        Random random = new Random(4);
        int objects = world.objects.size(), options = model.options().size();
        double[][] p = new double[objects][options];
        for (double[] row : p) {
            double sum = 0;
            for (int k = 0; k < options; k++) sum += row[k] = 0.1 + random.nextDouble();
            for (int k = 0; k < options; k++) row[k] /= sum;
        }
        double[][] gradient = model.gradient(p);
        double step = 1e-6;
        for (int i = 0; i < objects; i += 3)
            for (int k = 0; k < options; k++) {
                double kept = p[i][k];
                p[i][k] = kept + step;
                double up = model.predictSoft(p);
                p[i][k] = kept - step;
                double down = model.predictSoft(p);
                p[i][k] = kept;
                double numeric = (up - down) / (2 * step);
                assertEquals(numeric, gradient[i][k], 1e-6 * Math.max(1, Math.abs(numeric)) + 1e-9, i + "/" + k);
            }
    }

    @Test
    void aHardAllocationIsTheSoftOneOfItsIndicators() {
        Synthetic world = new Synthetic(8, 4, 1.1, 2.0, 5);
        FidelityModel model = FidelityModel.fit(data(world, 100, 3), 1e-3, world.priors, 4);
        Allocation allocation = world.random(new Random(8));
        double[][] p = new double[world.objects.size()][model.options().size()];
        for (int i = 0; i < p.length; i++)
            p[i][
                    model.options()
                            .indexOf(allocation.choices().get(model.objects().get(i)))] = 1;
        assertEquals(model.predict(allocation), model.predictSoft(p), 1e-12);
    }

    @Test
    void predictionsAreMonotoneInThePriorErrors() {
        Synthetic world = new Synthetic(8, 4, 1.1, 2.0, 5);
        FidelityModel model = FidelityModel.fit(data(world, 150, 3), 1e-3, world.priors, 4);
        Allocation best = new Allocation(
                Map.copyOf(world.objects.stream().collect(Collectors.toMap(o -> o, o -> Option.TRELLIS_4_0))));
        Allocation worst = new Allocation(
                Map.copyOf(world.objects.stream().collect(Collectors.toMap(o -> o, o -> Option.TRELLIS_2_5))));
        assertTrue(model.predict(best) < model.predict(worst));
    }

    @Test
    void fewNoisyMeasurementsNeverGiveAnAllocationANearZeroPrediction() {
        // Few measurements with noise: an unconstrained fit gives some allocations a non-positive linear part (a
        // predicted KL of nothing, with no gradient), which RCO would seek out.
        Synthetic world = new Synthetic(16, 4, 1.0, 1.0, 13);
        Random random = new Random(5);
        List<Measured> data = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            Allocation allocation = world.random(random);
            double noisy = world.truth(allocation) * Math.exp(0.5 * random.nextGaussian());
            data.add(new Measured(allocation, noisy, 0, 0, 0, 0, true, 1, "noisy"));
        }
        FidelityModel model = FidelityModel.fit(data, 1e-6, world.priors, 4);
        for (int family = 0; family < Synthetic.FAMILIES.size(); family++)
            for (int band = 0; band < 4; band++)
                assertTrue(model.weight(family, band) >= 0, "weight " + family + "/" + band);
        double least = Double.POSITIVE_INFINITY;
        for (Measured m : data) least = Math.min(least, m.kl());
        Random other = new Random(17);
        for (int i = 0; i < 200; i++) {
            double predicted = model.predict(world.random(other));
            assertTrue(predicted > 1e-3 * least, "prediction " + i + " = " + predicted);
        }
    }
}
