package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.quantization.mixed.SearchState.Measured;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;

class EnsembleTest {

    /// Allocations that vary only in the first half of the layers; the rest stay on NVFP4.
    private static Allocation front(Synthetic world, Random random, boolean varied) {
        Map<String, Option> choices = new TreeMap<>();
        for (String object : world.objects) {
            int layer = Integer.parseInt(object.split("/")[2]);
            boolean first = layer < world.layers / 2;
            choices.put(
                    object,
                    first == varied ? Synthetic.OPTIONS.get(random.nextInt(Synthetic.OPTIONS.size())) : Option.NVFP4);
        }
        return new Allocation(choices);
    }

    @Test
    void spreadIsLargerOffTheTrainingDistribution() {
        Synthetic world = new Synthetic(16, 4, 1.2, 3.0, 3);
        Random random = new Random(1);
        List<Measured> data = new ArrayList<>();
        for (int i = 0; i < 80; i++) data.add(world.measured(front(world, random, true), 1));
        Ensemble ensemble = Ensemble.fit(data, 8, 42, 1e-3, world.priors, 4);
        double inside = 0, outside = 0;
        for (int i = 0; i < 30; i++) {
            inside += ensemble.spread(front(world, random, true));
            outside += ensemble.spread(front(world, random, false));
        }
        assertTrue(outside > 1.5 * inside, "inside " + inside + " outside " + outside);
    }

    @Test
    void theMeanTracksTheTruthInDistributionAndHeldOutErrorIsSmall() {
        Synthetic world = new Synthetic(16, 4, 1.2, 3.0, 3);
        Random random = new Random(2);
        List<Measured> data = new ArrayList<>(), unseen = new ArrayList<>();
        for (int i = 0; i < 300; i++) data.add(world.measured(world.random(random), 1));
        for (int i = 0; i < 30; i++) unseen.add(world.measured(world.random(random), 2));
        Ensemble ensemble = Ensemble.fit(data, 6, 7, 1e-6, world.priors, 4);
        assertTrue(ensemble.heldOutError(unseen) < 0.05, "held-out " + ensemble.heldOutError(unseen));
        assertEquals(0.0, ensemble.heldOutError(List.of()), 0.0);
    }

    @Test
    void sameSeedSameEnsemble() {
        Synthetic world = new Synthetic(8, 4, 1.2, 3.0, 3);
        Random random = new Random(2);
        List<Measured> data = new ArrayList<>();
        for (int i = 0; i < 60; i++) data.add(world.measured(world.random(random), 1));
        Allocation probe = world.random(random);
        assertEquals(
                Ensemble.fit(data, 4, 5, 1e-3, world.priors, 4).mean(probe),
                Ensemble.fit(data, 4, 5, 1e-3, world.priors, 4).mean(probe),
                0.0);
    }
}
