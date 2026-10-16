package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.quantization.mixed.SearchState.Measured;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;
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

    @Test
    void fittingByMembersGivesTheSequentialEnsemble() throws Exception {
        Synthetic world = new Synthetic(8, 4, 1.2, 3.0, 3);
        Random random = new Random(3);
        List<Measured> data = new ArrayList<>();
        for (int i = 0; i < 60; i++) data.add(world.measured(world.random(random), 1));
        Ensemble sequential = Ensemble.fit(data, 5, 11, 1e-3, world.priors, 4);
        int[] frames = {0};
        // Each member is its own task, and they finish in the reverse of the order they were submitted.
        List<Runnable> pending = new ArrayList<>();
        List<CompletableFuture<?>> futures = new ArrayList<>();
        Ensemble.Frames deferred = new Ensemble.Frames() {
            @Override
            public <T> CompletableFuture<T> submit(Supplier<T> task) {
                frames[0]++;
                CompletableFuture<T> future = new CompletableFuture<>();
                futures.add(future);
                pending.add(() -> future.complete(task.get()));
                return future;
            }
        };
        CompletableFuture<Ensemble> split = Ensemble.fit(data, 5, 11, 1e-3, world.priors, 4, deferred);
        while (!pending.isEmpty()) pending.remove(pending.size() - 1).run();
        Ensemble byMembers = split.get();
        // One task per member, then the join.
        assertEquals(5 + 1, frames[0]);
        assertEquals(sequential.members().size(), byMembers.members().size());
        for (int m = 0; m < sequential.members().size(); m++) {
            FidelityModel a = sequential.members().get(m),
                    b = byMembers.members().get(m);
            assertEquals(a.gamma(), b.gamma(), 0.0);
            assertEquals(a.scale(), b.scale(), 0.0);
            for (int family = 0; family < a.families().size(); family++)
                for (int band = 0; band < 4; band++) assertEquals(a.weight(family, band), b.weight(family, band), 0.0);
        }
        for (int i = 0; i < 10; i++) {
            Allocation probe = world.random(random);
            assertEquals(sequential.mean(probe), byMembers.mean(probe), 0.0);
            assertEquals(sequential.spread(probe), byMembers.spread(probe), 0.0);
        }
    }
}
