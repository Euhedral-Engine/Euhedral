package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.quantization.CalibrationLattice;
import io.euhedral_execution.inference.quantization.Scorer;
import io.euhedral_execution.inference.quantization.mixed.SearchState.Measured;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The search end to end on tiny artifacts, with a scorer that reads the answer off the composite and host
/// work on a real lattice.
class SearchTest {

    private static final Cost.Shape SHAPE = new Cost.Shape(64, 64);
    private static final int VERIFY_ROWS = 4;
    private static CalibrationLattice lattice;

    @TempDir
    Path directory;

    @BeforeAll
    static void startLattice() {
        lattice = CalibrationLattice.start(3);
    }

    @AfterAll
    static void stopLattice() {
        lattice.close();
    }

    private Sources sources;
    private Priors priors;
    private CostTable costs;
    private final Map<String, Map<Option, Double>> truth = new TreeMap<>();

    private void setUp() throws Exception {
        sources = Sources.open(
                Map.of(
                        Option.TRELLIS_3_0, TinyArtifacts.write(directory.resolve("trellis.edrl"), 1, true),
                        Option.NVFP4, TinyArtifacts.write(directory.resolve("nvfp4.edrl"), 2, false)),
                Option.TRELLIS_3_0);
        Map<String, Map<Option, Double>> errors = new TreeMap<>();
        int n = 0;
        for (String name : sources.projections()) {
            n++;
            errors.put(name, Map.of(Option.NVFP4, 0.004 * n, Option.TRELLIS_3_0, 0.001 * n));
            // The scorer's own law: a sum with a pair term and a bend, which the model has to learn.
            truth.put(name, Map.of(Option.NVFP4, 0.004 * n + 0.001, Option.TRELLIS_3_0, 0.001 * n));
        }
        priors = new Priors(errors);
        Map<Cost.Key, Double> micros = new HashMap<>();
        micros.put(new Cost.Key(SHAPE, Option.NVFP4, 512), 51.2);
        micros.put(new Cost.Key(SHAPE, Option.TRELLIS_3_0, 512), 102.4);
        micros.put(new Cost.Key(SHAPE, Option.NVFP4, VERIFY_ROWS), 3.0);
        micros.put(new Cost.Key(SHAPE, Option.TRELLIS_3_0, VERIFY_ROWS), 5.0);
        costs = new CostTable(micros, 0, Map.of());
    }

    /// The KL the fake scorer reports: read off which source file each projection of the composite comes from.
    private double kl(Artifact composite) {
        double sum = 0;
        for (String name : sources.projections()) {
            TensorDescriptor tensor = null;
            for (TensorDescriptor candidate : composite.tensors())
                if (candidate.name().equals(name)) tensor = candidate;
            Path file = composite.pathOf(sources.path(sources.base()), tensor);
            Option option = file.equals(sources.path(Option.NVFP4)) ? Option.NVFP4 : Option.TRELLIS_3_0;
            sum += truth.get(name).get(option);
        }
        return Math.pow(sum, 1.1);
    }

    /// Fits unless every projection is on the trellis source (the larger one), so the budget is below the
    /// maximum.
    private static Feasibility notAllTrellis(Sources sources) {
        return new Feasibility() {
            @Override
            public long freeBytes() {
                return 1L << 30;
            }

            @Override
            public Check check(Allocation allocation) {
                long trellis = sources.projections().stream()
                        .filter(name -> allocation.choices().getOrDefault(name, sources.base()) == Option.TRELLIS_3_0)
                        .count();
                boolean fits = trellis < sources.projections().size();
                return new Check(fits, fits ? 1L << 20 : (1L << 30) + (1L << 20));
            }
        };
    }

    private Search.Settings settings(Path state) {
        return new Search.Settings(
                sources,
                priors,
                costs,
                4096,
                state,
                directory.resolve("out"),
                List.of(),
                3,
                0.3,
                2,
                3,
                2,
                VERIFY_ROWS,
                1L << 30,
                free -> notAllTrellis(sources),
                2,
                3,
                1e-3,
                List.of(0.0, 0.01),
                List.of(1.0, 0.5),
                new Rco.Settings(20, 0.1, 2, 1),
                64,
                4,
                11,
                "test");
    }

    @Test
    void measuresEachCandidateOnceAndWritesTheFront() throws Exception {
        setUp();
        Map<Allocation, Integer> scored = new HashMap<>();
        Search.Scoring scoring = (path, artifact) -> {
            Allocation allocation = allocationOf(artifact);
            scored.merge(allocation, 1, Integer::sum);
            return new Scorer.Result(kl(artifact), 0.001, 0.9, 0, 0, new double[0]);
        };
        Search.Outcome outcome = Search.run(settings(directory.resolve("search.resume")), scoring);
        assertTrue(scored.values().stream().allMatch(count -> count == 1), "each candidate scored once: " + scored);
        assertEquals(scored.size(), outcome.measured().size());
        assertTrue(
                outcome.measured().size() > 5,
                "the seeds and at least one round: " + outcome.measured().size());
        assertFalse(outcome.front().isEmpty());
        assertTrue(outcome.front().stream().allMatch(Measured::fits));
        assertTrue(Files.exists(directory.resolve("out/front.json")));
        assertTrue(Files.exists(directory.resolve("out/report.txt")));
        assertTrue(Files.exists(directory.resolve("out/allocations/front-00.json")));
        // The pure sources are measured first.
        Set<String> origins = new HashSet<>();
        outcome.measured().forEach(m -> origins.add(m.origin().split(":")[0]));
        assertTrue(origins.contains("seed"), origins.toString());
    }

    private Allocation allocationOf(Artifact composite) {
        Map<String, Option> choices = new TreeMap<>();
        for (String name : sources.projections()) {
            TensorDescriptor tensor = null;
            for (TensorDescriptor candidate : composite.tensors())
                if (candidate.name().equals(name)) tensor = candidate;
            Path file = composite.pathOf(sources.path(sources.base()), tensor);
            choices.put(name, file.equals(sources.path(Option.NVFP4)) ? Option.NVFP4 : Option.TRELLIS_3_0);
        }
        return new Allocation(choices);
    }

    @Test
    void aSearchStoppedMidBatchResumesWithoutScoringOrRecordingAnythingTwice() throws Exception {
        setUp();
        Path state = directory.resolve("search.resume");
        Map<Allocation, Integer> scored = new HashMap<>();
        int[] calls = {0};
        Search.Scoring failing = (path, artifact) -> {
            if (++calls[0] == 4) throw new IllegalStateException("stopped");
            scored.merge(allocationOf(artifact), 1, Integer::sum);
            return new Scorer.Result(kl(artifact), 0.001, 0.9, 0, 0, new double[0]);
        };
        IllegalStateException stop =
                assertThrows(IllegalStateException.class, () -> Search.run(settings(state), failing));
        assertEquals("stopped", stop.getMessage());
        // A restart with a scorer that counts: nothing scored before is scored again.
        Search.Scoring counting = (path, artifact) -> {
            scored.merge(allocationOf(artifact), 1, Integer::sum);
            return new Scorer.Result(kl(artifact), 0.001, 0.9, 0, 0, new double[0]);
        };
        Search.Outcome outcome = Search.run(settings(state), counting);
        assertTrue(scored.values().stream().allMatch(count -> count == 1), "each candidate scored once: " + scored);
        Set<Allocation> distinct = new HashSet<>();
        for (Measured m : outcome.measured()) assertTrue(distinct.add(m.a()), "no duplicate entry");
        assertEquals(scored.size(), outcome.measured().size());
        assertTrue(outcome.measured().size() > 3);
    }

    /// A scorer that scores several at once: it holds what is submitted and completes it, last submitted first, when
    /// the waiting thread services it; the `stopAt`th completion fails instead (never when 0).
    private final class OutOfOrder implements Search.Scoring {
        final Map<Allocation, Integer> scored = new HashMap<>();
        final List<Runnable> held = new ArrayList<>();
        int completions;
        final int stopAt;

        OutOfOrder(int stopAt) {
            this.stopAt = stopAt;
        }

        @Override
        public Scorer.Result score(Path path, Artifact artifact) {
            throw new AssertionError("scored one at a time");
        }

        @Override
        public boolean concurrent() {
            return true;
        }

        @Override
        public CompletableFuture<Scorer.Result> submit(Path path, Artifact artifact, Runnable started) {
            CompletableFuture<Scorer.Result> result = new CompletableFuture<>();
            started.run();
            held.add(() -> {
                if (++completions == stopAt) {
                    result.completeExceptionally(new IllegalStateException("stopped"));
                    return;
                }
                scored.merge(allocationOf(artifact), 1, Integer::sum);
                result.complete(new Scorer.Result(kl(artifact), 0.001, 0.9, 0, 0, new double[0]));
            });
            return result;
        }

        @Override
        public void service() {
            List<Runnable> now = new ArrayList<>(held);
            held.clear();
            for (int i = now.size() - 1; i >= 0; i--) now.get(i).run();
        }

        @Override
        public CompletableFuture<Void> attention() {
            return held.isEmpty() ? new CompletableFuture<>() : CompletableFuture.completedFuture(null);
        }
    }

    @Test
    void aBatchScoredAtOnceThatCompletesOutOfOrderRecordsEveryCandidateOnce() throws Exception {
        setUp();
        OutOfOrder scoring = new OutOfOrder(0);
        Search.Outcome outcome = Search.run(settings(directory.resolve("search.resume")), scoring);
        assertTrue(scoring.scored.values().stream().allMatch(count -> count == 1), scoring.scored.toString());
        assertEquals(scoring.scored.size(), outcome.measured().size());
        assertTrue(outcome.measured().size() > 5);
        Set<Allocation> distinct = new HashSet<>();
        for (Measured m : outcome.measured()) assertTrue(distinct.add(m.a()), "no duplicate entry");
        assertFalse(outcome.front().isEmpty());
        // The resume file holds exactly what the run measured.
        Search.Outcome resumed = Search.run(settings(directory.resolve("search.resume")), scoring);
        assertEquals(outcome.measured().size(), resumed.measured().size());
    }

    @Test
    void aSearchStoppedBetweenCompletionsResumesWithoutScoringOrRecordingAnythingTwice() throws Exception {
        setUp();
        Path state = directory.resolve("search.resume");
        OutOfOrder stopping = new OutOfOrder(4);
        IllegalStateException stop =
                assertThrows(IllegalStateException.class, () -> Search.run(settings(state), stopping));
        assertEquals("stopped", stop.getMessage());
        // Only completed scores were recorded: the ones that completed before the stop, each once.
        assertTrue(stopping.scored.values().stream().allMatch(count -> count == 1));
        OutOfOrder resuming = new OutOfOrder(0);
        Search.Outcome outcome = Search.run(settings(state), resuming);
        Map<Allocation, Integer> all = new HashMap<>(stopping.scored);
        resuming.scored.forEach((allocation, count) -> all.merge(allocation, count, Integer::sum));
        // A candidate that completed before the stop but was not recorded may be scored again; none is recorded twice.
        Set<Allocation> distinct = new HashSet<>();
        for (Measured m : outcome.measured()) assertTrue(distinct.add(m.a()), "no duplicate entry");
        assertTrue(outcome.measured().size() > 3);
        for (Measured m : outcome.measured()) assertTrue(all.containsKey(m.a()), "every record was scored");
    }

    @Test
    void aSearchOfAnotherConfigurationDoesNotUseTheFile() throws Exception {
        setUp();
        Path state = directory.resolve("search.resume");
        Search.Scoring scoring = (path, artifact) -> new Scorer.Result(kl(artifact), 0.001, 0.9, 0, 0, new double[0]);
        Search.run(settings(state), scoring);
        Search.Settings other = settings(state);
        other = new Search.Settings(
                other.sources(),
                other.priors(),
                other.costs(),
                8192,
                other.state(),
                other.out(),
                other.seeds(),
                other.batch(),
                other.explore(),
                other.patience(),
                other.random(),
                other.bands(),
                other.verifyRows(),
                other.freeBytes(),
                other.feasibility(),
                other.hostSinks(),
                other.members(),
                other.ridge(),
                other.lambdas(),
                other.budgetFractions(),
                other.rco(),
                other.knapsackSteps(),
                other.maxRounds(),
                other.seed(),
                other.identity());
        Search.Settings changed = other;
        IllegalStateException refused = assertThrows(IllegalStateException.class, () -> Search.run(changed, scoring));
        assertTrue(refused.getMessage().contains("different search"), refused.getMessage());
    }

    @Test
    void seedsNameTheirObjectAndUnknownOnesAreRefused() throws Exception {
        setUp();
        Search.Settings base = settings(directory.resolve("search.resume"));
        List<Allocation> seeds = new ArrayList<>(base.seeds());
        seeds.add(new Allocation(Map.of("text/layers/9/mlp/down", Option.NVFP4)));
        Search.Settings bad = new Search.Settings(
                base.sources(),
                base.priors(),
                base.costs(),
                base.context(),
                base.state(),
                base.out(),
                seeds,
                base.batch(),
                base.explore(),
                base.patience(),
                base.random(),
                base.bands(),
                base.verifyRows(),
                base.freeBytes(),
                base.feasibility(),
                base.hostSinks(),
                base.members(),
                base.ridge(),
                base.lambdas(),
                base.budgetFractions(),
                base.rco(),
                base.knapsackSteps(),
                base.maxRounds(),
                base.seed(),
                base.identity());
        Search.Scoring scoring = (path, artifact) -> new Scorer.Result(kl(artifact), 0.001, 0.9, 0, 0, new double[0]);
        Exception refused = assertThrows(Exception.class, () -> Search.run(bad, scoring));
        assertTrue(refused.toString().contains("text/layers/9/mlp/down"), refused.toString());
    }

    @Test
    void thePlannerFitsTinyArtifactsOnABigDeviceAndNotOnASmallOne() throws Exception {
        setUp();
        Allocation pure = new Allocation(Map.of());
        assertTrue(Feasibility.planner(sources, 1L << 40, 4096, 4).check(pure).fits());
        assertFalse(Feasibility.planner(sources, 1, 4096, 4).check(pure).fits());
    }
}
