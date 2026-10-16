package io.euhedral_execution.inference.quantization.mixed;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.quantization.Scorer;
import io.euhedral_execution.inference.quantization.mixed.SearchState.Measured;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.OptionalInt;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.function.LongFunction;

/// The search for the allocations on the quality and speed front of a set of calibrated sources, each candidate
/// scored by the engine against a reference.
///
/// A round fits an [Ensemble] of fidelity models to what has been measured, runs RCO ([Rco]) from several
/// starting points over a grid of cost weights and budgets, and picks a batch of candidates: most of them the
/// best the ensemble predicts, the rest where the ensemble is least sure. Each candidate is assembled, checked
/// against the residency plan at the target context and scored; the ensemble is refitted for the next round.
/// The search stops after several rounds without a new point on the [Pareto] front.
///
/// The optimization is Riemannian Constrained Optimization, Helcig and Alistarh, arXiv:2605.00649
/// (github.com/IST-DASLab/RCO).
///
/// The host work (assembling and checking the next candidate, fitting the next round's ensemble, running its
/// RCO) is done by ingest sinks on the lattice ([HostPool]) while the GPU scores; this thread starts it and
/// waits on its futures. Everything measured is kept in one [SearchState] file, so a stopped search resumes
/// without scoring a candidate twice.
public final class Search {

    /// Scores assembled candidates: the file their objects without a source of their own are read from, and the
    /// composite. A scoring that scores one candidate at a time implements [#score] alone; one that scores
    /// several at once ([#concurrent]) implements [#submit] too.
    public interface Scoring {
        Scorer.Result score(Path artifactPath, Artifact artifact) throws Exception;

        /// Starts scoring one candidate and returns its future. `started` is run once the candidate's scoring
        /// has begun (a candidate waiting for device memory has not begun). This default scores it before
        /// returning, on the calling thread.
        default CompletableFuture<Scorer.Result> submit(Path artifactPath, Artifact artifact, Runnable started) {
            started.run();
            try {
                return CompletableFuture.completedFuture(score(artifactPath, artifact));
            } catch (Exception failure) {
                return CompletableFuture.failedFuture(failure);
            }
        }

        /// Whether [#submit] returns before the candidate is scored.
        default boolean concurrent() {
            return false;
        }

        /// Runs the work a concurrent scoring leaves to the thread that waits (it starts and ends executions
        /// there, never on a lattice worker); nothing to do by default.
        default void service() throws Exception {}

        /// A future that completes when [#service] has work, replaced by each call of [#service].
        default CompletableFuture<Void> attention() {
            return new CompletableFuture<>();
        }
    }

    /// Everything a search is told.
    ///
    /// @param sources the calibrated artifacts and the base
    /// @param priors the proxy error of each option of each object
    /// @param costs the measured launch times
    /// @param context the target context the composite must fit with every object on the device
    /// @param state the resume file
    /// @param out the directory the front and the report are written to
    /// @param seeds allocations to measure first (objects they do not name keep the base's option)
    /// @param batch candidates per round
    /// @param explore the share of a batch proposed for what the ensemble is unsure of
    /// @param patience rounds without a new front point after which the search stops
    /// @param random feasible random mixes to measure first
    /// @param bands bands per family of the fidelity model
    /// @param verifyRows the rows of the verification step whose time is on the front
    /// @param freeBytes the device's free bytes, measured once at the start (a resumed search keeps its first)
    /// @param feasibility the fit check for those free bytes
    /// @param hostSinks sinks of each host pool (one pool per kind of host work)
    /// @param members models in the ensemble
    /// @param ridge the fidelity model's ridge
    /// @param lambdas weights of the normalized prefill and verification costs, in KL
    /// @param budgetFractions the budgets RCO is run at, as shares of the way from the cheapest allocation to
    /// the     largest that fits
    /// @param rco RCO's steps, learning rate, samples and restarts
    /// @param knapsackSteps the budget's resolution in the knapsack
    /// @param maxRounds the rounds after which the search stops whatever the front
    /// @param seed the seed of every random choice
    /// @param identity text that tells this search from another (references, cost table) for the resume file
    public record Settings(
            Sources sources,
            Priors priors,
            CostTable costs,
            int context,
            Path state,
            Path out,
            List<Allocation> seeds,
            int batch,
            double explore,
            int patience,
            int random,
            int bands,
            int verifyRows,
            long freeBytes,
            LongFunction<Feasibility> feasibility,
            int hostSinks,
            int members,
            double ridge,
            List<Double> lambdas,
            List<Double> budgetFractions,
            Rco.Settings rco,
            int knapsackSteps,
            int maxRounds,
            long seed,
            String identity) {}

    /// What a search ended with: everything measured, the front, the rounds run (not counting the seeds) and
    /// the bytes of projections RCO was budgeted.
    public record Outcome(List<Measured> measured, List<Measured> front, int rounds, long budget) {}

    /// A candidate proposed for scoring.
    record Candidate(Allocation allocation, String origin, double predicted, double spread, double value, int cell) {}

    private record Proposals(Ensemble ensemble, List<Candidate> candidates) {}

    private record Assembled(Artifact artifact, long bytes) {}

    private record Prepared(
            Candidate candidate, Artifact artifact, long bytes, double prefill, double decode, boolean fits) {}

    private static final ObjectMapper JSON = new ObjectMapper();
    private static final double EXPLORE_CAP = 0.6, EXPLORE_STEP = 0.1, WORSE = 1.1;
    private static final int MOVES_TRIED = 24, FIT_REPAIRS = 12;

    private final Settings s;
    private final Scoring scoring;
    /// One pool per kind of host work, so none queues behind another: proposals (ensemble fits, RCO samples,
    /// choosing a batch) and candidates (assembly and fit checks, the budget, the seeds). Each is attached to the
    /// lattice only while its work runs (this thread opens and closes them): idle attached sinks slow every
    /// worker's cycle, and with it the scorer's stage frames.
    private volatile HostPool proposing, candidates;
    /// Candidates another search measured, recorded before a new search's seeds.
    private List<Measured> imported = List.of();
    private final Space space;
    private final Priors priors;
    private SearchState state;
    private Feasibility feasibility;
    private long budget;
    private long seed;

    private Search(Settings settings, Scoring scoring) {
        this.s = settings;
        this.scoring = scoring;
        List<Option> options = new ArrayList<>(s.sources().options());
        options.retainAll(s.priors().options());
        options.sort(Comparator.naturalOrder());
        if (!options.contains(s.sources().base()))
            throw new IllegalArgumentException("the priors have no error for the base "
                    + s.sources().base().key() + " of every object");
        if (options.size() < 2)
            throw new IllegalArgumentException("a search needs two options with a source and a prior");
        this.space = new Space(s.sources(), options, s.costs(), s.verifyRows());
        this.priors = s.priors().restrict(space.objects, options);
    }

    /// The proposals pool, attached now if it is not.
    private HostPool proposing() {
        if (proposing == null) proposing = HostPool.start(s.hostSinks());
        return proposing;
    }

    /// The candidates pool, attached now if it is not.
    private HostPool candidates() {
        // Its few tasks run beside scoring: its sinks attach only while one is in flight.
        if (candidates == null) candidates = HostPool.startAttachingOnDemand(s.hostSinks());
        return candidates;
    }

    /// Detaches the proposals pool once its work is done.
    private void closeProposing() {
        if (proposing != null) {
            proposing.close();
            proposing = null;
        }
    }

    /// Detaches the candidates pool once its work is done.
    private void closeCandidates() {
        if (candidates != null) {
            candidates.close();
            candidates = null;
        }
    }

    /// Runs the search, resuming from `settings.state()` if it holds one.
    public static Outcome run(Settings settings, Scoring scoring) throws Exception {
        return run(settings, scoring, List.of());
    }

    /// As above; a new search first records `imported`, candidates another search measured, with their KL as
    /// measured and their size, costs and fit recomputed for this one's budget and context (its scores are the
    /// same bits: the reference and the scorer do not depend on them).
    public static Outcome run(Settings settings, Scoring scoring, List<Measured> imported) throws Exception {
        Search search = new Search(settings, scoring);
        search.imported = imported;
        try {
            return search.run();
        } finally {
            search.closeProposing();
            search.closeCandidates();
        }
    }

    private Outcome run() throws Exception {
        Files.createDirectories(s.out());
        try (SearchState opened = SearchState.open(s.state(), fingerprint(), s.seed(), s.freeBytes())) {
            state = opened;
            seed = state.seed();
            feasibility = s.feasibility().apply(state.freeBytes());
            budget = candidates().submit(this::derive).get();
            closeCandidates();
            System.out.printf(
                    Locale.ROOT,
                    "search: %d objects, %d options, projections %d..%d MiB, budget %d MiB, %d MiB free, %d measured%n",
                    space.objects.size(),
                    space.options.size(),
                    space.minimumBytes() >> 20,
                    space.maximumBytes() >> 20,
                    budget >> 20,
                    state.freeBytes() >> 20,
                    state.measured().size());
            if (Double.isNaN(state.explore())) state.explore(s.explore());

            CompletableFuture<Proposals> next = null;
            if (state.round() == 0 && state.measured().isEmpty() && !imported.isEmpty()) importMeasured();
            if (state.round() == 0) {
                List<Candidate> seeds =
                        candidates().submit(this::seedCandidates).get();
                next = measure(seeds, 0);
                state.round(1);
            }
            int rounds = 0;
            while (state.round() <= s.maxRounds() && state.stale() < s.patience()) {
                int round = state.round();
                Proposals proposals = (next != null ? next : propose(state.measured(), round)).get();
                next = null;
                int measuredHere = (int) state.measured().stream()
                        .filter(m -> m.round() == round)
                        .count();
                int need = s.batch() - measuredHere;
                List<Measured> data = state.measured();
                double share = state.explore();
                List<Candidate> chosen = need > 0
                        ? choose(proposing(), proposals, data, share, need, round)
                                .get()
                        : List.of();
                closeProposing();
                if (chosen.isEmpty() && need > 0)
                    System.out.printf(Locale.ROOT, "round %d: no candidate that was not measured%n", round);
                next = measure(chosen, round);
                finishRound(round, proposals.ensemble());
                rounds++;
            }
            writeOutputs();
            return new Outcome(state.measured(), Pareto.front(state.measured()), rounds, budget);
        }
    }

    /// The identity of this search for the resume file: what the measurements depend on.
    private String fingerprint() {
        StringBuilder text = new StringBuilder(s.identity()).append('\n');
        for (Option option : Option.values())
            if (s.sources().options().contains(option))
                text.append(option.key())
                        .append('=')
                        .append(s.sources().path(option))
                        .append('\n');
        text.append("base=").append(s.sources().base().key()).append('\n');
        text.append("bands=").append(s.bands()).append(" verifyRows=").append(s.verifyRows());
        text.append(" context=").append(s.context()).append(" objects=").append(space.objects.size());
        try {
            return HexFormat.of()
                    .formatHex(MessageDigest.getInstance("SHA-256")
                            .digest(text.toString().getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException failure) {
            throw new IllegalStateException(failure);
        }
    }

    /// The bytes of projections RCO may use: the largest allocation of a nested family (the first `k` objects
    /// on their largest option, the rest on their smallest) the planner accepts, found by bisection on `k`,
    /// plus the device bytes the plan leaves free there (capped below the next member of the family). A
    /// proposal is checked by the planner itself and, if it fails, re-solved with a smaller budget, so the
    /// figure only has to be
    /// near.
    private long derive() {
        int n = space.objects.size();
        if (!feasibility.check(family(0)).fits())
            throw new IllegalStateException("even the smallest allocation does not fit at a context of " + s.context()
                    + " tokens with " + (feasibility.freeBytes() >> 20) + " MiB free on the device");
        if (feasibility.check(family(n)).fits()) return space.bytesOf(familyChoice(n));
        int fits = 0, fails = n;
        while (fails - fits > 1) {
            int middle = (fits + fails) >>> 1;
            if (feasibility.check(family(middle)).fits()) fits = middle;
            else fails = middle;
        }
        long headroom = Math.max(
                0, feasibility.freeBytes() - feasibility.check(family(fits)).deviceBytes());
        return Math.min(space.bytesOf(familyChoice(fits)) + headroom, space.bytesOf(familyChoice(fails)));
    }

    private int[] familyChoice(int dearest) {
        int[] choice = new int[space.objects.size()];
        for (int i = 0; i < choice.length; i++) choice[i] = i < dearest ? space.dearest(i) : space.cheapest(i);
        return choice;
    }

    private Allocation family(int dearest) {
        return space.allocation(familyChoice(dearest));
    }

    // ------------------------------------------------------------------ seeds

    private List<Candidate> seedCandidates() {
        List<Candidate> seeds = new ArrayList<>();
        for (Allocation pure : space.pure()) seeds.add(new Candidate(pure, "seed:pure", Double.NaN, 0, 0, -1));
        for (Allocation given : s.seeds())
            seeds.add(new Candidate(space.normalize(given), "seed:file", Double.NaN, 0, 0, -1));
        Random random = new Random(seed);
        for (int i = 0; i < s.random(); i++) {
            Allocation mix = randomMix(random);
            if (mix != null) seeds.add(new Candidate(mix, "seed:random", Double.NaN, 0, 0, -1));
        }
        return seeds;
    }

    /// A random allocation within the budget that the planner accepts; null after repeated failures.
    private Allocation randomMix(Random random) {
        int k = space.options.size();
        for (int attempt = 0; attempt < 100; attempt++) {
            double[] weights = new double[k];
            double sum = 0;
            for (int j = 0; j < k; j++) sum += weights[j] = -Math.log(random.nextDouble() + 1e-300);
            int[] choice = new int[space.objects.size()];
            for (int i = 0; i < choice.length; i++) {
                double draw = random.nextDouble() * sum, at = 0;
                int picked = k - 1;
                for (int j = 0; j < k; j++) {
                    at += weights[j];
                    if (draw < at) {
                        picked = j;
                        break;
                    }
                }
                choice[i] = picked;
            }
            while (space.bytesOf(choice) > budget) {
                int i = random.nextInt(choice.length);
                choice[i] = space.cheapest(i);
            }
            Allocation allocation = space.allocation(choice);
            if (feasibility.check(allocation).fits()) return allocation;
        }
        return null;
    }

    // --------------------------------------------------------------- proposing

    /// An RCO sweep over the ensemble fitted to `data`: as futures on the proposals pool.
    private CompletableFuture<Proposals> propose(List<Measured> data, int round) {
        proposing();
        // Each member's fit is a frame of its own, then one frame joins them.
        return Ensemble.fit(data, s.members(), seed * 31 + round, s.ridge(), priors, s.bands(), proposing)
                .thenCompose(ensemble -> {
                    List<CompletableFuture<List<Candidate>>> runs = new ArrayList<>();
                    for (int lambda = 0; lambda < lambdaCells(); lambda++)
                        for (int fraction = 0; fraction < s.budgetFractions().size(); fraction++)
                            for (int restart = 0; restart < s.rco().restarts(); restart++) {
                                int l = lambda, f = fraction, r = restart;
                                runs.add(sweepCell(ensemble, round, l, f, r));
                            }
                    return CompletableFuture.allOf(runs.toArray(CompletableFuture[]::new))
                            .thenApply(done -> {
                                List<Candidate> all = new ArrayList<>();
                                Set<Allocation> seen = new HashSet<>();
                                for (var run : runs)
                                    for (Candidate candidate : run.join())
                                        if (seen.add(candidate.allocation())) all.add(candidate);
                                return new Proposals(ensemble, all);
                            });
                });
    }

    /// The sweep's lambda cells: every pair (prefill, decode) of the lambda list.
    private int lambdaCells() {
        return s.lambdas().size() * s.lambdas().size();
    }

    /// One RCO run of the sweep, its assignment made to fit the device.
    private CompletableFuture<List<Candidate>> sweepCell(
            Ensemble ensemble, int round, int lambdaIndex, int fractionIndex, int restart) {
        // The prefill and decode weights are swept independently: a cell is one pair of the lambda list.
        double lambdaPrefill = s.lambdas().get(lambdaIndex / s.lambdas().size());
        double lambdaDecode = s.lambdas().get(lambdaIndex % s.lambdas().size());
        long minimum = space.minimumBytes();
        long cellBudget = minimum + (long) (s.budgetFractions().get(fractionIndex) * Math.max(0, budget - minimum));
        long unit = Math.max(1, cellBudget / s.knapsackSteps());
        Rco.Objective objective = new Objective(ensemble, space, lambdaPrefill, lambdaDecode);
        long cellSeed = seed
                + 1_000_003L * round
                + 7919L * (lambdaIndex * s.budgetFractions().size() + fractionIndex);
        // Every sample of a step is a frame of its own: the pool holds many small frames for free workers.
        return proposing
                .submit(() -> new Rco.Run(objective, space.bytes, cellBudget, unit, s.rco(), cellSeed, restart))
                .thenCompose(this::advance)
                .thenCompose(result -> proposing.submit(() -> repaired(
                        ensemble,
                        objective,
                        result,
                        cellBudget,
                        unit,
                        minimum,
                        lambdaPrefill,
                        lambdaDecode,
                        fractionIndex,
                        lambdaIndex)));
    }

    /// `run` advanced to its end: each step's samples as frames of their own, all available at once, then the
    /// step's update as one more.
    private CompletableFuture<Rco.Result> advance(Rco.Run run) {
        CompletableFuture<?>[] samples = new CompletableFuture<?>[run.samples()];
        for (int i = 0; i < samples.length; i++) {
            int index = i;
            samples[i] = proposing.submit(() -> {
                run.sample(index);
                return null;
            });
        }
        return CompletableFuture.allOf(samples)
                .thenCompose(drawn -> proposing.submit(run::update))
                .thenCompose(done -> done ? proposing.submit(run::result) : advance(run));
    }

    /// The assignment of `result` made to fit the device, as a candidate (none when it cannot be made to fit).
    private List<Candidate> repaired(
            Ensemble ensemble,
            Rco.Objective objective,
            Rco.Result result,
            long cellBudget,
            long unit,
            long minimum,
            double lambdaPrefill,
            double lambdaDecode,
            int fractionIndex,
            int lambdaIndex) {
        int[] choice = result.choice();
        Feasibility.Check check = feasibility.check(space.allocation(choice));
        double[][] logP = new double[result.p().length][];
        for (int i = 0; i < logP.length; i++) {
            logP[i] = new double[result.p()[i].length];
            for (int j = 0; j < logP[i].length; j++) logP[i][j] = Math.log(Math.max(result.p()[i][j], 1e-300));
        }
        long limit = Math.min(space.bytesOf(choice), cellBudget);
        for (int attempt = 0; attempt < FIT_REPAIRS && !check.fits(); attempt++) {
            limit -= Math.max(check.deviceBytes() - feasibility.freeBytes(), unit);
            if (limit < minimum) return List.of();
            choice = Knapsack.solve(logP, space.bytes, limit, unit);
            check = feasibility.check(space.allocation(choice));
        }
        if (!check.fits()) return List.of();
        Allocation allocation = space.allocation(choice);
        double[][] hard = new double[choice.length][space.options.size()];
        for (int i = 0; i < choice.length; i++) hard[i][choice[i]] = 1;
        return List.of(new Candidate(
                allocation,
                String.format(
                        Locale.ROOT,
                        "rco:prefill=%g,decode=%g,budget=%g",
                        lambdaPrefill,
                        lambdaDecode,
                        s.budgetFractions().get(fractionIndex)),
                ensemble.mean(allocation),
                ensemble.spread(allocation),
                objective.value(hard),
                lambdaIndex));
    }

    /// The objective RCO minimizes: the ensemble's KL plus `lambdas` times the normalized prefill and
    /// verification costs, each the share of the way between the cheapest and the dearest allocation.
    private static final class Objective implements Rco.Objective {
        private final Ensemble ensemble;
        private final Space space;
        private final double lambdaPrefill, lambdaDecode;
        private final double prefillLow, prefillRange, decodeLow, decodeRange;

        Objective(Ensemble ensemble, Space space, double lambdaPrefill, double lambdaDecode) {
            this.ensemble = ensemble;
            this.space = space;
            this.lambdaPrefill = lambdaPrefill;
            this.lambdaDecode = lambdaDecode;
            double[] prefill = extent(space.prefill), decode = extent(space.decode);
            prefillLow = prefill[0];
            prefillRange = prefill[1] - prefill[0];
            decodeLow = decode[0];
            decodeRange = decode[1] - decode[0];
        }

        private static double[] extent(double[][] cost) {
            double low = 0, high = 0;
            for (double[] row : cost) {
                low += Arrays.stream(row).min().orElse(0);
                high += Arrays.stream(row).max().orElse(0);
            }
            return new double[] {low, high};
        }

        @Override
        public double value(double[][] p) {
            double value = ensemble.meanSoft(p);
            if (lambdaPrefill != 0 && prefillRange > 0)
                value += lambdaPrefill * (expected(space.prefill, p) - prefillLow) / prefillRange;
            if (lambdaDecode != 0 && decodeRange > 0)
                value += lambdaDecode * (expected(space.decode, p) - decodeLow) / decodeRange;
            return value;
        }

        @Override
        public double[][] gradient(double[][] p) {
            double[][] gradient = ensemble.gradient(p);
            for (int i = 0; i < gradient.length; i++)
                for (int k = 0; k < gradient[i].length; k++) {
                    if (lambdaPrefill != 0 && prefillRange > 0)
                        gradient[i][k] += lambdaPrefill * space.prefill[i][k] / prefillRange;
                    if (lambdaDecode != 0 && decodeRange > 0)
                        gradient[i][k] += lambdaDecode * space.decode[i][k] / decodeRange;
                }
            return gradient;
        }

        private static double expected(double[][] cost, double[][] p) {
            double sum = 0;
            for (int i = 0; i < p.length; i++) for (int k = 0; k < p[i].length; k++) sum += p[i][k] * cost[i][k];
            return sum;
        }
    }

    /// The batch: `1 - share` of it the best the ensemble predicts for each weight in turn, the rest
    /// alternately the feasible RCO results with the largest ensemble spread and the feasible single and
    /// double moves around the front with the largest spread. Nothing measured, nothing twice. The sorting of
    /// the proposals is one frame; each pick that needs the ensemble or the planner is frames of its own (a move
    /// tries each of its neighbours in a frame), and the picks follow one another, as the draws of one random
    /// stream do.
    private CompletableFuture<List<Candidate>> choose(
            HostPool pool, Proposals proposals, List<Measured> data, double share, int need, int round) {
        return pool.submit(() -> exploiting(proposals, data, share, need, round))
                .thenCompose(choosing -> exploring(pool, proposals, choosing, 0));
    }

    /// What the choice has so far.
    private static final class Choosing {
        final Set<Allocation> taken = new HashSet<>();
        final List<Candidate> chosen = new ArrayList<>();
        final List<Candidate> bySpread;
        final List<Measured> anchors;
        final Random random;
        final int exploiting, exploring, need;
        int spreadAt;

        Choosing(
                List<Candidate> bySpread,
                List<Measured> anchors,
                Random random,
                int exploiting,
                int exploring,
                int need) {
            this.bySpread = bySpread;
            this.anchors = anchors;
            this.random = random;
            this.exploiting = exploiting;
            this.exploring = exploring;
            this.need = need;
        }

        /// The next candidate by spread that is not taken, relabelled; null when there is none.
        Candidate bySpread() {
            while (spreadAt < bySpread.size()
                    && taken.contains(bySpread.get(spreadAt).allocation())) spreadAt++;
            return spreadAt < bySpread.size() ? relabel(bySpread.get(spreadAt++), "explore") : null;
        }

        void add(Candidate pick) {
            taken.add(pick.allocation());
            chosen.add(pick);
        }
    }

    /// The exploiting picks, and the ordering the exploring ones draw from.
    private Choosing exploiting(Proposals proposals, List<Measured> data, double share, int need, int round) {
        Set<Allocation> taken = new HashSet<>();
        for (Measured m : data) taken.add(m.a());
        int exploring = (int) Math.round(share * need);
        int exploiting = need - exploring;
        List<Candidate> chosen = new ArrayList<>();

        List<List<Candidate>> byLambda = new ArrayList<>();
        for (int l = 0; l < lambdaCells(); l++) byLambda.add(new ArrayList<>());
        for (Candidate c : proposals.candidates()) byLambda.get(c.cell()).add(c);
        for (var list : byLambda) list.sort(Comparator.comparingDouble(Candidate::value));
        int[] at = new int[byLambda.size()];
        for (boolean progressed = true; chosen.size() < exploiting && progressed; ) {
            progressed = false;
            for (int l = 0; l < byLambda.size() && chosen.size() < exploiting; l++) {
                List<Candidate> list = byLambda.get(l);
                while (at[l] < list.size() && taken.contains(list.get(at[l]).allocation())) at[l]++;
                if (at[l] == list.size()) continue;
                Candidate pick = list.get(at[l]++);
                taken.add(pick.allocation());
                chosen.add(relabel(pick, "exploit"));
                progressed = true;
            }
        }

        List<Candidate> bySpread = new ArrayList<>(proposals.candidates());
        bySpread.sort(Comparator.comparingDouble(Candidate::spread).reversed());
        Choosing choosing =
                new Choosing(bySpread, anchors(data), new Random(seed * 17 + round), exploiting, exploring, need);
        choosing.taken.addAll(taken);
        choosing.chosen.addAll(chosen);
        return choosing;
    }

    /// The exploring picks from iteration `e`, then, short of candidates (a small space, or most of it
    /// measured), moves around the front for the rest.
    private CompletableFuture<List<Candidate>> exploring(HostPool pool, Proposals proposals, Choosing choosing, int e) {
        if (choosing.chosen.size() >= choosing.exploiting + choosing.exploring)
            return filling(pool, proposals, choosing);
        Candidate first = e % 2 == 0 ? choosing.bySpread() : null;
        CompletableFuture<Candidate> picked =
                first != null ? CompletableFuture.completedFuture(first) : move(pool, proposals.ensemble(), choosing);
        return picked.thenCompose(pick -> {
            if (pick == null && e % 2 == 1) pick = choosing.bySpread();
            if (pick == null) return filling(pool, proposals, choosing);
            choosing.add(pick);
            return exploring(pool, proposals, choosing, e + 1);
        });
    }

    private CompletableFuture<List<Candidate>> filling(HostPool pool, Proposals proposals, Choosing choosing) {
        if (choosing.chosen.size() >= choosing.need) return CompletableFuture.completedFuture(choosing.chosen);
        return move(pool, proposals.ensemble(), choosing).thenCompose(pick -> {
            if (pick == null) return CompletableFuture.completedFuture(choosing.chosen);
            choosing.add(pick);
            return filling(pool, proposals, choosing);
        });
    }

    private static Candidate relabel(Candidate c, String kind) {
        return new Candidate(c.allocation(), kind + ":" + c.origin(), c.predicted(), c.spread(), c.value(), c.cell());
    }

    /// The front, or the feasible measured candidates if there is none yet, or all of them.
    private static List<Measured> anchors(List<Measured> data) {
        List<Measured> front = Pareto.front(data);
        if (!front.isEmpty()) return front;
        List<Measured> fitting = data.stream().filter(Measured::fits).toList();
        return fitting.isEmpty() ? data : fitting;
    }

    /// The feasible single or double move around a random anchor with the largest ensemble spread, among
    /// [#MOVES_TRIED] tried; null if none is new. One frame draws the moves, one frame per move checks it
    /// against the planner and the ensemble, and the best is taken in the order they were drawn.
    private CompletableFuture<Candidate> move(HostPool pool, Ensemble ensemble, Choosing choosing) {
        if (choosing.anchors.isEmpty()) return CompletableFuture.completedFuture(null);
        return pool.submit(() -> {
                    List<int[]> moves = new ArrayList<>();
                    for (int tried = 0; tried < MOVES_TRIED; tried++) {
                        int[] choice = space.choice(choosing.anchors
                                .get(choosing.random.nextInt(choosing.anchors.size()))
                                .a());
                        int changes = 1 + choosing.random.nextInt(2);
                        for (int c = 0; c < changes; c++) {
                            int i = choosing.random.nextInt(choice.length);
                            int other = choosing.random.nextInt(space.options.size() - 1);
                            choice[i] = other >= choice[i] ? other + 1 : other;
                        }
                        moves.add(choice);
                    }
                    return moves;
                })
                .thenCompose(moves -> {
                    List<CompletableFuture<Candidate>> tries = new ArrayList<>();
                    for (int[] choice : moves)
                        tries.add(pool.submit(() -> {
                            Allocation allocation = space.allocation(choice);
                            if (choosing.taken.contains(allocation)
                                    || !feasibility.check(allocation).fits()) return null;
                            return new Candidate(
                                    allocation,
                                    "explore:move",
                                    ensemble.mean(allocation),
                                    ensemble.spread(allocation),
                                    0,
                                    -1);
                        }));
                    return CompletableFuture.allOf(tries.toArray(CompletableFuture[]::new))
                            .thenApply(done -> {
                                Candidate best = null;
                                for (var future : tries) {
                                    Candidate found = future.join();
                                    if (found != null && (best == null || found.spread() > best.spread())) best = found;
                                }
                                return best;
                            });
                });
    }

    // --------------------------------------------------------------- measuring

    /// A candidate submitted for scoring.
    private record Submitted(int index, Prepared prepared, CompletableFuture<Scorer.Result> result) {}

    /// Scores the candidates of `batch` not measured yet: each is submitted as it is assembled and checked on the
    /// candidates pool (the next while the previous ones are scored), and recorded when its score is complete,
    /// whatever order they complete in. Once every one is recorded, the next round's proposals start on the
    /// proposals pool, and are returned as a future.
    private CompletableFuture<Proposals> measure(List<Candidate> batch, int round) throws Exception {
        List<Candidate> todo = new ArrayList<>();
        Set<Allocation> listed = new HashSet<>();
        for (Candidate candidate : batch)
            if (!state.contains(candidate.allocation()) && listed.add(candidate.allocation())) todo.add(candidate);
        List<Submitted> pending = new ArrayList<>();
        CompletableFuture<Prepared> prepared = todo.isEmpty() ? null : prepare(candidates(), todo.get(0));
        for (int k = 0; k < todo.size(); k++) {
            Prepared ready = prepared.get();
            if (k + 1 < todo.size()) {
                Candidate following = todo.get(k + 1);
                prepared = prepare(candidates(), following);
            }
            pending.add(new Submitted(
                    k, ready, scoring.submit(s.sources().path(s.sources().base()), ready.artifact(), () -> {})));
            record(pending, todo.size(), round);
        }
        while (!pending.isEmpty()) {
            scoring.service();
            record(pending, todo.size(), round);
            if (pending.isEmpty()) break;
            List<CompletableFuture<?>> waiting = new ArrayList<>();
            for (Submitted submitted : pending) waiting.add(submitted.result());
            waiting.add(scoring.attention());
            try {
                CompletableFuture.anyOf(waiting.toArray(new CompletableFuture<?>[0]))
                        .get();
            } catch (ExecutionException failed) {
                // A failed score is found, and thrown, when the finished are recorded.
            }
        }
        closeCandidates();
        // The next round's proposals start only now: RCO beside scoring slowed the candidate it overlapped
        // (81 s against 30 s) by more than the overlap saved.
        return round + 1 <= s.maxRounds() ? propose(state.measured(), round + 1) : null;
    }

    /// Records, in the order they finished, the submitted candidates whose scores are complete; a failed one stops
    /// the search with its failure.
    private void record(List<Submitted> pending, int total, int round) throws Exception {
        for (var it = pending.iterator(); it.hasNext(); ) {
            Submitted done = it.next();
            if (!done.result().isDone()) continue;
            it.remove();
            Scorer.Result result;
            try {
                result = done.result().get();
            } catch (ExecutionException failure) {
                Throwable cause = failure.getCause();
                if (cause instanceof Exception exception) throw exception;
                if (cause instanceof Error error) throw error;
                throw failure;
            }
            Prepared ready = done.prepared();
            Measured measured = new Measured(
                    ready.candidate().allocation(),
                    result.meanKl(),
                    result.klSe(),
                    ready.bytes(),
                    ready.prefill(),
                    ready.decode(),
                    ready.fits(),
                    round,
                    ready.candidate().origin());
            state.add(measured);
            System.out.printf(
                    Locale.ROOT,
                    "round %d, %d/%d: %s kl %.5f +- %.5f, %d MiB, prefill %.2f us/token, verify %.1f us, %s%n",
                    round,
                    done.index() + 1,
                    total,
                    measured.origin(),
                    measured.kl(),
                    measured.klSe(),
                    measured.bytes() >> 20,
                    measured.prefill(),
                    measured.decode(),
                    measured.fits() ? "fits" : "does not fit");
        }
    }

    /// A candidate assembled and checked, in frames of their own: its composite view (and the bytes of its
    /// tensors), the residency planner's verdict, and the cost table's two predictions.
    private CompletableFuture<Prepared> prepare(HostPool pool, Candidate candidate) {
        Allocation allocation = candidate.allocation();
        CompletableFuture<Assembled> view = pool.submit(() -> {
            Artifact artifact = Assemble.view(s.sources(), allocation, OptionalInt.of(s.verifyRows() - 1));
            long bytes = 0;
            for (var tensor : artifact.tensors()) bytes += tensor.byteSize();
            return new Assembled(artifact, bytes);
        });
        CompletableFuture<Boolean> fits =
                pool.submit(() -> feasibility.check(allocation).fits());
        CompletableFuture<Double> prefill = pool.submit(() -> s.costs().prefillMicrosPerToken(s.sources(), allocation));
        CompletableFuture<Double> decode =
                pool.submit(() -> s.costs().verifyMicros(s.sources(), allocation, s.verifyRows()));
        return CompletableFuture.allOf(view, fits, prefill, decode)
                .thenApply(done -> new Prepared(
                        candidate,
                        view.join().artifact(),
                        view.join().bytes(),
                        prefill.join(),
                        decode.join(),
                        fits.join()));
    }

    // ------------------------------------------------------------------ import

    /// Records the imported candidates: each one's measured KL, with size, costs and fit prepared for this search.
    private void importMeasured() throws Exception {
        List<CompletableFuture<Prepared>> preparing = new ArrayList<>();
        for (Measured m : imported)
            preparing.add(prepare(
                    candidates(), new Candidate(space.normalize(m.a()), "import:" + m.origin(), Double.NaN, 0, 0, -1)));
        int fitting = 0;
        for (int i = 0; i < preparing.size(); i++) {
            Prepared ready = preparing.get(i).get();
            Measured m = imported.get(i);
            if (state.add(new Measured(
                            ready.candidate().allocation(),
                            m.kl(),
                            m.klSe(),
                            ready.bytes(),
                            ready.prefill(),
                            ready.decode(),
                            ready.fits(),
                            0,
                            ready.candidate().origin()))
                    && ready.fits()) fitting++;
        }
        closeCandidates();
        System.out.printf(Locale.ROOT, "imported %d measured candidates, %d fit%n", imported.size(), fitting);
    }

    // ---------------------------------------------------------------- finishing

    /// The round's bookkeeping: the held-out error of the ensemble that proposed it on what it proposed, the
    /// explore share, whether the front gained a point, the round counter.
    private void finishRound(int round, Ensemble proposer) throws IOException {
        List<Measured> mine =
                state.measured().stream().filter(m -> m.round() == round).toList();
        double heldOut = proposer.heldOutError(mine);
        double previous = state.heldOut();
        if (!Double.isNaN(previous) && heldOut > WORSE * previous)
            state.explore(Math.min(EXPLORE_CAP, state.explore() + EXPLORE_STEP));
        state.heldOut(heldOut);
        boolean grew = Pareto.front(state.measured()).stream().anyMatch(m -> m.round() == round);
        state.stale(grew ? 0 : state.stale() + 1);
        state.round(round + 1);
        System.out.printf(
                Locale.ROOT,
                "round %d: %d measured, held-out error %.4f, explore %.1f, front %d points%s%n",
                round,
                mine.size(),
                heldOut,
                state.explore(),
                Pareto.front(state.measured()).size(),
                grew ? "" : " (no new point)");
        writeOutputs();
    }

    /// `front.json`, the front's allocations and the text report.
    private void writeOutputs() throws IOException {
        List<Measured> front = Pareto.front(state.measured());
        Path allocations = Files.createDirectories(s.out().resolve("allocations"));
        ObjectNode root = JSON.createObjectNode();
        ArrayNode points = root.putArray("front");
        StringBuilder report = new StringBuilder();
        report.append(String.format(
                Locale.ROOT,
                "mixed allocation search: %d measured, %d on the front, round %d, budget %d MiB, %d MiB free%n%n",
                state.measured().size(),
                front.size(),
                state.round(),
                budget >> 20,
                state.freeBytes() >> 20));
        report.append(
                String.format(Locale.ROOT, "      kl    +-se   MiB  prefill us/token  verify us  round  origin%n"));
        for (int i = 0; i < front.size(); i++) {
            Measured m = front.get(i);
            String file = String.format(Locale.ROOT, "front-%02d.json", i);
            m.a().write(allocations.resolve(file));
            ObjectNode point = points.addObject();
            point.put("kl", m.kl());
            point.put("klSe", m.klSe());
            point.put("bytes", m.bytes());
            point.put("prefillMicrosPerToken", m.prefill());
            point.put("verifyMicros", m.decode());
            point.put("round", m.round());
            point.put("origin", m.origin());
            point.put("allocation", "allocations/" + file);
            report.append(String.format(
                    Locale.ROOT,
                    "%8.5f %7.5f %5d %17.2f %10.1f %6d  %s%n",
                    m.kl(),
                    m.klSe(),
                    m.bytes() >> 20,
                    m.prefill(),
                    m.decode(),
                    m.round(),
                    m.origin()));
        }
        Files.writeString(
                s.out().resolve("front.json"),
                JSON.writerWithDefaultPrettyPrinter().writeValueAsString(root) + "\n");
        Files.writeString(s.out().resolve("report.txt"), report.toString());
    }
}
