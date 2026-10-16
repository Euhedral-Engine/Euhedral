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

    /// Scores one assembled candidate: the file its objects without a source of their own are read from, and
    /// the composite.
    public interface Scoring {
        Scorer.Result score(Path artifactPath, Artifact artifact) throws Exception;
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

    private record Prepared(
            Candidate candidate, Artifact artifact, long bytes, double prefill, double decode, boolean fits) {}

    private static final ObjectMapper JSON = new ObjectMapper();
    private static final double EXPLORE_CAP = 0.6, EXPLORE_STEP = 0.1, WORSE = 1.1;
    private static final int MOVES_TRIED = 24, FIT_REPAIRS = 12;

    private final Settings s;
    private final Scoring scoring;
    /// One pool per kind of host work, so none queues behind another: proposals (ensemble fits, RCO samples,
    /// choosing a batch) and candidates (assembly and fit checks, the budget, the seeds).
    private final HostPool proposing, candidates;
    private final Space space;
    private final Priors priors;
    private SearchState state;
    private Feasibility feasibility;
    private long budget;
    private long seed;

    private Search(Settings settings, Scoring scoring, HostPool proposing, HostPool candidates) {
        this.s = settings;
        this.scoring = scoring;
        this.proposing = proposing;
        this.candidates = candidates;
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

    /// Runs the search, resuming from `settings.state()` if it holds one.
    public static Outcome run(Settings settings, Scoring scoring) throws Exception {
        try (HostPool proposing = HostPool.start(settings.hostSinks());
                HostPool candidates = HostPool.start(settings.hostSinks())) {
            Search search = new Search(settings, scoring, proposing, candidates);
            return search.run();
        }
    }

    private Outcome run() throws Exception {
        Files.createDirectories(s.out());
        try (SearchState opened = SearchState.open(s.state(), fingerprint(), s.seed(), s.freeBytes())) {
            state = opened;
            seed = state.seed();
            feasibility = s.feasibility().apply(state.freeBytes());
            budget = candidates.submit(this::derive).get();
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
            if (state.round() == 0) {
                List<Candidate> seeds = candidates.submit(this::seedCandidates).get();
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
                        ? proposing
                                .submit(() -> choose(proposals, data, share, need, round))
                                .get()
                        : List.of();
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
        return proposing
                .submit(() -> Ensemble.fit(data, s.members(), seed * 31 + round, s.ridge(), priors, s.bands()))
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
        // One sample per host task: a lattice worker polling the pool never holds a whole run, so the engine's
        // frames on the same workers keep moving while the device scores.
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

    /// `run` advanced to its end, one sample per host task.
    private CompletableFuture<Rco.Result> advance(Rco.Run run) {
        return proposing.submit(run::advance).thenCompose(done -> done ? proposing.submit(run::result) : advance(run));
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
    /// double moves around the front with the largest spread. Nothing measured, nothing twice.
    private List<Candidate> choose(Proposals proposals, List<Measured> data, double share, int need, int round) {
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
        int spreadAt = 0;
        Random random = new Random(seed * 17 + round);
        List<Measured> anchors = anchors(data);
        for (int e = 0; chosen.size() < exploiting + exploring; e++) {
            Candidate pick = null;
            if (e % 2 == 0) {
                while (spreadAt < bySpread.size()
                        && taken.contains(bySpread.get(spreadAt).allocation())) spreadAt++;
                if (spreadAt < bySpread.size()) pick = relabel(bySpread.get(spreadAt++), "explore");
            }
            if (pick == null) pick = move(proposals.ensemble(), anchors, taken, random);
            if (pick == null && e % 2 == 1) {
                while (spreadAt < bySpread.size()
                        && taken.contains(bySpread.get(spreadAt).allocation())) spreadAt++;
                if (spreadAt < bySpread.size()) pick = relabel(bySpread.get(spreadAt++), "explore");
            }
            if (pick == null) break;
            taken.add(pick.allocation());
            chosen.add(pick);
        }
        // Short of candidates (a small space, or most of it measured): moves around the front make up the rest.
        while (chosen.size() < need) {
            Candidate pick = move(proposals.ensemble(), anchors, taken, random);
            if (pick == null) break;
            taken.add(pick.allocation());
            chosen.add(pick);
        }
        return chosen;
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
    /// [#MOVES_TRIED] tried; null if none is new.
    private Candidate move(Ensemble ensemble, List<Measured> anchors, Set<Allocation> taken, Random random) {
        if (anchors.isEmpty()) return null;
        Candidate best = null;
        for (int tried = 0; tried < MOVES_TRIED; tried++) {
            int[] choice =
                    space.choice(anchors.get(random.nextInt(anchors.size())).a());
            int changes = 1 + random.nextInt(2);
            for (int c = 0; c < changes; c++) {
                int i = random.nextInt(choice.length);
                int other = random.nextInt(space.options.size() - 1);
                choice[i] = other >= choice[i] ? other + 1 : other;
            }
            Allocation allocation = space.allocation(choice);
            if (taken.contains(allocation) || !feasibility.check(allocation).fits()) continue;
            double spread = ensemble.spread(allocation);
            if (best == null || spread > best.spread())
                best = new Candidate(allocation, "explore:move", ensemble.mean(allocation), spread, 0, -1);
        }
        return best;
    }

    // --------------------------------------------------------------- measuring

    /// Scores the candidates of `batch` not measured yet, in order. While one is scored the next is assembled and
    /// checked on the candidates pool. When the last one starts scoring, the next round's proposals start on the
    /// proposals pool (from what is measured before it), and are returned as a future.
    private CompletableFuture<Proposals> measure(List<Candidate> batch, int round) throws Exception {
        List<Candidate> todo = new ArrayList<>();
        Set<Allocation> listed = new HashSet<>();
        for (Candidate candidate : batch)
            if (!state.contains(candidate.allocation()) && listed.add(candidate.allocation())) todo.add(candidate);
        CompletableFuture<Proposals> next = null;
        CompletableFuture<Prepared> prepared = todo.isEmpty() ? null : candidates.submit(() -> prepare(todo.get(0)));
        for (int k = 0; k < todo.size(); k++) {
            Prepared ready = prepared.get();
            if (k + 1 < todo.size()) {
                Candidate following = todo.get(k + 1);
                prepared = candidates.submit(() -> prepare(following));
            }
            if (k == todo.size() - 1 && round + 1 <= s.maxRounds()) next = propose(state.measured(), round + 1);
            Scorer.Result result = scoring.score(s.sources().path(s.sources().base()), ready.artifact());
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
                    k + 1,
                    todo.size(),
                    measured.origin(),
                    measured.kl(),
                    measured.klSe(),
                    measured.bytes() >> 20,
                    measured.prefill(),
                    measured.decode(),
                    measured.fits() ? "fits" : "does not fit");
        }
        if (next == null && round + 1 <= s.maxRounds()) next = propose(state.measured(), round + 1);
        return next;
    }

    private Prepared prepare(Candidate candidate) {
        Allocation allocation = candidate.allocation();
        Artifact artifact = Assemble.view(s.sources(), allocation, OptionalInt.of(s.verifyRows() - 1));
        long bytes = 0;
        for (var tensor : artifact.tensors()) bytes += tensor.byteSize();
        return new Prepared(
                candidate,
                artifact,
                bytes,
                s.costs().prefillMicrosPerToken(s.sources(), allocation),
                s.costs().verifyMicros(s.sources(), allocation, s.verifyRows()),
                feasibility.check(allocation).fits());
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
